"""Optional wgpu rasterizer, adapted from tools/spike_3d_backends.py.

RGBA uses float32 when float32-blendable is available, otherwise half precision.
Data outputs always use float32. Textures are half precision with CPU-generated
mips and a trilinear sampler; explicit per-triangle LOD matches the reference's
rounded area heuristic rather than hardware derivative-based LOD.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import threading
from dataclasses import replace

import numpy as np

from . import gpupicture, raytrace, scene3d
from .playback_trace import mark as _mark

log = logging.getLogger(__name__)


class Unsupported(Exception):
    """The caller should render this scene with the CPU backend."""


_lock = threading.RLock()
_states = {}
_adapter_logged = []
_errors = {}

# Calibration (2026-09-19). The first budgets were derived from whole-render times, which include large
# host-side costs (per-triangle Python preparation: ~270 ms at 10k triangles, ~1.1 s at 40k), so they
# understated GPU shadow throughput. Re-measured by subtracting a no-shadow render of the same scene
# (tools: see TASKLOG), 960x540, brute-force loop, one directional light:
#   RTX 3080 Ti   ~40-300e9 pair tests/s (noisy, 40k tris: 209e9)   Radeon 8060S  ~19-60e9/s (40k tris: 19e9)
#   llvmpipe      ~0.4-0.5e9/s
# BVH traversal in the fragment shader (units: rays x 16 x log2(triangles+2), as on the CPU):
#   RTX 3080 Ti   ~0.4-1.4e9 units/s   Radeon 8060S ~0.45-1.6e9/s   llvmpipe ~0.01-0.2e9/s
# Re-measured 10/22 on the furnished room (one point light, 1920 x 1080, GPU time from submit to done, RTX 3080 Ti /
# Radeon 8060S): 264 triangles 15.1 / 4.9 ms brute, 12.2 / 3.9 ms BVH; 1,080 triangles 26.8 / 18.3 brute, 12.2 / 5.7 BVH;
# 4,440 triangles 86 / 80 brute, 7.5 / 5.6 BVH; 19,624 triangles 214 / 184 brute, 9.3 / 15.5 BVH. The BVH wins from about
# 300 triangles on both cards (the 9/19 calibration above found the opposite on a different shader and scene).
# Budgets bound one submission to about a second on each adapter type (display-driver timeouts; a
# submitted job cannot be cancelled); 'other' is a guess. Windows and other adapters are unmeasured.
SHADOW_WORK_BUDGETS = {'discrete': 4e10, 'integrated': 1e10, 'cpu': 3e8, 'other': 2e9}
# Re-measured 10/22: a 1080p furnished room is 4.0e8 BVH units and took 6.4 ms on the RTX 3080 Ti and 5.6 ms on the Radeon 8060S
# (6e10 and 7e10 units a second, against the 0.4-1.4e9 above), so the old 4e8 budget cut every 1080p frame into two
# submissions. 2e9 is about 30 ms of that work on either card.
SHADOW_BVH_WORK_BUDGETS = {'discrete': 2e9, 'integrated': 2e9, 'cpu': 1e7, 'other': 5e8}
# Triangle count above which the BVH path is chosen (when the adapter is capable). None = per-adapter table.
SHADOW_BVH_THRESHOLDS = {'discrete': 300, 'integrated': 300, 'cpu': 500, 'other': 1000}
SHADOW_BVH_THRESHOLD = None
SHADOW_BVH_STACK_SIZE = 64
GPU_MAX_BANDS = 64
GPU_FORCE_BANDS = None
last_shadow_path = 'brute'


def _bvh_capable(limits, node_bytes, order_bytes):
    # Fragment storage: lights, triangles, nodes, primitive order.
    return (limits.get('max-storage-buffers-per-shader-stage', 0) >= 4
            and max(node_bytes, order_bytes) <= limits.get('max-storage-buffer-binding-size', 0))


def _pack_bvh(bvh, cancel=None):
    pending = [(0, 1)] if len(bvh.left) else []
    while pending:
        _cancel(cancel)
        node, depth = pending.pop()
        if depth > SHADOW_BVH_STACK_SIZE:
            raise ValueError(f'Shadow BVH depth {depth} exceeds traversal stack {SHADOW_BVH_STACK_SIZE}')
        if bvh.prim_count[node] == 0:
            pending.extend(((int(bvh.left[node]), depth+1), (int(bvh.right[node]), depth+1)))
    dtype = np.dtype([('lo', '<f4', 3), ('left', '<i4'), ('hi', '<f4', 3),
                      ('right', '<i4'), ('offset', '<u4'), ('count', '<u4'), ('pad', '<u4', 2)])
    assert dtype.itemsize == 48
    nodes = np.zeros(len(bvh.left), dtype=dtype)
    nodes['lo'] = np.nextafter(bvh.node_lo.astype('f4'), np.float32(-np.inf))
    nodes['hi'] = np.nextafter(bvh.node_hi.astype('f4'), np.float32(np.inf))
    for name, value in [('left', bvh.left), ('right', bvh.right),
                        ('offset', bvh.prim_offset), ('count', bvh.prim_count)]:
        nodes[name] = value
    return nodes, bvh.prim_order.astype('u4')


def _adapter_kind(state):
    normalized = str(state['info'].get('adapter_type', 'unknown')).lower().replace('_', '').replace(' ', '')
    return {'discretegpu': 'discrete', 'integratedgpu': 'integrated', 'cpu': 'cpu'}.get(normalized, 'other')


def _shadow_budget(state, work, path='brute'):
    reported = str(state['info'].get('adapter_type', 'unknown'))
    kind = _adapter_kind(state)
    budget = (SHADOW_BVH_WORK_BUDGETS if path == 'bvh' else SHADOW_WORK_BUDGETS)[kind]
    if work > budget:
        raise ValueError(f'Shadow rays exceed the GPU budget: adapter {reported} ({kind}), '
                         f'{work:,.0f} > {budget:,.0f} {path} work units; '
                         'reduce resolution/samples/triangles or switch shadows off')
    return budget


def _band_plan(state, work, path, height):
    """Return contiguous (start, stop) row bands within the submission budget."""
    if height < 1:
        raise ValueError('Render dimensions must be positive')
    budget = _shadow_budget(state, 0, path)
    required = max(1, math.ceil(work / budget)) if budget > 0 else (1 if work == 0 else math.inf)
    cap = min(height, GPU_MAX_BANDS)
    if required > cap:
        reported = state['info'].get('adapter_type', 'unknown')
        raise ValueError(f'Shadow rays exceed the GPU budget: adapter {reported} ({_adapter_kind(state)}), '
                         f'{work:,.0f} total {path} work units, {budget:,.0f} per-submission budget; '
                         f'even split into {GPU_MAX_BANDS} bands (band cap, at most {height} rows); '
                         'reduce resolution/samples/triangles or switch shadows off')
    bands = required
    if GPU_FORCE_BANDS is not None:
        if not isinstance(GPU_FORCE_BANDS, int) or not 1 <= GPU_FORCE_BANDS <= GPU_MAX_BANDS:
            raise ValueError(f'GPU_FORCE_BANDS must be an integer from 1 to {GPU_MAX_BANDS}')
        bands = min(height, GPU_FORCE_BANDS)
    _shadow_budget(state, work / bands, path)
    return [(i * height // bands, (i + 1) * height // bands) for i in range(bands)]


ADAPTER_ENV = 'NODEBASED_ADAPTER'
ADAPTER_SETTING = 'interface/graphics_adapter'
_KIND_RANK = {'discretegpu': 0, 'integratedgpu': 1, 'virtualgpu': 2, 'cpu': 3}
_KIND_WORDS = {'discretegpu': 'discrete', 'integratedgpu': 'integrated', 'virtualgpu': 'virtual', 'cpu': 'software'}
_adapter_cache = {}


def _kind_of(info):
    return str(info.get('adapter_type', '')).lower().replace('_', '').replace(' ', '')


def _name_of(info):
    return str(info.get('device') or info.get('description') or 'unnamed adapter')


def _capacity(adapter):
    limits = getattr(adapter, 'limits', None) or {}
    return (limits.get('max-buffer-size', 0) or 0, limits.get('max-storage-buffer-binding-size', 0) or 0)


def rank_adapters(adapters):
    """Prefer discrete GPUs, then integrated, virtual and software adapters. This is a
    startup heuristic, not a measured speed ranking; workloads and eGPU links can reverse it."""
    return sorted(adapters, key=lambda a: (_KIND_RANK.get(_kind_of(a.info), 4),
                                           str(a.info.get('backend_type', '')).lower() == 'opengl',
                                           tuple(-v for v in _capacity(a))))


def adapter_preference():
    """What the artist asked for: NODEBASED_ADAPTER (a part of an adapter's name), else the Graphics adapter
    setting; '' means Automatic."""
    value = os.environ.get(ADAPTER_ENV, '').strip()
    if value:
        return value
    try:
        from PySide6.QtCore import QSettings
        stored = QSettings('NodeBased', 'NodeBased').value(ADAPTER_SETTING, '')
        return str(stored or '').strip()
    except Exception:
        return ''


def select_adapter(adapters, preference=''):
    """The adapter the app runs on: the one the preference names (case-insensitive part of its name), else the
    best ranked. None when there is nothing to choose from."""
    ranked = rank_adapters(adapters)
    wanted = (preference or '').strip().lower()
    if wanted and wanted != 'automatic':
        for adapter in ranked:
            if wanted in _name_of(adapter.info).lower():
                return adapter
        log.warning('graphics adapter %r matches no adapter on this machine; choosing automatically', preference)
    return ranked[0] if ranked else None


def _summary(adapter):
    info = adapter.info
    return dict(name=_name_of(info), type=_KIND_WORDS.get(_kind_of(info), str(info.get('adapter_type', 'unknown'))),
                backend=str(info.get('backend_type', 'unknown')), driver=str(info.get('description') or 'unknown'))


def enumerate_adapters():
    """Every adapter wgpu can see, best first, as plain dicts (name, type, backend, driver); cached. [] without wgpu."""
    if 'list' not in _adapter_cache:
        try:
            import wgpu
            _adapter_cache['list'] = [_summary(a) for a in rank_adapters(wgpu.gpu.enumerate_adapters_sync())]
        except Exception:
            _adapter_cache['list'] = []
    return _adapter_cache['list']


def adapter_warning(chosen_name, adapters=None):
    """Suggest an available discrete adapter when the current one is integrated or software."""
    adapters = enumerate_adapters() if adapters is None else adapters
    chosen = next((a for a in adapters if a['name'] == chosen_name), None)
    if chosen is None or chosen['type'] not in ('integrated', 'software'):
        return None
    if not any(a['type'] == 'discrete' for a in adapters):
        return None
    return f"{chosen_name}: a discrete adapter is available, Settings \u2192 Graphics adapter"


def adapter_preference_source():
    """Where the preference named by `adapter_preference` comes from: 'environment', 'settings' or '' (Automatic)."""
    if os.environ.get(ADAPTER_ENV, '').strip():
        return 'environment'
    return 'settings' if adapter_preference() else ''


def adapter_choice(chosen_name=None, adapters=None, preference=None, source=None):
    """Which adapter the app runs on and why, in words an artist can read: a dict of `name`, `type` and `reason`.
    On a hybrid laptop that lists a discrete card next to the integrated one, Automatic picks the discrete card;
    when only the integrated card is listed the reason says so (Windows hides the discrete one from an application
    it has not been told to give it to)."""
    adapters = enumerate_adapters() if adapters is None else adapters
    name = chosen_name if chosen_name is not None else chosen_adapter_name()
    chosen = next((a for a in adapters if a['name'] == name), None)
    kind = chosen['type'] if chosen else ''
    preference = adapter_preference() if preference is None else preference
    source = (adapter_preference_source() if source is None else source) if preference else ''
    wanted = (preference or '').strip().lower()
    others = []         # the other graphics cards; the same card listed again through OpenGL ('<name>/PCIe/SSE2') is not one
    for a in adapters:
        if a['type'] in ('discrete', 'integrated') and not (a['name'].startswith(name) or name.startswith(a['name'])) \
                and a['name'] not in others:
            others.append(a['name'])
    if wanted and wanted != 'automatic' and wanted in name.lower():
        where = 'the NODEBASED_ADAPTER setting' if source == 'environment' else 'Settings \u2192 Graphics adapter'
        reason = f'chosen in {where}'
    elif wanted and wanted != 'automatic':
        reason = f'{preference!r} matches no adapter on this machine, so the best one was chosen automatically'
    elif not chosen:
        reason = 'the graphics driver chose it'
    elif kind == 'discrete':
        reason = 'Automatic: a discrete graphics card is preferred' + (f' over {", ".join(others)}' if others else '')
    elif kind == 'integrated':
        reason = 'Automatic: the only graphics card this application can see; a discrete card is not listed'
    else:
        reason = 'Automatic: no graphics card is available, so the processor draws'
    return dict(name=name, type=kind, reason=reason)


ADAPTER_WARNING_HELP = ('A discrete graphics card is installed but this application is not using it. Its speed depends '
                        'on the workload and connection. Pick it in Settings '
                        '\u2192 Graphics adapter (takes effect after a restart). If the card is not listed, Windows is hiding '
                        'it from the application: open Settings \u2192 System \u2192 Display \u2192 Graphics, add NodeBased '
                        'and set it to High performance.')


def _log_adapters(adapters, chosen):
    lines = [f"  {'*' if a['name'] == chosen else ' '} {a['name']} ({a['type']}, {a['backend']}, driver {a['driver']})"
             for a in adapters]
    log.info('graphics adapters, chosen marked:\n%s', '\n'.join(lines))


def _state(choice=None):
    choice = choice or 'default'
    if choice not in ('default', 'discrete', 'integrated', 'cpu'):
        raise ValueError(f'Unknown wgpu adapter {choice!r}')
    with _lock:
        if choice in _states:
            return _states[choice]
        if choice in _errors:
            raise RuntimeError(_errors[choice])
        try:
            import wgpu
            if choice == 'default':
                try:
                    adapter = select_adapter(list(wgpu.gpu.enumerate_adapters_sync()), adapter_preference())
                except Exception as exc:       # an enumeration that fails is the same as one that finds nothing
                    log.warning('could not enumerate graphics adapters (%s); asking for high performance', exc)
                    adapter = None
                if adapter is None:
                    adapter = wgpu.gpu.request_adapter_sync(power_preference='high-performance')
                if not _adapter_logged and enumerate_adapters():
                    _adapter_logged.append(True)
                    _log_adapters(enumerate_adapters(), _name_of(adapter.info))
            else:
                target = {'discrete': 'discretegpu', 'integrated': 'integratedgpu', 'cpu': 'cpu'}[choice]
                adapter = next((a for a in wgpu.gpu.enumerate_adapters_sync()
                                if str(a.info.get('adapter_type', '')).lower().replace('_', '').replace(' ', '') == target), None)
            if adapter is None:
                raise RuntimeError(f'no {choice} adapter')
            storage_buffers = adapter.limits.get('max-storage-buffers-per-shader-stage', 2)
            if storage_buffers < 2:
                raise RuntimeError('adapter allows fewer than 2 storage buffers per shader stage')
            features = ['float32-blendable'] if 'float32-blendable' in adapter.features else []
            device = adapter.request_device_sync(required_features=features, required_limits={
                'max-storage-buffer-binding-size': adapter.limits['max-storage-buffer-binding-size'],
                'max-storage-buffers-per-shader-stage': min(8, storage_buffers)})
            from . import gpumemory
            memory = gpumemory.instrument(device)
            # Data outputs render to float32; downlevel adapters (GL/GLES class) reject that attachment.
            try:
                device.create_texture(size=(1, 1, 1), format='rgba32float',
                    usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC).destroy()
            except Exception as exc:
                raise RuntimeError(f'adapter cannot render to rgba32float (downlevel): {exc}') from exc
            state = dict(wgpu=wgpu, device=device, info=dict(adapter.info), pipelines={},
                         format='rgba32float' if features else 'rgba16float',
                         features=sorted(adapter.features), memory=memory)
            _states[choice] = state
            return state
        except Exception as exc:
            reason = f'wgpu unavailable ({choice}): {type(exc).__name__}: {exc}'
            _errors[choice] = reason
            raise RuntimeError(reason) from exc


def available():
    """Cached adapter/device probe; safe when the optional dependency is absent."""
    try:
        _state()
        return True
    except Exception:
        return False


def chosen_adapter_name():
    """The adapter the app runs on ('' before the first device exists or when there is none)."""
    state = _states.get('default')
    if state:
        return _name_of(state['info'])
    adapters = enumerate_adapters()      # no device yet: name the adapter the first device will use
    if not adapters:
        return ''
    wanted = adapter_preference().lower()
    match = next((a for a in adapters if wanted and wanted != 'automatic' and wanted in a['name'].lower()), None)
    return (match or adapters[0])['name']


def describe():
    try:
        state = _state()
        info = state['info']
        return f"{info.get('device', info.get('description', 'wgpu adapter'))} / {info.get('backend_type', 'unknown')} ({state['format']})"
    except Exception as exc:
        return str(exc)


def adapter_report(choice=None):
    """Multi-line adapter, backend and precision summary for test and CI logs."""
    try:
        state = _state(choice)
    except Exception as exc:
        return f'wgpu adapter: none ({exc})'
    info, limits = state['info'], state['device'].limits
    try:
        import wgpu
        version = wgpu.__version__
    except Exception:
        version = 'unknown'
    blendable = 'float32-blendable' in state.get('features', ())
    return '\n'.join((
        f"wgpu adapter: {info.get('device') or info.get('description') or 'unnamed'}",
        f"  type {info.get('adapter_type', 'unknown')}, backend {info.get('backend_type', 'unknown')}, "
        f"vendor {info.get('vendor') or info.get('vendor_id', 'unknown')}, driver {info.get('description') or 'unknown'}, wgpu {version}",
        f"  rgba target {state['format']} (float32-blendable {'present' if blendable else 'MISSING: beauty and splat layers blend in half precision'})",
        f"  storage buffers per stage {limits.get('max-storage-buffers-per-shader-stage')}, "
        f"binding size {limits.get('max-storage-buffer-binding-size')}, max texture {limits.get('max-texture-dimension-2d')}"))


_SHADER = '''
struct Params { projection: vec4<f32>, eye: vec4<f32>, settings: vec4<f32>, shadow: vec4<f32>, view0: vec4<f32>, view1: vec4<f32>, view2: vec4<f32> };
struct Light { position: vec4<f32>, direction: vec4<f32>, colour: vec4<f32>, cone: vec4<f32>, shadow: vec4<f32> };  // colour.w = falloff power; shadow = (bias scale, tan blur, samples, 0)
var<private> near_bias: f32 = 0.0;
// Light linking: a triangle's `e1.w` is the bit mask of the lights its mesh excludes, a fragment's `scales.z` the mask of
// the lights its mesh excludes (light i is bit i: the table's lights, then the area lights, then the environment).
// `shadow_bit` is the light a shadow ray goes to; a triangle that excludes it casts no shadow from it.
var<private> shadow_bit: u32 = 99u;
// Per-object render switches: a triangle whose mesh casts no shadow carries every bit in `e1.w` (see `NO_CAST_MASK`); a
// fragment whose mesh takes no shadow has `scales.w` set, and every shadow ray it would trace returns "unblocked".
var<private> shadow_off: bool = false;
fn excl_bit(mask: u32, bit: u32) -> bool { return bit < 24u && ((mask >> bit) & 1u) != 0u; }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> lights: array<Light>;
@group(0) @binding(2) var tex: texture_2d<f32>;
@group(0) @binding(3) var filtering: sampler;
struct Triangle { v0: vec4<f32>, e1: vec4<f32>, e2: vec4<f32> };
@group(0) @binding(4) var<storage, read> triangles: array<Triangle>;
// Y3 of 3, part 1: the four remaining PBR texture maps, matching `scene3d._shade_fragments`. Each is
// a single (top) level, sampled with the same `filtering` sampler as the mip-chained base colour
// texture; a geometry missing a map still binds a harmless 1x1 dummy so every pipeline variant
// compiles and draws alike, but its `maps` flag (packed in `_prepare`) keeps the fragment shader from
// ever sampling it.
@group(0) @binding(7) var mr_tex: texture_2d<f32>;
@group(0) @binding(8) var normal_tex: texture_2d<f32>;
@group(0) @binding(9) var occlusion_tex: texture_2d<f32>;
@group(0) @binding(10) var emissive_tex: texture_2d<f32>;
// Z1 of 2 finish: Rect/Disc/Sphere area lights, matching `scene3d._area_light_contribution`'s fixed
// low-discrepancy sample set exactly -- `_area_light_resources` computes the same sample points and
// normals on the host (`scene3d._area_light_samples`) and uploads them once, reused by every
// fragment, the way the CPU reference reuses them across every shading point. `center.w` is the
// two-sided flag; `radiance_area.w` is the light's surface area; `info` is (sample offset, sample
// count, shadow bias scale matching `light.shadow.x` above, shadows-on flag). A scene with no area
// lights still binds a harmless one-row dummy so every pipeline variant compiles alike.
struct AreaLight { center: vec4<f32>, radiance_area: vec4<f32>, info: vec4<f32> };
struct AreaSample { point: vec4<f32>, normal: vec4<f32> };
@group(0) @binding(11) var<storage, read> area_lights: array<AreaLight>;
@group(0) @binding(12) var<storage, read> area_samples: array<AreaSample>;
// A single environment light on meshes (docs/3D_FOUNDATION.md "Left out" of Y1 of 2 finish (1)):
// image-based diffuse (9 SH coefficients, matching envlight._sh_irradiance) and specular (a six-tile
// vertical atlas, one GGX-roughness level per tile from envlight.LEVEL_ROUGHNESS, sampled with a
// manual two-tile lerp, matching envlight.Prefiltered.lookup). `rot0`/`rot1`/`rot2` are the columns of
// envlight.Environment._local's combined rotation (the environment's own turn and its parent's,
// unlike viewportgpu.py's dome which assumes a root-level environment) so a parented or rotated
// Environment still matches the CPU reference. `enabled.x` is 0 with no environment bound (a scene
// with more than one Environment, or the ray-traced render mode, still refuses on the CPU: group(2) is
// always bound so the shader compiles either way, but its atlas is a harmless 1-texel-wide default then).
struct EnvGlobals { sh: array<vec4<f32>, 9>, rot0: vec4<f32>, rot1: vec4<f32>, rot2: vec4<f32>, enabled: vec4<f32> };
@group(2) @binding(0) var<uniform> envg: EnvGlobals;
@group(2) @binding(1) var env_tex: texture_2d<f32>;
@group(2) @binding(2) var env_sampler: sampler;
fn env_local(d: vec3<f32>) -> vec3<f32> {
    return mat3x3<f32>(envg.rot0.xyz, envg.rot1.xyz, envg.rot2.xyz) * d;
}
fn env_uv(d: vec3<f32>) -> vec2<f32> {
    let u = 0.5 + atan2(d.x, -d.z) / (2.0 * 3.14159265);
    let v = acos(clamp(d.y, -1.0, 1.0)) / 3.14159265;
    return vec2<f32>(u, v);
}
// envlight._sh_irradiance: cosine-convolved radiance, folded with the environment's gain already.
fn env_diffuse(n: vec3<f32>) -> vec3<f32> {
    if (envg.enabled.x < 0.5) { return vec3<f32>(0.0); }
    let d = env_local(n);
    let x = d.x; let y = d.y; let z = d.z;
    var total = envg.sh[0].rgb * 0.282095;
    total += envg.sh[1].rgb * (0.488603 * y) * (2.0 / 3.0);
    total += envg.sh[2].rgb * (0.488603 * z) * (2.0 / 3.0);
    total += envg.sh[3].rgb * (0.488603 * x) * (2.0 / 3.0);
    total += envg.sh[4].rgb * (1.092548 * x * y) * 0.25;
    total += envg.sh[5].rgb * (1.092548 * y * z) * 0.25;
    total += envg.sh[6].rgb * (0.315392 * (3.0 * z * z - 1.0)) * 0.25;
    total += envg.sh[7].rgb * (1.092548 * x * z) * 0.25;
    total += envg.sh[8].rgb * (0.546274 * (x * x - y * y)) * 0.25;
    return total;
}
// envlight.Prefiltered.lookup: a two-tile lerp over the atlas's six GGX-roughness levels. Each
// `textureSampleLevel` call is inset by half a texel from its tile's own top/bottom row so hardware
// bilinear filtering cannot blend in the next tile's edge (a different roughness level) the way
// `envlight.sample_map`'s per-level clamp never would -- otherwise a reflection direction near the
// map's pole (v near 0 or 1) picks up a sliver of the wrong level.
fn env_specular(direction: vec3<f32>, roughness: f32) -> vec3<f32> {
    if (envg.enabled.x < 0.5) { return vec3<f32>(0.0); }
    let uv = env_uv(env_local(direction));
    let pos = clamp(roughness, 0.0, 1.0) * 5.0;
    let lo = floor(pos);
    let hi = min(lo + 1.0, 5.0);
    let half_texel = 0.5 / f32(textureDimensions(env_tex).y);
    let row_lo = clamp((lo + uv.y) / 6.0, lo / 6.0 + half_texel, (lo + 1.0) / 6.0 - half_texel);
    let row_hi = clamp((hi + uv.y) / 6.0, hi / 6.0 + half_texel, (hi + 1.0) / 6.0 - half_texel);
    let a = textureSampleLevel(env_tex, env_sampler, vec2<f32>(uv.x, row_lo), 0.0).rgb;
    let b = textureSampleLevel(env_tex, env_sampler, vec2<f32>(uv.x, row_hi), 0.0).rgb;
    return mix(a, b, pos - lo);
}
fn triangle_transmission(index: u32, origin: vec3<f32>, ray: vec3<f32>, limit: f32, point: f32) -> f32 {
    let tri = triangles[index];
    if (excl_bit(u32(tri.e1.w), shadow_bit)) { return 1.0; }
    let h = cross(ray, tri.e2.xyz);
    let det = dot(h, tri.e1.xyz);
    if (abs(det) > 1e-10) {
        let inverse = 1.0 / det;
        let delta = origin - tri.v0.xyz;
        let u = dot(delta, h) * inverse;
        let q = cross(delta, tri.e1.xyz);
        let v = dot(ray, q) * inverse;
        let t = dot(tri.e2.xyz, q) * inverse;
        if (u >= 0.0 && v >= 0.0 && u + v <= 1.0 &&
            t > near_bias && (point == 0.0 || t < limit)) {
            return 1.0 - tri.v0.w;
        }
    }
    return 1.0;
}
// Uniform-controlled loops keep compilation independent of scene complexity.
fn attenuation(lp: vec4<f32>, ld: vec4<f32>, cone: vec4<f32>, power: f32, point: vec3<f32>) -> f32 {
    // scene3d.light_attenuation: distance falloff, times the spot cone; cone = (is spot, inner, outer, exponent) in degrees.
    if (lp.w <= 0.0) { return 1.0; }
    let offset = point - lp.xyz;
    let dist = length(offset);
    var result = 1.0;
    if (power > 0.0) { result = min(1.0, pow(max(dist, 1e-8), -power)); }
    if (cone.x > 0.0) {
        let angle = degrees(acos(clamp(dot(offset, ld.xyz) / max(dist, 1e-12), -1.0, 1.0)));
        var k = select(0.0, 1.0, angle <= cone.y);
        if (cone.z > cone.y) {
            let t = clamp((cone.z - angle) / (cone.z - cone.y), 0.0, 1.0);
            k = select(0.0, pow(t * t * (3.0 - 2.0 * t), cone.w), t > 0.0);
        }
        result = result * k;
    }
    return result;
}
fn seed_hash(p: vec3<f32>) -> u32 {
    // scene3d._point_seeds: the same hash of the float32 bits, so CPU and GPU share the sample pattern.
    var h = (bitcast<u32>(p.x) * 73856093u) ^ (bitcast<u32>(p.y) * 19349663u) ^ (bitcast<u32>(p.z) * 83492791u);
    h ^= h >> 16u; h *= 0x85ebca6bu; h ^= h >> 13u; h *= 0xc2b2ae35u; h ^= h >> 16u;
    return h;
}
fn trace_visibility(origin: vec3<f32>, ray: vec3<f32>, limit: f32, light: Light) -> f32 {
    if (shadow_off) { return 1.0; }
    var transmission = 1.0;
    // BVH_TRAVERSAL
    for (var j = 0u; j < u32(params.shadow.y); j += 1u) {
        transmission *= triangle_transmission(j, origin, ray, limit, light.position.w);
    }
    return transmission;
}
fn visibility(position: vec3<f32>, normal: vec3<f32>, light: Light) -> f32 {
    let bias = params.shadow.x * light.shadow.x;
    near_bias = bias * 0.01;
    let origin = position + normal * bias;
    var ray = -light.direction.xyz;
    var limit = 0.0;
    if (light.position.w > 0.0) {
        ray = light.position.xyz - origin;
        limit = length(ray);
        ray = ray / max(limit, 1e-8);
    }
    if (light.shadow.y <= 0.0) { return trace_visibility(origin, ray, limit, light); }
    // Soft shadow: scene3d._shadow_trace, a Vogel spiral of jittered rays turned by a per-point hash angle.
    let count = max(u32(light.shadow.z), 1u);
    let phi = f32(seed_hash(origin)) * (6.2831853 / 4294967296.0);
    let axis = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(ray.y) < 0.9);
    let a = normalize(cross(ray, axis));
    let b = cross(ray, a);
    var total = 0.0;
    for (var k = 0u; k < count; k += 1u) {
        let r = sqrt((f32(k) + 0.5) / f32(count));
        let theta = f32(k) * 2.3999632 + phi;
        let spread = a * (r * cos(theta)) + b * (r * sin(theta));
        var jray = normalize(ray + spread * light.shadow.y);
        var jlimit = 0.0;
        if (light.position.w > 0.0) {
            let delta = light.position.xyz + spread * (limit * light.shadow.y) - origin;
            jlimit = length(delta);
            jray = delta / max(jlimit, 1e-8);
        }
        total += trace_visibility(origin, jray, jlimit, light);
    }
    return total / f32(count);
}
// Z1 of 2 finish: scene3d._area_light_contribution/_area_light_shading for a Rect/Disc/Sphere light --
// a Monte Carlo diffuse irradiance over the light's fixed sample set (`area_samples`, uploaded once by
// `_area_light_resources` and reused by every fragment, like the CPU reference reuses them across every
// shading point), and a centre-point, inverse-square specular colour (no test exercises area-light
// specular directly on the CPU reference either; see `_area_light_shading`'s own docstring).
struct AreaResult { irradiance: vec3<f32>, to_light: vec3<f32>, spec_colour: vec3<f32> };
fn area_light_shade(light_index: u32, position: vec3<f32>, normal: vec3<f32>) -> AreaResult {
    shadow_bit = u32(params.settings.y) + light_index;
    let al = area_lights[light_index];
    let two_sided = al.center.w > 0.5;
    let offset = u32(al.info.x);
    let count = u32(al.info.y);
    let bias = params.shadow.x * al.info.z;
    near_bias = bias * 0.01;
    let origin = position + normal * bias;
    let shadows_on = al.info.w > 0.5 && params.shadow.y > 0.0;
    let point_light = Light(vec4<f32>(0.0, 0.0, 0.0, 1.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
    var total = vec3<f32>(0.0);
    var vis_sum = 0.0;
    for (var s = 0u; s < count; s += 1u) {
        let smp = area_samples[offset + s];
        let to_recv = position - smp.point.xyz;
        let dist2 = max(dot(to_recv, to_recv), 1e-10);
        let dist = sqrt(dist2);
        let wi = to_recv / dist;
        var cos_light = dot(wi, smp.normal.xyz);
        cos_light = select(max(cos_light, 0.0), abs(cos_light), two_sided);
        let cos_surface = max(dot(-wi, normal), 0.0);
        let weight = cos_light * cos_surface / dist2;
        var vis = 1.0;
        if (shadows_on && weight > 0.0) {
            let ray = smp.point.xyz - origin;
            let limit = length(ray);
            vis = trace_visibility(origin, ray / max(limit, 1e-8), limit, point_light);
        }
        total += (weight * vis) * al.radiance_area.xyz;
        vis_sum += vis;
    }
    let fcount = f32(max(count, 1u));
    var result: AreaResult;
    result.irradiance = total * (al.radiance_area.w / fcount);
    let to_centre = al.center.xyz - position;
    let dist2c = max(dot(to_centre, to_centre), 1e-6);
    result.to_light = to_centre / sqrt(dist2c);
    result.spec_colour = (vis_sum / fcount / dist2c) * (al.radiance_area.xyz * al.radiance_area.w);
    return result;
}
// scene3d._shade_pbr_mesh / splatshade._cook_torrance: Cook-Torrance GGX specular response times n.l,
// with a per-channel Fresnel `f0` (Y3 of 3, part 1: `pbr` mesh materials on the GPU raster path, factors
// only -- the texture maps and an Environment together with `pbr` are still CPU-only, see `render`'s
// `pbr_geometries` checks in gpu3d.py).
// scene3d._orthonormal_tangent: Duff et al. branchless ONB, the fallback frame when a triangle's
// UV-gradient tangent degenerates (parallel to the normal).
fn orthonormal_tangent(n: vec3<f32>) -> vec3<f32> {
    let sign = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    return vec3<f32>(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
}
// scene3d._shade_fragments' tangent-space normal map decode: Gram-Schmidt the per-triangle tangent
// against the (possibly bump-mapped-so-far) normal, build the bitangent, and reproject the texel.
fn apply_normal_map(n: vec3<f32>, tangent: vec3<f32>, uv: vec2<f32>, scale: f32) -> vec3<f32> {
    let texel = textureSampleLevel(normal_tex, filtering, uv, 0.0).rgb;
    let local_n = (texel * 2.0 - 1.0) * vec3<f32>(scale, scale, 1.0);
    let t_ortho = tangent - n * dot(tangent, n);
    let t_norm = length(t_ortho);
    var t = orthonormal_tangent(n);
    if (t_norm >= 1e-12) { t = t_ortho / t_norm; }
    let b = cross(n, t);
    let result = t * local_n.x + b * local_n.y + n * local_n.z;
    return result / max(length(result), 1e-8);
}
fn ggx_response(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, roughness: f32, f0: vec3<f32>) -> vec3<f32> {
    let h = normalize(l + v);
    let nl = max(dot(n, l), 0.0);
    let nv = max(dot(n, v), 1e-4);
    let nh = max(dot(n, h), 0.0);
    let vh = max(dot(v, h), 0.0);
    let alpha = max(roughness, 0.05) * max(roughness, 0.05);
    let alpha2 = alpha * alpha;
    let denom = nh * nh * (alpha2 - 1.0) + 1.0;
    let d = alpha2 / (3.14159265 * denom * denom);
    let k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    let vis = 1.0 / (max(nl * (1.0 - k) + k, 1e-4) * max(nv * (1.0 - k) + k, 1e-4) * 4.0);
    let fresnel = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - vh, 5.0);
    return vec3<f32>(3.14159265 * d * vis * nl) * fresnel;
}
// envlight.dfg (Z1 of 2 finish): Karis's analytic fit of the split-sum BRDF table, `(A, B)` with
// `F0 * A + B` the specular albedo -- the same closed form as the CPU reference, so no lookup texture.
fn dfg(n_dot_v: f32, roughness: f32) -> vec2<f32> {
    let nv = clamp(n_dot_v, 1e-4, 1.0);
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let t = roughness * c0 + c1;
    let a004 = min(t.x * t.x, pow(2.0, -9.28 * nv)) * t.x + t.y;
    return vec2<f32>(-1.04 * a004 + t.z, 1.04 * a004 + t.w);
}
// scene3d._mesh_pbr_environment: split-sum image-based light for a `pbr` mesh. `base_rgb` is the F0
// used for the metal term's own split-sum weight (the metallic-workflow convention: a full metal's
// specular colour is its base colour), independent of the `f0` already mixed for direct lighting.
fn pbr_env_diffuse(n: vec3<f32>, nv: f32, roughness: f32, metallic: f32, f0_dielectric: f32) -> vec3<f32> {
    if (envg.enabled.x < 0.5) { return vec3<f32>(0.0); }
    let fit = dfg(nv, roughness);
    let dielectric_w = (f0_dielectric * fit.x + fit.y) * (1.0 + f0_dielectric * (1.0 / max(fit.x + fit.y, 1e-4) - 1.0));
    let kd = (1.0 - metallic) * (1.0 - dielectric_w);
    return env_diffuse(n) * kd;
}
fn pbr_env_specular(n: vec3<f32>, to_eye: vec3<f32>, roughness: f32, metallic: f32,
                    f0_dielectric: f32, base_rgb: vec3<f32>) -> vec3<f32> {
    if (envg.enabled.x < 0.5) { return vec3<f32>(0.0); }
    let nv = max(dot(n, to_eye), 0.0);
    let fit = dfg(nv, roughness);
    let compensation = 1.0 / max(fit.x + fit.y, 1e-4);
    let dielectric_w = vec3<f32>((f0_dielectric * fit.x + fit.y) * (1.0 + f0_dielectric * (compensation - 1.0)));
    let metal_w = (base_rgb * fit.x + fit.y) * (vec3<f32>(1.0) + base_rgb * (compensation - 1.0));
    let total_w = (1.0 - metallic) * dielectric_w + metallic * metal_w;
    let reflected = 2.0 * nv * n - to_eye;
    return total_w * env_specular(reflected, roughness);
}
override PASS: u32 = 0u;
struct Vertex {
    @builtin(position) position: vec4<f32>,
    @location(0) depth: f32,
    @location(1) world: vec3<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) @interpolate(flat) colour: vec4<f32>,
    @location(5) @interpolate(flat) lod: f32,
    @location(6) @interpolate(flat) material: vec3<f32>,
    @location(7) @interpolate(flat) object_id: f32,
    // (metallic, roughness, dielectric F0, is-pbr flag); scene3d.Geometry's metallic/pbr_roughness/
    // pbr_specular, packed only when `material` is "pbr" (`_prepare` below).
    @location(8) @interpolate(flat) pbr: vec4<f32>,
    // Y3 of 3, part 1: a per-triangle UV-gradient tangent (world space, flat -- see `_prepare`), which
    // of the four texture maps are bound (metallic-roughness, normal, occlusion, emissive), the flat
    // `emissive_color` tint, and (normal_scale, occlusion_strength) -- every value is zero for a
    // non-`pbr` or untextured geometry, so every earlier render is untouched.
    @location(9) @interpolate(flat) tangent: vec3<f32>,
    @location(10) @interpolate(flat) maps: vec4<f32>,
    @location(11) @interpolate(flat) emissive_color: vec3<f32>,
    @location(12) @interpolate(flat) scales: vec4<f32>,
};
@vertex fn vs(@location(0) p_unused: vec3<f32>, @location(1) world: vec3<f32>,
             @location(2) normal: vec3<f32>, @location(3) uv: vec2<f32>,
             @location(4) colour: vec4<f32>, @location(5) lod: f32, @location(6) material: vec3<f32>,
             @location(7) object_id: f32, @location(8) pbr: vec4<f32>,
             @location(9) tangent: vec3<f32>, @location(10) maps: vec4<f32>,
             @location(11) emissive_color: vec3<f32>, @location(12) scales: vec4<f32>) -> Vertex {
    let q = params.projection;
    // The camera-space position is worked out here, from the world position and the view rows in `params`, so the vertex
    // buffer holds nothing that depends on the camera and a camera move needs no new upload (location 0 is no longer read).
    let rel = world - params.eye.xyz;
    let p = vec3<f32>(dot(params.view0.xyz, rel), dot(params.view1.xyz, rel), dot(params.view2.xyz, rel));
    var v: Vertex;
    // WebGPU depth is 0..w: crossing triangles are clipped, not rejected.
    v.position = vec4<f32>(p.x*q.x, p.y*q.y,
        -q.w/(q.w-q.z)*p.z - q.w*q.z/(q.w-q.z), -p.z);
    v.depth = -p.z; v.world = world; v.normal = normal;
    v.uv = uv; v.colour = colour; v.lod = lod; v.material = material; v.object_id = object_id;
    v.pbr = pbr; v.tangent = tangent; v.maps = maps; v.emissive_color = emissive_color; v.scales = scales;
    return v;
}
@fragment fn fs(v: Vertex) -> @location(0) vec4<f32> {
    // `scales.w` is a bit set: 1 takes no shadow, 2 a holdout (opaque to the depth test, nothing in the picture or the data)
    let flags = u32(v.scales.w + 0.5);
    if ((flags & 2u) != 0u) {
        if (PASS == 1u) { discard; }
        return vec4<f32>(0.0);
    }
    let flipped_uv = vec2<f32>(v.uv.x, 1.0-v.uv.y);
    var source = textureSampleLevel(tex, filtering, flipped_uv, v.lod) * v.colour;
    if (source.a <= 0.0) { discard; }
    if (PASS == 0u && source.a < 0.999) { discard; }
    if (PASS == 1u && source.a >= 0.999) { discard; }
    shadow_off = (flags & 1u) != 0u;
    var normal = v.normal / max(length(v.normal), 1e-8);
    if (v.maps.y > 0.5) {
        // Y3 of 3, part 1: tangent-space normal map (`scene3d._shade_fragments`'s normal-map block
        // runs before the eye-facing flip below, same order here).
        normal = apply_normal_map(normal, v.tangent, flipped_uv, v.scales.x);
    }
    if (dot(normal, params.eye.xyz-v.world) < 0.0) { normal = -normal; }
    if (params.settings.z == 1.0) { return vec4<f32>(vec3<f32>(v.depth), 1.0); }
    if (params.settings.z == 2.0) { return vec4<f32>(normal, 1.0); }
    if (params.settings.z == 7.0) { return vec4<f32>(v.world, 1.0); }
    if (params.settings.z == 8.0) { return vec4<f32>(v.uv, 0.0, 1.0); }
    if (params.settings.z == 9.0) { return vec4<f32>(v.object_id, 0.0, 0.0, 1.0); }
    if (params.settings.z == 3.0) { return source; }
    var emission = source.rgb * v.material.z;
    if (v.pbr.w > 0.5) {
        // scene3d._shade_fragments' emissive block: a flat `emissive_color`, tinted by an emissive
        // texture when one is bound, added independently of the base colour.
        var emissive_tint = v.emissive_color;
        if (v.maps.w > 0.5) {
            emissive_tint = emissive_tint * textureSampleLevel(emissive_tex, filtering, flipped_uv, 0.0).rgb;
        }
        emission += emissive_tint;
    }
    if (params.settings.z == 6.0) { return vec4<f32>(emission, source.a); }
    if (params.settings.y > 0.0 || params.settings.w > 0.5 || envg.enabled.x > 0.5) {
        var specular = vec3<f32>(0.0);
        let link = u32(v.scales.z);
        let area_base = u32(params.settings.y);
        let env_gate = select(1.0, 0.0, excl_bit(link, area_base + u32(params.settings.w)));
        let eye_delta = params.eye.xyz-v.world;
        let to_eye = eye_delta / max(length(eye_delta), 1e-8);
        var radiance: vec3<f32>;
        if (v.pbr.w > 0.5) {
            // scene3d._shade_pbr_mesh (Y3 of 3): metallic/roughness/dielectric-F0 factors, optionally
            // overridden per-texel by a metallic-roughness map (G roughness, B metallic, glTF packing,
            // matching `scene3d._shade_fragments`), lit by Directional/Point/Spot/area lights and
            // (Z1 of 2 finish) a single Environment's split-sum diffuse/specular cross term
            // (`pbr_env_diffuse`/`pbr_env_specular`, matching `scene3d._mesh_pbr_environment`).
            var metallic = v.pbr.x;
            var roughness = v.pbr.y;
            if (v.maps.x > 0.5) {
                let mr = textureSampleLevel(mr_tex, filtering, flipped_uv, 0.0);
                roughness = mr.g; metallic = mr.b;
            }
            let base_rgb = source.rgb / max(source.a, 1e-6);
            let f0 = mix(vec3<f32>(v.pbr.z), base_rgb, metallic);
            radiance = vec3<f32>(params.settings.x) * (1.0 - metallic);
            for (var i = 0u; i < u32(params.settings.y); i += 1u) {
                if (excl_bit(link, i)) { continue; }
                var toward = -lights[i].direction.xyz;
                if (lights[i].position.w > 0.0) {
                    toward = lights[i].position.xyz-v.world;
                    toward = toward/max(length(toward), 1e-8);
                }
                var transmission = 1.0;
                if (params.shadow.y > 0.0 && lights[i].direction.w > 0.0) {
                    shadow_bit = i;
                    transmission = visibility(v.world, normal, lights[i]);
                    // VOLUME_SHADOW
                }
                let factor = attenuation(lights[i].position, lights[i].direction, lights[i].cone, lights[i].colour.w, v.world);
                let scale = transmission * factor;
                let nl = max(dot(normal, toward), 0.0);
                let half_delta = toward + to_eye;
                let half_vector = half_delta / max(length(half_delta), 1e-8);
                let vh = max(dot(to_eye, half_vector), 0.0);
                let kd = (1.0 - metallic) * (1.0 - (0.04 + 0.96 * pow(1.0 - vh, 5.0)));
                radiance += nl * kd * scale * lights[i].colour.xyz;
                specular += ggx_response(normal, to_eye, toward, roughness, f0) * scale * lights[i].colour.xyz;
            }
            for (var i = 0u; i < u32(params.settings.w); i += 1u) {
                if (excl_bit(link, area_base + i)) { continue; }
                // scene3d._area_light_shading, through `_shade_pbr_mesh`'s area branch: the same
                // Cook-Torrance response as a direct light, fed the Monte Carlo diffuse irradiance and
                // the centre-point specular colour instead of a light's own (position, colour).
                let area = area_light_shade(i, v.world, normal);
                let half_delta = area.to_light + to_eye;
                let half_vector = half_delta / max(length(half_delta), 1e-8);
                let vh = max(dot(to_eye, half_vector), 0.0);
                let kd = (1.0 - metallic) * (1.0 - (0.04 + 0.96 * pow(1.0 - vh, 5.0)));
                radiance += kd * area.irradiance;
                specular += ggx_response(normal, to_eye, area.to_light, roughness, f0) * area.spec_colour;
            }
            let env_nv = max(dot(normal, to_eye), 0.0);
            radiance += pbr_env_diffuse(normal, env_nv, roughness, metallic, v.pbr.z) * env_gate;
            specular += pbr_env_specular(normal, to_eye, roughness, metallic, v.pbr.z, base_rgb) * env_gate;
            if (v.maps.z > 0.5) {
                // scene3d._shade_fragments: occlusion attenuates the diffuse response only (ambient,
                // direct lights and the environment's diffuse term alike), never the specular term --
                // `_mesh_pbr_environment`'s diffuse is folded into `diffuse_radiance` before
                // `_shade_fragments` applies `occlusion_factor`, so it is here too.
                let occ = textureSampleLevel(occlusion_tex, filtering, flipped_uv, 0.0).r;
                let strength = clamp(v.scales.y, 0.0, 1.0);
                radiance *= 1.0 - strength * (1.0 - occ);
            }
        } else {
            radiance = vec3<f32>(params.settings.x) + env_diffuse(normal) * env_gate;
            for (var i = 0u; i < u32(params.settings.y); i += 1u) {
                if (excl_bit(link, i)) { continue; }
                var toward = -lights[i].direction.xyz;
                if (lights[i].position.w > 0.0) {
                    toward = lights[i].position.xyz-v.world;
                    toward = toward/max(length(toward), 1e-8);
                }
                var transmission = 1.0;
                if (params.shadow.y > 0.0 && lights[i].direction.w > 0.0) {
                    shadow_bit = i;
                    transmission = visibility(v.world, normal, lights[i]);
                    // VOLUME_SHADOW
                }
                let factor = attenuation(lights[i].position, lights[i].direction, lights[i].cone, lights[i].colour.w, v.world);
                radiance += max(dot(normal, toward), 0.0)*transmission*factor*lights[i].colour.xyz;
                if (v.material.x > 0.0 && dot(normal, toward) > 0.0) {
                    let half_delta = toward + to_eye;
                    let half_vector = half_delta / max(length(half_delta), 1e-8);
                    specular += v.material.x * pow(max(dot(normal, half_vector), 0.0), v.material.y)
                        * transmission * factor * lights[i].colour.xyz;
                }
            }
            for (var i = 0u; i < u32(params.settings.w); i += 1u) {
                if (excl_bit(link, area_base + i)) { continue; }
                // scene3d._shade_fragments' non-`pbr` area branch: no `front`/visibility gate beyond
                // what `area_light_shade`'s own Monte Carlo visibility already folded in.
                let area = area_light_shade(i, v.world, normal);
                radiance += area.irradiance;
                if (v.material.x > 0.0) {
                    let half_delta = area.to_light + to_eye;
                    let half_vector = half_delta / max(length(half_delta), 1e-8);
                    let lobe = pow(max(dot(normal, half_vector), 0.0), v.material.y);
                    specular += v.material.x * lobe * area.spec_colour;
                }
            }
            if (v.material.x > 0.0) {
                // scene3d._mesh_environment_specular: Blinn-Phong shininess mapped to a GGX roughness.
                let refl = 2.0 * dot(normal, to_eye) * normal - to_eye;
                specular += env_specular(refl, sqrt(2.0 / (v.material.y + 2.0))) * v.material.x * env_gate;
            }
        }
        if (params.settings.z == 4.0) { return vec4<f32>(source.rgb*radiance, source.a); }
        if (params.settings.z == 5.0) { return vec4<f32>(specular*source.a, source.a); }
        source = vec4<f32>(source.rgb*radiance + specular*source.a, source.a);
    }
    if (params.settings.z == 4.0) { return source; }
    if (params.settings.z == 5.0) { return vec4<f32>(vec3<f32>(0.0), source.a); }
    return vec4<f32>(source.rgb + emission, source.a);
}
'''


_BVH_DECL = """
struct BvhNode { lo: vec3<f32>, left: i32, hi: vec3<f32>, right: i32,
                 offset: u32, count: u32, pad0: u32, pad1: u32 };
@group(0) @binding(5) var<storage, read> nodes: array<BvhNode>;
@group(0) @binding(6) var<storage, read> prim_order: array<u32>;
fn box_hit(node: BvhNode, origin: vec3<f32>, ray: vec3<f32>, limit: f32) -> bool {
    var near = near_bias;
    var far = limit;
    for (var axis = 0u; axis < 3u; axis += 1u) {
        if (ray[axis] == 0.0) {
            if (origin[axis] < node.lo[axis] || origin[axis] > node.hi[axis]) { return false; }
        } else {
            let a = (node.lo[axis] - origin[axis]) / ray[axis];
            let b = (node.hi[axis] - origin[axis]) / ray[axis];
            near = max(near, min(a, b));
            far = min(far, max(a, b));
            if (near > far) { return false; }
        }
    }
    return true;
}
"""
_BVH_TRAVERSAL = """
    if (params.shadow.z > 0.0) {
        var stack: array<u32, 64>;
        stack[0] = 0u;
        var size = 1u;
        var box_limit = 3.402823e38;
        if (light.position.w > 0.0) { box_limit = limit; }
        loop {
            if (size == 0u) { break; }
            size -= 1u;
            let node = nodes[stack[size]];
            if (!box_hit(node, origin, ray, box_limit)) { continue; }
            if (node.count > 0u) {
                for (var k = 0u; k < node.count; k += 1u) {
                    transmission *= triangle_transmission(prim_order[node.offset+k], origin, ray, limit, light.position.w);
                }
            } else {
                stack[size] = u32(node.left);
                stack[size+1u] = u32(node.right);
                size += 2u;
            }
        }
        return transmission;
    }
"""


_PARTICLE_SHADER = '''
struct PParams { screen: vec4<f32>, planes: vec4<f32>, light: vec4<f32>,
                  mode: vec4<f32>, view0: vec4<f32>, view1: vec4<f32>, view2: vec4<f32> };
// (width, height, 0, 0), (near, far, 0, 0), view-space light, (output code, id_base, 0, 0), the view
// rotation matrix's rows (scene3d._view_basis), read as columns to undo it: world = VT * view_space.
struct PInst { centre: vec2<f32>, radius: f32, z: f32, colour: vec4<f32>,
               world_radius: f32, shape: u32, tex_offset: u32, tex_w: u32, tex_h: u32, pad0: u32, pad1: u32, pad2: u32,
               world_centre: vec3<f32>, particle_id: u32 };
@group(0) @binding(0) var<uniform> pp: PParams;
@group(0) @binding(1) var<storage, read> instances: array<PInst>;
@group(0) @binding(2) var<storage, read> texels: array<vec4<f32>>;
struct POut { @builtin(position) position: vec4<f32>, @location(0) @interpolate(flat) index: u32 };
@vertex fn vs(@builtin(vertex_index) vertex: u32, @builtin(instance_index) index: u32) -> POut {
    var corners = array<vec2<f32>, 6>(vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
                                      vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0));
    let inst = instances[index];
    let pixel = inst.centre + corners[vertex] * inst.radius;
    var out: POut;
    out.position = vec4<f32>(pixel.x / pp.screen.x * 2.0 - 1.0, 1.0 - pixel.y / pp.screen.y * 2.0, 0.0, 1.0);
    out.index = index;
    return out;
}
struct PFrag { @location(0) colour: vec4<f32>, @builtin(frag_depth) depth: f32 };
@fragment fn fs(v: POut) -> PFrag {
    // scene3d._composite_particle_chunk: u, v run -1..1 across the sprite, v down; pixel centres are at +0.5.
    let inst = instances[v.index];
    let offset = (v.position.xy - inst.centre) / inst.radius;
    let square = inst.shape == 2u;
    if (square) {
        if (abs(offset.x) > 1.0 || abs(offset.y) > 1.0) { discard; }
    } else {
        if (dot(offset, offset) > 1.0) { discard; }
    }
    let sphere = inst.shape == 1u || inst.shape == 3u;
    let facing = sqrt(max(0.0, 1.0 - dot(offset, offset)));
    var colour = inst.colour;
    var z = inst.z;
    if (sphere) {
        z = z - facing * inst.world_radius;
    }
    var out: PFrag;
    // The mesh pass writes far*(z-near)/((far-near)*z); the particle depth test compares against it.
    out.depth = clamp(pp.planes.y * (z - pp.planes.x) / ((pp.planes.y - pp.planes.x) * max(z, 1e-8)), 0.0, 1.0);
    if (pp.mode.x == 1.0) {
        // depth: scene3d._composite_particle_data_chunk repeats the same view-space z across rgb.
        out.colour = vec4<f32>(z, z, z, 1.0);
        return out;
    }
    if (pp.mode.x == 2.0) {
        // position: scene3d._composite_particle_data_chunk's world_center + world_normal * offset.
        let normal_view = vec3<f32>(select(0.0, offset.x, sphere), select(0.0, -offset.y, sphere),
                                    select(1.0, facing, sphere));
        let vt = mat3x3<f32>(pp.view0.xyz, pp.view1.xyz, pp.view2.xyz);
        let world_normal = normalize(vt * normal_view);
        let world_pos = inst.world_centre + world_normal * select(0.0, inst.world_radius, sphere);
        out.colour = vec4<f32>(world_pos, 1.0);
        return out;
    }
    if (pp.mode.x == 3.0) {
        out.colour = vec4<f32>(pp.mode.y + f32(inst.particle_id), 0.0, 0.0, 1.0);
        return out;
    }
    if (sphere) {
        let lit = max(0.0, offset.x * pp.light.x - offset.y * pp.light.y + facing * pp.light.z);
        colour = vec4<f32>(colour.rgb * (0.25 + 0.75 * lit), colour.a);
    }
    if (inst.shape == 3u) {
        // foam: a soft rim, premultiplied colour and alpha fade together (scene3d._composite_particle_chunk)
        let rim = 1.0 - min(1.0, dot(offset, offset));
        colour = colour * (rim * rim);
    }
    if (square && inst.tex_w > 0u) {
        let column = clamp(i32(floor((offset.x + 1.0) * 0.5 * f32(inst.tex_w))), 0, i32(inst.tex_w) - 1);
        let line = clamp(i32(floor((offset.y + 1.0) * 0.5 * f32(inst.tex_h))), 0, i32(inst.tex_h) - 1);
        colour = colour * texels[inst.tex_offset + u32(line) * inst.tex_w + u32(column)];
    }
    out.colour = colour;
    return out;
}
'''


def particle_pipeline(state, target=None, depth='depth32float', samples=1, data=False):
    """The instanced particle pipeline: blended and depth-tested-but-not-written for `rgba` (painter's
    order, sorted far to near by `particle_data`), or opaque and depth-written for a data output
    (`depth`/`position`/`object_id`, R7 of 7 finish (2)) so hardware depth resolves nearest-wins the same
    way it already does for the mesh passes, in any draw order. The editor viewport asks for its own
    target formats."""
    target = target or state['format']
    key = ('particles', target, depth, samples, data)
    if key in state['pipelines']:
        return state['pipelines'][key]
    device = state['device']
    module = device.create_shader_module(code=_PARTICLE_SHADER)
    fragment_target = {'format': target}
    if not data:
        blend = {'src_factor': 'one', 'dst_factor': 'one-minus-src-alpha', 'operation': 'add'}
        fragment_target['blend'] = {'color': blend, 'alpha': blend}
    pipeline = device.create_render_pipeline(layout='auto',
        vertex={'module': module, 'entry_point': 'vs', 'buffers': []},
        primitive={'topology': 'triangle-list', 'cull_mode': 'none'},
        depth_stencil={'format': depth, 'depth_write_enabled': data, 'depth_compare': 'less'},
        multisample={'count': samples},
        fragment={'module': module, 'entry_point': 'fs', 'targets': [fragment_target]})
    state['pipelines'][key] = pipeline
    return pipeline


_PARTICLE_OUTPUT_MODES = {'rgba': 0.0, 'depth': 1.0, 'position': 2.0, 'object_id': 3.0}


def particle_data(scene, camera, width, height, limits, cancel, output='rgba'):
    """Instance and sprite-texel arrays for the particle draw, sorted far to near, or None. `output`
    "rgba" draws the old fixed look (a "pbr" particle_material is refused earlier, in `render`); `depth`,
    `position` and `object_id` (R7 of 7 finish (2)) instead write that data output exactly like
    `scene3d._composite_particle_data_chunk`, matching a mesh's own first-hit, unantialiased data pass."""
    _cancel(cancel)
    eye, view = scene3d._view_basis(camera)
    focal = 1 / math.tan(math.radians(camera.fov) / 2)
    sprites = scene3d.particle_sprites(scene, camera, width, height, eye, view, focal, width / height)
    if sprites is None:
        return None
    # The "pbr" material/ramp fields (R7 of 7) are CPU-only for now (docs/3D_FOUNDATION.md "Particles"
    # known limits): the GPU draw keeps its old fixed look and simply ignores them here.
    (z, centre, radius, color, shape, world_radius, texture_id, textures,
     world_center, _pbr, _metallic, _roughness, _specular, _emission, particle_id) = sprites
    count = len(z)
    offsets, dims, chunks, total = [], [], [], 0
    for image in textures:
        offsets.append(total)
        dims.append(image.shape[:2])
        chunks.append(np.ascontiguousarray(image, 'f4').reshape(-1, 4))
        total += len(chunks[-1])
    if count * 80 > limits['max-storage-buffer-binding-size'] or total * 16 > limits['max-storage-buffer-binding-size']:
        raise Unsupported('particle data exceeds the adapter storage buffer limit')
    packed = np.zeros((count, 20), 'f4')
    packed[:, 0:2], packed[:, 2], packed[:, 3], packed[:, 4:8] = centre, radius, z, color
    packed[:, 8] = world_radius
    packed[:, 16:19] = world_center
    words = packed.view('u4')
    words[:, 9] = shape
    words[:, 19] = particle_id
    textured = texture_id >= 0
    if textured.any():
        table = np.array(offsets, 'u4'), np.array([d[1] for d in dims], 'u4'), np.array([d[0] for d in dims], 'u4')
        words[textured, 10] = table[0][texture_id[textured]]
        words[textured, 11] = table[1][texture_id[textured]]
        words[textured, 12] = table[2][texture_id[textured]]
    texel_data = np.concatenate(chunks) if chunks else np.zeros((1, 4), 'f4')
    params = np.array([width, height, 0, 0, camera.near, camera.far, 0, 0,
                       *scene3d._VIEW_LIGHT, 0, _PARTICLE_OUTPUT_MODES[output], scene3d.particle_id_base(scene), 0, 0,
                       *view[0], 0, *view[1], 0, *view[2], 0], 'f4')
    return packed, texel_data, params


_playing = threading.local()


class playing:
    """``with gpu3d.playing(True):`` a render on this thread draws splats with meshes as the playback tier does: the meshes
    with the raster shader and the splats as a layer over them, instead of the ray tracer the shadows between the two
    need (see ``playback.PLAYING_SPLAT_SEED``)."""

    def __init__(self, on=True):
        self.on = bool(on)

    def __enter__(self):
        self.before = getattr(_playing, 'on', False)
        _playing.on = self.on

    def __exit__(self, *exc):
        _playing.on = self.before


def is_playing():
    return getattr(_playing, 'on', False)


def _view_rows(camera):
    """The camera's rotation as three padded rows (`Params.view0..2`): the vertex shader turns a world position into
    camera space with them, so no vertex buffer holds a camera-space value."""
    _, view = scene3d._view_basis(camera)
    rows = np.zeros((3, 4), 'f4')
    rows[:, :3] = view
    return rows.reshape(-1)


def _pipeline(state, data, phase, bvh=False, smoke=False, hold=False):
    """`hold`: the variant that draws holdout triangles in the beauty's first pass, replacing what is under them with zero.
    `phase` 0 draws the opaque triangles, 1 the translucent ones, 2 the data passes, 3 writes the depth of the translucent ones only
    (the glass depth the splat layer is ordered against)."""
    key = (data, phase, bvh, smoke, hold)
    if key in state['pipelines']:
        return state['pipelines'][key]
    device = state['device']
    code = _SHADER
    if bvh:
        code = _BVH_DECL + code.replace('// BVH_TRAVERSAL', _BVH_TRAVERSAL)
    if smoke:
        from . import gpuvolume
        code = gpuvolume.MESH_SHADOW_WGSL + code.replace(
            '// VOLUME_SHADOW', 'if (!shadow_off) { transmission *= volume_transmission(v.world, lights[i], u32(i)); }')
    module = device.create_shader_module(code=code)
    target = {'format': 'rgba32float' if data else state['format']}
    if phase == 3:
        target['write_mask'] = 0      # the translucent meshes' depth only: the colour stays as it is
    if not data:
        blend = ({'src_factor': 'zero', 'dst_factor': 'zero', 'operation': 'add'} if hold else
                 {'src_factor': 'one', 'dst_factor': 'one-minus-src-alpha', 'operation': 'add'})
        target['blend'] = {'color': blend, 'alpha': blend}
    attributes = [dict(format=f, offset=o, shader_location=i) for i, (f, o) in enumerate(
        [('float32x3', 0), ('float32x3', 12), ('float32x3', 24), ('float32x2', 36), ('float32x4', 44), ('float32', 60), ('float32x3', 64), ('float32', 76), ('float32x4', 80),
         ('float32x3', 96), ('float32x4', 108), ('float32x3', 124), ('float32x4', 136)])]
    pipeline = device.create_render_pipeline(layout='auto',
        vertex={'module': module, 'entry_point': 'vs', 'buffers': [
            {'array_stride': 152, 'step_mode': 'vertex', 'attributes': attributes}]},
        primitive={'topology': 'triangle-list', 'cull_mode': 'none'},
        depth_stencil={'format': 'depth32float', 'depth_write_enabled': phase != 1, 'depth_compare': 'less'},
        fragment={'module': module, 'entry_point': 'fs', 'constants': {'PASS': 1 if phase == 3 else phase}, 'targets': [target]})
    state['pipelines'][key] = pipeline
    return pipeline


def _cancel(event):
    if event is not None and event.is_set():
        from .cancellation import Cancelled
        raise Cancelled()


def _check_holdouts(scene):
    """The raster and ray-traced modes cut holdout meshes out of mesh pictures (beauty and data passes). What would let a
    holdout be seen, or leave something behind it uncut, stays on the CPU (`Unsupported`)."""
    if not any(g.holdout for g in scene.geometries):
        return
    if scene.splats or getattr(scene, 'particles', ()) or getattr(scene, 'volumes', ()):
        raise Unsupported('holdouts together with splats, particles or smoke are CPU-only in the raster and ray-traced modes')
    if any(g.material == 'liquid' for g in scene.geometries):
        raise Unsupported('holdouts together with liquid surfaces are CPU-only (refraction would see them)')
    if any(getattr(e, 'visible_to_camera', False) for e in getattr(scene, 'environments', ())) or any(
            light.visible_to_camera and light.kind in scene3d._AREA and light.intensity > 0 for light in scene.lights):
        raise Unsupported('holdouts with an environment or area light that the camera sees are CPU-only here')


def link_order(scene):
    """The lights as the raster shader numbers them for light linking: the table's point-like lights, then the area
    lights, then the first environment (its bit is one past the last light)."""
    active = [light for light in scene.lights if light.intensity > 0]
    order = ([l for l in active if l.kind not in scene3d._AREA] + [l for l in active if l.kind in scene3d._AREA]
             + list(getattr(scene, 'environments', ())[:1]))
    if len(order) > 24 and scene3d.has_shadow_links(scene):
        raise Unsupported('light linking and per-object shadow switches on the GPU take at most 24 lights and '
                          'environments together')
    return order


# A mesh with `cast_shadows` off excludes every light from its shadow triangles (all 24 bits `excl_bit` reads).
NO_CAST_MASK = float((1 << 24) - 1)


def caster_mask(geometry, order):
    """The shadow-triangle mask of `geometry`: the lights its link excludes, or every light when it casts no shadow."""
    return NO_CAST_MASK if not geometry.cast_shadows else link_mask(geometry.light_link, order)


def link_mask(link, order):
    """The float the shaders read: the bit mask of the lights in `order` that `link` excludes."""
    return float(sum(1 << k for k, light in enumerate(order[:24]) if not scene3d.light_reaches(link, light)))


VECTOR_PREPARE = True      # pack a geometry's triangles with NumPy when none of them needs near-plane clipping


def _pack_triangles(geometry, local, world, normals, uvs, camera, focal, width, height, object_id, tint, pbr, maps,
                    emissive_color, scales):
    """`_prepare`'s per-triangle loop for a geometry without textures, done on whole arrays: (packed triangles
    (K,3,38) float32, mean depth per triangle), or None when any visible triangle crosses the near plane (the loop clips
    those). The same float32 and float64 operations in the same order as the loop, so the packed bytes are equal; the
    loop took 32 ms and the BVH around it 150 ms of a 475 ms furnished-room frame."""
    tris = np.asarray(geometry.triangles)
    if not len(tris):
        return np.empty((0, 3, 38), 'f4'), np.empty(0)
    z = -local[tris, 2]
    visible = ~((z <= camera.near).all(axis=1) | (z >= camera.far).all(axis=1))
    if not visible.any():
        return np.empty((0, 3, 38), 'f4'), np.empty(0)
    tris, z = tris[visible], z[visible]
    if not (z > camera.near).all():
        return None
    w0, w1, w2 = world[tris[:, 0]], world[tris[:, 1]], world[tris[:, 2]]
    e1, e2 = w1 - w0, w2 - w0
    if normals is None:
        face = np.cross(e1, e2)
        face = face / np.maximum(np.linalg.norm(face, axis=1, keepdims=True), 1e-8)
        ns = np.repeat(face[:, None, :], 3, axis=1)
    else:
        ns = normals[tris]
    uv = uvs[tris]
    duv1, duv2 = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
    det = duv1[:, 0]*duv2[:, 1] - duv2[:, 0]*duv1[:, 1]
    good = np.abs(det) > 1e-12
    safe = np.where(good, det, 1.0)[:, None]
    tangent = np.where(good[:, None], (e1*duv2[:, 1:2] - e2*duv1[:, 1:2]) / safe, e1)
    attrs = np.concatenate((local[tris], world[tris], ns, uv, np.repeat(tangent[:, None, :], 3, axis=1)),
                           axis=2).astype('f4')
    zs = -attrs[:, :, 2]
    count = len(attrs)
    xy = scene3d._to_pixels(attrs[:, :, :3].reshape(-1, 3), zs.reshape(-1), focal, width/height, width, height)
    xy = xy.reshape(count, 3, 2)
    a, b, c = xy[:, 0], xy[:, 1], xy[:, 2]
    den = (b[:, 1]-c[:, 1])*(a[:, 0]-c[:, 0]) + (c[:, 0]-b[:, 0])*(a[:, 1]-c[:, 1])
    drawn = ~(np.abs(den) < 1e-8)
    if not drawn.all():
        attrs, zs = attrs[drawn], zs[drawn]
        count = len(attrs)
    packed = np.empty((count, 3, 38), 'f4')
    packed[:, :, :11] = attrs[:, :, :11]
    packed[:, :, 11:15] = tint
    packed[:, :, 15] = 0                 # one mip level: the loop's lod clips to 0
    packed[:, :, 16:19] = (geometry.specular, geometry.shininess, geometry.emission)
    packed[:, :, 19] = object_id
    packed[:, :, 20:24] = pbr
    packed[:, :, 24:27] = attrs[:, :, 11:14]
    packed[:, :, 27:31] = maps
    packed[:, :, 31:34] = emissive_color
    packed[:, :, 34:38] = scales
    return packed, zs.mean(axis=1)


def _geometry_terms(geometry, order):
    """What one geometry contributes to the vertex buffer whatever the camera does: (world vertices, world normals or
    None, uvs, mip chain, the four PBR maps, premultiplied tint, pbr row, map flags, emissive colour, scales row)."""
    matrix = geometry.world_matrix()
    world = (matrix[:3, :3] @ geometry.vertices.T + matrix[:3, 3:4]).T
    normals = None
    if geometry.normals is not None:
        normals = (np.linalg.inv(matrix[:3, :3]).T @ geometry.normals.T).T
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    uvs = geometry.uvs if geometry.uvs is not None else np.zeros((len(world), 2), 'f4')
    texture = geometry.texture if geometry.uvs is not None else None
    mips = scene3d._mip_chain(texture) if texture is not None else [np.ones((1, 1, 4), 'f4')]
    # Y3 of 3, part 1: the four remaining PBR texture maps (base colour already shades through
    # `mips` above, pbr or not) plus flat `emissive_color`, matching `scene3d._shade_fragments`'
    # glTF-style metallic-roughness packing (G roughness, B metallic) and tangent-space normal
    # decode. Only read for a `pbr` geometry with UVs, like the CPU reference's own checks.
    is_pbr = geometry.material == 'pbr' and geometry.uvs is not None
    mr_texture = geometry.metallic_roughness_texture if is_pbr else None
    normal_texture = geometry.normal_texture if is_pbr else None
    occlusion_texture = geometry.occlusion_texture if is_pbr else None
    emissive_texture = geometry.emissive_texture if is_pbr else None
    tint = np.asarray(geometry.color, 'f4').copy()
    tint[:3] *= tint[3]
    # Y3 of 3, part 1: (metallic, roughness, dielectric F0, is-pbr flag), matching
    # scene3d._shade_fragments' `0.08 * clip(pbr_specular, 0, 1)` dielectric F0 derivation; zero
    # (the is-pbr flag off) reproduces every earlier (Blinn-Phong) render exactly.
    if geometry.material == 'pbr':
        pbr = (float(np.clip(geometry.metallic, 0, 1)), float(np.clip(geometry.pbr_roughness, 0, 1)),
              0.08 * float(np.clip(geometry.pbr_specular, 0, 1)), 1.0)
    else:
        pbr = (0.0, 0.0, 0.0, 0.0)
    maps = (1.0 if mr_texture is not None else 0.0, 1.0 if normal_texture is not None else 0.0,
            1.0 if occlusion_texture is not None else 0.0, 1.0 if emissive_texture is not None else 0.0)
    emissive_color = np.asarray(geometry.emissive_color, 'f4') if geometry.material == 'pbr' else np.zeros(3, 'f4')
    scales = (float(geometry.normal_scale) if is_pbr else 0.0, float(geometry.occlusion_strength) if is_pbr else 0.0,
              link_mask(geometry.light_link, order),
              (0.0 if geometry.receive_shadows else 1.0) + (2.0 if geometry.holdout else 0.0))
    return (world, normals, uvs, mips, (mr_texture, normal_texture, occlusion_texture, emissive_texture), tint, pbr, maps,
            emissive_color, scales)


def _prepare(scene, camera, width, height, cancel):
    eye, view = scene3d._view_basis(camera)
    order = link_order(scene)
    focal = 1 / math.tan(math.radians(camera.fov) / 2)
    vertices, queue, materials = [], [], []
    emitted = 0      # triangles packed so far: the first vertex of the next one is 3 * emitted
    for object_id, geometry in enumerate(scene.geometries, 1):
        _cancel(cancel)
        if not geometry.visible_to_camera:
            continue        # drawn by no pass; `_shadow_data` still holds its triangles, so it casts its shadow
        (world, normals, uvs, mips, (mr_texture, normal_texture, occlusion_texture, emissive_texture), tint, pbr, maps,
         emissive_color, scales) = _geometry_terms(geometry, order)
        local = (view @ (world-eye).T).T
        material = len(materials)
        materials.append((mips, mr_texture, normal_texture, occlusion_texture, emissive_texture))
        if VECTOR_PREPARE and len(mips) == 1:
            fast = _pack_triangles(geometry, local, world, normals, uvs, camera, focal, width, height, object_id,
                                   tint, pbr, maps, emissive_color, scales)
            if fast is not None:
                chunk, depths = fast
                queue.extend(zip(depths.tolist(), (emitted*3 + 3*np.arange(len(chunk))).tolist(),
                                 [material]*len(chunk), [geometry.holdout]*len(chunk)))
                vertices.append(chunk.reshape(-1, 38))
                emitted += len(chunk)
                continue
        for index, tri in enumerate(geometry.triangles):
            if index % 256 == 0:
                _cancel(cancel)
            z = -local[tri, 2]
            if (z <= camera.near).all() or (z >= camera.far).all():
                continue
            if normals is None:
                face = np.cross(world[tri[1]]-world[tri[0]], world[tri[2]]-world[tri[0]])
                ns = np.broadcast_to(face / max(float(np.linalg.norm(face)), 1e-8), (3, 3))
            else:
                ns = normals[tri]
            # scene3d.render's raster reference `_flat_tangents`-style UV-gradient tangent: one per
            # triangle (world space), broadcast to its three vertices, so normal mapping needs no
            # separate per-vertex tangent pass.
            e1, e2 = world[tri[1]]-world[tri[0]], world[tri[2]]-world[tri[0]]
            duv1, duv2 = uvs[tri[1]]-uvs[tri[0]], uvs[tri[2]]-uvs[tri[0]]
            det = duv1[0]*duv2[1]-duv2[0]*duv1[1]
            tri_tangent = (e1*duv2[1]-e2*duv1[1])/det if abs(det) > 1e-12 else e1
            ts = np.broadcast_to(tri_tangent, (3, 3))
            attrs = np.concatenate((local[tri], world[tri], ns, uvs[tri], ts), axis=1).astype('f4')
            # Reference clipping also defines sorting and the area-based mip footprint.
            # Hardware still performs homogeneous near/far and viewport clipping.
            for clipped in scene3d._clip_near(attrs, z, camera.near):
                zs = -clipped[:, 2]
                a, b, c = scene3d._to_pixels(clipped[:, :3], zs, focal, width/height, width, height)
                den = (b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
                if abs(den) < 1e-8:
                    continue
                e, f = clipped[1, 9:11]-clipped[0, 9:11], clipped[2, 9:11]-clipped[0, 9:11]
                area = abs(float(e[0]*f[1]-e[1]*f[0]))*mips[0].shape[0]*mips[0].shape[1]
                lod = np.clip(round(.5*math.log2(max(area/max(abs(den), 1e-8), 1))), 0, len(mips)-1)
                packed = np.empty((3, 38), 'f4')
                packed[:, :11] = clipped[:, :11]
                # All three vertices agree, regardless of the provoking vertex.
                packed[:, 11:15] = tint
                packed[:, 15] = lod
                packed[:, 16:19] = (geometry.specular, geometry.shininess, geometry.emission)
                packed[:, 19] = object_id
                packed[:, 20:24] = pbr
                packed[:, 24:27] = clipped[:, 11:14]
                packed[:, 27:31] = maps
                packed[:, 31:34] = emissive_color
                packed[:, 34:38] = scales
                queue.append((float(zs.mean()), emitted*3, material, geometry.holdout))
                vertices.append(packed)
                emitted += 1
    return eye, focal, vertices, sorted(queue, key=lambda q: q[0], reverse=True), materials


def _shadow_data(scene, count, limit, cancel):
    """Pack all world triangles, including invisible/off-camera geometry."""
    size = max(1, count) * 48
    if size > limit:
        raise ValueError(f'Shadow triangle buffer exceeds max_storage_buffer_binding_size: {size} > {limit} bytes')
    packed = np.zeros((max(1, count), 3, 4), 'f4')
    low, high = np.full(3, np.inf, 'f4'), np.full(3, -np.inf, 'f4')
    offset = 0
    order = link_order(scene)
    if count:
        for geometry in scene.geometries:
            _cancel(cancel)
            n = len(geometry.triangles)
            if not n:
                continue
            matrix = geometry.world_matrix()
            world = (matrix[:3, :3] @ geometry.vertices.T + matrix[:3, 3:4]).T
            triangles = world[geometry.triangles]
            low = np.minimum(low, triangles.min(axis=(0, 1)))
            high = np.maximum(high, triangles.max(axis=(0, 1)))
            block = packed[offset:offset+n]
            block[:, 0, :3] = triangles[:, 0]
            block[:, 1, :3] = triangles[:, 1] - triangles[:, 0]
            block[:, 2, :3] = triangles[:, 2] - triangles[:, 0]
            block[:, 0, 3] = np.clip(geometry.color[3], 0, 1)
            block[:, 1, 3] = caster_mask(geometry, order)
            offset += n
    return packed, 1e-3 * max(1.0, float((high-low).max()) if count else 1.0)


HASH_BYTES = 16 << 20      # arrays over this are told apart by identity (a hash of 16 MB costs about 15 ms)


def _array_key(array):
    """What tells one array from another in a cache key. The Evaluator builds a scene's geometry again for every frame, so the
    arrays of an unchanged mesh are new objects each time: a writeable array is identified by a hash of its bytes (about 0.2 ms
    for a few thousand triangles). A read-only array, or one too big to hash on every frame, by itself (the cache entry keeps it
    alive, so the identity stays unique)."""
    if array is None:
        return 0
    if not array.flags.writeable or array.nbytes > HASH_BYTES:
        return ('id', id(array))
    return ('hash', array.shape, array.dtype.str, hashlib.blake2b(np.ascontiguousarray(array), digest_size=16).digest())


def _shadow_key(scene):
    """What the packed shadow triangles and their BVH are made of, without reading a vertex: each geometry's arrays (by
    `_array_key`), its world matrix, opacity and caster mask. A camera that moves leaves all of it alone."""
    order = link_order(scene)
    return tuple((_array_key(g.vertices), _array_key(g.triangles), g.world_matrix().tobytes(), float(g.color[3]),
                  caster_mask(g, order)) for g in scene.geometries)


def _shadow_cache(state, scene, count):
    """The state's single cached shadow entry when it is this scene's (a dict), else a fresh empty one for the caller to
    fill and `_shadow_cache_keep` to store. Rebuilding the packed triangles and the BVH on every frame of a camera move
    cost 150 ms of a 475 ms furnished-room frame (docs/PERFORMANCE.md, Lane 4 playback profile)."""
    key = (count, _shadow_key(scene))
    entry = state.get('shadow_cache')
    if entry is not None and entry['key'] == key:
        state['shadow_cache_hits'] = state.get('shadow_cache_hits', 0) + 1
        return entry
    state['shadow_cache_misses'] = state.get('shadow_cache_misses', 0) + 1
    return {'key': key, 'refs': [(g.vertices, g.triangles) for g in scene.geometries]}


def _shadow_cache_keep(state, entry):
    state['shadow_cache'] = entry


def _splat_extras(scene, ambient=0.0, provider=None, cancel=None):
    """What splat shading reads beyond the scene's lights, or None: the environments and, for instances
    with Indirect samples, the CPU `splatindirect.IndirectLight` (its hemisphere rays run on the CPU
    casters; the drawer then gets the same per-splat colours the reference renderer computes). `provider`
    is the scene's `_SplatShadows` when one exists, so hit splats reuse its visibility cache."""
    from .envlight import SplatLighting
    environments = getattr(scene, 'environments', ())
    indirect = None
    if scene3d._indirect_instances(scene) > 0:
        from .splatindirect import IndirectLight
        scene3d._indirect_budget(scene, 0)
        if provider is None:
            provider = scene3d._SplatShadows(scene.splats, scene.lights, None, None, .001, cancel)
            provider.relit_shadows = False
        indirect = IndirectLight(provider, ambient, environments, cancel=cancel)
    return SplatLighting(environments, None, indirect) if (environments or indirect) else None


def render(scene, camera, width, height, background=(0, 0, 0, 0), ambient=0.0,
           samples=1, output='rgba', cancel=None, adapter=None, *, mode='raster', volume=None):
    """Render a read-only premultiplied float32 image; raise on unavailable GPUs.

    Projection and viewport shade rendering are unsupported. Callers can catch
    Unsupported/RuntimeError and use scene3d.render as their fallback.
    """
    _mark('render_begin')
    if samples > 1 and scene3d.is_empty_scene(scene):
        samples = 1   # nothing to antialias: the frame is the flat background
    if scene3d.has_shadow_catchers(scene):
        # Shadow catcher (step R3): the picture with the catchers as holdouts, the shadow layer under it (nodebased/shadowcatch.py)
        from . import shadowcatch
        bound = dict(background=background, ambient=ambient, samples=samples, output=output, cancel=cancel,
                     adapter=adapter, mode=mode, volume=volume)
        return shadowcatch.render(lambda sc, **more: render(sc, camera, width, height, **{**bound, **more}), scene, camera,
                                  width, height, background=background, output=output, ambient=ambient, engine='raster',
                                  samples=samples, volume=volume, cancel=cancel)
    if getattr(scene, 'instances', ()):
        # gpuinstance traces InstanceSet items on a two-level GPU BVH without flattening them
        # (docs/3D_ROADMAP.md "Instancing"); everything else here reads only scene.geometries,
        # so a scene with instances alongside other content would otherwise silently drop them.
        pure = not (scene.geometries or scene.splats or getattr(scene, 'particles', ())
                    or getattr(scene, 'volumes', ()))
        if mode == 'raytrace' and pure:
            if output == 'splats':
                raise Unsupported('splats output is CPU-only')
            if output == 'relight':
                raise Unsupported('the relight bundle output is CPU-only for now')
            if output not in scene3d.RENDER_OUTPUTS:
                raise ValueError(f'Unknown 3D render output {output!r}')
            _cancel(cancel)
            from . import gpuinstance
            with _lock:
                state = _state(adapter)
                return gpuinstance.render(state, scene, camera, width, height, background,
                                          ambient, output, samples, cancel=cancel)
        raise Unsupported('GPU rendering of Instance3D instances needs raytrace mode and a scene made '
                          'entirely of instances (no ordinary geometry, splats, particles or volumes '
                          'alongside them); the CPU renderer draws every combination')
    _check_holdouts(scene)
    pbr_geometries = [g for g in scene.geometries if g.material == 'pbr']
    # Y3 of 3 (docs/3D_FOUNDATION.md "Materials"): the wgpu rasterizer shades a `pbr` geometry's
    # metallic/roughness/specular factors with the same Cook-Torrance GGX as the CPU reference
    # (`ggx_response` in `_SHADER`, matching `scene3d._shade_pbr_mesh`), lit by Directional/Point/Spot
    # lights, its own texture bindings and (Z1 of 2 finish) a single Environment's split-sum image-based
    # diffuse and specular cross term (`pbr_env_diffuse`/`pbr_env_specular` in `_SHADER`, matching
    # `scene3d._mesh_pbr_environment`) and Rect/Disc/Sphere area lights (the same Monte Carlo estimator
    # as `scene3d._area_light_contribution`). The ray-traced mode draws all of it too (Z3 of 3: the
    # material table in `gpurt_render._prepare`); a scene with more than one Environment stays CPU-only
    # in both modes (each only ever binds `environments[0]`).
    if mode == 'raytrace' and any(g.material == 'liquid' for g in scene.geometries) and (
            pbr_geometries or getattr(scene, 'environments', ()) or any(light.kind in scene3d._AREA for light in scene.lights)):
        # The liquid renderer's secondary rays shade solids with a simplified light loop in the compute shader
        # (`shade_solid`); the CPU reference shades them with the full `_shade_fragments`.
        raise Unsupported('liquid surfaces together with pbr materials, an Environment or area lights are CPU-only')
    has_scene_volumes = bool(getattr(scene, 'volumes', ()))
    if output in scene3d.VOLUME_OUTPUTS or (output == 'depth' and has_scene_volumes):
        return _render_volume_passes(scene, camera, width, height, background, output, cancel, adapter, mode, volume)
    volumes = output == 'rgba' and has_scene_volumes
    if (volumes and scene.geometries and len(scene.volumes) > 4 and volume is not None and volume.shadow_density > 0
            and any(light.shadows and light.intensity > 0 for light in scene.lights)):
        raise Unsupported('smoke shadows on meshes from more than four volumes are CPU-only')
    if volumes and mode == 'raytrace':
        raise Unsupported('volumes drawn with the ray tracer are CPU-only')
    environments = getattr(scene, 'environments', ())
    if environments and scene.geometries and len(environments) > 1:
        # Y1 of 2 finish (2): a mesh lit by a single Environment shades in the GPU raster path
        # (`env_diffuse`/`env_specular` in `_SHADER`, matching `scene3d._shade_fragments` and
        # `_mesh_environment_specular`), and Z3 of 3 gave the ray-traced mode the same terms. More than
        # one Environment (like `gpupathtrace`'s own "one environment" scope) is still the CPU reference.
        raise Unsupported('more than one environment light on meshes is CPU-only')
    if scene.geometries and any(getattr(i, 'relight', 0) > 0 and getattr(i, 'reflection_samples', 0) > 0
                                for i in scene.splats):
        raise Unsupported('splat reflections of meshes are CPU-only')
    if scene.geometries and scene3d._indirect_instances(scene) > 0:
        raise Unsupported('splat indirect light and occlusion with meshes in the scene is CPU-only')
    particles = bool(getattr(scene, 'particles', ()))
    if particles and (mode == 'raytrace' or scene.splats):
        raise Unsupported('particles drawn with the ray tracer or together with splats are CPU-only')
    if particles and output == 'rgba' and any(i.material == 'pbr' for i in scene.particles):
        # R7 of 7 finish (2), like the mesh "pbr" check above: the instanced sprite pipeline's fragment
        # shader only has the old fixed "headlight" look (`_PARTICLE_SHADER`), so a "pbr" particle used
        # to draw with that fixed look silently instead of its lit/shadowed material; raising here sends
        # `auto` scenes to the CPU reference (`scene3d._draw_particles`, "pbr" branch) for a correct picture.
        raise Unsupported('particle_material "pbr" lighting and shadows are CPU-only for now')
    if mode == 'raytrace':
        if output == 'splats':
            raise Unsupported('splats output is CPU-only')
        if output == 'relight':
            raise Unsupported('the relight bundle output is CPU-only for now')
        if output not in scene3d.RENDER_OUTPUTS:
            raise ValueError(f'Unknown 3D render output {output!r}')
        if scene.splats:
            if output != 'rgba':
                raise Unsupported('splat data passes are CPU-only')
        if any(g.projection is not None for g in scene.geometries):
            raise Unsupported('Camera-projected geometry is not implemented by wgpu')
        _cancel(cancel)
        from . import gpurt_render
        with _lock:
            state = _state(adapter)
            _release_staging(state)
            reason = gpurt_render.check_capability(state)
            if reason is not None:
                raise Unsupported(reason)
            seen_by_camera = any(getattr(e, 'visible_to_camera', False) for e in environments) or any(
                light.visible_to_camera and light.kind in scene3d._AREA and light.intensity > 0
                for light in scene.lights)
            if output == 'rgba' and scene.geometries and seen_by_camera:
                # Z3 of 3: the same post pass as the raster path below (`scene3d._visible_background` on the
                # readback, over the depth of the ray-traced first hits), run before the supersample average.
                ss = max(1, min(int(samples), 4))
                iw, ih = int(width)*ss, int(height)*ss
                result = np.array(gpurt_render.render(
                    state, scene, camera, iw, ih, background, ambient, 'rgba', 1, cancel=cancel))
                depth_image = gpurt_render.render(
                    state, scene, camera, iw, ih, (0, 0, 0, 0), ambient, 'depth', 1, cancel=cancel)
                depth = np.where(depth_image[..., 3] > 0, depth_image[..., 0], np.inf)
                view_eye, view_matrix = scene3d._view_basis(camera)
                focal = 1 / math.tan(math.radians(camera.fov) / 2)
                scene3d._visible_background(scene, iw, ih, result, depth, view_eye, view_matrix, focal,
                                            iw / ih, float(np.clip(background[3], 0, 1)))
                if ss > 1:
                    result = scene3d.box_resolve(result, ss)
                result.flags.writeable = False
                return result
            if output == 'rgba':
                return gpurt_render.render_beauty(
                    state, scene, camera, width, height, background, ambient, samples, cancel=cancel)
            return gpurt_render.render(
                state, scene, camera, width, height, background, ambient, output, samples, cancel=cancel)
    if output == 'rgba' and any(g.material == 'liquid' for g in scene.geometries):
        raise Unsupported('the raster approximation of liquid surfaces is CPU-only (the ray tracer does them on the GPU)')
    glass_on_card = False
    if scene.splats:
        if output != 'rgba' and output not in ('depth', 'position', 'object_id'):
            raise Unsupported('splat data passes other than depth, position and object_id, and the `splats` output, '
                              'are CPU-only')
        if not scene3d._opaque_meshes(scene):
            if output != 'rgba':
                raise Unsupported('splat data passes with transparent meshes are CPU-only')
            # Transparent meshes with splats: the CPU takes the mesh surfaces from primary rays in either mode
            # (the two give identical results), so the GPU path is the ray tracer's layer recording merged with
            # the splat tiles (`gpurt_render`, `gpusplat.LayeredResolve`). Camera-projected geometry stays on the
            # CPU: the ray-traced branch raises for it.
            # A played frame takes the card route instead (`_render_resident` draws the splat layer in two parts around the
            # translucent meshes); the ray tracer keeps every stopped frame, and a played one the card could not take.
            if not (is_playing() and SPLATS_ON_CARD):
                return render(scene, camera, width, height, background, ambient, samples, output,
                              cancel, adapter, mode='raytrace')
            glass_on_card = True
        # The raster shader has no splat casters, so shadow work that involves splats (meshes shadowing
        # relit or caught splats, splats shadowing meshes and each other) goes to the ray tracer, whose
        # opaque-mesh result equals the raster one.
        shadowed = any(light.shadows and light.intensity > 0 for light in scene.lights)
        if (output == 'rgba' and shadowed and not is_playing()
                and (scene.geometries or any(getattr(i, 'relight', 0) > 0 for i in scene.splats))):
            return render(scene, camera, width, height, background, ambient, samples, output,
                          cancel, adapter, mode='raytrace')
    # The splat contribution layer is CPU-only, including empty scenes.
    if output == 'splats':
        raise Unsupported('splats output is CPU-only')
    # The relight bundle is raster-only on the CPU for now (docs/3D_ROADMAP.md "Design: relight
    # passes"); it has no GPU raster shader branch, so reject it explicitly rather than let an
    # unrecognised output index reach the WGSL output-mode switch.
    if output == 'relight':
        raise Unsupported('the relight bundle output is CPU-only for now')
    if mode != 'raster':
        raise ValueError(f'Unknown 3D render mode {mode!r}')
    if output == 'normals_blend':
        output = 'normals'   # without splats (they returned above) the blend is the plain first-hit pass
    if output == 'shade':
        raise Unsupported('Viewport shade mode is not implemented by wgpu')
    if output not in scene3d.RENDER_OUTPUTS:
        raise ValueError(f'Unknown 3D render output {output!r}')
    if any(g.projection is not None for g in scene.geometries):
        raise Unsupported('Camera-projected geometry is not implemented by wgpu')
    if sum(len(g.triangles) for g in scene.geometries) > scene3d.MAX_TRIANGLES:
        raise Unsupported(f'Scene exceeds {scene3d.MAX_TRIANGLES} triangles')
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError('Render dimensions must be positive')
    samples = max(1, min(int(samples), 4)) if output in scene3d.LIGHT_OUTPUTS else 1
    _cancel(cancel)
    # Z1 of 2 finish: an area light's Monte Carlo loop traces one shadow ray per sample per fragment
    # (`area_light_shade`), so it costs `light_samples` shadow-ray units, not one, in the same work
    # estimate a Directional/Point/Spot light uses (`_band_plan` below then still picks the right BVH
    # threshold and submission band count for the real ray count).
    shadow_count = sum(
        (int(np.clip(light.light_samples, 1, scene3d.SHADOW_SAMPLES_MAX * 4)) if light.kind in scene3d._AREA else 1)
        for light in scene.lights if light.shadows and light.intensity > 0
    ) if output in ('rgba', 'diffuse', 'specular') else 0
    triangles = sum(len(g.triangles) for g in scene.geometries) if shadow_count else 0
    work = width * height * samples ** 2 * shadow_count * triangles
    with _lock:
        state = _state(adapter)
        volume_bands = 0
        if volumes:
            from . import gpuvolume, volumerender
            volume = volume if volume is not None else volumerender.VolumeSettings()
            gpuvolume.check(state, scene, volume)
        if scene.splats:
            from . import gpusplat
            reason = (gpusplat.check_data_capability(state) if output in gpusplat.DATA_OUTPUTS
                      else gpusplat.check_capability(state))
            if reason is not None:
                raise Unsupported(reason)
        global last_shadow_path
        last_shadow_path = 'brute'
        shadow_prepared = None
        bvh_data = None
        limits = state['device'].limits if 'device' in state else {}
        bvh_threshold = (SHADOW_BVH_THRESHOLD if SHADOW_BVH_THRESHOLD is not None
                         else SHADOW_BVH_THRESHOLDS[_adapter_kind(state)])
        if triangles > bvh_threshold and _bvh_capable(limits, 48, triangles*4):
            entry = _shadow_cache(state, scene, triangles)
            if 'prepared' not in entry:
                entry['prepared'] = _shadow_data(scene, triangles, limits['max-storage-buffer-binding-size'], cancel)
            shadow_prepared = entry['prepared']
            if 'bvh' not in entry:
                packed = shadow_prepared[0]
                tri_set = raytrace.TriangleSet(packed[:, 0, :3], packed[:, 1, :3],
                                              packed[:, 2, :3], packed[:, 0, 3])
                bvh = raytrace.Bvh.build(*tri_set.aabbs(), cancel=cancel)
                entry['bvh'] = _pack_bvh(bvh, cancel) if _bvh_capable(limits, len(bvh.left)*48, triangles*4) else None
            _shadow_cache_keep(state, entry)
            if entry['bvh'] is not None:
                bvh_data = entry['bvh']
                last_shadow_path = 'bvh'
                levels = math.log2(triangles+2)
                work = width * height * samples**2 * shadow_count * 16 * levels + 16 * triangles * levels
        if volumes:
            # The meshes' shadows on the smoke cost a walk of the triangle tree per lit sample (sixteen box tests a level) when the
            # raster shadows use the tree, else a test of every triangle: the work the frame is cut into bands by.
            lit_lights = sum(light.intensity > 0 for light in scene.lights)
            tree_cost = 16 * math.log2(triangles + 2) if last_shadow_path == 'bvh' else triangles
            volume_bands = gpuvolume.band_plan(state, gpuvolume.work_estimate(
                scene, camera, width*samples, height*samples, volume, lit_lights, tree_cost), height*samples)
        bands = _band_plan(state, work, last_shadow_path, height*samples)
        if volume_bands > len(bands):
            rows = height*samples
            bands = [(i * rows // volume_bands, (i + 1) * rows // volume_bands) for i in range(volume_bands)]
        _cancel(cancel)
        state['picture_rgba'] = None
        state['splats_on_card'] = False
        splat_job = None
        if volumes and scene.splats:
            # Smoke together with splats (the CPU draws the smoke over the meshes and splats, cut at the mesh depth): the splat layer
            # goes onto the card between the mesh passes and the smoke march. A shadow of the smoke on relit or shadow-catching
            # splats stays with the CPU outside playback; a played frame lights its splats without shadows of any kind, as it does
            # without smoke. The old readback route stays with the CPU.
            smoke_shades_splats = (volume.shadow_density > 0 and any(light.shadows and light.intensity > 0 for light in scene.lights)
                                   and any(getattr(i, 'relight', 0) > 0 or getattr(i, 'shadow_catch', 0) > 0 for i in scene.splats))
            if (not SPLATS_ON_CARD or (smoke_shades_splats and not is_playing()) or any(
                    getattr(e, 'visible_to_camera', False) for e in environments) or any(
                    light.visible_to_camera and light.kind in scene3d._AREA and light.intensity > 0 for light in scene.lights)):
                raise Unsupported('smoke together with splats is drawn on the card without smoke shadows on relit splats (outside playback) and '
                                  'without a visible environment or area light; the CPU draws the rest')
        if (scene.splats and output == 'rgba' and SPLATS_ON_CARD and not any(
                getattr(e, 'visible_to_camera', False) for e in environments) and not any(
                light.visible_to_camera and light.kind in scene3d._AREA and light.intensity > 0 for light in scene.lights)):
            # The splat layer drawn over the meshes on the card, in the mesh frame's own submission (`gpusplat.prepare_on_card`)
            lighting = ((scene.lights, ambient, None, _splat_extras(scene, ambient, None, cancel)) if any(
                getattr(i, 'relight', 0) > 0 for i in scene.splats) else None)
            splat_job = (scene.splats, lighting)
        result = _render(state, scene, camera, width*samples, height*samples,
                         background, ambient, output, cancel, triangles, shadow_prepared, bvh_data, bands=bands,
                         volume=volume if volumes else None, splat_job=splat_job)
        picture, state['picture_rgba'] = state['picture_rgba'], None     # made beside this frame, valid while nothing changes it
        splats_drawn = state.pop('splats_on_card', False)
        if volumes and scene.splats and not splats_drawn:
            # the smoke is over the splat layer, which only the card route composites in that order
            raise Unsupported('smoke together with splats is drawn only on the card route; the CPU draws this frame')
        if scene.splats and output in gpusplat.DATA_OUTPUTS:
            _splat_data_pass(state, scene, camera, width, height, result, output, cancel)
        elif scene.splats and not splats_drawn and not glass_on_card:
            mesh_depth = None
            if scene.geometries:
                depth = _render(state, scene, camera, width*samples, height*samples,
                                (0, 0, 0, 0), ambient, 'depth', cancel)
                # The depth shader interpolates -view.z, already positive
                # camera-forward distance (not radial ray distance).
                mesh_depth = np.where(depth[..., 3] > 0, depth[..., 0], np.inf)
            lighting = ((scene.lights, ambient, None, _splat_extras(scene, ambient, None, cancel)) if any(
                getattr(i, 'relight', 0) > 0 for i in scene.splats) else None)
            splat_rgb, splat_alpha = gpusplat.render_layer(
                state, scene.splats, camera, width*samples, height*samples,
                mesh_depth, lighting=lighting, cancel=cancel)
            result[..., :3] = splat_rgb + (1-splat_alpha[..., None])*result[..., :3]
            result[..., 3] = splat_alpha + (1-splat_alpha)*result[..., 3]
        visible_background = output == 'rgba' and (any(getattr(e, 'visible_to_camera', False) for e in environments) or any(
                light.visible_to_camera and light.kind in scene3d._AREA and light.intensity > 0
                for light in scene.lights))
        if visible_background:
            # Y1 of 2 finish (2), the "background image" part: a `visible_to_camera` Environment
            # replaces the flat background on camera rays that hit nothing, exactly like
            # `scene3d._visible_background` -- reused directly (not reimplemented) on this GPU
            # readback, since it is plain NumPy; a `visible_to_camera` area light seen straight on is
            # drawn by it too.
            depth_image = _render(state, scene, camera, width*samples, height*samples,
                                  (0, 0, 0, 0), ambient, 'depth', cancel)
            depth = np.where(depth_image[..., 3] > 0, depth_image[..., 0], np.inf)
            view_eye, view_matrix = scene3d._view_basis(camera)
            focal = 1 / math.tan(math.radians(camera.fov) / 2)
            scene3d._visible_background(scene, width*samples, height*samples, result, depth, view_eye,
                                        view_matrix, focal, (width*samples)/(height*samples),
                                        float(np.clip(background[3], 0, 1)))
    if glass_on_card and not splats_drawn:
        # The card could not take the translucent meshes with the splats (several bands, a coloured background, a mesh held back
        # from the resident path): the ray tracer's layer merge makes the frame.
        return render(scene, camera, width, height, background, ambient, samples, output, cancel, adapter, mode='raytrace')
    if samples > 1:
        result = scene3d.box_resolve(result, samples)
    result.flags.writeable = False
    if (picture is not None and output == 'rgba' and samples == 1 and (splats_drawn or not scene.splats) and not visible_background
            and picture.shape[:2] == result.shape[:2]):
        gpupicture.register(result, gpupicture.armed_view(), picture)
    _mark('resolve_done')
    return result


def _splat_data_pass(state, scene, camera, width, height, result, output, cancel):
    """Splats in the `depth`, `position` and `object_id` outputs, written over `result` (the GPU mesh render of
    the same output) where a splat is the first hit, as `scene3d.render` does on the CPU: the first splat
    where accumulated opacity reaches one half in front of the opaque mesh depth wins the pixel at full alpha,
    meshes the splats do not cover keep their own values. Caller holds the state lock."""
    from . import gpusplat
    mesh_depth = None
    if scene.geometries:
        # The depth shader interpolates -view.z, the positive camera-forward distance the splat pass compares.
        depth = result if output == 'depth' else _render(state, scene, camera, width, height,
                                                         (0, 0, 0, 0), 0.0, 'depth', cancel)
        mesh_depth = np.where(depth[..., 3] > 0, depth[..., 0], np.inf)
    hit, splat_depth, object_id = gpusplat._render_data(
        state, scene.splats, camera, width, height, mesh_depth, len(scene.geometries), cancel, None)
    if not hit.any():
        return
    if output == 'depth':
        value = np.repeat(splat_depth[hit][:, None], 3, axis=1)
    elif output == 'object_id':
        value = np.zeros((int(hit.sum()), 3), 'f4')
        value[:, 0] = object_id[hit]
    else:
        eye, view = scene3d._view_basis(camera)
        basis = view.astype(np.float64).copy()
        basis[2] *= -1
        tan = np.tan(np.deg2rad(camera.fov)/2)
        focal = height/(2*tan)
        rows, cols = np.nonzero(hit)
        rays = np.column_stack(((cols+.5-width/2)/focal, (height/2-(rows+.5))/focal, np.ones(len(rows))))
        value = eye + (rays @ np.linalg.inv(basis).T)*splat_depth[hit][:, None].astype(np.float64)
    result[hit, :3] = value
    result[hit, 3] = 1


def _render_volume_passes(scene, camera, width, height, background, output, cancel, adapter, mode, volume):
    """A volume control pass, or the depth output merged with the volumes' first hit, raymarched on the GPU
    (gpuvolume.render_passes) over the opaque meshes' depth, which the GPU raster path renders first."""
    from . import gpuvolume, volumerender
    if mode == 'raytrace' or scene.splats:
        raise Unsupported('volume passes drawn with the ray tracer or together with splats are CPU-only')
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError('Render dimensions must be positive')
    settings = volume if volume is not None else volumerender.VolumeSettings()
    if not scene.volumes:
        return np.zeros((height, width, 4), np.float32)
    meshes = replace(scene, volumes=(), particles=(), lights=(), splats=())
    if meshes.geometries:
        image = render(meshes, camera, width, height, output='depth', cancel=cancel, adapter=adapter, mode=mode)
        mesh_depth = np.where(image[..., 3] > 0, image[..., 0], np.inf).astype(np.float32)
    else:
        image = np.zeros((height, width, 4), np.float32)
        mesh_depth = np.full((height, width), np.inf, np.float32)
    with _lock:
        state = _state(adapter)
        got = gpuvolume.render_passes(state, scene, camera, width, height, mesh_depth, settings, output, cancel)
    if output == 'depth':
        first = (got['first_t'] / got['length']).astype(np.float32)
        out = np.array(image, np.float32)
        hit = first < mesh_depth
        out[hit, :3] = first[hit, None]
        out[hit, 3] = 1
    else:
        out = volumerender.finish_pass(got, output.split('_', 1)[1], camera, settings)
    out.flags.writeable = False
    return out


def light_table(lights):
    """The 20-float-per-light storage table the raster and volume shaders read; `lights` are (light, position, direction)."""
    light_data = np.zeros((max(1, len(lights)), 20), 'f4')
    for i, (light, position, direction) in enumerate(lights):
        light_data[i, :3] = position
        light_data[i, 3] = light.kind in scene3d._POSITIONAL
        light_data[i, 4:7] = direction
        light_data[i, 7] = light.shadows
        light_data[i, 8:11] = np.asarray(light.color)*light.intensity
        light_data[i, 11], light_data[i, 12:16] = scene3d._falloff_power(light), scene3d._cone_terms(light)
        light_data[i, 16:19] = scene3d._shadow_terms(light)
    return light_data


def _area_light_resources(lights):
    """(lights_table (L, 12) f32, samples_table (S, 8) f32) for the `AreaLight`/`AreaSample` storage
    buffers `_SHADER` reads (Z1 of 2 finish): the same fixed low-discrepancy sample points and normals
    `scene3d._area_light_contribution` computes on the CPU (`scene3d._area_light_samples`), uploaded
    once and reused by every fragment. One row per light even with none, like `light_table` above.

    `lights_table` columns: center.xyz, two_sided, radiance.xyz, area, sample offset, sample count,
    shadow bias scale (matching `scene3d._shadow_terms`'s own bias-scale column), shadows-on flag."""
    lights_data = np.zeros((max(1, len(lights)), 12), 'f4')
    sample_rows = []
    offset = 0
    for i, light in enumerate(lights):
        center, _ = light.world()
        radiance, area = scene3d._area_light_radiance(light)
        count = int(np.clip(light.light_samples, 1, scene3d.SHADOW_SAMPLES_MAX * 4))
        points, normals = scene3d._area_light_samples(light, count)
        lights_data[i, 0:3] = center
        lights_data[i, 3] = float(bool(light.two_sided))
        lights_data[i, 4:7] = radiance
        lights_data[i, 7] = area
        lights_data[i, 8] = offset
        lights_data[i, 9] = count
        lights_data[i, 10] = light.shadow_bias / scene3d.SHADOW_BIAS_DEFAULT
        lights_data[i, 11] = float(bool(light.shadows))
        rows = np.zeros((count, 8), 'f4')
        rows[:, 0:3] = points
        rows[:, 4:7] = normals
        sample_rows.append(rows)
        offset += count
    samples_data = np.concatenate(sample_rows) if sample_rows else np.zeros((1, 8), 'f4')
    return lights_data, samples_data


def _environment_resources(state, keep, environments):
    """The `EnvGlobals` uniform block and the atlas texture `_SHADER`'s group(2) reads: `environments[0]`
    (raster mode allows at most one, checked by the caller), or a disabled/zeroed block when there is
    none -- the shader always declares group(2), so a bind group is always needed. `rot0`/`rot1`/`rot2`
    are the columns of `envlight.Environment._local`'s combined rotation (the environment's own turn
    and its parent's), read back into a `mat3x3<f32>` by `env_local`, so a parented or rotated
    Environment matches the CPU reference exactly, unlike viewportgpu.py's root-level-only dome.

    The overwhelmingly common case has no Environment at all, so that disabled block/atlas/sampler is
    built once per adapter `state` and cached there (`env_disabled`), not rebuilt (and not `keep`-ed
    for per-call destruction) on every render -- the ordinary per-material textures are already rebuilt
    fresh each call, but this one never varies, and a fresh 64x32x6 texture upload on every mesh render
    would be pure waste."""
    from . import envlight
    wgpu, device = state['wgpu'], state['device']
    if not environments:
        cached = state.get('env_disabled')
        if cached is None:
            block = np.zeros(9 * 4 + 4 * 4, np.float32)
            atlas = np.ones((6, 1, 4), np.float32)
            buffer = device.create_buffer_with_data(data=block, usage=wgpu.BufferUsage.UNIFORM)
            texture = device.create_texture(size=(1, 6, 1), format='rgba16float',
                usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
            device.queue.write_texture({'texture': texture, 'mip_level': 0, 'origin': (0, 0, 0)},
                np.ascontiguousarray(atlas, 'f2'), {'bytes_per_row': 8, 'rows_per_image': 6}, (1, 6, 1))
            sampler = device.create_sampler(mag_filter='linear', min_filter='linear', address_mode_u='repeat')
            cached = state['env_disabled'] = (buffer, texture.create_view(), sampler)
        return cached
    tile_w, tile_h = envlight.PREFILTER_SIZE
    block = np.zeros(9 * 4 + 4 * 4, np.float32)
    env = environments[0]
    pre = env._pre()
    gain = float(env.intensity) * np.asarray(env.tint, np.float64)
    dirs, _ = envlight.direction_grid(tile_w, tile_h)
    level0 = envlight.sample_map(pre.levels[0], dirs.reshape(-1, 3)).reshape(tile_h, tile_w, 3)
    levels = [level0] + [np.asarray(level, np.float64) for level in pre.levels[1:]]
    rgb = (np.concatenate(levels, axis=0) * gain[None, None, :]).astype(np.float32)
    atlas = np.concatenate([rgb, np.ones(rgb.shape[:2] + (1,), np.float32)], axis=-1)
    # `array<vec4<f32>, 9>` pads every SH coefficient to 16 bytes; only the first 3 floats of
    # each 4 are used (see `env_diffuse`'s `envg.sh[i].rgb`).
    block[:36].reshape(9, 4)[:, :3] = (pre.sh * gain[None, :]).astype(np.float32)
    angle = math.radians(float(env.rotation))
    turn = np.array(((math.cos(angle), 0, -math.sin(angle)), (0, 1, 0), (math.sin(angle), 0, math.cos(angle))))
    parent = np.asarray(env.parent, np.float64)[:3, :3]
    norms = np.linalg.norm(parent, axis=0)
    rot = parent / np.where(norms > 1e-12, norms, 1.0)
    matrix = (rot.T @ turn).astype(np.float32)   # envlight.Environment._local: d @ matrix.T
    block[36:39], block[40:43], block[44:47] = matrix[:, 0], matrix[:, 1], matrix[:, 2]
    block[48] = 1.0
    env_buffer = keep(device.create_buffer_with_data(data=block, usage=wgpu.BufferUsage.UNIFORM))
    atlas_texture = keep(device.create_texture(size=(atlas.shape[1], atlas.shape[0], 1), format='rgba16float',
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST))
    device.queue.write_texture({'texture': atlas_texture, 'mip_level': 0, 'origin': (0, 0, 0)},
        np.ascontiguousarray(atlas, 'f2'), {'bytes_per_row': atlas.shape[1] * 8, 'rows_per_image': atlas.shape[0]},
        (atlas.shape[1], atlas.shape[0], 1))
    # `envlight.sample_map` wraps in U (longitude) but clamps in V; `address_mode_u='repeat'` matches
    # the wrap (the default is clamp-to-edge, which would seam at the u=0/1 meridian). V clamps to the
    # whole atlas, which is harmless on its own -- `env_specular` insets its own V reads by half a
    # texel so hardware bilinear never bleeds across two tiles' shared row into the wrong roughness level.
    sampler = device.create_sampler(mag_filter='linear', min_filter='linear', address_mode_u='repeat')
    return env_buffer, atlas_texture.create_view(), sampler


def _cpus():
    """The processors this process may run on (a laptop stand-in pinned with taskset reports its four, not the machine's thirty-two)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


READBACK_MIN_BYTES = 4 * 1024 * 1024      # a band smaller than this is copied by the calling thread alone
READBACK_PARTS = max(1, min(4, _cpus() // 2))      # threads that copy one band: half the processors, at most four
_readback_pool = {}


def _readback_threads():
    if 'pool' not in _readback_pool:
        from concurrent.futures import ThreadPoolExecutor
        _readback_pool['pool'] = ThreadPoolExecutor(max_workers=max(1, READBACK_PARTS - 1), thread_name_prefix='nb-readback')
    return _readback_pool['pool']


def _copy_band(staging, result, y0, y1, dtype, stride, width):
    """Copy rows y0..y1 of a mapped staging buffer into ``result`` with NumPy copies straight from the mapping.
    ``read_mapped()`` copies by default, into a Python bytearray with the GIL held (33 ms of the Qt thread's
    time at 1080p in a playing window, on top of the copy NumPy makes next); the view is dropped before returning
    so the caller can unmap. A band of 4 MB or more is copied in READBACK_PARTS slices of rows by as many threads
    (NumPy lets go of the GIL for a copy this size): the 33 MB of a 1080p float picture took 2.7 to 3.0 ms in one
    thread on every adapter, the one stage the timelines of all four test scenes share (10/23, Lane 4 step B)."""
    rows = y1 - y0
    view = staging.read_mapped(copy=False)
    raw = np.frombuffer(view, dtype, count=rows*stride//dtype.itemsize).reshape(rows, stride//dtype.itemsize)

    def part(a, b):
        result[y0 + a:y0 + b] = raw[a:b, :width*4].reshape(b - a, width, 4)
    parts = READBACK_PARTS if rows * width * 4 * dtype.itemsize >= READBACK_MIN_BYTES else 1
    if parts <= 1 or rows < parts:
        part(0, rows)
    else:
        step = -(-rows // parts)
        pool = _readback_threads()
        futures = [pool.submit(part, a, min(rows, a + step)) for a in range(step, rows, step)]
        part(0, step)
        for future in futures:
            future.result()
    del raw, view


def release_staging(state=None):
    """Destroy the readback buffer a state is holding (the next render makes a new one)."""
    state = _state() if state is None else state
    held = state.pop('staging', None)
    if held is not None:
        held.destroy()


STAGING_KEEP_BYTES = 256 * 1024 * 1024      # the readback buffer a state holds on to between renders


STAGING_RELEASE_BYTES = 16 * 1024 * 1024


def _release_staging(state):
    """Let go of the readback buffer a played frame kept, before a ray-traced render. With a mapped-read buffer of 24 MB or more alive
    (a played 1080p frame and its 8 MB display picture), the ray tracer's own band readbacks on the RTX 3080 Ti took 3.2 times as long
    (the stop's full-quality picture of the mixed scene: 1.7 s with 16 MB held, 5.5 s with 25 MB, 10/23); a played frame makes it again
    in one allocation. Call with the GPU lock held."""
    held = state.get('staging')
    if held is not None and held.size > STAGING_RELEASE_BYTES:
        held.destroy()
        state['staging'] = None


def _staging_buffer(state, size, keep):
    """The MAP_READ buffer a render reads its pixels back through. Creating and destroying one per render cost
    209 ms for the 133 MB of a 1080p picture drawn at 2 samples (docs/PERFORMANCE.md, Lane 4 playback profile), so a
    state keeps the last one and hands it out again while it is big enough. Held buffers are bounded by
    STAGING_KEEP_BYTES; a bigger one lives for its render only. Call with the GPU lock held."""
    wgpu, device = state['wgpu'], state['device']
    held = state.get('staging')
    if held is not None and held.size >= size:
        state['staging_reuses'] = state.get('staging_reuses', 0) + 1
        return held
    if held is not None:
        held.destroy()
        state['staging'] = None
    buffer = device.create_buffer(size=size, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
    state['staging_creates'] = state.get('staging_creates', 0) + 1
    if size <= STAGING_KEEP_BYTES:
        state['staging'] = buffer
        return buffer
    return keep(buffer)


# ---- a scene whose camera moves keeps what it uploaded ---------------------------------------------------------------
#
# Playing a 3D comp under an animated camera redraws the same scene from a new place every frame. The vertex buffer holds
# world-space attributes only (the vertex shader turns them into camera space with `Params.view0..2`), so a frame where
# nothing but the camera changed needs one small uniform write and the draw commands, not a repack and an upload of every
# triangle, texture, light table, bind group and pipeline binding (about 35 ms of a furnished-room frame, docs/PERFORMANCE.md).
# Scenes with textures that carry mip chains, holdouts, particles or volumes keep the per-frame path: their vertex data or
# passes depend on the camera.

RESIDENT = True      # off: every frame packs and uploads its scene again (the comparison tests use it)
SPLATS_ON_CARD = True      # a splat layer is drawn over the resident meshes on the card; False: the layer comes back to the processor
RESIDENT_SCENES = 2  # a scene that also draws a depth pass for its splats keeps one entry for each of the two passes
OPAQUE_ALPHA = np.float32(0.999)        # the shader's PASS 0 / PASS 1 split of `source.a`


def _resident_ok(scene, output, volume):
    """True when the scene can be drawn from resident buffers: nothing depends on the camera but the uniform. Smoke is drawn after
    the meshes from the volume's own cached textures, so a scene with volumes is resident too (it took 1.3 s a frame on the
    per-frame path, the meshes' packing and upload, and 22 ms on this one: the mixed scene with a smoke puff, 10/23)."""
    if not RESIDENT:
        return False
    if output in _PARTICLE_OUTPUT_MODES and getattr(scene, 'particles', ()):
        return False
    drawn = 0
    for geometry in scene.geometries:
        if geometry.holdout:
            return False
        if not geometry.visible_to_camera:
            continue
        drawn += 1
        if geometry.texture is not None and geometry.uvs is not None and max(geometry.texture.shape[:2]) > 1:
            return False          # a mip chain: the level depends on each triangle's size on screen
    return drawn > 0


def _mesh_key(scene, order):
    """Everything the packed vertices, the materials and the draw lists are made of: the arrays (by `_array_key`; the
    resident entry keeps the geometries alive, so an identity cannot be reused) and the scalar fields."""
    return tuple((_array_key(g.vertices), _array_key(g.triangles), _array_key(g.normals), _array_key(g.uvs), _array_key(g.texture),
                  _array_key(g.metallic_roughness_texture), _array_key(g.normal_texture), _array_key(g.occlusion_texture),
                  _array_key(g.emissive_texture), g.world_matrix().tobytes(), tuple(float(v) for v in g.color),
                  float(g.specular), float(g.shininess), float(g.emission), g.material, float(g.metallic),
                  float(g.pbr_roughness), float(g.pbr_specular), tuple(float(v) for v in g.emissive_color),
                  float(g.normal_scale), float(g.occlusion_strength), bool(g.receive_shadows),
                  bool(g.visible_to_camera), link_mask(g.light_link, order)) for g in scene.geometries)


def _environment_key(environments):
    if not environments:
        return None
    e = environments[0]
    return (e.fingerprint, float(e.intensity), float(e.rotation), tuple(float(v) for v in e.tint),
            np.asarray(e.parent, 'f8').tobytes())


def _pack_static(geometry, world, normals, uvs, object_id, tint, pbr, maps, emissive_color, scales):
    """`_pack_triangles` without the camera: (packed (K,3,38) float32 for every triangle, world-space triangle centres).
    Columns 0 to 2 (the camera-space position) stay zero; the vertex shader works it out."""
    tris = np.asarray(geometry.triangles)
    if not len(tris):
        return np.empty((0, 3, 38), 'f4'), np.empty((0, 3), 'f4')
    w0, w1, w2 = world[tris[:, 0]], world[tris[:, 1]], world[tris[:, 2]]
    e1, e2 = w1 - w0, w2 - w0
    if normals is None:
        face = np.cross(e1, e2)
        face = face / np.maximum(np.linalg.norm(face, axis=1, keepdims=True), 1e-8)
        ns = np.repeat(face[:, None, :], 3, axis=1)
    else:
        ns = normals[tris]
    uv = uvs[tris]
    duv1, duv2 = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
    det = duv1[:, 0]*duv2[:, 1] - duv2[:, 0]*duv1[:, 1]
    good = np.abs(det) > 1e-12
    safe = np.where(good, det, 1.0)[:, None]
    tangent = np.where(good[:, None], (e1*duv2[:, 1:2] - e2*duv1[:, 1:2]) / safe, e1)
    count = len(tris)
    packed = np.zeros((count, 3, 38), 'f4')
    packed[:, :, 3:6] = world[tris]
    packed[:, :, 6:9] = ns
    packed[:, :, 9:11] = uv
    packed[:, :, 11:15] = tint
    packed[:, :, 16:19] = (geometry.specular, geometry.shininess, geometry.emission)
    packed[:, :, 19] = object_id
    packed[:, :, 20:24] = pbr
    packed[:, :, 24:27] = np.repeat(tangent[:, None, :], 3, axis=1)
    packed[:, :, 27:31] = maps
    packed[:, :, 31:34] = emissive_color
    packed[:, :, 34:38] = scales
    centres = ((w0 + w1 + w2) / 3).astype('f4')
    return packed, centres


def _prepare_static(scene, order, cancel):
    """The camera-free half of `_prepare`: (vertices (3K,38) float32, materials, opaque draws [(first vertex, vertex count,
    material)], translucent triangles {start, centre, material} or None, the draws of every geometry for the data passes).
    Triangles are never dropped here (the hardware
    clips what is behind the camera or beyond the far plane, as it did for the triangles `_prepare` kept)."""
    chunks, materials, opaque, everything, starts, centres, mats = [], [], [], [], [], [], []
    emitted = 0
    for object_id, geometry in enumerate(scene.geometries, 1):
        _cancel(cancel)
        if not geometry.visible_to_camera:
            continue
        (world, normals, uvs, mips, maps_textures, tint, pbr, maps, emissive_color, scales) = _geometry_terms(geometry, order)
        packed, centre = _pack_static(geometry, world, normals, uvs, object_id, tint, pbr, maps, emissive_color, scales)
        if not len(packed):
            continue
        material = len(materials)
        materials.append((mips, *maps_textures))
        everything.append((emitted*3, len(packed)*3, material))
        alpha = np.float32(tint[3])
        if geometry.texture is not None and geometry.uvs is not None:
            alpha = np.float32(alpha * np.float32(np.float16(geometry.texture[..., 3].flat[0])))
        if alpha >= OPAQUE_ALPHA:
            opaque.append((emitted*3, len(packed)*3, material))
        else:
            starts.append(emitted*3 + 3*np.arange(len(packed)))
            centres.append(centre)
            mats.append(np.full(len(packed), material))
        chunks.append(packed.reshape(-1, 38))
        emitted += len(packed)
    vertices = np.concatenate(chunks) if chunks else np.empty((0, 38), 'f4')
    translucent = ({'start': np.concatenate(starts), 'centre': np.concatenate(centres), 'material': np.concatenate(mats)}
                   if starts else None)
    if translucent is not None:
        # the translucent triangles' vertices again, one row per triangle: a frame writes them far to near into one buffer
        translucent['vertices'] = np.ascontiguousarray(vertices.reshape(-1, 3, 38)[translucent['start'] // 3])
    return vertices, materials, opaque, translucent, everything


def _translucent_runs(translucent, eye, view, ordered=True):
    """The translucent triangles far to near as (first vertex, vertex count, material) draws; neighbours in the vertex
    buffer with the same material go out as one draw. `ordered` False: in vertex order, where the neighbours are the whole mesh
    (a pass that only writes depth, the glass depth the splat layer is ordered against, does not care and needs a draw per
    material instead of one per triangle: 2,300 draws of a glass sphere a frame)."""
    if not ordered:
        order = np.argsort(translucent['start'], kind='stable')
    else:
        depth = -((translucent['centre'] - eye) @ view[2])
        order = np.argsort(-depth, kind='stable')
    starts, mats = translucent['start'][order], translucent['material'][order]
    new = np.ones(len(order), bool)
    new[1:] = (mats[1:] != mats[:-1]) | (starts[1:] != starts[:-1] + 3)
    first = np.flatnonzero(new)
    last = np.append(first[1:], len(order))
    return [(int(starts[i]), 3*int(j - i), int(mats[i])) for i, j in zip(first, last)]


def _sorted_translucent(res, state, eye, view):
    """(draws, vertex buffer) of the translucent triangles far to near: the frame's order is written into a buffer of their own
    (their vertices gathered in that order: ~1 ms for 2,300 triangles) so a run of one material is a single draw, where drawing
    from the shared buffer in depth order was a draw call per triangle (14 ms a frame of a glass sphere in Python)."""
    translucent = res['translucent']
    depth = -((translucent['centre'] - eye) @ view[2])
    order = np.argsort(-depth, kind='stable')
    rows = translucent['vertices'][order]
    buffer = res.get('sorted')
    if buffer is None:
        wgpu = state['wgpu']
        buffer = res['sorted'] = state['device'].create_buffer(size=rows.nbytes, usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST)
    state['device'].queue.write_buffer(buffer, 0, rows)
    mats = translucent['material'][order]
    new = np.ones(len(order), bool)
    new[1:] = mats[1:] != mats[:-1]
    first = np.flatnonzero(new)
    last = np.append(first[1:], len(order))
    return [(3*int(i), 3*int(j - i), int(mats[i])) for i, j in zip(first, last)], buffer


def _light_tables(scene):
    """(point/spot/directional lights with their world pose, their table, the area-light table, the area samples)."""
    all_lights = [light for light in scene.lights if light.intensity > 0]
    area = [light for light in all_lights if light.kind in scene3d._AREA]
    lights = [(light, *light.world()) for light in all_lights if light.kind not in scene3d._AREA]
    area_lights_data, area_samples_data = _area_light_resources(area)
    return lights, light_table(lights), area, area_lights_data, area_samples_data


def drop_resident(state=None, targets=True):
    """Let go of what a state keeps for its resident scene (and, unless ``targets`` is off, its colour and depth targets).
    The buffers and textures are released with their last reference rather than destroyed by hand: a state dict that a
    test restores from a snapshot may still point at them."""
    state = _state() if state is None else state
    state.pop('residents', None)
    if targets:
        state.pop('targets', None)


def _build_resident(state, scene, order, key, light_data, area_lights_data, area_samples_data, shadow_data, bvh_data,
                    shadow_triangles, cancel):
    """Pack the scene without the camera and upload everything a frame binds. The buffers and textures are released
    with the entry (see `drop_resident`), so nothing is tracked for destruction here."""
    wgpu, device = state['wgpu'], state['device']
    keep = lambda resource: resource
    vertices, materials, opaque, translucent, everything = _prepare_static(scene, order, cancel)
    res = {'key': key, 'refs': (list(scene.geometries), shadow_data, bvh_data), 'opaque': opaque,
           'translucent': translucent, 'everything': everything, 'triangles': len(vertices) // 3, 'binds': {}}
    res['uniform'] = device.create_buffer(size=112, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
    res['lights'] = device.create_buffer_with_data(data=light_data, usage=wgpu.BufferUsage.STORAGE)
    res['area_lights'] = device.create_buffer_with_data(data=area_lights_data, usage=wgpu.BufferUsage.STORAGE)
    res['area_samples'] = device.create_buffer_with_data(data=area_samples_data, usage=wgpu.BufferUsage.STORAGE)
    res['shadow'] = device.create_buffer_with_data(data=shadow_data, usage=wgpu.BufferUsage.STORAGE)
    res['bvh'] = []
    if bvh_data is not None:
        for binding, array in zip((5, 6), bvh_data):
            buffer = device.create_buffer_with_data(data=array, usage=wgpu.BufferUsage.STORAGE)
            res['bvh'].append({'binding': binding, 'resource': {'buffer': buffer}})
    res['vertices'] = device.create_buffer_with_data(data=vertices, usage=wgpu.BufferUsage.VERTEX)
    res['sampler'] = device.create_sampler(mag_filter='linear', min_filter='linear', mipmap_filter='linear')
    res['env'] = _environment_resources(state, keep, getattr(scene, 'environments', ()))
    _, res['textures'] = _upload_materials(wgpu, device, keep, materials, cancel)
    return res


def _resident_binds(state, res, phase, smoke=False):
    """(pipeline, per-material bind groups, environment bind group) for one pass; rebuilt when the pipeline cache was reset.
    `smoke`: the variant that reads the volumes' density for their shadows on the meshes (group 1 is bound by the caller)."""
    bvh = bool(res['bvh'])
    pipeline = _pipeline(state, phase == 2, phase, bvh, smoke)
    held = res['binds'].get((phase, smoke) if smoke else phase)
    if held is not None and held[0] is pipeline:
        return held
    device = state['device']
    groups = [device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=[
        {'binding': 0, 'resource': {'buffer': res['uniform']}}, {'binding': 1, 'resource': {'buffer': res['lights']}},
        {'binding': 2, 'resource': texture}, {'binding': 3, 'resource': res['sampler']},
        {'binding': 4, 'resource': {'buffer': res['shadow']}},
        {'binding': 7, 'resource': mr_tex}, {'binding': 8, 'resource': normal_tex},
        {'binding': 9, 'resource': occlusion_tex}, {'binding': 10, 'resource': emissive_tex},
        {'binding': 11, 'resource': {'buffer': res['area_lights']}},
        {'binding': 12, 'resource': {'buffer': res['area_samples']}}] + res['bvh'])
        for texture, mr_tex, normal_tex, occlusion_tex, emissive_tex in res['textures']]
    env_buffer, env_atlas, env_sampler = res['env']
    env_group = device.create_bind_group(layout=pipeline.get_bind_group_layout(2), entries=[
        {'binding': 0, 'resource': {'buffer': env_buffer}}, {'binding': 1, 'resource': env_atlas},
        {'binding': 2, 'resource': env_sampler}])
    slot = (phase, smoke) if smoke else phase
    res['binds'][slot] = (pipeline, groups, env_group)
    return res['binds'][slot]


def _frame_targets(state, width, height, fmt):
    """The colour and depth textures of a frame size and format, kept between renders (a camera move does not change
    them; a scene that also draws a depth pass for its splats alternates two formats)."""
    wgpu, device = state['wgpu'], state['device']
    held = state.setdefault('targets', {})
    key = (width, height, fmt)
    if key not in held:
        while len(held) >= 3:
            held.pop(next(iter(held)))
        held[key] = (device.create_texture(size=(width, height, 1), format=fmt,
                                           usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC
                                           | wgpu.TextureUsage.TEXTURE_BINDING),
                     device.create_texture(size=(width, height, 1), format='depth32float',
                                           usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING))
    target, depth = held[key]
    return target.create_view(), depth.create_view(), target


def _glass_depth(state, width, height):
    """The depth buffer of the translucent meshes' nearest surfaces (see `_render_resident`), kept between renders."""
    wgpu, device = state['wgpu'], state['device']
    held = state.setdefault('glass_depth', {})
    key = (width, height)
    if key not in held:
        while len(held) >= 2:
            held.pop(next(iter(held))).destroy()
        held[key] = device.create_texture(size=(width, height, 1), format='depth32float',
                                          usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING)
    return held[key].create_view()


def _card_finishes_frame(scene, splats_drawn=False):
    """True when nothing is drawn over the card's mesh frame afterwards (a splat layer, a visible environment or area light):
    only then is the picture made beside it the frame's picture. A scene with splats made the splat layer 5 times slower when the
    picture's 8 MB of readback memory was allocated beside it (mixed scene 7.3 to 2.8 fps, 10/23), and its picture is thrown away anyway."""
    return (splats_drawn or not scene.splats) and not any(getattr(e, 'visible_to_camera', False) for e in getattr(scene, 'environments', ())) \
        and not any(light.visible_to_camera and light.kind in scene3d._AREA and light.intensity > 0 for light in scene.lights)


def _render_resident(state, scene, camera, width, height, bg, ambient, output, cancel, shadow_triangles, shadow_data, bias,
                     bvh_data, bands, splat_job=None, volume=None):
    """`_render` for a scene `_resident_ok` accepts: the same passes over resident buffers. `splat_job` (the splat instances
    and their lighting) asks for the splat layer to be drawn over the meshes in the same submission (`gpusplat.prepare_on_card`);
    `state['splats_on_card']` says afterwards whether it was."""
    wgpu, device = state['wgpu'], state['device']
    order = link_order(scene)
    lights, light_data, area, area_lights_data, area_samples_data = _light_tables(scene)
    key = (_mesh_key(scene, order), light_data.tobytes(), area_lights_data.tobytes(), area_samples_data.tobytes(),
           _environment_key(getattr(scene, 'environments', ())), shadow_triangles,
           id(shadow_data) if shadow_triangles else None, id(bvh_data[0]) if bvh_data is not None else None)
    held = state.setdefault('residents', {})
    res = held.get(key)
    if res is None:
        res = _build_resident(state, scene, order, key, light_data, area_lights_data, area_samples_data, shadow_data,
                              bvh_data, shadow_triangles, cancel)
        held[key] = res
        while len(held) > RESIDENT_SCENES:          # oldest first (insertion order)
            held.pop(next(iter(held)))
        state['resident_builds'] = state.get('resident_builds', 0) + 1
    else:
        held[key] = held.pop(key)                   # most recent last
        state['resident_reuses'] = state.get('resident_reuses', 0) + 1
    _mark('scene_ready')
    if not res['triangles']:
        return np.broadcast_to(bg, (height, width, 4)).copy()
    data = output in scene3d.DATA_OUTPUTS
    eye, view = scene3d._view_basis(camera)
    focal = 1 / math.tan(math.radians(camera.fov) / 2)
    params = np.array([focal/(width/height), focal, camera.near, camera.far, *eye, 0, ambient, len(lights),
                       scene3d.RENDER_OUTPUTS.index(output), len(area), bias, shadow_triangles, bvh_data is not None, 0,
                       *_view_rows(camera)], 'f4')
    device.queue.write_buffer(res['uniform'], 0, params)
    fmt = 'rgba32float' if data else state['format']
    target_view, depth_view, target = _frame_targets(state, width, height, fmt)
    has_volumes = volume is not None and output == 'rgba' and bool(getattr(scene, 'volumes', ()))
    # The smoke shadows the meshes: the shadowed lights' rays are marched through the volumes' density (gpuvolume).
    smoke = bool(has_volumes and shadow_triangles and volume.shadow_density > 0
                 and any(light.shadows and light.intensity > 0 for light in scene.lights))
    if data:
        # One pass over every triangle, depth tested: the order does not matter.
        runs_by_pass = [res['everything']]
        passes = [_resident_binds(state, res, 2)]
    else:
        sorted_runs, sorted_buffer = _sorted_translucent(res, state, eye, view) if res['translucent'] else ([], None)
        runs_by_pass = [res['opaque'], sorted_runs]
        passes = [_resident_binds(state, res, 0, smoke)] + ([_resident_binds(state, res, 1, smoke)] if runs_by_pass[1] else [])
    _mark('upload_done')
    dtype = np.dtype('f4' if fmt == 'rgba32float' else 'f2')
    stride = ((width*4*dtype.itemsize+255)//256)*256
    held = []
    def keep(resource):
        held.append(resource)
        return resource
    # The mesh depth is the pass's own depth buffer, which holds the opaque surfaces only. A scene with translucent meshes also
    # draws their nearest surface into a second depth buffer, and the splat layer is drawn in two parts around the translucent
    # pass: the splats behind the glass first (the glass tints them), the ones in front of it after.
    splats_here = (splat_job is not None and not data
                   and ((len(bands) == 1 and not bg.any()) or has_volumes))      # a smoke march in bands: the background goes under the finished frame
    glass = splats_here and bool(runs_by_pass[1])
    state['splats_on_card'] = splats_here
    used = set()
    light_order = [light for light, *_ in lights]
    smoke_groups = []
    volume_pass = None
    if has_volumes:
        from . import gpuvolume
        if smoke:
            smoke_groups = [gpuvolume.mesh_shadow_group(state, binds[0], scene, volume, used, keep, light_order) for binds in passes]
        volume_pass = gpuvolume.prepare(state, scene, camera, width, height, ambient, volume, res['lights'], len(lights),
                                        bool(scene.lights), depth_view, keep, target=fmt, used=used, shadow_buffer=res['shadow'],
                                        shadow_count=shadow_triangles, shadow_bias=bias, light_order=light_order,
                                        bvh=[entry['resource']['buffer'] for entry in res['bvh']] if res['bvh'] else None)
    # The display picture of a played frame (nodebased/gpupicture.py) is made by one more pass in the last band's submission and
    # read back through the same buffer, after the float rows.
    picture_view = (gpupicture.armed_view() if output == 'rgba' and not data and not bg.any()
                    and _card_finishes_frame(scene, state['splats_on_card']) else None)
    recorder = gpupicture.prepare(state, picture_view, width, height) if picture_view is not None else None
    if picture_view is None and gpupicture.warm_view() is not None:
        gpupicture.warm(state, gpupicture.warm_view())       # a frame that is not played builds the display pass the played ones will use
    float_bytes = stride*max(y1-y0 for y0, y1 in bands)
    picture_offset = (float_bytes + 255)//256*256
    staging = _staging_buffer(state, picture_offset + (recorder.bytes if recorder is not None else 0), keep)
    result = np.empty((height, width, 4), 'f4')
    state['picture_rgba'] = None
    splat_layer = False            # a frame with smoke in several bands prepares its splat layer once: False until then, then it or None
    try:
        for y0, y1 in bands:
            _cancel(cancel)
            rows = y1-y0
            encoder = device.create_command_encoder()
            def mesh_pass(encoder, pass_number, pipeline, groups, env_group, depth_target=None, runs=None):
                rp = encoder.begin_render_pass(color_attachments=[{'view': target_view,
                    'resolve_target': None, 'clear_value': (0, 0, 0, 0),
                    'load_op': 'clear' if pass_number == 0 else 'load', 'store_op': 'store'}],
                    depth_stencil_attachment={'view': depth_view if depth_target is None else depth_target, 'depth_clear_value': 1.0,
                        'depth_load_op': 'clear' if pass_number == 0 or depth_target is not None else 'load', 'depth_store_op': 'store'})
                rp.set_scissor_rect(0, y0, width, rows)
                rp.set_pipeline(pipeline)
                if smoke_groups and depth_target is None:
                    rp.set_bind_group(1, smoke_groups[pass_number])
                rp.set_bind_group(2, env_group)
                rp.set_vertex_buffer(0, res['vertices'] if pass_number == 0 or runs is not None else sorted_buffer)
                bound = None
                for first, count, material in (runs if runs is not None else runs_by_pass[0 if pass_number == 0 else 1]):
                    if bound != material:
                        rp.set_bind_group(0, groups[material])
                        bound = material
                    rp.draw(count, 1, first, 0)
                rp.end()
            for pass_number, (pipeline, groups, env_group) in enumerate(passes):
                if glass and pass_number == 1:
                    glass_view = _glass_depth(state, width, height)
                    mesh_pass(encoder, 1, *_resident_binds(state, res, 3), depth_target=glass_view,      # their depth, no colour
                              runs=_translucent_runs(res['translucent'], eye, view, ordered=False))
                    continue            # the translucent meshes themselves are drawn between the two parts of the splat layer
                mesh_pass(encoder, pass_number, pipeline, groups, env_group)
            if splats_here:
                # The card starts on the meshes now; the processor prepares the splat layer (order, colours, buffers: 8 ms
                # at 20,000 relit splats) while it draws, and the layer goes out in a second submission over the same frame.
                from . import gpusplat
                split = {'rows': (y0, rows)} if len(bands) > 1 else {}
                between = (lambda enc: mesh_pass(enc, 1, *passes[1])) if glass else None
                if splat_layer is False:
                    _mark('submit')
                    device.queue.submit([encoder.finish()])
                    splat_layer = gpusplat.prepare_on_card(state, splat_job[0], camera, width, height, depth_view,
                                                           lighting=splat_job[1], cancel=cancel, keep=keep,
                                                           glass_view=glass_view if glass else None)
                    encoder = device.create_command_encoder()
                    if splat_layer is not None:
                        splat_layer(encoder, target_view, between, **split)
                    elif glass:
                        mesh_pass(encoder, 1, *passes[1])
                elif splat_layer is not None:
                    splat_layer(encoder, target_view, between, project_now=False, **split)
                elif glass:
                    mesh_pass(encoder, 1, *passes[1])
            if volume_pass is not None:
                rp = encoder.begin_render_pass(color_attachments=[{'view': target_view, 'resolve_target': None,
                    'clear_value': (0, 0, 0, 0), 'load_op': 'load', 'store_op': 'store'}])
                rp.set_scissor_rect(0, y0, width, rows)
                volume_pass.record(rp)
                rp.end()
            encoder.copy_texture_to_buffer({'texture': target, 'origin': (0, y0, 0)},
                {'buffer': staging, 'bytes_per_row': stride, 'rows_per_image': rows}, (width, rows, 1))
            fused = recorder is not None and y1 == bands[-1][1]
            if fused:
                recorder.record(encoder, target_view, staging, picture_offset)
            _cancel(cancel)
            if not splats_here:
                _mark('submit')
            device.queue.submit([encoder.finish()])
            staging.map_sync(wgpu.MapMode.READ)
            _mark('gpu_done')
            try:
                _copy_band(staging, result, y0, y1, dtype, stride, width)
                if fused:
                    state['picture_rgba'] = recorder.read(staging)
            finally:
                staging.unmap()
            _mark('readback_done')
            _cancel(cancel)
    finally:
        for resource in held:
            resource.destroy()
    if bg.any():
        result += bg*(1-result[..., 3:4])
    return result


def _upload_materials(wgpu, device, keep, materials, cancel):
    """(the 1x1 stand-in view, one tuple of views per material): base colour with its mip chain, then the four PBR maps
    (the stand-in where a map is missing, so every pipeline variant binds something valid at group(0) bindings 7..10)."""
    dummy = keep(device.create_texture(size=(1, 1, 1), format='rgba16float',
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST))
    device.queue.write_texture({'texture': dummy, 'mip_level': 0}, np.zeros((1, 1, 4), 'f2'),
        {'bytes_per_row': 8, 'rows_per_image': 1}, (1, 1, 1))
    dummy_view = dummy.create_view()
    def single_level(array):
        if array is None:
            return dummy_view
        h, w = array.shape[:2]
        t = keep(device.create_texture(size=(w, h, 1), format='rgba16float',
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST))
        device.queue.write_texture({'texture': t, 'mip_level': 0}, np.ascontiguousarray(array, 'f2'),
            {'bytes_per_row': w*8, 'rows_per_image': h}, (w, h, 1))
        return t.create_view()
    textures = []
    for mips, mr_array, normal_array, occlusion_array, emissive_array in materials:
        _cancel(cancel)
        h, w = mips[0].shape[:2]
        # Complete the rectangular tail too; reference LOD is capped at its last mip.
        mips = list(mips)
        while max(mips[-1].shape[:2]) > 1:
            t = mips[-1]
            if t.shape[0] == 1:
                t = t[:, :t.shape[1]//2*2].reshape(1, -1, 2, 4).mean(axis=2)
            else:
                t = t[:t.shape[0]//2*2].reshape(-1, 2, 1, 4).mean(axis=1)
            mips.append(t)
        texture = keep(device.create_texture(size=(w, h, 1), format='rgba16float', mip_level_count=len(mips),
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST))
        for level, mip in enumerate(mips):
            mh, mw = mip.shape[:2]
            device.queue.write_texture({'texture': texture, 'mip_level': level}, np.ascontiguousarray(mip, 'f2'),
                {'bytes_per_row': mw*8, 'rows_per_image': mh}, (mw, mh, 1))
        textures.append((texture.create_view(), single_level(mr_array), single_level(normal_array),
                         single_level(occlusion_array), single_level(emissive_array)))
    return dummy_view, textures


def _render(state, scene, camera, width, height, background, ambient, output, cancel, shadow_triangles=0, shadow_prepared=None, bvh_data=None, *, bands=None, volume=None, splat_job=None):
    wgpu, device = state['wgpu'], state['device']
    data = output in scene3d.DATA_OUTPUTS
    if bands is None:
        bands = _band_plan(state, 0, 'brute', height)
    if shadow_prepared is None and shadow_triangles:
        entry = _shadow_cache(state, scene, shadow_triangles)
        if 'prepared' not in entry:
            entry['prepared'] = _shadow_data(scene, shadow_triangles, device.limits['max-storage-buffer-binding-size'], cancel)
        _shadow_cache_keep(state, entry)
        shadow_prepared = entry['prepared']
    shadow_data, bias = shadow_prepared if shadow_prepared is not None else _shadow_data(scene, shadow_triangles,
        device.limits['max-storage-buffer-binding-size'], cancel)
    _cancel(cancel)
    bg = np.asarray(background, 'f4').copy()
    bg[3] = np.clip(bg[3], 0, 1)
    bg[:3] *= bg[3]
    if output != 'rgba':
        bg[:] = 0
    if _resident_ok(scene, output, volume):
        return _render_resident(state, scene, camera, width, height, bg, ambient, output, cancel, shadow_triangles,
                                shadow_data, bias, bvh_data, bands, splat_job, volume)
    eye, focal, vertices, queue, materials = _prepare(scene, camera, width, height, cancel)
    _mark('scene_ready')
    _cancel(cancel)
    sprites = None
    if output in _PARTICLE_OUTPUT_MODES and getattr(scene, 'particles', ()):
        sprites = particle_data(scene, camera, width, height, device.limits, cancel, output=output)
    has_volumes = volume is not None and output == 'rgba' and bool(getattr(scene, 'volumes', ()))
    if not vertices and sprites is None and not has_volumes:
        return np.broadcast_to(bg, (height, width, 4)).copy()
    resources = []
    def keep(resource):
        resources.append(resource)
        return resource
    try:
        all_lights = [light for light in scene.lights if light.intensity > 0]
        # Z1 of 2 finish: Rect/Disc/Sphere lights go through the separate `AreaLight`/`AreaSample`
        # buffers below (`_area_light_resources`), not the Directional/Point/Spot `lights` table --
        # `light_table` has no "kind" column, so a mis-filed area light would shade as a directional one.
        area = [light for light in all_lights if light.kind in scene3d._AREA]
        lights = [(light, *light.world()) for light in all_lights if light.kind not in scene3d._AREA]
        light_data = light_table(lights)
        area_lights_data, area_samples_data = _area_light_resources(area)
        params = np.array([focal/(width/height), focal, camera.near, camera.far,
                           *eye, 0, ambient, len(lights), scene3d.RENDER_OUTPUTS.index(output), len(area),
                           bias, shadow_triangles, bvh_data is not None, 0, *_view_rows(camera)], 'f4')
        uniform = keep(device.create_buffer_with_data(data=params, usage=wgpu.BufferUsage.UNIFORM))
        light_buffer = keep(device.create_buffer_with_data(data=light_data, usage=wgpu.BufferUsage.STORAGE))
        area_light_buffer = keep(device.create_buffer_with_data(data=area_lights_data, usage=wgpu.BufferUsage.STORAGE))
        area_sample_buffer = keep(device.create_buffer_with_data(data=area_samples_data, usage=wgpu.BufferUsage.STORAGE))
        shadow_buffer = keep(device.create_buffer_with_data(data=shadow_data, usage=wgpu.BufferUsage.STORAGE))
        bvh_entries = []
        if bvh_data is not None:
            for binding, array in zip((5, 6), bvh_data):
                buffer = keep(device.create_buffer_with_data(data=array, usage=wgpu.BufferUsage.STORAGE))
                bvh_entries.append({'binding': binding, 'resource': {'buffer': buffer}})
        vertex_buffer = keep(device.create_buffer_with_data(data=np.concatenate(vertices), usage=wgpu.BufferUsage.VERTEX)) if vertices else None
        sampler = device.create_sampler(mag_filter='linear', min_filter='linear', mipmap_filter='linear')
        # `_SHADER` always declares group(2) (the environment), so every mesh pipeline needs a bind
        # group there even with no Environment in the scene (`_environment_resources` then returns
        # the disabled/zeroed block and a harmless 1x1 atlas).
        env_buffer, env_atlas, env_sampler = (
            _environment_resources(state, keep, getattr(scene, 'environments', ())) if vertices else (None, None, None))
        # Y3 of 3, part 1: a single harmless 1x1 texture, shared by every material missing a given PBR
        # map, so every pipeline variant binds something valid at group(0) bindings 7..10 even though
        # the fragment shader's `maps` flags (`_prepare`) keep it from ever sampling the dummy.
        dummy_view, textures = _upload_materials(wgpu, device, keep, materials, cancel)
        fmt = 'rgba32float' if data else state['format']
        target = keep(device.create_texture(size=(width, height, 1), format=fmt,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC))
        depth = keep(device.create_texture(size=(width, height, 1), format='depth32float',
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | (wgpu.TextureUsage.TEXTURE_BINDING if has_volumes else 0)))
        def bind(pipeline):
            groups = [device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=[
                {'binding': 0, 'resource': {'buffer': uniform}}, {'binding': 1, 'resource': {'buffer': light_buffer}},
                {'binding': 2, 'resource': texture}, {'binding': 3, 'resource': sampler},
                {'binding': 4, 'resource': {'buffer': shadow_buffer}},
                {'binding': 7, 'resource': mr_tex}, {'binding': 8, 'resource': normal_tex},
                {'binding': 9, 'resource': occlusion_tex}, {'binding': 10, 'resource': emissive_tex},
                {'binding': 11, 'resource': {'buffer': area_light_buffer}},
                {'binding': 12, 'resource': {'buffer': area_sample_buffer}}]
                + bvh_entries) for texture, mr_tex, normal_tex, occlusion_tex, emissive_tex in textures]
            env_group = device.create_bind_group(layout=pipeline.get_bind_group_layout(2), entries=[
                {'binding': 0, 'resource': {'buffer': env_buffer}}, {'binding': 1, 'resource': env_atlas},
                {'binding': 2, 'resource': env_sampler}])
            return groups, env_group
        passes = []
        used = set()
        # The smoke shadows meshes: the shadowed lights' rays are marched through the volumes' density (gpuvolume).
        smoke = bool(has_volumes and shadow_triangles and volume.shadow_density > 0
                     and any(light.shadows and light.intensity > 0 for light in scene.lights))
        smoke_groups = []      # per mesh pass: an auto layout belongs to its own pipeline
        for phase in (() if not vertices else (2,) if data else (0, 1)):
            _cancel(cancel)
            pipeline = _pipeline(state, data, phase, bvh_data is not None, smoke)
            if smoke:
                from . import gpuvolume
                smoke_groups.append(gpuvolume.mesh_shadow_group(state, pipeline, scene, volume, used, keep,
                                                                [light for light, *_ in lights]))
            passes.append((pipeline, *bind(pipeline)))
        hold_pass = None
        if vertices and not data and any(entry[3] for entry in queue):
            # holdout triangles are drawn by a pipeline that replaces the colour under them with zero (beauty, first pass)
            pipeline = _pipeline(state, data, 0, bvh_data is not None, False, hold=True)
            hold_pass = (pipeline, *bind(pipeline))
        particle_pass = None
        if sprites is not None:
            instance_data, texel_data, particle_params = sprites
            pipeline = particle_pipeline(state, target=fmt, data=data)
            group = device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=[
                {'binding': i, 'resource': {'buffer': keep(device.create_buffer_with_data(data=array, usage=usage))}}
                for i, (array, usage) in enumerate(((particle_params, wgpu.BufferUsage.UNIFORM),
                    (instance_data, wgpu.BufferUsage.STORAGE), (texel_data, wgpu.BufferUsage.STORAGE)))])
            particle_pass = (pipeline, group, None)
        volume_pass = None
        if has_volumes:
            from . import gpuvolume
            volume_pass = gpuvolume.prepare(state, scene, camera, width, height, ambient, volume, light_buffer,
                                            len(lights), bool(scene.lights), depth.create_view(), keep, target=fmt, used=used,
                                            shadow_buffer=shadow_buffer, shadow_count=shadow_triangles, shadow_bias=bias,
                                            light_order=[light for light, *_ in lights],
                                            bvh=[entry['resource']['buffer'] for entry in bvh_entries] if bvh_entries else None)
        target_view, depth_view = target.create_view(), depth.create_view()
        _mark('upload_done')
        dtype = np.dtype('f4' if fmt == 'rgba32float' else 'f2')
        stride = ((width*4*dtype.itemsize+255)//256)*256
        staging = _staging_buffer(state, stride*max(y1-y0 for y0, y1 in bands), keep)
        result = np.empty((height, width, 4), 'f4')
        # Replaying the draw list multiplies host recording cost, acceptable only
        # for heavy-shadow scenes. Ordinary renders retain one submission.
        splat_layer = False            # the splat layer of a frame with smoke: False until it is prepared, then it or None
        for y0, y1 in bands:
            _cancel(cancel)
            rows = y1-y0
            encoder = device.create_command_encoder()
            if not passes and particle_pass is None:
                # Volumes only: the first pass still has to clear the colour and the depth the volumes read.
                encoder.begin_render_pass(color_attachments=[{'view': target_view, 'resolve_target': None,
                    'clear_value': (0, 0, 0, 0), 'load_op': 'clear', 'store_op': 'store'}],
                    depth_stencil_attachment={'view': depth_view, 'depth_clear_value': 1.0,
                        'depth_load_op': 'clear', 'depth_store_op': 'store'}).end()
            for pass_number, (pipeline, groups, env_group) in enumerate(passes + ([particle_pass] if particle_pass else [])):
                _cancel(cancel)
                rp = encoder.begin_render_pass(color_attachments=[{'view': target_view,
                    'resolve_target': None, 'clear_value': (0, 0, 0, 0),
                    'load_op': 'clear' if pass_number == 0 else 'load', 'store_op': 'store'}],
                    depth_stencil_attachment={'view': depth_view, 'depth_clear_value': 1.0,
                        'depth_load_op': 'clear' if pass_number == 0 else 'load', 'depth_store_op': 'store'})
                rp.set_scissor_rect(0, y0, width, rows)
                rp.set_pipeline(pipeline)
                if smoke_groups and pass_number < len(passes):
                    rp.set_bind_group(1, smoke_groups[pass_number])
                if env_group is not None:
                    rp.set_bind_group(2, env_group)
                if pass_number == len(passes) and particle_pass:
                    # Drawn last, over the meshes: one instanced quad per sprite, far to near.
                    rp.set_bind_group(0, groups)
                    rp.draw(6, len(instance_data), 0, 0)
                else:
                    rp.set_vertex_buffer(0, vertex_buffer)
                    use_hold = hold_pass is not None and pass_number == 0
                    drawing_holdout, current = False, groups
                    # Triangles that follow each other in the vertex buffer with the same material go out as one draw,
                    # and the bind group is set when the material changes: 4,400 triangles were 8,800 Python calls.
                    run_start, run_count, bound = 0, 0, None
                    for _, start, material, holdout in queue:
                        if use_hold and holdout != drawing_holdout:
                            if run_count:
                                rp.draw(3*run_count, 1, run_start, 0)
                                run_count = 0
                            drawing_holdout = holdout
                            bound = None
                            if holdout:
                                rp.set_pipeline(hold_pass[0])
                                rp.set_bind_group(2, hold_pass[2])
                                current = hold_pass[1]
                            else:
                                rp.set_pipeline(pipeline)
                                rp.set_bind_group(2, env_group)
                                current = groups
                        if run_count and bound == material and start == run_start + 3*run_count:
                            run_count += 1
                            continue
                        if run_count:
                            rp.draw(3*run_count, 1, run_start, 0)
                        if bound != material:
                            rp.set_bind_group(0, current[material])
                            bound = material
                        run_start, run_count = start, 1
                    if run_count:
                        rp.draw(3*run_count, 1, run_start, 0)
                rp.end()
            if splat_job is not None and has_volumes and not data:
                # The meshes go out first; the processor prepares the splat layer while the card draws them (once, for the first
                # band), and the layer is drawn over the meshes (reading their depth) before the smoke is marched over both.
                from . import gpusplat
                if splat_layer is False:
                    _mark('submit')
                    device.queue.submit([encoder.finish()])
                    splat_layer = gpusplat.prepare_on_card(state, splat_job[0], camera, width, height, depth_view,
                                                           lighting=splat_job[1], cancel=cancel, keep=keep)
                    encoder = device.create_command_encoder()
                    if splat_layer is not None:
                        splat_layer(encoder, target_view, rows=(y0, rows))
                elif splat_layer is not None:
                    splat_layer(encoder, target_view, rows=(y0, rows), project_now=False)
                state['splats_on_card'] = True
            if volume_pass is not None:
                _cancel(cancel)
                rp = encoder.begin_render_pass(color_attachments=[{'view': target_view, 'resolve_target': None,
                    'clear_value': (0, 0, 0, 0), 'load_op': 'load', 'store_op': 'store'}])
                rp.set_scissor_rect(0, y0, width, rows)
                volume_pass.record(rp)
                rp.end()
            encoder.copy_texture_to_buffer({'texture': target, 'origin': (0, y0, 0)},
                {'buffer': staging, 'bytes_per_row': stride, 'rows_per_image': rows}, (width, rows, 1))
            # A submitted band cannot be interrupted; cancellation bounds the next submission.
            _cancel(cancel)
            if not state.get('splats_on_card'):
                _mark('submit')
            device.queue.submit([encoder.finish()])
            staging.map_sync(wgpu.MapMode.READ)
            _mark('gpu_done')
            try:
                _copy_band(staging, result, y0, y1, dtype, stride, width)
            finally:
                staging.unmap()
            _mark('readback_done')
            _cancel(cancel)
        if not data and bg.any():
            # A transparent black background (the default) adds exactly zero; the three full-frame temporaries
            # of this line cost about 15 ms at 1080p.
            result += bg*(1-result[..., 3:4])
        return result
    finally:
        for resource in reversed(resources):
            resource.destroy()
