"""Peak GPU memory tracking and a bounded-VRAM budget check (M3 gate, "bounded VRAM").

wgpu-native's own `wgpuGenerateReport` (wrapped by wgpu-py's `generate_report()`) only counts
*objects* (buffers allocated/kept/released) -- its own diagnostics text says so directly:
"Reported memory does not include buffer/texture data." It cannot answer "how many bytes." Nor
does wgpu-py expose a per-adapter OS/vendor memory counter (DXGI budget, VK_EXT_memory_budget,
NVML): those are platform-specific and outside wgpu's own API. So "the adapter's own report" is
unavailable in bytes; the number this module reports instead is NodeBased's own byte-accurate
account of every `wgpu.GPUBuffer`/`wgpu.GPUTexture` it allocates and frees, which is what the
renderer actually controls and the only VRAM number it can make precise and portable across the
three adapters. This is a finding, not an assumption going in: `instrument()`'s own test
(`tests/test_m3_vram_budget.py`) asserts `wgpu.backends.wgpu_native._helpers.generate_report()`
indeed carries no byte sizes, so the gap cannot silently reappear if a future wgpu-py version
starts to report bytes and this module should switch to trusting it instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Bytes per texel, keyed by every wgpu texture format this codebase creates (nodebased/gpu3d.py,
# gpurt_render.py, gpupathtrace.py, gpusplat.py, gpuvolume.py). An unlisted format is a real gap,
# not a guess: `_texture_bytes` raises rather than silently under-counting it.
TEXEL_BYTES = {
    'r8unorm': 1, 'r8uint': 1, 'r8sint': 1,
    'r16float': 2, 'r16uint': 2, 'r16sint': 2, 'rg8unorm': 2,
    'r32float': 4, 'r32uint': 4, 'r32sint': 4, 'rg16float': 4, 'rgba8unorm': 4, 'rgba8unorm-srgb': 4,
    'bgra8unorm': 4, 'depth32float': 4, 'depth24plus': 4,
    'rg32float': 8, 'rgba16float': 8, 'rgba16uint': 8,
    'rgba32float': 16, 'rgba32uint': 16,
}


class UnknownTextureFormat(ValueError):
    """A texture format with no entry in TEXEL_BYTES: the tracker cannot count its bytes."""


def _texture_bytes(size, format, mip_level_count=1, sample_count=1):
    """Total bytes for every mip level of one texture, the same geometric mip chain gpu3d.py's
    own `_mip_chain` halves down to a 1x1 base (CPU mips are generated the same way)."""
    try:
        texel = TEXEL_BYTES[format]
    except KeyError:
        raise UnknownTextureFormat(
            f'gpumemory.TEXEL_BYTES has no entry for texture format {format!r}; add it rather '
            f'than let this texture under-count the VRAM budget') from None
    width, height, depth = size
    total = 0
    for _level in range(max(1, int(mip_level_count))):
        total += max(width, 1) * max(height, 1) * max(depth, 1) * texel
        width, height = max(width // 2, 1), max(height // 2, 1)
    return total * max(1, int(sample_count))


@dataclass
class MemoryTracker:
    """Live byte accounting for one wgpu device: every buffer/texture this module wrapped."""
    current: int = 0
    peak: int = 0
    _sizes: dict = field(default_factory=dict)

    def _add(self, resource, nbytes):
        self._sizes[id(resource)] = nbytes
        self.current += nbytes
        self.peak = max(self.peak, self.current)

    def _remove(self, resource):
        nbytes = self._sizes.pop(id(resource), None)
        if nbytes is not None:
            self.current -= nbytes

    def reset_peak(self):
        """Start a fresh measurement window without losing track of what is still allocated."""
        self.peak = self.current


class BudgetExceeded(ValueError):
    """A render stayed under the per-scene VRAM budget's 30% headroom but not its budget."""


def instrument(device):
    """Wrap one GPUDevice's buffer/texture creation and destruction to tally live bytes.

    Idempotent: calling this twice on the same device returns the existing tracker rather than
    wrapping the wrappers. Every `.destroy()` on a tracked resource (explicit, or `keep()`-driven
    cleanup, as `gpu3d._render` already does at the end of every render) removes its bytes, so
    `tracker.current` reflects what is actually still resident, and `tracker.peak` is the high
    -water mark since the device (or the last `reset_peak()`) was created.
    """
    existing = getattr(device, '_nb_memory_tracker', None)
    if existing is not None:
        return existing
    tracker = MemoryTracker()
    original_create_buffer = device.create_buffer
    original_create_buffer_with_data = device.create_buffer_with_data
    original_create_texture = device.create_texture

    def _tracked(resource, nbytes):
        tracker._add(resource, nbytes)
        original_destroy = resource.destroy

        def destroy():
            tracker._remove(resource)
            original_destroy()

        resource.destroy = destroy
        return resource

    def create_buffer(*, size, **kwargs):
        return _tracked(original_create_buffer(size=size, **kwargs), int(size))

    def create_buffer_with_data(*, data, **kwargs):
        nbytes = data.nbytes if hasattr(data, 'nbytes') else len(bytes(data))
        return _tracked(original_create_buffer_with_data(data=data, **kwargs), nbytes)

    def create_texture(*, size, format, mip_level_count=1, sample_count=1, **kwargs):
        nbytes = _texture_bytes(size, format, mip_level_count, sample_count)
        return _tracked(original_create_texture(size=size, format=format,
                                                mip_level_count=mip_level_count,
                                                sample_count=sample_count, **kwargs), nbytes)

    device.create_buffer = create_buffer
    device.create_buffer_with_data = create_buffer_with_data
    device.create_texture = create_texture
    device._nb_memory_tracker = tracker
    return tracker


def render_bounded(render_fn, *args, tracker, budget_bytes, renderer_name, scene_label, **kwargs):
    """Call ``render_fn(*args, **kwargs)``; raise BudgetExceeded if its peak VRAM use (relative to
    the tracker's level when this started) exceeds ``budget_bytes``. Never leaves a device-side
    exception in place of a named one: the render itself has already finished (or raised its own
    error) by the time this checks, so the only new failure mode this adds is the clean refusal.
    """
    baseline = tracker.current
    tracker.reset_peak()
    result = render_fn(*args, **kwargs)
    used = tracker.peak - min(baseline, tracker.peak)
    if used > budget_bytes:
        raise BudgetExceeded(
            f'{renderer_name} exceeded its VRAM budget on {scene_label}: used {used:,} bytes, '
            f'budget {budget_bytes:,} bytes; lower resolution/samples, instance count or volume '
            f'resolution, or raise the budget for this scene class')
    return result
