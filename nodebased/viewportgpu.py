"""Interactive wgpu renderer for the 3D editor viewport.

This is not the Render3D backend. ``gpu3d`` reproduces the CPU reference renderer's per-triangle
sorting and mip choice, which costs one draw call per triangle; that is right for a final render
and far too slow to orbit. The viewport wants the opposite trade: mesh buffers are uploaded once
per geometry and cached, every object is one draw call against a real depth buffer, and orbiting
only rewrites a camera matrix. Shading follows ``scene3d._shade_fragments`` (headlight, Lambert,
Blinn-Phong specular, emission, textures and camera projection), with these viewport limits:

- transparency is sorted per object, not per triangle or per pixel;
- projection depth occlusion is not evaluated (the projection shows through occluders);
- at most ``MAX_LIGHTS`` lights shade the view, and shadows are never shown;
- Gaussian splats are a layout proxy, not the Render3D look: each splat is an opaque
  camera-facing disc in its SH-DC colour (no view-dependent colour, no blending), at most
  ``MAX_SPLATS`` per cloud with an even stride beyond that. ``Relight`` is followed per splat
  with the viewport's lights and ambient, without shadows;
- particles use Render3D's draw (gpu3d.particle_pipeline: sprites sorted far to near, blended over
  the meshes, depth tested, points and spheres as discs, cards with their texture), at most
  ``MAX_PARTICLES`` per set with an even stride beyond that;
- a Spot light shows its cone and falloff, and every light its distance falloff, as Render3D lights
  them (``gpu3d`` uniform layout: falloff power in ``color.w``, direction and cone terms);
- display uses the sRGB transfer curve and 4x multisampling.

Frames come back as 8-bit RGBA for QPainter. Editor lines (grid, axes, camera frustum, light
direction) are drawn here too, depth tested against the geometry.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from . import gpu3d, scene3d
from .splats import C0, _rotation, to_linear_color
from .splatshade import normal_confidence

MAX_LIGHTS = 16
MAX_SPLATS = 1_000_000    # per cloud; larger clouds are shown with an even stride
MAX_PARTICLES = 250_000   # per particle set; larger sets are shown with an even stride (sorted every frame)
SPLAT_SIGMA = 1.5         # disc radius, in standard deviations of the splat's middle axis
SPLAT_MAX_PIXELS = 2.5     # disc radius on screen never exceeds this
SPLAT_MIN_OPACITY = 0.05  # splats fainter than this (after the node's opacity scale) are hidden
SAMPLES = 4
_OBJECT_STRIDE = 512  # one Object struct, padded to a multiple of every adapter's offset alignment
_LOCK_TIMEOUT = 0.02  # seconds to wait for a Render3D job that holds the shared device

_SHADER = """
// place: position (positional) or the direction the light travels (Directional), w = positional;
// color: rgb x intensity, w = distance falloff power; direction: where a Spot points; cone: (is spot,
// inner, outer, exponent) in degrees, as gpu3d and scene3d.light_attenuation take them.
struct Light { place: vec4<f32>, color: vec4<f32>, direction: vec4<f32>, cone: vec4<f32> };
struct Globals {
    view_proj: mat4x4<f32>,
    eye: vec4<f32>,
    settings: vec4<f32>,      // ambient, mode (0 lit, 1 headlight, 2 unlit), light count, unused
    view_light: vec4<f32>,
    lights: array<Light, 16>,
    splat: vec4<f32>,         // view -> clip scale x, y; one pixel in clip units x, y
};
struct Object {
    model: mat4x4<f32>,
    normal: mat4x4<f32>,
    color: vec4<f32>,
    material: vec4<f32>,      // specular, shininess, emission, textured
    projector: mat4x4<f32>,
    projector_eye: vec4<f32>,
    projector_flags: vec4<f32>, // enabled, outside transparent, skip backfaces, unused
    projector_range: vec4<f32>, // near, far
};
@group(0) @binding(0) var<uniform> globals: Globals;
@group(1) @binding(0) var<uniform> object: Object;
@group(1) @binding(1) var surface: texture_2d<f32>;
@group(1) @binding(2) var surface_sampler: sampler;

fn attenuation(light: Light, point: vec3<f32>) -> f32 {
    // scene3d.light_attenuation: distance falloff, times the Spot cone.
    if (light.place.w < 0.5) { return 1.0; }
    let offset = point - light.place.xyz;
    let dist = length(offset);
    var result = 1.0;
    if (light.color.w > 0.0) { result = min(1.0, pow(max(dist, 1e-8), -light.color.w)); }
    if (light.cone.x > 0.0) {
        let angle = degrees(acos(clamp(dot(offset, light.direction.xyz) / max(dist, 1e-12), -1.0, 1.0)));
        var k = select(0.0, 1.0, angle <= light.cone.y);
        if (light.cone.z > light.cone.y) {
            let t = clamp((light.cone.z - angle) / (light.cone.z - light.cone.y), 0.0, 1.0);
            k = select(0.0, pow(t * t * (3.0 - 2.0 * t), light.cone.w), t > 0.0);
        }
        result = result * k;
    }
    return result;
}

struct Fragment {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

@vertex
fn mesh_vertex(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>,
               @location(2) uv: vec2<f32>) -> Fragment {
    var out: Fragment;
    let world = object.model * vec4<f32>(position, 1.0);
    out.clip = globals.view_proj * world;
    out.world = world.xyz;
    out.normal = (object.normal * vec4<f32>(normal, 0.0)).xyz;
    out.uv = uv;
    return out;
}

@fragment
fn mesh_fragment(in: Fragment) -> @location(0) vec4<f32> {
    var normal = in.normal / max(length(in.normal), 1e-8);
    let tint = vec4<f32>(object.color.rgb * object.color.a, object.color.a);
    var source = tint;
    var uv = in.uv;
    var rejected = false;
    if (object.projector_flags.x > 0.5) {
        let projected = object.projector * vec4<f32>(in.world, 1.0);
        let depth = projected.w;
        uv = projected.xy / max(depth, 1e-8) * 0.5 + 0.5;
        rejected = depth <= 0.0;
        if (object.projector_flags.y > 0.5) {
            rejected = rejected || depth <= object.projector_range.x || depth >= object.projector_range.y
                || uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0;
        }
        if (object.projector_flags.z > 0.5) {
            rejected = rejected || dot(normal, object.projector_eye.xyz - in.world) <= 0.0;
        }
    }
    // Sample outside branches: derivatives must stay in uniform control flow.
    let texel = textureSample(surface, surface_sampler, vec2<f32>(uv.x, 1.0 - uv.y));
    if (object.material.w > 0.5) {
        source = texel * tint;
    }
    if (rejected) {
        discard;
    }
    let toward_eye = globals.eye.xyz - in.world;
    if (dot(normal, toward_eye) < 0.0) {
        normal = -normal;
    }
    let emissive = source.rgb * object.material.z;
    var rgb = source.rgb;
    if (globals.settings.y > 0.5 && globals.settings.y < 1.5) {
        rgb = rgb * (0.25 + 0.75 * abs(dot(normal, globals.view_light.xyz)));
    } else if (globals.settings.y < 0.5) {
        let to_eye = toward_eye / max(length(toward_eye), 1e-8);
        var radiance = vec3<f32>(globals.settings.x);
        var specular = vec3<f32>(0.0);
        for (var i = 0u; i < u32(globals.settings.z); i = i + 1u) {
            let light = globals.lights[i];
            var to_light = -light.place.xyz;
            if (light.place.w > 0.5) {
                let delta = light.place.xyz - in.world;
                to_light = delta / max(length(delta), 1e-8);
            }
            let lambert = dot(normal, to_light);
            let factor = attenuation(light, in.world);
            radiance = radiance + max(lambert, 0.0) * factor * light.color.rgb;
            if (object.material.x > 0.0 && lambert > 0.0) {
                let half_vector = to_light + to_eye;
                let lobe = pow(max(dot(normal, half_vector / max(length(half_vector), 1e-8)), 0.0),
                               object.material.y);
                specular = specular + object.material.x * lobe * factor * light.color.rgb;
            }
        }
        rgb = rgb * radiance + specular * source.a;
    }
    return vec4<f32>(rgb + emissive, source.a);
}

struct SplatFragment {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) corner: vec2<f32>,
};

// One camera-facing disc per splat. object.color carries relight, opacity scale, world radius
// scale and the hide threshold. Shading follows splatshade.shade_splats, without shadows.
@vertex
fn splat_vertex(@location(0) corner: vec2<f32>, @location(1) place: vec4<f32>,
                @location(2) paint: vec4<f32>, @location(3) facing: vec4<f32>) -> SplatFragment {
    var out: SplatFragment;
    let world = object.model * vec4<f32>(place.xyz, 1.0);
    var clip = globals.view_proj * world;
    // Between one pixel and material.x pixels: captures carry huge soft splats (sky, haze) that
    // an opaque disc would turn into a wall in front of everything else.
    let extent = clamp(vec2<f32>(place.w * object.color.z) * globals.splat.xy,
                       globals.splat.zw * clip.w, globals.splat.zw * clip.w * object.material.x);
    clip = vec4<f32>(clip.xy + corner * extent, clip.zw);
    if (paint.a * object.color.y < object.color.w) {
        clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);  // beyond the far plane: clipped
    }
    var rgb = paint.rgb;
    if (object.color.x > 0.0) {
        let toward_eye = globals.eye.xyz - world.xyz;
        let to_eye = toward_eye / max(length(toward_eye), 1e-8);
        var normal = (object.normal * vec4<f32>(facing.xyz, 0.0)).xyz;
        normal = normal / max(length(normal), 1e-8);
        if (dot(normal, to_eye) < 0.0) {
            normal = -normal;
        }
        var effective = facing.w * normal + (1.0 - facing.w) * to_eye;
        effective = effective / max(length(effective), 1e-8);
        var radiance = vec3<f32>(globals.settings.x);
        if (globals.settings.y > 0.5 && globals.settings.y < 1.5) {
            radiance = vec3<f32>(0.25 + 0.75 * abs(dot(effective, globals.view_light.xyz)));
        } else {
            for (var i = 0u; i < u32(globals.settings.z); i = i + 1u) {
                let light = globals.lights[i];
                var to_light = -light.place.xyz;
                if (light.place.w > 0.5) {
                    let delta = light.place.xyz - world.xyz;
                    to_light = delta / max(length(delta), 1e-8);
                }
                radiance = radiance + max(dot(effective, to_light), 0.0) * attenuation(light, world.xyz) * light.color.rgb;
            }
        }
        rgb = mix(paint.rgb, paint.rgb * radiance, object.color.x);
    }
    out.clip = clip;
    out.color = rgb;
    out.corner = corner;
    return out;
}

@fragment
fn splat_fragment(in: SplatFragment) -> @location(0) vec4<f32> {
    if (dot(in.corner, in.corner) > 1.0) {
        discard;
    }
    return vec4<f32>(in.color, 1.0);
}

struct LineFragment { @builtin(position) clip: vec4<f32>, @location(0) color: vec4<f32> };

@vertex
fn line_vertex(@location(0) position: vec3<f32>, @location(1) color: vec4<f32>) -> LineFragment {
    var out: LineFragment;
    // Pull editor lines a hair toward the eye so they win against coplanar geometry.
    let nudged = globals.eye.xyz + (position - globals.eye.xyz) * 0.999;
    out.clip = globals.view_proj * vec4<f32>(nudged, 1.0);
    out.color = color;
    return out;
}

@fragment
fn line_fragment(in: LineFragment) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color.rgb * in.color.a, in.color.a);
}
"""


def view_projection(camera, width, height):
    """World -> clip matrix agreeing with scene3d.project() pixel for pixel (depth range 0..1)."""
    eye, view = scene3d._view_basis(camera)
    look = np.eye(4, dtype=np.float64)
    look[:3, :3] = view
    look[:3, 3] = -(view.astype(np.float64) @ eye.astype(np.float64))
    focal = 1.0 / math.tan(math.radians(camera.fov) / 2)
    aspect = width / max(height, 1)
    near, far = float(camera.near), float(camera.far)
    lens = np.zeros((4, 4), np.float64)
    lens[0, 0], lens[1, 1] = focal / aspect, focal
    lens[2, 2], lens[2, 3] = far / (near - far), near * far / (near - far)
    lens[3, 2] = -1.0
    return eye, lens @ look


def _soup(geometry):
    """(N*3, 8) float32 position | normal | uv, one vertex per triangle corner."""
    triangles = np.asarray(geometry.triangles, np.int64).reshape(-1, 3)
    corners = np.asarray(geometry.vertices, np.float32)[triangles]            # (T,3,3)
    if geometry.normals is not None:
        normals = np.asarray(geometry.normals, np.float32)[triangles]
    else:
        face = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        normals = np.repeat(face[:, None, :], 3, axis=1)
    if geometry.uvs is not None:
        uvs = np.asarray(geometry.uvs, np.float32)[triangles]
    else:
        uvs = np.zeros((len(triangles), 3, 2), np.float32)
    return np.ascontiguousarray(np.concatenate((corners, normals, uvs), axis=2).reshape(-1, 8), np.float32)


def splat_proxy(cloud, limit=MAX_SPLATS):
    """(N, 12) float32 disc instances in the cloud's local space, plus the stride that was used.

    Columns: position, disc radius | linear SH-DC colour, opacity | normal (shortest axis),
    normal confidence. Local space keeps the proxy valid while the node's transform changes.
    """
    stride = max(1, -(-len(cloud) // max(int(limit), 1)))
    index = np.arange(0, len(cloud), stride)
    scales = cloud.scales[index]
    out = np.empty((len(index), 12), np.float32)
    out[:, :3] = cloud.positions[index]
    out[:, 3] = np.sort(scales, axis=1)[:, 1] * SPLAT_SIGMA
    out[:, 4:7] = to_linear_color(np.maximum(0.5 + C0 * cloud.sh[index, 0, :], 0), cloud.colorspace)
    out[:, 7] = cloud.opacity[index]
    out[:, 8:11] = _rotation(cloud.rotations[index])[np.arange(len(index)), :, scales.argmin(axis=1)]
    out[:, 11] = normal_confidence(scales)
    return out, stride


def splat_radius_scale(instance, stride):
    """World size of one local unit of disc radius. A strided cloud has 1/stride of its discs,
    so they grow by sqrt(stride) (capped) to keep surfaces closed."""
    volume = abs(float(np.linalg.det(np.asarray(instance.matrix, np.float64)[:3, :3])))
    return volume ** (1 / 3) * float(instance.scale_scale) * min(math.sqrt(stride), 4.0)


class ViewportRenderer:
    """Owns the viewport's pipelines, targets and per-geometry caches on the shared wgpu device."""

    def __init__(self):
        state = gpu3d._state()
        self.wgpu, self.device = state["wgpu"], state["device"]
        self.description = gpu3d.describe()
        self.uploads = 0   # mesh uploads so far; tests and the status line read it
        self._meshes, self._textures, self._splats = {}, {}, {}
        self.splat_stride = 1  # largest stride among the clouds in the last frame (1: all shown)
        self.particle_stride = 1  # largest stride among the particle sets in the last frame
        self._state = state
        self._particle_buffers = {}
        self._targets = None
        self._lines = (None, None, 0)
        self._objects = None
        self._build()

    # --- setup ----------------------------------------------------------------------------------

    def _build(self):
        wgpu, device = self.wgpu, self.device
        module = device.create_shader_module(code=_SHADER)
        stage = wgpu.ShaderStage
        self._global_layout = device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": stage.VERTEX | stage.FRAGMENT, "buffer": {"type": "uniform"}}])
        self._object_layout = device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": stage.VERTEX | stage.FRAGMENT,
             "buffer": {"type": "uniform", "has_dynamic_offset": True, "min_binding_size": 272}},
            {"binding": 1, "visibility": stage.FRAGMENT, "texture": {"sample_type": "float"}},
            {"binding": 2, "visibility": stage.FRAGMENT, "sampler": {"type": "filtering"}}])
        self._globals = device.create_buffer(size=64 + 16 * 4 + 64 * MAX_LIGHTS,
                                             usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self._global_group = device.create_bind_group(layout=self._global_layout, entries=[
            {"binding": 0, "resource": {"buffer": self._globals}}])
        self._sampler = device.create_sampler(min_filter="linear", mag_filter="linear", mipmap_filter="linear",
                                              address_mode_u="clamp-to-edge", address_mode_v="clamp-to-edge")
        self._white = self._upload_texture(np.ones((1, 1, 4), np.float32))
        blend = {"color": {"src_factor": "one", "dst_factor": "one-minus-src-alpha", "operation": "add"},
                 "alpha": {"src_factor": "one", "dst_factor": "one-minus-src-alpha", "operation": "add"}}
        target = {"format": "rgba8unorm-srgb", "blend": blend}

        self._corners = device.create_buffer_with_data(
            data=np.array(((-1, -1), (1, -1), (-1, 1), (1, 1)), np.float32), usage=wgpu.BufferUsage.VERTEX)

        def pipeline(vertex, fragment, layouts, buffers, topology, depth_write):
            return device.create_render_pipeline(
                layout=device.create_pipeline_layout(bind_group_layouts=layouts),
                vertex={"module": module, "entry_point": vertex, "buffers": buffers},
                primitive={"topology": topology, "cull_mode": "none"},
                depth_stencil={"format": "depth24plus", "depth_write_enabled": depth_write,
                               "depth_compare": "less-equal"},
                multisample={"count": SAMPLES},
                fragment={"module": module, "entry_point": fragment, "targets": [target]})

        mesh_buffers = [{"array_stride": 32, "step_mode": "vertex", "attributes": [
            {"format": "float32x3", "offset": 0, "shader_location": 0},
            {"format": "float32x3", "offset": 12, "shader_location": 1},
            {"format": "float32x2", "offset": 24, "shader_location": 2}]}]
        line_buffers = [{"array_stride": 28, "step_mode": "vertex", "attributes": [
            {"format": "float32x3", "offset": 0, "shader_location": 0},
            {"format": "float32x4", "offset": 12, "shader_location": 1}]}]
        splat_buffers = [
            {"array_stride": 8, "step_mode": "vertex", "attributes": [
                {"format": "float32x2", "offset": 0, "shader_location": 0}]},
            {"array_stride": 48, "step_mode": "instance", "attributes": [
                {"format": "float32x4", "offset": 0, "shader_location": 1},
                {"format": "float32x4", "offset": 16, "shader_location": 2},
                {"format": "float32x4", "offset": 32, "shader_location": 3}]}]
        layouts = [self._global_layout, self._object_layout]
        self._splat_pipeline = pipeline("splat_vertex", "splat_fragment", layouts, splat_buffers,
                                        "triangle-strip", True)
        self._opaque = pipeline("mesh_vertex", "mesh_fragment", layouts, mesh_buffers, "triangle-list", True)
        self._blended = pipeline("mesh_vertex", "mesh_fragment", layouts, mesh_buffers, "triangle-list", False)
        self._line_pipeline = pipeline("line_vertex", "line_fragment", [self._global_layout], line_buffers,
                                       "line-list", False)

    def _upload_texture(self, image):
        """Mip-mapped rgba16float texture view from premultiplied float RGBA, row zero at the top."""
        wgpu, device = self.wgpu, self.device
        levels = scene3d._mip_chain(np.asarray(image, np.float32))
        height, width = levels[0].shape[:2]
        texture = device.create_texture(size=(width, height, 1), format="rgba16float",
                                        mip_level_count=len(levels),
                                        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        for index, level in enumerate(levels):
            h, w = level.shape[:2]
            data = np.ascontiguousarray(np.clip(level, -65504, 65504), np.float16)
            device.queue.write_texture({"texture": texture, "mip_level": index, "origin": (0, 0, 0)}, data,
                                       {"offset": 0, "bytes_per_row": w * 8, "rows_per_image": h}, (w, h, 1))
        return texture.create_view()

    def _ensure_targets(self, width, height):
        if self._targets and self._targets["size"] == (width, height):
            return self._targets
        wgpu, device = self.wgpu, self.device
        usage = wgpu.TextureUsage.RENDER_ATTACHMENT
        stride = (width * 4 + 255) // 256 * 256
        self._targets = dict(
            size=(width, height), stride=stride,
            color=device.create_texture(size=(width, height, 1), format="rgba8unorm-srgb",
                                        sample_count=SAMPLES, usage=usage).create_view(),
            depth=device.create_texture(size=(width, height, 1), format="depth24plus",
                                        sample_count=SAMPLES, usage=usage).create_view(),
            resolve=device.create_texture(size=(width, height, 1), format="rgba8unorm-srgb",
                                          usage=usage | wgpu.TextureUsage.COPY_SRC),
            readback=device.create_buffer(size=stride * height,
                                          usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ))
        self._targets["resolve_view"] = self._targets["resolve"].create_view()
        return self._targets

    # --- caches ---------------------------------------------------------------------------------

    def _mesh(self, geometry, used):
        # Keyed on the arrays, not the Geometry: moving an object re-evaluates the node but
        # keeps its mesh arrays when the generator caches them, so no upload happens.
        arrays = (geometry.vertices, geometry.triangles, geometry.normals, geometry.uvs)
        key = tuple(id(a) for a in arrays)
        used.add(key)
        entry = self._meshes.get(key)
        if entry is None:
            data = _soup(geometry)
            buffer = self.device.create_buffer_with_data(
                data=data if len(data) else np.zeros((3, 8), np.float32), usage=self.wgpu.BufferUsage.VERTEX)
            # The arrays are held so their ids cannot be recycled while the entry lives.
            entry = self._meshes[key] = (buffer, len(data), arrays)
            self.uploads += 1
        return entry

    def _splat(self, cloud, used):
        key = id(cloud)  # clouds are immutable and come from the loader's cache
        used.add(key)
        entry = self._splats.get(key)
        if entry is None:
            data, stride = splat_proxy(cloud)
            buffer = self.device.create_buffer_with_data(data=data, usage=self.wgpu.BufferUsage.VERTEX)
            entry = self._splats[key] = (buffer, len(data), stride, cloud)
            self.uploads += 1
        return entry

    def _texture(self, image, used):
        key = id(image)
        used.add(key)
        entry = self._textures.get(key)
        if entry is None:
            opaque = bool(np.asarray(image)[..., 3].min() >= 0.999)
            entry = self._textures[key] = (self._upload_texture(image), opaque, image)
        return entry

    def _object_group(self, view, count):
        size = max(count, 1) * _OBJECT_STRIDE
        if self._objects is None or self._objects[1] < size:
            buffer = self.device.create_buffer(size=size * 2, usage=self.wgpu.BufferUsage.UNIFORM
                                               | self.wgpu.BufferUsage.COPY_DST)
            self._objects = (buffer, size * 2, {})
        buffer, _size, groups = self._objects
        if id(view) not in groups:
            groups[id(view)] = (self.device.create_bind_group(layout=self._object_layout, entries=[
                {"binding": 0, "resource": {"buffer": buffer, "offset": 0, "size": 272}},
                {"binding": 1, "resource": view},
                {"binding": 2, "resource": self._sampler}]), view)
        return groups[id(view)][0]

    # --- frame ----------------------------------------------------------------------------------

    def render(self, scene, camera, width, height, background, lines=None, headlight=False, ambient=0.0):
        """Return (H, W, 4) uint8 sRGB-encoded RGBA, or None while a Render3D job holds the device.

        ``lines`` is an (N, 7) float32 array of line-list vertices: world xyz, straight RGBA.
        """
        if not gpu3d._lock.acquire(timeout=_LOCK_TIMEOUT):
            return None
        try:
            return self._render(scene, camera, int(width), int(height), background, lines, headlight, ambient)
        finally:
            gpu3d._lock.release()

    def _render(self, scene, camera, width, height, background, lines, headlight, ambient):
        wgpu, device = self.wgpu, self.device
        targets = self._ensure_targets(width, height)
        eye, view_proj = view_projection(camera, width, height)

        lights = [(light, *light.world()) for light in scene.lights if light.intensity > 0][:MAX_LIGHTS]
        block = np.zeros(32 + 16 * MAX_LIGHTS, np.float32)
        block[:16] = view_proj.T.ravel()
        focal = 1.0 / math.tan(math.radians(camera.fov) / 2)
        block[-4:] = focal * height / max(width, 1), focal, 2.0 / width, 2.0 / height
        block[16:19] = eye
        # As scene3d: the headlight when asked, else scene lights, else the authored colour unlit.
        block[20:23] = ambient, 1.0 if headlight else (0.0 if lights else 2.0), len(lights)
        block[24:27] = scene3d._VIEW_LIGHT
        for index, (light, position, direction) in enumerate(lights):
            at = 28 + index * 16
            positional = light.kind in scene3d._POSITIONAL
            block[at:at + 4] = (*(position if positional else direction), float(positional))
            block[at + 4:at + 7] = np.asarray(light.color, np.float32) * light.intensity
            block[at + 7] = scene3d._falloff_power(light)
            block[at + 8:at + 11] = direction
            block[at + 12:at + 16] = scene3d._cone_terms(light)
        device.queue.write_buffer(self._globals, 0, block)

        used_meshes, used_textures, used_splats, draws, clouds = set(), set(), set(), [], []
        object_count = len(scene.geometries) + len(scene.splats)
        uniforms = np.zeros((max(object_count, 1), _OBJECT_STRIDE // 4), np.float32)
        for index, geometry in enumerate(scene.geometries):
            buffer, count, _arrays = self._mesh(geometry, used_meshes)
            if not count:
                continue
            matrix = np.asarray(geometry.world_matrix(), np.float64)
            normal = np.eye(4)
            try:
                normal[:3, :3] = np.linalg.inv(matrix[:3, :3]).T
            except np.linalg.LinAlgError:
                pass  # a zero scale has no surface to shade
            projection = geometry.projection
            image = projection.texture if projection is not None else geometry.texture
            view, texture_opaque = self._texture(image, used_textures)[:2] if image is not None else (self._white, True)
            row = uniforms[index]
            row[:16], row[16:32] = matrix.T.ravel(), normal.T.ravel()
            row[32:36] = geometry.color
            row[36:40] = geometry.specular, geometry.shininess, geometry.emission, float(image is not None)
            if projection is not None:
                h, w = projection.texture.shape[:2]
                projector_eye, projector = view_projection(projection.camera, w, h)
                row[40:56] = projector.T.ravel()
                row[56:59] = projector_eye
                row[60:63] = 1.0, float(projection.outside == "transparent"), float(projection.backfaces == "skip")
                row[64:66] = projection.camera.near, projection.camera.far
            opaque = geometry.color[3] >= 0.999 and texture_opaque and projection is None
            centre = matrix[:3, 3] - eye
            draws.append((opaque, -float(centre @ centre), index, buffer, count, view))
        self.splat_stride = 1
        for index, instance in enumerate(scene.splats, len(scene.geometries)):
            if not len(instance.cloud):
                continue
            buffer, count, stride, _cloud = self._splat(instance.cloud, used_splats)
            self.splat_stride = max(self.splat_stride, stride)
            matrix = np.asarray(instance.matrix, np.float64)
            normal = np.eye(4)
            try:
                normal[:3, :3] = np.linalg.inv(matrix[:3, :3]).T
            except np.linalg.LinAlgError:
                pass  # a zero scale leaves nothing to see
            row = uniforms[index]
            row[:16], row[16:32] = matrix.T.ravel(), normal.T.ravel()
            row[32:36] = (instance.relight, instance.opacity_scale, splat_radius_scale(instance, stride),
                          SPLAT_MIN_OPACITY)
            row[36] = SPLAT_MAX_PIXELS
            clouds.append((index, buffer, count))
        device.queue.write_buffer(self._object_buffer(object_count), 0, uniforms)

        line_count = self._upload_lines(lines)
        encoder = device.create_command_encoder()
        render_pass = encoder.begin_render_pass(
            color_attachments=[{"view": targets["color"], "resolve_target": targets["resolve_view"],
                                "clear_value": _linear(background), "load_op": "clear", "store_op": "discard"}],
            depth_stencil_attachment={"view": targets["depth"], "depth_clear_value": 1.0,
                                      "depth_load_op": "clear", "depth_store_op": "discard"})
        render_pass.set_bind_group(0, self._global_group)

        def draw(items, pipeline):
            if items:
                render_pass.set_pipeline(pipeline)
            for _opaque, _order, index, buffer, count, view in items:
                render_pass.set_bind_group(1, self._object_group(view, object_count),
                                           dynamic_offsets_data=[index * _OBJECT_STRIDE])
                render_pass.set_vertex_buffer(0, buffer)
                render_pass.draw(count)

        draw([d for d in draws if d[0]], self._opaque)
        if clouds:
            render_pass.set_pipeline(self._splat_pipeline)
            render_pass.set_vertex_buffer(0, self._corners)
        for index, buffer, count in clouds:
            render_pass.set_bind_group(1, self._object_group(self._white, object_count),
                                       dynamic_offsets_data=[index * _OBJECT_STRIDE])
            render_pass.set_vertex_buffer(1, buffer)
            render_pass.draw(4, count)
        if line_count:
            render_pass.set_pipeline(self._line_pipeline)
            render_pass.set_vertex_buffer(0, self._lines[1])
            render_pass.draw(line_count)
        draw(sorted((d for d in draws if not d[0]), key=lambda d: d[1]), self._blended)  # far to near
        self._draw_particles(render_pass, scene, camera, width, height)  # last, and with its own group 0
        render_pass.end()
        encoder.copy_texture_to_buffer(
            {"texture": targets["resolve"], "mip_level": 0, "origin": (0, 0, 0)},
            {"buffer": targets["readback"], "offset": 0, "bytes_per_row": targets["stride"], "rows_per_image": height},
            (width, height, 1))
        device.queue.submit([encoder.finish()])
        readback = targets["readback"]
        readback.map_sync(wgpu.MapMode.READ)
        try:
            pixels = np.frombuffer(readback.read_mapped(), np.uint8).reshape(height, targets["stride"])
        finally:
            readback.unmap()
        for cache, used in ((self._meshes, used_meshes), (self._textures, used_textures),
                            (self._splats, used_splats)):
            for key in [k for k in cache if k not in used]:
                del cache[key]
        if self._objects is not None:
            groups = self._objects[2]
            for key in [k for k, (_group, view) in groups.items() if view is not self._white
                        and not any(view is entry[0] for entry in self._textures.values())]:
                del groups[key]
        # Readback rows are padded to a 256-byte stride. Whenever the width is not a multiple of 64
        # the slice below is a strided view, and QImage refuses a non-contiguous buffer: that
        # took the editor down on a real display while every 64-pixel-wide test passed.
        return np.ascontiguousarray(pixels[:, :width * 4].reshape(height, width, 4))

    def _draw_particles(self, render_pass, scene, camera, width, height):
        """Draw the scene's particle sets through gpu3d's instanced sprite pipeline (docs/SIMULATION.md)."""
        self.particle_stride = 1
        if not scene.particles:
            return
        sets = []
        for instance in scene.particles:
            stride = max(1, -(-len(instance) // MAX_PARTICLES))
            self.particle_stride = max(self.particle_stride, stride)
            if stride > 1:
                instance = replace(instance, positions=instance.positions[::stride],
                                   sizes=instance.sizes[::stride], colors=instance.colors[::stride])
            sets.append(instance)
        data = gpu3d.particle_data(scene3d.Scene(particles=tuple(sets)), camera, width, height,
                                   self.device.limits, None)
        if data is None:
            return
        wgpu, device = self.wgpu, self.device
        pipeline = gpu3d.particle_pipeline(self._state, "rgba8unorm-srgb", "depth24plus", SAMPLES)
        entries = []
        instances, texels, params = data
        for binding, (array, usage) in enumerate(((params, wgpu.BufferUsage.UNIFORM),
                                                  (instances, wgpu.BufferUsage.STORAGE),
                                                  (texels, wgpu.BufferUsage.STORAGE))):
            array = np.ascontiguousarray(array)
            buffer = self._particle_buffers.get(binding)
            if buffer is None or buffer.size < array.nbytes:
                if buffer is not None:
                    buffer.destroy()
                buffer = device.create_buffer(size=max(array.nbytes * 2, 256), usage=usage | wgpu.BufferUsage.COPY_DST)
                self._particle_buffers[binding] = buffer
            device.queue.write_buffer(buffer, 0, array)
            entries.append({"binding": binding, "resource": {"buffer": buffer, "offset": 0, "size": array.nbytes}})
        render_pass.set_pipeline(pipeline)
        render_pass.set_bind_group(0, device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=entries))
        render_pass.draw(6, len(instances))

    def _object_buffer(self, count):
        self._object_group(self._white, count)
        return self._objects[0]

    def _upload_lines(self, lines):
        if lines is None or not len(lines):
            return 0
        data = np.ascontiguousarray(lines, np.float32)
        source, buffer, count = self._lines
        if source is not None and source.shape == data.shape and np.array_equal(source, data):
            return count
        if buffer is None or buffer.size < data.nbytes:
            buffer = self.device.create_buffer(size=max(data.nbytes * 2, 4096), usage=self.wgpu.BufferUsage.VERTEX
                                               | self.wgpu.BufferUsage.COPY_DST)
        self.device.queue.write_buffer(buffer, 0, data)
        self._lines = (data, buffer, len(data))
        return len(data)


def _linear(color):
    """Clear values go through the sRGB target's encode, so they are given as linear light."""
    return tuple(float(c) for c in color)


_renderer = None
_failure = None


def renderer():
    """The process-wide viewport renderer, or None (with ``failure()`` saying why)."""
    global _renderer, _failure
    if _renderer is None and _failure is None:
        try:
            _renderer = ViewportRenderer()
        except Exception as error:  # no adapter, no wgpu, or a driver that refuses the pipelines
            _failure = f"{type(error).__name__}: {error}"
    return _renderer


def failure():
    return _failure
