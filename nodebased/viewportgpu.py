"""Interactive wgpu renderer for the 3D editor viewport.

This is not the Render3D backend. ``gpu3d`` reproduces the CPU reference renderer's per-triangle
sorting and mip choice, which costs one draw call per triangle; that is right for a final render
and far too slow to orbit. The viewport wants the opposite trade: mesh buffers are uploaded once
per geometry and cached, every object is one draw call against a real depth buffer, and orbiting
only rewrites a camera matrix. Shading follows ``scene3d._shade_fragments`` (headlight, Lambert,
Blinn-Phong specular, emission, textures and camera projection), with these viewport limits:

- transparency is sorted per object, not per triangle or per pixel;
- projection depth occlusion is not evaluated (the projection shows through occluders);
- at most ``MAX_LIGHTS`` lights shade the view. One key light -- the brightest shadow-enabled
  Directional or Spot light (``_shadow_light``; Point is not supported, a stated limit) -- casts a
  ``SHADOW_MAP_SIZE`` depth map from opaque meshes only (``_render_shadow_map``), sampled with a
  3x3 texel box filter and a fixed bias by every shaded mesh and splat (R6 "next", closed); blended
  meshes and every other light are unshadowed. Negligible cost measured (a two-mesh scene at
  1920x1080 on an RTX 3080 Ti: about the same either way, within noise of run to run);
- Gaussian splats are a layout proxy, not the Render3D look: each splat is an opaque
  camera-facing disc in its SH-DC colour (no view-dependent colour, no blending), at most
  ``MAX_SPLATS`` per cloud with an even stride beyond that. ``Relight`` is followed per splat
  with Cook-Torrance GGX, the dome and the key light's shadow (R6); when the cloud has a de-lit layer
  (``nodebased.intrinsics``) the proxy carries its per-splat albedo and roughness too (R6
  "next", closed), blended toward the SH-DC colour and a neutral roughness of 1 by
  ``Intrinsics mix`` and ``Roughness`` (now a multiplier, matching ``splatshade.material_roughness``);
  ``metallic`` is still one constant per cloud, as the CPU renderer's own fit also keeps it (a
  capture cannot show metallic). A relit cloud with ``Indirect samples`` and at most
  ``INDIRECT_MAX_SPLATS`` splats also shows its traced occlusion and one-bounce indirect light, computed once
  at the ``preview`` quality preset from the scene's splats (no meshes, no shadows) and reused while the
  scene, lights and ambient stay the same;
- meshes with a ``pbr`` material are lit with the same Cook-Torrance GGX and the dome (R6,
  ``scene3d._shade_pbr_mesh``); standard (Blinn-Phong) meshes also pick up the dome's diffuse and a
  mirror-reflection specular, as the CPU renderer's ``elif lit`` path does. The dome is a single
  ``Environment`` (a scene with more than one shows only the first), prefiltered on the CPU
  (``envlight``) into 9 SH coefficients and a six-tile GGX-roughness atlas resampled to 64x32
  per tile, uploaded once per fingerprint/intensity/rotation/blur/tint and sampled with a
  two-tile lerp. ``show_background`` (the widget's
  ``B`` key, R6 "next", closed) paints the dome behind everything empty instead of the solid
  clear colour, from the same atlas's sharpest tile (still the 64x32 prefilter, not a separate
  full-resolution upload) sampled per pixel through a camera ray reconstructed from the four
  screen corners (``_background_ray_corners``); real geometry always wins over it. No
  material-ball preview yet (open, R6 "next");
- particles use Render3D's draw (gpu3d.particle_pipeline: sprites sorted far to near, blended over
  the meshes, depth tested, points and spheres as discs, cards with their texture), at most
  ``MAX_PARTICLES`` per set with an even stride beyond that;
- volumes (Plume3D and other `Volume` members) are raymarched by gpuvolume, the shader Render3D uses, over the
  finished frame and cut at its depth buffer: lit by the scene's lights with shadow rays, at a coarse step
  count while orbiting (``VOLUME_FAST``) and a finer one after the ``quality`` toggle (``VOLUME_QUALITY``),
  coarsened further when a frame would cost more than ``VOLUME_FRAME_FRACTION`` of one submission budget;
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

from . import gpu3d, gpuvolume, scene3d, volumerender
from .splats import C0, _rotation, to_linear_color
from .splatshade import normal_confidence

MAX_LIGHTS = 16
MAX_SPLATS = 1_000_000    # per cloud; larger clouds are shown with an even stride
MAX_PARTICLES = 250_000   # per particle set; larger sets are shown with an even stride (sorted every frame)
SPLAT_SIGMA = 1.5         # disc radius, in standard deviations of the splat's middle axis
SPLAT_MAX_PIXELS = 2.5     # disc radius on screen never exceeds this
SPLAT_MIN_OPACITY = 0.05  # splats fainter than this (after the node's opacity scale) are hidden
INDIRECT_MAX_SPLATS = 30_000   # larger relit clouds skip the viewport's indirect preview (rays run on the CPU)
INDIRECT_RESIGN = 0.02    # recompute the preview once this fraction of splats face the other way from the eye
# March steps across the largest volume's diagonal and shadow steps per sample: (steps, shadow steps).
VOLUME_FAST = (48, 6)
VOLUME_QUALITY = (192, 16)
VOLUME_FRAME_FRACTION = 0.25   # of gpuvolume.VOLUME_WORK_BUDGETS: an interactive frame stays well inside one submission
SAMPLES = 4
_OBJECT_STRIDE = 512  # one Object struct, padded to a multiple of every adapter's offset alignment
_INSTANCE_STRIDE = 128  # one Instance3D copy: model (4 vec4) + normal matrix (3 vec4, padded) + tint (vec4)
_LOCK_TIMEOUT = 0.02  # seconds to wait for a Render3D job that holds the shared device
SHADOW_MAP_SIZE = 1024    # one key light's depth map, opaque meshes only (R6 "next", closed)
SHADOW_BIAS = 0.0015      # fixed NDC-depth bias; a real-time approximation, not the light's own Shadow Bias knob
_SHADOW_KINDS = ("Directional", "Spot")   # Point needs a cube map; not built (a stated limit)

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
    // Meshes (materials 1): is-pbr, metallic, roughness, dielectric F0. Splats reuse `material`
    // instead (max disc pixels, roughness-scale multiplier, metallic, intrinsics mix) and leave
    // this field zero.
    pbr: vec4<f32>,
};
// A single dome (docs/SPLAT_RELIGHTING.md, R6): split-sum image-based light, folded with the
// dome's intensity and tint on upload so the shader only ever multiplies by 1. `env_tex` is a
// six-tile vertical atlas (one GGX-roughness level per tile, `envlight.LEVEL_ROUGHNESS`), sampled
// with a manual two-tile lerp instead of a texture array (this project's wgpu binding has no
// precedent for one). A scene with more than one Environment shows only the first; that is this
// step's known limit, not an oversight.
struct EnvGlobals { sh: array<vec4<f32>, 9>, params: vec4<f32> };  // params: enabled, rotation cos, sin, unused
@group(0) @binding(0) var<uniform> globals: Globals;
@group(1) @binding(0) var<uniform> object: Object;
@group(1) @binding(1) var surface: texture_2d<f32>;
@group(1) @binding(2) var surface_sampler: sampler;
@group(2) @binding(0) var<uniform> envg: EnvGlobals;
@group(2) @binding(1) var env_tex: texture_2d<f32>;
@group(2) @binding(2) var env_sampler: sampler;
// One key light's shadow map (R6 "next", closed): opaque meshes only cast (`shadow_vertex`, its own
// fragmentless pipeline), meshes and splats both receive. `params`: enabled, unused (bias is the
// fixed `SHADOW_BIAS` above), map size in texels, the index into `globals.lights` this map belongs
// to (-1 disables every light's lookup even if `enabled` is left set). Point lights are not
// supported (a cube map); the brightest shadow-enabled Directional or Spot light wins.
struct ShadowGlobals { view_proj: mat4x4<f32>, params: vec4<f32> };
@group(3) @binding(0) var<uniform> shadowd: ShadowGlobals;
@group(3) @binding(1) var shadow_tex: texture_depth_2d;

// A 3x3 texel box filter (soft-edged, cheap): 1 fully lit .. 0 fully shadowed. Outside the map's
// frustum (nothing was rendered there) reads as lit, the safe default for an approximation.
fn shadow_factor(world: vec3<f32>) -> f32 {
    if (shadowd.params.x < 0.5) { return 1.0; }
    let clip = shadowd.view_proj * vec4<f32>(world, 1.0);
    if (clip.w <= 0.0) { return 1.0; }
    let ndc = clip.xyz / clip.w;
    if (abs(ndc.x) > 1.0 || abs(ndc.y) > 1.0 || ndc.z < 0.0 || ndc.z > 1.0) { return 1.0; }
    let size = i32(shadowd.params.z);
    let texel = vec2<i32>(vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * shadowd.params.z);
    var lit = 0.0;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let coord = clamp(texel + vec2<i32>(dx, dy), vec2<i32>(0), vec2<i32>(size - 1));
            if (ndc.z - 0.0015 <= textureLoad(shadow_tex, coord, 0)) { lit = lit + 1.0; }
        }
    }
    return lit / 9.0;
}

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

// Cook-Torrance GGX specular response times n.l, matching splatshade._cook_torrance so a mesh and
// a splat under one light agree; fresnel here is the material's own (used for the response), while
// callers use a fixed dielectric Schlick term for kd, as the CPU shader does.
fn cook_torrance(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, roughness: f32, f0: vec3<f32>) -> vec3<f32> {
    let h = normalize(l + v);
    let nl = max(dot(n, l), 0.0);
    let nv = max(dot(n, v), 1e-4);
    let nh = max(dot(n, h), 0.0);
    let vh = max(dot(v, h), 0.0);
    let a = max(roughness, 0.05) * max(roughness, 0.05);
    let d = (a * a) / (3.14159265 * pow(nh * nh * (a * a - 1.0) + 1.0, 2.0));
    let k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    let vis = 1.0 / (max(nl * (1.0 - k) + k, 1e-4) * max(nv * (1.0 - k) + k, 1e-4) * 4.0);
    let fresnel = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - vh, 5.0);
    return (3.14159265 * d * vis * nl) * fresnel;
}

// Karis's analytic fit of the split-sum BRDF table, matching envlight.dfg exactly.
fn dfg_approx(nv: f32, roughness: f32) -> vec2<f32> {
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let t = roughness * c0 + c1;
    let a004 = min(t.x * t.x, exp2(-9.28 * nv)) * t.x + t.y;
    return vec2<f32>(-1.04 * a004 + t.z, 1.04 * a004 + t.w);
}

// envlight.Environment._local with no parent transform (the viewport dome is root-level).
fn env_local(d: vec3<f32>) -> vec3<f32> {
    let c = envg.params.y;
    let s = envg.params.z;
    return vec3<f32>(c * d.x - s * d.z, d.y, s * d.x + c * d.z);
}

// envlight._uv: row 0 up, map centre looks down -Z.
fn env_uv(d: vec3<f32>) -> vec2<f32> {
    let u = 0.5 + atan2(d.x, -d.z) / (2.0 * 3.14159265);
    let v = acos(clamp(d.y, -1.0, 1.0)) / 3.14159265;
    return vec2<f32>(u, v);
}

// envlight._sh_irradiance: cosine-convolved radiance, folded with the dome's gain already.
fn env_diffuse(n: vec3<f32>) -> vec3<f32> {
    if (envg.params.x < 0.5) { return vec3<f32>(0.0); }
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

// envlight.Prefiltered.lookup: a two-tile lerp over the atlas's six GGX-roughness levels.
fn env_specular(direction: vec3<f32>, roughness: f32) -> vec3<f32> {
    if (envg.params.x < 0.5) { return vec3<f32>(0.0); }
    let uv = env_uv(env_local(direction));
    let pos = clamp(roughness, 0.0, 1.0) * 5.0;
    let lo = floor(pos);
    let hi = min(lo + 1.0, 5.0);
    let a = textureSampleLevel(env_tex, env_sampler, vec2<f32>(uv.x, (lo + uv.y) / 6.0), 0.0).rgb;
    let b = textureSampleLevel(env_tex, env_sampler, vec2<f32>(uv.x, (hi + uv.y) / 6.0), 0.0).rgb;
    return mix(a, b, pos - lo);
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
    } else if (globals.settings.y < 0.5 && object.pbr.x > 0.5) {
        // materials 1 (R1), lit here for the first time (R6): Cook-Torrance GGX plus the dome,
        // matching scene3d._shade_pbr_mesh so a simple lit scene agrees with the final render.
        let to_eye = toward_eye / max(length(toward_eye), 1e-8);
        let base_rgb = source.rgb / max(source.a, 1e-6);
        let m = object.pbr.y;
        let rough = object.pbr.z;
        let f0d = object.pbr.w;
        let f0 = mix(vec3<f32>(f0d), base_rgb, m);
        var diffuse_light = vec3<f32>(globals.settings.x) * (1.0 - m);
        var specular = vec3<f32>(0.0);
        for (var i = 0u; i < u32(globals.settings.z); i = i + 1u) {
            let light = globals.lights[i];
            var to_light = -light.place.xyz;
            if (light.place.w > 0.5) {
                let delta = light.place.xyz - in.world;
                to_light = delta / max(length(delta), 1e-8);
            }
            var factor = attenuation(light, in.world);
            if (f32(i) == shadowd.params.w) { factor = factor * shadow_factor(in.world); }
            let colour = light.color.rgb;
            let nl = max(dot(normal, to_light), 0.0);
            let half_vector = normalize(to_light + to_eye);
            let vh = max(dot(to_eye, half_vector), 0.0);
            let kd = (1.0 - m) * (1.0 - (0.04 + 0.96 * pow(1.0 - vh, 5.0)));
            diffuse_light += nl * kd * factor * colour;
            specular += cook_torrance(normal, to_eye, to_light, rough, f0) * factor * colour;
        }
        if (envg.params.x > 0.5) {
            let nv = max(dot(normal, to_eye), 1e-4);
            let ab = dfg_approx(nv, rough);
            let comp = 1.0 / max(ab.x + ab.y, 1e-4);
            let dielectric = vec3<f32>(f0d * ab.x + ab.y) * (1.0 + f0d * (comp - 1.0));
            let conductor = (base_rgb * ab.x + vec3<f32>(ab.y)) * (vec3<f32>(1.0) + base_rgb * (comp - 1.0));
            let total = mix(dielectric, conductor, m);
            let refl = 2.0 * dot(normal, to_eye) * normal - to_eye;
            diffuse_light += env_diffuse(normal) * ((1.0 - m) * (1.0 - dielectric.x));
            specular += total * env_specular(refl, rough);
        }
        rgb = base_rgb * diffuse_light * source.a + specular * source.a;
    } else if (globals.settings.y < 0.5) {
        let to_eye = toward_eye / max(length(toward_eye), 1e-8);
        var radiance = vec3<f32>(globals.settings.x) + env_diffuse(normal);
        var specular = vec3<f32>(0.0);
        for (var i = 0u; i < u32(globals.settings.z); i = i + 1u) {
            let light = globals.lights[i];
            var to_light = -light.place.xyz;
            if (light.place.w > 0.5) {
                let delta = light.place.xyz - in.world;
                to_light = delta / max(length(delta), 1e-8);
            }
            let lambert = dot(normal, to_light);
            var factor = attenuation(light, in.world);
            if (f32(i) == shadowd.params.w) { factor = factor * shadow_factor(in.world); }
            radiance = radiance + max(lambert, 0.0) * factor * light.color.rgb;
            if (object.material.x > 0.0 && lambert > 0.0) {
                let half_vector = to_light + to_eye;
                let lobe = pow(max(dot(normal, half_vector / max(length(half_vector), 1e-8)), 0.0),
                               object.material.y);
                specular = specular + object.material.x * lobe * factor * light.color.rgb;
            }
        }
        if (object.material.x > 0.0) {
            let refl = 2.0 * dot(normal, to_eye) * normal - to_eye;
            specular += env_specular(refl, object.pbr.z) * object.material.x;
        }
        rgb = rgb * radiance + specular * source.a;
    }
    return vec4<f32>(rgb + emissive, source.a);
}

// Instance3D copies (step X2 of 2, "Instances drawn in the viewport"): one hardware-instanced draw
// call per (InstanceSet, source variant), instead of the one-draw-call-per-object path `mesh_vertex`
// takes. Model and normal matrices are supplied per instance (`_build_instance_buffer`, CPU-side,
// cached like every other upload here) rather than through the `object` uniform's dynamic offset,
// since that uniform is one row per draw call, not per instance. `object` still carries the source
// mesh's own material (colour, specular, texture, pbr): `in.tint` (straight rgba, `InstanceSet.colors`)
// multiplies it the same way `scene3d.expand_instances` tints the CPU/final-render path, so a scene
// with and without GPU instancing agrees. A stated limit: instanced copies do not cast into the key
// light's shadow map (`_render_shadow_map` only walks `draws`, built from `scene.geometries`), and
// carry no projector (`mesh_fragment`'s `object.projector_flags` path is not read here).
struct InstanceFragment {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tint: vec4<f32>,
};

@vertex
fn instance_vertex(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>,
                    @location(3) model0: vec4<f32>, @location(4) model1: vec4<f32>,
                    @location(5) model2: vec4<f32>, @location(6) model3: vec4<f32>,
                    @location(7) normal0: vec4<f32>, @location(8) normal1: vec4<f32>,
                    @location(9) normal2: vec4<f32>, @location(10) tint: vec4<f32>) -> InstanceFragment {
    var out: InstanceFragment;
    let model = mat4x4<f32>(model0, model1, model2, model3);
    let normal_mat = mat3x3<f32>(normal0.xyz, normal1.xyz, normal2.xyz);
    let world = model * vec4<f32>(position, 1.0);
    out.clip = globals.view_proj * world;
    out.world = world.xyz;
    out.normal = normal_mat * normal;
    out.uv = uv;
    out.tint = tint;
    return out;
}

@fragment
fn instance_fragment(in: InstanceFragment) -> @location(0) vec4<f32> {
    var normal = in.normal / max(length(in.normal), 1e-8);
    let straight = object.color * in.tint;
    let tint = vec4<f32>(straight.rgb * straight.a, straight.a);
    var source = tint;
    let texel = textureSample(surface, surface_sampler, vec2<f32>(in.uv.x, 1.0 - in.uv.y));
    if (object.material.w > 0.5) {
        source = texel * tint;
    }
    let toward_eye = globals.eye.xyz - in.world;
    if (dot(normal, toward_eye) < 0.0) {
        normal = -normal;
    }
    let emissive = source.rgb * object.material.z;
    var rgb = source.rgb;
    if (globals.settings.y > 0.5 && globals.settings.y < 1.5) {
        rgb = rgb * (0.25 + 0.75 * abs(dot(normal, globals.view_light.xyz)));
    } else if (globals.settings.y < 0.5 && object.pbr.x > 0.5) {
        let to_eye = toward_eye / max(length(toward_eye), 1e-8);
        let base_rgb = source.rgb / max(source.a, 1e-6);
        let m = object.pbr.y;
        let rough = object.pbr.z;
        let f0d = object.pbr.w;
        let f0 = mix(vec3<f32>(f0d), base_rgb, m);
        var diffuse_light = vec3<f32>(globals.settings.x) * (1.0 - m);
        var specular = vec3<f32>(0.0);
        for (var i = 0u; i < u32(globals.settings.z); i = i + 1u) {
            let light = globals.lights[i];
            var to_light = -light.place.xyz;
            if (light.place.w > 0.5) {
                let delta = light.place.xyz - in.world;
                to_light = delta / max(length(delta), 1e-8);
            }
            var factor = attenuation(light, in.world);
            if (f32(i) == shadowd.params.w) { factor = factor * shadow_factor(in.world); }
            let colour = light.color.rgb;
            let nl = max(dot(normal, to_light), 0.0);
            let half_vector = normalize(to_light + to_eye);
            let vh = max(dot(to_eye, half_vector), 0.0);
            let kd = (1.0 - m) * (1.0 - (0.04 + 0.96 * pow(1.0 - vh, 5.0)));
            diffuse_light += nl * kd * factor * colour;
            specular += cook_torrance(normal, to_eye, to_light, rough, f0) * factor * colour;
        }
        if (envg.params.x > 0.5) {
            let nv = max(dot(normal, to_eye), 1e-4);
            let ab = dfg_approx(nv, rough);
            let comp = 1.0 / max(ab.x + ab.y, 1e-4);
            let dielectric = vec3<f32>(f0d * ab.x + ab.y) * (1.0 + f0d * (comp - 1.0));
            let conductor = (base_rgb * ab.x + vec3<f32>(ab.y)) * (vec3<f32>(1.0) + base_rgb * (comp - 1.0));
            let total = mix(dielectric, conductor, m);
            let refl = 2.0 * dot(normal, to_eye) * normal - to_eye;
            diffuse_light += env_diffuse(normal) * ((1.0 - m) * (1.0 - dielectric.x));
            specular += total * env_specular(refl, rough);
        }
        rgb = base_rgb * diffuse_light * source.a + specular * source.a;
    } else if (globals.settings.y < 0.5) {
        let to_eye = toward_eye / max(length(toward_eye), 1e-8);
        var radiance = vec3<f32>(globals.settings.x) + env_diffuse(normal);
        var specular = vec3<f32>(0.0);
        for (var i = 0u; i < u32(globals.settings.z); i = i + 1u) {
            let light = globals.lights[i];
            var to_light = -light.place.xyz;
            if (light.place.w > 0.5) {
                let delta = light.place.xyz - in.world;
                to_light = delta / max(length(delta), 1e-8);
            }
            let lambert = dot(normal, to_light);
            var factor = attenuation(light, in.world);
            if (f32(i) == shadowd.params.w) { factor = factor * shadow_factor(in.world); }
            radiance = radiance + max(lambert, 0.0) * factor * light.color.rgb;
            if (object.material.x > 0.0 && lambert > 0.0) {
                let half_vector = to_light + to_eye;
                let lobe = pow(max(dot(normal, half_vector / max(length(half_vector), 1e-8)), 0.0),
                               object.material.y);
                specular = specular + object.material.x * lobe * factor * light.color.rgb;
            }
        }
        if (object.material.x > 0.0) {
            let refl = 2.0 * dot(normal, to_eye) * normal - to_eye;
            specular += env_specular(refl, object.pbr.z) * object.material.x;
        }
        rgb = rgb * radiance + specular * source.a;
    }
    return vec4<f32>(rgb + emissive, source.a);
}

// The shadow map's own depth-only pass (R6 "next", closed): one draw call per opaque mesh, exactly
// like `mesh_vertex`/`draw()` in the main pass, but through the key light's `view_proj` (its own
// group 0, `ShadowPassGlobals`) instead of the camera's. Fragmentless: only depth is written.
struct ShadowPassGlobals { view_proj: mat4x4<f32> };
@group(0) @binding(0) var<uniform> shadow_pass: ShadowPassGlobals;

@vertex
fn shadow_vertex(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    return shadow_pass.view_proj * object.model * vec4<f32>(position, 1.0);
}

struct SplatFragment {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) corner: vec2<f32>,
};

// One camera-facing disc per splat. object.color carries relight, opacity scale, world radius
// scale and the hide threshold. Shading follows splatshade.shade_splats, without shadows.
// `delit` is the cloud's own per-splat de-lit albedo (rgb) and roughness (a) from
// nodebased.intrinsics, or a copy of `paint` and a neutral 1.0 when the cloud has no de-lit
// layer, so the mix below collapses to the pre-R6 per-cloud approximation exactly.
@vertex
fn splat_vertex(@location(0) corner: vec2<f32>, @location(1) place: vec4<f32>,
                @location(2) paint: vec4<f32>, @location(3) facing: vec4<f32>,
                @location(4) indirect: vec4<f32>, @location(5) delit: vec4<f32>) -> SplatFragment {
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
        // indirect: traced occlusion (scales the ambient) and the one-bounce light (added), see splatindirect.
        // Metallic (materials 1/R6) rides in object.material.z, one constant per cloud, matching the CPU
        // fit's own limit (a capture cannot show metallic). Albedo and roughness are per splat: `delit` is
        // the cloud's de-lit layer (or a copy of `paint` and 1.0 without one), mixed toward it by
        // object.material.w (`Intrinsics mix`); object.material.y is now the `Roughness` multiplier
        // (splatshade.material_roughness), not the final value.
        let m = object.material.z;
        let albedo = mix(paint.rgb, delit.rgb, object.material.w);
        let rough = clamp(mix(1.0, delit.a, object.material.w) * object.material.y, 0.0, 1.0);
        var diffuse_light = vec3<f32>(globals.settings.x * indirect.x);
        var specular = vec3<f32>(0.0);
        if (globals.settings.y > 0.5 && globals.settings.y < 1.5) {
            diffuse_light = vec3<f32>(0.25 + 0.75 * abs(dot(effective, globals.view_light.xyz)));
        } else {
            diffuse_light = diffuse_light * (1.0 - m);
            let f0 = mix(vec3<f32>(0.04), albedo, m);
            for (var i = 0u; i < u32(globals.settings.z); i = i + 1u) {
                let light = globals.lights[i];
                var to_light = -light.place.xyz;
                if (light.place.w > 0.5) {
                    let delta = light.place.xyz - world.xyz;
                    to_light = delta / max(length(delta), 1e-8);
                }
                var factor = attenuation(light, world.xyz);
                if (f32(i) == shadowd.params.w) { factor = factor * shadow_factor(world.xyz); }
                let colour = light.color.rgb;
                let nl = max(dot(effective, to_light), 0.0);
                let half_vector = normalize(to_light + to_eye);
                let vh = max(dot(to_eye, half_vector), 0.0);
                let kd = (1.0 - m) * (1.0 - (0.04 + 0.96 * pow(1.0 - vh, 5.0)));
                diffuse_light += nl * kd * factor * colour;
                specular += cook_torrance(effective, to_eye, to_light, rough, f0) * factor * colour;
            }
            if (envg.params.x > 0.5) {
                let nv = max(dot(effective, to_eye), 1e-4);
                let ab = dfg_approx(nv, rough);
                let comp = 1.0 / max(ab.x + ab.y, 1e-4);
                let dielectric = vec3<f32>(0.04 * ab.x + ab.y) * (1.0 + 0.04 * (comp - 1.0));
                let conductor = (albedo * ab.x + vec3<f32>(ab.y)) * (vec3<f32>(1.0) + albedo * (comp - 1.0));
                let total = mix(dielectric, conductor, m);
                let refl = 2.0 * dot(effective, to_eye) * effective - to_eye;
                diffuse_light += env_diffuse(effective) * ((1.0 - m) * (1.0 - dielectric.x));
                specular += total * env_specular(refl, rough);
            }
        }
        rgb = mix(paint.rgb, albedo * (diffuse_light + indirect.yzw) + specular, object.color.x);
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

// The look-dev background (R6 "next", closed): the dome shown behind everything instead of the
// solid clear colour, toggled by the viewport's `B` key (`ViewportRenderer.show_background`).
// A dedicated group(0), since this pipeline draws nothing else and so needs no `object` group:
// the same buffer, texture and sampler as `envg`/`env_tex`/`env_sampler` (group 2, the shading
// passes' dome), bound here at group 0 instead. Sampled from the atlas's sharpest tile (roughness
// 0, `envlight.LEVEL_ROUGHNESS[0]`) -- still the prefilter's 64x32 resample, so a background pixel
// is as blurry as a mirror-roughness reflection already is elsewhere in this shader, not the raw
// map at full resolution (no separate high-resolution upload exists yet).
@group(0) @binding(0) var<uniform> bg_env: EnvGlobals;
@group(0) @binding(1) var bg_env_tex: texture_2d<f32>;
@group(0) @binding(2) var bg_env_sampler: sampler;

struct BgFragment { @builtin(position) clip: vec4<f32>, @location(0) direction: vec3<f32> };

// `corner` is the same clip-space NDC quad the splat discs use (`_corners`); here it is the
// screen itself, drawn at the far plane (`less-equal` against the cleared depth of 1 lets real
// geometry win, since this pipeline never writes depth). `direction` is the world-space camera
// ray through that corner, uploaded fresh every frame (view_projection is bilinear in NDC for a
// pinhole camera, so interpolating the four corners' rays across the quad reconstructs every
// pixel's ray exactly, with no matrix inverse needed in the shader).
@vertex
fn background_vertex(@location(0) corner: vec2<f32>, @location(1) direction: vec3<f32>) -> BgFragment {
    var out: BgFragment;
    out.clip = vec4<f32>(corner, 1.0, 1.0);
    out.direction = direction;
    return out;
}

@fragment
fn background_fragment(in: BgFragment) -> @location(0) vec4<f32> {
    let c = bg_env.params.y;
    let s = bg_env.params.z;
    let d = normalize(in.direction);
    let local = vec3<f32>(c * d.x - s * d.z, d.y, s * d.x + c * d.z);   // env_local, against bg_env
    let uv = env_uv(local);
    let color = textureSampleLevel(bg_env_tex, bg_env_sampler, vec2<f32>(uv.x, uv.y / 6.0), 0.0).rgb;
    return vec4<f32>(color, 1.0);
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


_NDC_CORNERS = np.array(((-1, -1), (1, -1), (-1, 1), (1, 1)), np.float64)


def _background_ray_corners(camera, width, height):
    """(4, 4) float32 world-space camera ray directions (xyz, w=0 padding) through the NDC quad
    corners `_corners` draws (see `background_vertex`): the projection's inverse mapping, done once
    per corner in Python instead of inverting a matrix in the shader."""
    _eye, view = scene3d._view_basis(camera)
    focal = 1.0 / math.tan(math.radians(camera.fov) / 2)
    aspect = width / max(height, 1)
    camera_space = np.concatenate(
        (_NDC_CORNERS[:, :1] * aspect / focal, _NDC_CORNERS[:, 1:] / focal, np.full((4, 1), -1.0)), axis=1)
    world = camera_space @ view.astype(np.float64)  # view's rows are right/up/-forward: v @ view undoes it
    world /= np.linalg.norm(world, axis=1, keepdims=True)
    out = np.zeros((4, 4), np.float32)
    out[:, :3] = world
    return out


def _mesh_bounds(scene):
    """(low, high) float64 world AABB of the scene's mesh geometries, or None (nothing casts)."""
    corners = []
    for geometry in scene.geometries:
        vertices = np.asarray(geometry.vertices, np.float64)
        if not len(vertices):
            continue
        matrix = np.asarray(geometry.world_matrix(), np.float64)
        corners.append(vertices @ matrix[:3, :3].T + matrix[:3, 3])
    if not corners:
        return None
    cloud = np.concatenate(corners)
    return cloud.min(axis=0), cloud.max(axis=0)


def _shadow_light(lights):
    """The brightest shadow-enabled Directional or Spot light in `lights` (as built in `_render`:
    `(light, position, direction)` tuples), as `(list index, light, position, direction)`, or None.
    Point is skipped (a cube map is not built); area kinds do not cast shadows anywhere yet."""
    luma = np.array((.2126, .7152, .0722))
    best = None
    for index, (light, position, direction) in enumerate(lights):
        if not light.shadows or light.kind not in _SHADOW_KINDS:
            continue
        weight = float(light.intensity) * float(np.dot(np.asarray(light.color, np.float64), luma))
        if best is None or weight > best[0]:
            best = (weight, index, light, position, direction)
    return best[1:] if best is not None else None


def _light_basis(direction):
    """(right, up, forward) orthonormal world axes for a light looking along `direction` (unit),
    the same construction as `scene3d._view_basis` but from a direction instead of a target."""
    forward = np.asarray(direction, np.float64)
    forward = forward / max(np.linalg.norm(forward), 1e-9)
    up = np.array((0.0, 1.0, 0.0))
    if abs(float(forward @ up)) > 0.9999:
        up = np.array((0.0, 0.0, -1.0 if forward[1] < 0 else 1.0))
    right = np.cross(forward, up)
    right = right / max(np.linalg.norm(right), 1e-9)
    up = np.cross(right, forward)
    return right, up, forward


def shadow_view_proj(light, position, direction, bounds):
    """World -> clip matrix for `light`'s shadow map, framing `bounds` (a mesh AABB, or None for a
    small default extent at the origin so an empty scene never divides by zero): orthographic for
    Directional, perspective for Spot (fov the cone plus its penumbra, so the whole falloff is
    covered). Follows `view_projection`'s depth-range and forward-sign convention exactly, with
    ``forward`` here the light's own travel direction (``light.world()``'s second return)."""
    right, up, forward = _light_basis(direction)
    view = np.stack((right, up, -forward))
    low, high = bounds if bounds is not None else (np.array((-1.0, -1.0, -1.0)), np.array((1.0, 1.0, 1.0)))
    center = (np.asarray(low, np.float64) + np.asarray(high, np.float64)) / 2
    radius = max(float(np.linalg.norm(np.asarray(high, np.float64) - np.asarray(low, np.float64))) / 2, 1e-3)
    if light.kind == "Directional":
        eye = center - forward * radius * 2
        look = np.eye(4, dtype=np.float64)
        look[:3, :3], look[:3, 3] = view, -(view @ eye)
        near, far = 0.01, radius * 4
        ortho = np.zeros((4, 4), np.float64)
        ortho[0, 0] = ortho[1, 1] = 1.0 / radius
        ortho[2, 2], ortho[2, 3] = -1.0 / (far - near), -near / (far - near)
        ortho[3, 3] = 1.0
        return ortho @ look
    eye = np.asarray(position, np.float64)
    look = np.eye(4, dtype=np.float64)
    look[:3, :3], look[:3, 3] = view, -(view @ eye)
    fov = min(max(float(light.cone_angle) + 2 * float(light.cone_penumbra_angle), 1.0), 170.0)
    focal = 1.0 / math.tan(math.radians(fov) / 2)
    distance = max(float(np.linalg.norm(center - eye)) + radius, radius, 1.0)
    near, far = max(distance * 0.01, 0.01), distance + radius * 2
    lens = np.zeros((4, 4), np.float64)
    lens[0, 0] = lens[1, 1] = focal
    lens[2, 2], lens[2, 3] = far / (near - far), near * far / (near - far)
    lens[3, 2] = -1.0
    return lens @ look


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
    """(N, 16) float32 disc instances in the cloud's local space, plus the stride that was used.

    Columns: position, disc radius | linear SH-DC colour, opacity | normal (shortest axis),
    normal confidence | de-lit albedo, de-lit roughness. The last four (R6 "next", closed) are the
    cloud's own per-splat intrinsic fit (``nodebased.intrinsics``) when it has one, else a copy of
    the SH-DC colour and a neutral roughness of 1, so a cloud without one shades exactly as the
    pre-R6 per-cloud approximation did. Local space keeps the proxy valid while the node's
    transform changes.
    """
    stride = max(1, -(-len(cloud) // max(int(limit), 1)))
    index = np.arange(0, len(cloud), stride)
    scales = cloud.scales[index]
    out = np.empty((len(index), 16), np.float32)
    out[:, :3] = cloud.positions[index]
    out[:, 3] = np.sort(scales, axis=1)[:, 1] * SPLAT_SIGMA
    paint = to_linear_color(np.maximum(0.5 + C0 * cloud.sh[index, 0, :], 0), cloud.colorspace)
    out[:, 4:7] = paint
    out[:, 7] = cloud.opacity[index]
    out[:, 8:11] = _rotation(cloud.rotations[index])[np.arange(len(index)), :, scales.argmin(axis=1)]
    out[:, 11] = normal_confidence(scales)
    intrinsics = getattr(cloud, 'intrinsics', None)
    if intrinsics is not None:
        out[:, 12:15] = np.asarray(intrinsics.albedo, np.float32)[index]
        out[:, 15] = np.asarray(intrinsics.roughness, np.float32)[index]
    else:
        out[:, 12:15] = paint
        out[:, 15] = 1.0
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
        self._instance_buffers, self._instance_meshes = {}, {}
        self._indirect = {}    # id(cloud) -> (key, eye-side signs, vertex buffer, cloud): the indirect preview
        self.splat_stride = 1  # largest stride among the clouds in the last frame (1: all shown)
        self.particle_stride = 1  # largest stride among the particle sets in the last frame
        self._state = state
        self._particle_buffers = {}
        self.volume_quality = False   # the toggle: finer march steps (the viewport's `V` key)
        self.volume_note = ""         # what the last frame did with volumes, for the status line
        self.volume_steps = 0         # march step count across the largest volume in the last frame (0: no volumes)
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
             "buffer": {"type": "uniform", "has_dynamic_offset": True, "min_binding_size": 288}},
            {"binding": 1, "visibility": stage.FRAGMENT, "texture": {"sample_type": "float"}},
            {"binding": 2, "visibility": stage.FRAGMENT, "sampler": {"type": "filtering"}}])
        # The dome (materials 1/R6): sampled from both stages (mesh_fragment and splat_vertex).
        self._env_layout = device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": stage.VERTEX | stage.FRAGMENT, "buffer": {"type": "uniform"}},
            {"binding": 1, "visibility": stage.VERTEX | stage.FRAGMENT, "texture": {"sample_type": "float"}},
            {"binding": 2, "visibility": stage.VERTEX | stage.FRAGMENT, "sampler": {"type": "filtering"}}])
        self._env_buffer = device.create_buffer(size=9 * 16 + 16,
                                                usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self._env_key = "unset"
        self._env_group = None
        self._globals = device.create_buffer(size=64 + 16 * 4 + 64 * MAX_LIGHTS,
                                             usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self._global_group = device.create_bind_group(layout=self._global_layout, entries=[
            {"binding": 0, "resource": {"buffer": self._globals}}])
        self._sampler = device.create_sampler(min_filter="linear", mag_filter="linear", mipmap_filter="linear",
                                              address_mode_u="clamp-to-edge", address_mode_v="clamp-to-edge")
        # Wraps in U (longitude seam); clamps in V so bilinear filtering never bleeds past the atlas's
        # top and bottom tiles (a little bleed between adjacent tiles is the interactive approximation).
        self._env_sampler = device.create_sampler(min_filter="linear", mag_filter="linear",
                                                   address_mode_u="repeat", address_mode_v="clamp-to-edge")
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
            {"array_stride": 64, "step_mode": "instance", "attributes": [
                {"format": "float32x4", "offset": 0, "shader_location": 1},
                {"format": "float32x4", "offset": 16, "shader_location": 2},
                {"format": "float32x4", "offset": 32, "shader_location": 3},
                {"format": "float32x4", "offset": 48, "shader_location": 5}]},
            {"array_stride": 16, "step_mode": "instance", "attributes": [
                {"format": "float32x4", "offset": 0, "shader_location": 4}]}]
        # One key light's shadow map (R6 "next", closed): the shading passes read it at group 3.
        self._shading_shadow_layout = device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": stage.VERTEX | stage.FRAGMENT, "buffer": {"type": "uniform"}},
            {"binding": 1, "visibility": stage.VERTEX | stage.FRAGMENT, "texture": {"sample_type": "depth"}}])
        self._shading_shadow_buffer = device.create_buffer(
            size=64 + 16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        layouts = [self._global_layout, self._object_layout, self._env_layout, self._shading_shadow_layout]
        self._splat_pipeline = pipeline("splat_vertex", "splat_fragment", layouts, splat_buffers,
                                        "triangle-strip", True)
        self._opaque = pipeline("mesh_vertex", "mesh_fragment", layouts, mesh_buffers, "triangle-list", True)
        self._blended = pipeline("mesh_vertex", "mesh_fragment", layouts, mesh_buffers, "triangle-list", False)
        # Instance3D copies (step X2 of 2): buffer 0 is the source mesh's own per-vertex data
        # (`mesh_buffers[0]`, shared with `_opaque`/`_blended`); buffer 1 is per-instance, one entry
        # per copy: model (4 columns), normal matrix (3 columns, padded to vec4) and straight tint.
        instance_buffers = [mesh_buffers[0], {"array_stride": _INSTANCE_STRIDE, "step_mode": "instance",
            "attributes": [{"format": "float32x4", "offset": offset, "shader_location": location}
                          for location, offset in enumerate((0, 16, 32, 48, 64, 80, 96, 112), start=3)]}]
        self._instance_opaque = pipeline("instance_vertex", "instance_fragment", layouts, instance_buffers,
                                         "triangle-list", True)
        self._instance_blended = pipeline("instance_vertex", "instance_fragment", layouts, instance_buffers,
                                          "triangle-list", False)
        self._line_pipeline = pipeline("line_vertex", "line_fragment", [self._global_layout], line_buffers,
                                       "line-list", False)
        background_buffers = [
            {"array_stride": 8, "step_mode": "vertex", "attributes": [
                {"format": "float32x2", "offset": 0, "shader_location": 0}]},
            {"array_stride": 16, "step_mode": "vertex", "attributes": [
                {"format": "float32x3", "offset": 0, "shader_location": 1}]}]
        self._background_pipeline = pipeline("background_vertex", "background_fragment", [self._env_layout],
                                             background_buffers, "triangle-strip", False)
        self._background_directions = device.create_buffer(
            size=4 * 16, usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST)
        self.show_background = False   # the viewport's `B` key: the dome behind everything, or the solid clear colour

        # The shadow map's own depth-only pass: fragmentless, its own tiny group 0 (`shadow_pass`,
        # just the key light's view_proj), reusing `_object_layout` at group 1 for the model matrix
        # exactly as the main pass does (same per-object uniform buffer, same dynamic offsets).
        self._shadow_pass_layout = device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": stage.VERTEX, "buffer": {"type": "uniform"}}])
        self._shadow_pass_buffer = device.create_buffer(
            size=64, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self._shadow_pass_group = device.create_bind_group(layout=self._shadow_pass_layout, entries=[
            {"binding": 0, "resource": {"buffer": self._shadow_pass_buffer}}])
        self._shadow_pipeline = device.create_render_pipeline(
            layout=device.create_pipeline_layout(
                bind_group_layouts=[self._shadow_pass_layout, self._object_layout]),
            vertex={"module": module, "entry_point": "shadow_vertex", "buffers": mesh_buffers},
            primitive={"topology": "triangle-list", "cull_mode": "none"},
            depth_stencil={"format": "depth24plus", "depth_write_enabled": True, "depth_compare": "less-equal"},
            multisample={"count": 1})
        shadow_texture = device.create_texture(
            size=(SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 1), format="depth24plus",
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING)
        self._shadow_attach_view = shadow_texture.create_view()
        self._shadow_sample_view = shadow_texture.create_view()
        self._shading_shadow_group = device.create_bind_group(layout=self._shading_shadow_layout, entries=[
            {"binding": 0, "resource": {"buffer": self._shading_shadow_buffer}},
            {"binding": 1, "resource": self._shadow_sample_view}])

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

    def _upload_flat(self, image):
        """A single-level rgba16float texture view from a float RGBA image; no mip chain."""
        wgpu, device = self.wgpu, self.device
        height, width = image.shape[:2]
        texture = device.create_texture(size=(width, height, 1), format="rgba16float", mip_level_count=1,
                                        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        data = np.ascontiguousarray(np.clip(image, -65504, 65504), np.float16)
        device.queue.write_texture({"texture": texture, "mip_level": 0, "origin": (0, 0, 0)}, data,
                                   {"offset": 0, "bytes_per_row": width * 8, "rows_per_image": height}, (width, height, 1))
        return texture.create_view()

    def _update_environment(self, scene):
        """Rebuild the dome's uniform and atlas when the scene's first Environment changes (see the
        `EnvGlobals` comment in `_SHADER`: one dome, folded gain, a six-tile roughness atlas)."""
        from . import envlight
        environments = getattr(scene, 'environments', ())
        key = None
        if environments:
            env = environments[0]
            key = (env.fingerprint, float(env.intensity), float(env.rotation), float(env.blur),
                  tuple(float(t) for t in env.tint))
        if key == self._env_key and self._env_group is not None:
            return
        self._env_key = key
        block = np.zeros(9 * 4 + 4, np.float32)
        tile_w, tile_h = envlight.PREFILTER_SIZE
        if key is None:
            atlas = np.zeros((6 * tile_h, tile_w, 4), np.float32)
            atlas[..., 3] = 1.0
        else:
            pre = env._pre()
            gain = float(env.intensity) * np.asarray(env.tint, np.float64)
            dirs, _ = envlight.direction_grid(tile_w, tile_h)   # (tile_h, tile_w, 3)
            level0 = envlight.sample_map(pre.levels[0], dirs.reshape(-1, 3)).reshape(tile_h, tile_w, 3)
            levels = [level0] + [np.asarray(level, np.float64) for level in pre.levels[1:]]
            rgb = (np.concatenate(levels, axis=0) * gain[None, None, :]).astype(np.float32)
            atlas = np.concatenate([rgb, np.ones(rgb.shape[:2] + (1,), np.float32)], axis=-1)
            # `array<vec4<f32>, 9>` pads every SH coefficient to 16 bytes; only the first 3 floats of
            # each 4 are used (see `env_diffuse`'s `envg.sh[i].rgb`).
            block[:36].reshape(9, 4)[:, :3] = (pre.sh * gain[None, :]).astype(np.float32)
            angle = math.radians(float(env.rotation))
            block[36:40] = 1.0, math.cos(angle), math.sin(angle), 0.0
        self.device.queue.write_buffer(self._env_buffer, 0, block)
        self._env_group = self.device.create_bind_group(layout=self._env_layout, entries=[
            {"binding": 0, "resource": {"buffer": self._env_buffer}},
            {"binding": 1, "resource": self._upload_flat(atlas)},
            {"binding": 2, "resource": self._env_sampler}])

    def _ensure_targets(self, width, height):
        if self._targets and self._targets["size"] == (width, height):
            return self._targets
        wgpu, device = self.wgpu, self.device
        usage = wgpu.TextureUsage.RENDER_ATTACHMENT
        stride = (width * 4 + 255) // 256 * 256
        # Sampled as well as attached: the volume pass reads the depth the geometry drew.
        depth_texture = device.create_texture(size=(width, height, 1), format="depth24plus", sample_count=SAMPLES,
                                              usage=usage | wgpu.TextureUsage.TEXTURE_BINDING)
        self._targets = dict(
            size=(width, height), stride=stride,
            color=device.create_texture(size=(width, height, 1), format="rgba8unorm-srgb",
                                        sample_count=SAMPLES, usage=usage).create_view(),
            depth=depth_texture.create_view(), depth_sample=depth_texture.create_view(),
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

    def _instance_mesh(self, geometry, used):
        """(vertex buffer, uint32 index buffer, index count) for one Instance3D source mesh, deduplicated
        from `_soup`'s per-corner triangle list (measured: a sphere's shared vertex normals collapse
        ~3 corners to ~1 unique vertex on average). A hardware-instanced draw pays this per-vertex
        transform cost once per copy, so at 100,000 copies the 3x-corner triangle soup `_mesh` uses
        for the one-draw-per-object path became the bottleneck (measured: 55 ms/frame at 990
        triangles x 100,000 with the soup, 18 ms indexed, on an RTX 3080 Ti); the plain per-object
        path stays on the soup since it never pays that cost more than once a frame regardless."""
        arrays = (geometry.vertices, geometry.triangles, geometry.normals, geometry.uvs)
        key = tuple(id(a) for a in arrays)
        used.add(key)
        entry = self._instance_meshes.get(key)
        if entry is None:
            soup = _soup(geometry)
            if len(soup):
                unique, inverse = np.unique(soup, axis=0, return_inverse=True)
                indices = np.ascontiguousarray(inverse, np.uint32)
            else:
                unique, indices = np.zeros((3, 8), np.float32), np.zeros(3, np.uint32)
            vertex_buffer = self.device.create_buffer_with_data(
                data=np.ascontiguousarray(unique, np.float32), usage=self.wgpu.BufferUsage.VERTEX)
            index_buffer = self.device.create_buffer_with_data(data=indices, usage=self.wgpu.BufferUsage.INDEX)
            # The arrays are held so their ids cannot be recycled while the entry lives.
            entry = self._instance_meshes[key] = (vertex_buffer, index_buffer, len(indices), arrays)
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

    def _instances(self, instance_set, used):
        """(buffer, [(variant, first_instance, count)]) for one `InstanceSet`: a single vertex
        buffer holding every copy's model matrix, normal matrix and tint, grouped by source variant
        so each group draws in one hardware-instanced call (`draw(..., first_instance=...)`).
        Cached on the identity of the set's own arrays, like every other upload here: an unchanged
        InstanceSet (nothing moved, nothing re-evaluated) costs nothing after its first frame."""
        key = (id(instance_set.matrices), id(instance_set.variant), id(instance_set.parent),
              id(instance_set.colors) if instance_set.colors is not None else None, id(instance_set.sources))
        used.add(key)
        entry = self._instance_buffers.get(key)
        if entry is None:
            entry = self._instance_buffers[key] = self._build_instances(instance_set)
            self.uploads += 1
        return entry

    def _build_instances(self, instance_set):
        variant = np.asarray(instance_set.variant, np.int64)
        order = np.argsort(variant, kind="stable")
        variant_sorted = variant[order]
        parent = np.asarray(instance_set.parent, np.float64)
        matrices = np.asarray(instance_set.matrices, np.float64)[order]
        bases = np.stack([np.asarray(s.world_matrix(), np.float64) for s in instance_set.sources])
        model = parent[None] @ matrices @ bases[variant_sorted]   # (N,4,4)
        linear = model[:, :3, :3]
        normal = np.tile(np.eye(3), (len(linear), 1, 1))
        nonsingular = np.abs(np.linalg.det(linear)) > 1e-12
        if np.any(nonsingular):
            # transpose(inverse(linear)): the world normal transform, batch-inverted for every
            # non-singular instance at once. A zero-scale instance (singular) keeps the identity
            # above; it has no surface to light correctly either way.
            normal[nonsingular] = np.linalg.inv(linear[nonsingular]).transpose(0, 2, 1)
        data = np.zeros((len(model), _INSTANCE_STRIDE // 4), np.float32)
        data[:, 0:4], data[:, 4:8] = model[:, :, 0], model[:, :, 1]
        data[:, 8:12], data[:, 12:16] = model[:, :, 2], model[:, :, 3]
        data[:, 16:19], data[:, 20:23], data[:, 24:27] = normal[:, :, 0], normal[:, :, 1], normal[:, :, 2]
        data[:, 28:32] = instance_set.colors[order] if instance_set.colors is not None else 1.0
        buffer = self.device.create_buffer_with_data(data=data, usage=self.wgpu.BufferUsage.VERTEX)
        groups, counts = [], np.bincount(variant_sorted, minlength=len(instance_set.sources))
        first = 0
        for source_index, count in enumerate(counts):
            if count:
                groups.append((source_index, first, int(count)))
                first += int(count)
        # `instance_set` (and its arrays) are held so the cache key's ids cannot be recycled.
        return buffer, groups, instance_set

    def _neutral_indirect(self, count):
        """A vertex buffer of `count` rows (occlusion 1, no bounce): what a splat without indirect light reads."""
        if getattr(self, '_neutral', (None, 0))[1] < count:
            data = np.zeros((count, 4), np.float32)
            data[:, 0] = 1.0
            self._neutral = (self.device.create_buffer_with_data(data=data, usage=self.wgpu.BufferUsage.VERTEX), count)
        return self._neutral[0]

    def _splat_indirect(self, instance, scene, ambient, eye, count, stride):
        """The vertex buffer of `(occlusion, bounce rgb)` per splat for the viewport preview, else the neutral one.

        Traced once on the CPU at the `preview` preset (`splatindirect`), from the scene's splats alone, and
        cached; it is recomputed when the scene, lights, ambient or knobs change or when more than
        `INDIRECT_RESIGN` of the splats would now face the other way from the eye (the rays leave the eye side)."""
        from . import envlight, splatindirect, splatshade
        if (instance.relight <= 0 or stride != 1 or count > INDIRECT_MAX_SPLATS
                or splatindirect.effective_samples(replace(instance, quality='preview'), 'indirect_samples') <= 0
                or float(instance.indirect_distance) <= 0):
            return self._neutral_indirect(count)
        world = instance.cloud.transformed(instance.matrix)
        intrinsics = splatshade.uses_intrinsics(instance, world)
        normals = np.asarray(world.normals() if intrinsics is None else intrinsics.normal, np.float64)
        signs = np.sum(normals * (np.asarray(eye, np.float64) - world.positions), axis=1) >= 0
        lights = tuple((l.kind, tuple(l.color), l.intensity, l.world()[0].tobytes(), l.world()[1].tobytes(),
                        l.cone_angle, l.cone_penumbra_angle, l.falloff_type) for l in scene.lights)
        key = (id(instance.cloud), np.asarray(instance.matrix, np.float64).tobytes(), instance.indirect_samples,
               float(instance.indirect_distance), float(instance.denoise), instance.use_intrinsics,
               float(instance.metallic), float(ambient), lights,
               tuple((id(i.cloud), np.asarray(i.matrix, np.float64).tobytes(), i.relight > 0) for i in scene.splats))
        entry = self._indirect.get(id(instance.cloud))
        if entry is not None and entry[0] == key and float(np.mean(entry[1] != signs)) <= INDIRECT_RESIGN:
            return entry[2]
        preview = replace(instance, quality='preview')
        shadows = scene3d._SplatShadows(scene.splats, scene.lights, None, None, .001)
        shadows.relit_shadows = False
        tracer = splatindirect.IndirectLight(shadows, ambient, scene.environments)
        extras = envlight.SplatLighting(indirect=tracer)
        albedo = splatshade.splat_albedo(world) if intrinsics is None else intrinsics.albedo
        occlusion, bounce = splatshade.indirect_terms(preview, world, eye, extras, albedo, normals)
        bounce = (1 - float(np.clip(instance.metallic, 0, 1))) * splatindirect.guided_denoise(
            bounce, world.positions, normals, albedo, float(instance.denoise))
        data = np.concatenate((occlusion[:, None], bounce), axis=1)
        buffer = self.device.create_buffer_with_data(data=np.ascontiguousarray(data, np.float32),
                                                     usage=self.wgpu.BufferUsage.VERTEX)
        self._indirect[id(instance.cloud)] = (key, signs, buffer, instance.cloud)
        return buffer

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
                {"binding": 0, "resource": {"buffer": buffer, "offset": 0, "size": 288}},
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

        (used_meshes, used_textures, used_splats, used_instances,
         used_instance_meshes, draws, clouds) = set(), set(), set(), set(), set(), [], []
        # Instance3D groups (step X2 of 2) are gathered before `uniforms` is sized, since each
        # (InstanceSet, source variant) present needs its own object row same as a geometry does.
        instance_groups = []
        for instance_set in scene.instances:
            if not len(instance_set) or not instance_set.sources:
                continue
            buffer, groups, _held = self._instances(instance_set, used_instances)
            for source_index, first, count in groups:
                source = instance_set.sources[source_index]
                mesh_buffer, index_buffer, index_count, _arrays = self._instance_mesh(source, used_instance_meshes)
                if not index_count:
                    continue
                image = source.texture
                view, texture_opaque = (self._texture(image, used_textures)[:2] if image is not None
                                        else (self._white, True))
                opaque = source.color[3] >= 0.999 and texture_opaque
                instance_groups.append((opaque, source, mesh_buffer, index_buffer, index_count, view,
                                        buffer, first, count))
        object_count = len(scene.geometries) + len(scene.splats) + len(instance_groups)
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
            if geometry.material == "pbr":
                row[68:72] = (1.0, geometry.metallic, geometry.pbr_roughness,
                             0.08 * float(np.clip(geometry.pbr_specular, 0, 1)))
            else:
                row[68:72] = 0.0, 0.0, math.sqrt(2.0 / (float(geometry.shininess) + 2.0)), 0.0
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
            # material.y/z/w (materials 1/R6): the `Roughness` multiplier, metallic (still one
            # constant per cloud, as the CPU fit's own limit is) and how much of the proxy's
            # per-splat de-lit albedo/roughness to mix in (0 when the cloud has none, or
            # `use_intrinsics` is off, so the shader falls back to the SH-DC/neutral columns).
            has_intrinsics = getattr(instance.cloud, 'intrinsics', None) is not None
            mix = float(np.clip(instance.intrinsics_mix, 0, 1)) if (
                has_intrinsics and instance.use_intrinsics) else 0.0
            row[36:40] = (SPLAT_MAX_PIXELS, float(np.clip(instance.roughness_scale, 0.0, 4.0)),
                         float(np.clip(instance.metallic, 0, 1)), mix)
            clouds.append((index, buffer, count, self._splat_indirect(instance, scene, ambient, eye, count, stride)))
        instance_draws = []
        for offset, (opaque, source, mesh_buffer, index_buffer, index_count, view, buffer, first, count) in enumerate(
                instance_groups, len(scene.geometries) + len(scene.splats)):
            row = uniforms[offset]
            row[:16] = row[16:32] = np.eye(4, dtype=np.float32).ravel()  # unused by instance_vertex; identity is inert
            row[32:36] = source.color
            row[36:40] = source.specular, source.shininess, source.emission, float(source.texture is not None)
            if source.material == "pbr":
                row[68:72] = (1.0, source.metallic, source.pbr_roughness,
                             0.08 * float(np.clip(source.pbr_specular, 0, 1)))
            else:
                row[68:72] = 0.0, 0.0, math.sqrt(2.0 / (float(source.shininess) + 2.0)), 0.0
            instance_draws.append((opaque, offset, mesh_buffer, index_buffer, index_count, view, buffer, first, count))
        device.queue.write_buffer(self._object_buffer(object_count), 0, uniforms)

        self._update_environment(scene)
        line_count = self._upload_lines(lines)
        frame_resources = []
        volume_pass = self._prepare_volumes(scene, camera, width, height, ambient, lights, targets, frame_resources)
        encoder = device.create_command_encoder()
        self._render_shadow_map(encoder, scene, lights, draws, object_count)
        # With volumes the geometry pass keeps its colour and depth for a second pass that raymarches over them.
        render_pass = encoder.begin_render_pass(
            color_attachments=[{"view": targets["color"],
                                "resolve_target": None if volume_pass else targets["resolve_view"],
                                "clear_value": _linear(background), "load_op": "clear",
                                "store_op": "store" if volume_pass else "discard"}],
            depth_stencil_attachment={"view": targets["depth"], "depth_clear_value": 1.0,
                                      "depth_load_op": "clear", "depth_store_op": "store" if volume_pass else "discard"})
        render_pass.set_bind_group(0, self._global_group)
        render_pass.set_bind_group(2, self._env_group)
        render_pass.set_bind_group(3, self._shading_shadow_group)

        def draw(items, pipeline):
            if items:
                render_pass.set_pipeline(pipeline)
            for _opaque, _order, index, buffer, count, view in items:
                render_pass.set_bind_group(1, self._object_group(view, object_count),
                                           dynamic_offsets_data=[index * _OBJECT_STRIDE])
                render_pass.set_vertex_buffer(0, buffer)
                render_pass.draw(count)

        def draw_instances(items, pipeline):
            # One hardware-instanced, indexed draw call per (InstanceSet, source variant) group:
            # `first` selects that group's slice of the one vertex buffer `_build_instances` packed
            # every copy into, grouped by variant already, so no per-instance CPU work happens here.
            if items:
                render_pass.set_pipeline(pipeline)
            for _opaque, index, mesh_buffer, index_buffer, index_count, view, instance_buffer, first, count in items:
                render_pass.set_bind_group(1, self._object_group(view, object_count),
                                           dynamic_offsets_data=[index * _OBJECT_STRIDE])
                render_pass.set_vertex_buffer(0, mesh_buffer)
                render_pass.set_vertex_buffer(1, instance_buffer)
                render_pass.set_index_buffer(index_buffer, "uint32")
                render_pass.draw_indexed(index_count, count, 0, 0, first)

        draw([d for d in draws if d[0]], self._opaque)
        draw_instances([d for d in instance_draws if d[0]], self._instance_opaque)
        if clouds:
            render_pass.set_pipeline(self._splat_pipeline)
            render_pass.set_vertex_buffer(0, self._corners)
        for index, buffer, count, extra in clouds:
            render_pass.set_bind_group(1, self._object_group(self._white, object_count),
                                       dynamic_offsets_data=[index * _OBJECT_STRIDE])
            render_pass.set_vertex_buffer(1, buffer)
            render_pass.set_vertex_buffer(2, extra)
            render_pass.draw(4, count)
        if line_count:
            render_pass.set_pipeline(self._line_pipeline)
            render_pass.set_vertex_buffer(0, self._lines[1])
            render_pass.draw(line_count)
        if self.show_background and scene.environments:
            device.queue.write_buffer(self._background_directions, 0, _background_ray_corners(camera, width, height))
            render_pass.set_pipeline(self._background_pipeline)
            render_pass.set_bind_group(0, self._env_group)
            render_pass.set_vertex_buffer(0, self._corners)
            render_pass.set_vertex_buffer(1, self._background_directions)
            render_pass.draw(4)
            render_pass.set_bind_group(0, self._global_group)  # restore group 0 for the draws below
        draw(sorted((d for d in draws if not d[0]), key=lambda d: d[1]), self._blended)  # far to near
        draw_instances([d for d in instance_draws if not d[0]], self._instance_blended)
        self._draw_particles(render_pass, scene, camera, width, height)  # last, and with its own group 0
        render_pass.end()
        if volume_pass:
            render_pass = encoder.begin_render_pass(color_attachments=[{
                "view": targets["color"], "resolve_target": targets["resolve_view"], "clear_value": (0, 0, 0, 0),
                "load_op": "load", "store_op": "discard"}])
            volume_pass.record(render_pass)
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
            for resource in frame_resources:
                resource.destroy()
        for cache, used in ((self._meshes, used_meshes), (self._textures, used_textures),
                            (self._splats, used_splats), (self._indirect, used_splats),
                            (self._instance_buffers, used_instances),
                            (self._instance_meshes, used_instance_meshes)):
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

    def _volume_settings(self, scene, camera, width, height, light_count):
        """March settings for the frame: the toggle's step count over the largest diagonal, coarsened to the budget."""
        steps, shadow_steps = VOLUME_QUALITY if self.volume_quality else VOLUME_FAST
        diagonal = max(float(np.linalg.norm(np.asarray(v.matrix, np.float64)[:3, :3] @ (np.array(v.shape) * v.voxel_size)))
                       for v in scene.volumes)
        settings = volumerender.VolumeSettings(step_size=max(diagonal / steps, 1e-4), shadow_steps=shadow_steps)
        target = gpuvolume.VOLUME_WORK_BUDGETS[gpuvolume.adapter_kind(self._state)] * VOLUME_FRAME_FRACTION
        work = gpuvolume.work_estimate(scene, camera, width, height, settings, light_count)
        if work > target:
            settings = replace(settings, step_size=settings.step_size * work / target)
        self.volume_steps = max(1, round(diagonal / settings.step_size))
        return settings

    def _prepare_volumes(self, scene, camera, width, height, ambient, lights, targets, resources):
        """The frame's volume draw (gpuvolume, shared with Render3D), or None with `volume_note` saying why not."""
        self.volume_note, self.volume_steps = "", 0
        if not scene.volumes:
            return None
        def keep(resource):
            resources.append(resource)
            return resource

        try:
            settings = self._volume_settings(scene, camera, width, height, len(lights))
            gpuvolume.check(self._state, scene, settings)
            buffer = keep(self.device.create_buffer_with_data(data=gpu3d.light_table(lights),
                                                              usage=self.wgpu.BufferUsage.STORAGE))
            return gpuvolume.prepare(
                self._state, scene, camera, width, height, ambient, settings, buffer, len(lights), bool(scene.lights),
                targets["depth_sample"], keep, target="rgba8unorm-srgb", samples=SAMPLES, multisampled_depth=True)
        except (gpu3d.Unsupported, ValueError) as error:
            self.volume_note = f"volumes hidden: {error}"
            return None

    def _render_shadow_map(self, encoder, scene, lights, draws, object_count):
        """The key light's depth-only pass into `_shadow_attach_view`, then the shading passes'
        `_shading_shadow_buffer` (view_proj, enabled, the map size, the light's `globals.lights`
        index). Only opaque meshes cast (`draws`, already built by `_render`); splats and blended
        meshes do not, a stated limit. With no shadow-enabled Directional or Spot light this only
        disables the shading lookup -- the depth texture is left however it last was."""
        choice = _shadow_light(lights)
        if choice is None:
            self.device.queue.write_buffer(self._shading_shadow_buffer, 64, np.zeros(4, np.float32))
            return
        index, light, position, direction = choice
        view_proj = shadow_view_proj(light, position, direction, _mesh_bounds(scene))
        matrix = np.asarray(view_proj.T.ravel(), np.float32)
        self.device.queue.write_buffer(self._shadow_pass_buffer, 0, matrix)
        self.device.queue.write_buffer(self._shading_shadow_buffer, 0, matrix)
        self.device.queue.write_buffer(self._shading_shadow_buffer, 64,
                                       np.array((1.0, 0.0, float(SHADOW_MAP_SIZE), float(index)), np.float32))
        opaque = [d for d in draws if d[0]]
        if not opaque:
            return
        pass_ = encoder.begin_render_pass(color_attachments=[], depth_stencil_attachment={
            "view": self._shadow_attach_view, "depth_clear_value": 1.0,
            "depth_load_op": "clear", "depth_store_op": "store"})
        pass_.set_pipeline(self._shadow_pipeline)
        pass_.set_bind_group(0, self._shadow_pass_group)
        for _opaque, _order, obj_index, buffer, count, view in opaque:
            pass_.set_bind_group(1, self._object_group(view, object_count),
                                 dynamic_offsets_data=[obj_index * _OBJECT_STRIDE])
            pass_.set_vertex_buffer(0, buffer)
            pass_.draw(count)
        pass_.end()

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
