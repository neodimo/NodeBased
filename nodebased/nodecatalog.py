"""The one place every node kind is filed under a toolbar category with a plain-English
description, so an artist can browse and search node types instead of having to already know
their name (`docs/PARITY_2D.md`'s section names are these categories for the 2D ones; the NODES
dock and the Tab search's category hint in `app.py` both read this module).

Categories follow Nuke's own toolbar groups where the node is a 2D one, with 3D, Particles and
Fluids added for this app's non-2D nodes. `NODE_CATEGORIES` is an ordered mapping of category name
to an ordered mapping of node type to its one-line description, so both the category order (this
dict's own order) and the node order within a category (each inner dict's own order) are exactly
what a lane wrote here -- tests/test_nodecatalog.py asserts every `core.SPECS` type is filed in
exactly one category with a non-empty description, so a lane that ships a node without filing it
here fails the suite.
"""

NODE_CATEGORIES = {
    "Image": {
        "Read": "Loads an image or an image sequence from disk.",
        "Constant": "A flat, uniform colour at a given size.",
        "Checker": "A checkerboard test pattern.",
        "Viewer": "Shows the graph's picture on screen, with A/B compare and channel controls.",
        "Write": "Renders its input to disk as EXR or PNG.",
        "ReadBundle": "Reads a diffusion or transform model's rendered output for the current frame.",
    },
    "Draw": {
        "Roto": "Draws and animates bezier or B-spline shapes as a matte.",
        "Text": "Renders text onto the image or a transparent frame.",
        "Rectangle": "A soft-edged rectangle, optionally over an image.",
        "Ramp": "A linear gradient between two colours.",
        "Radial": "A soft-edged disc, optionally over an image.",
        "Noise": "Fractal noise, with octaves and a seed.",
        "LightWrap": "Wraps the background's light around the foreground's edge.",
        "Grain": "Adds synthetic film grain per channel.",
        "Dither": "Adds noise before quantising, to hide banding.",
        "Grid": "Draws an evenly spaced line grid.",
    },
    "Time": {
        "TimeOffset": "Shifts the input's frame by a fixed offset.",
        "FrameHold": "Holds on one frame, or steps forward every few frames.",
        "Retime": "Maps an output frame range onto a different input range.",
        "TimeClip": "Clips the input to a frame range, with a policy for frames outside it.",
        "FrameRange": "Clamps, loops or bounces the input outside a frame range.",
        "AppendClip": "Plays up to eight clips head to tail.",
    },
    "Channel": {
        "Shuffle": "Routes channels or a named layer into R, G, B and A.",
        "ChannelShuffle": "Routes explicit channels from two inputs into R, G, B and A.",
        "Copy": "Copies chosen channels from one input onto another.",
        "ChannelMerge": "Merges one channel of each input using a merge operation.",
    },
    "Color": {
        "Grade": "Exposure, multiply and offset colour correction.",
        "ColorCorrect": "Lift, gamma, gain and saturation colour correction.",
        "Invert": "Inverts selected channels.",
        "Clamp": "Clamps values to a minimum and maximum.",
        "Multiply": "Multiplies selected channels by a value.",
        "Add": "Adds a value to selected channels.",
        "Gamma": "Applies a gamma curve to selected channels.",
        "Saturation": "Scales colour saturation.",
        "Exposure": "Exposure adjustment in stops or densities.",
        "HueCorrect": "Per-hue saturation and luminance adjustment.",
        "ColorMatrix": "A 3x3 colour matrix applied to RGB.",
        "Log2Lin": "Converts Cineon log code values and linear light.",
        "PLogLin": "Converts between linear and density-coded log values.",
        "CrossTalk": "Mixes channel-dependent response curves.",
        "Toe": "Lifts shadows with a smooth toe curve.",
        "Expression": "Evaluates per-channel pixel formulas.",
        "Posterize": "Reduces the image to a fixed number of levels per channel.",
        "SoftClip": "Compresses highlights above a threshold instead of clipping them.",
        "HSVTool": "Adjusts hue, saturation and brightness within chosen ranges.",
    },
    "Filter": {
        "Blur": "A box blur.",
        "Erode": "Shrinks (or, negative, grows) the image with a box filter.",
        "Dilate": "Grows the image with a box filter.",
        "Median": "A despeckling median filter.",
        "Sharpen": "An unsharp-mask sharpen.",
        "Matrix": "A user-defined 3x3, 5x5 or 7x7 convolution kernel.",
        "Laplacian": "A four-neighbour edge-response filter.",
        "Convolve": "Convolves the image with a kernel read from a second image.",
        "EdgeDetect": "Sobel, Prewitt or Laplacian edge detection with a threshold.",
        "Emboss": "An embossed relief lit from an angle.",
        "BumpBoss": "Embosses using a chosen channel as a height map.",
        "ErodeFilter": "Shrinks a matte by a size, with box or Gaussian filtering.",
        "Glow": "Blurs and adds back the bright areas of the image.",
        "Soften": "A Gaussian blur.",
        "Defocus": "A disc blur that mimics an out-of-focus lens.",
        "Bilateral": "Smooths noise while preserving colour edges.",
        "Denoise": "A practical bilateral-plus-temporal noise reducer (not Nuke Denoise).",
        "DegrainSimple": "Blurs RGB channels by separately chosen amounts.",
        "ZDefocus": "Blurs by depth around a chosen focal plane.",
        "DirBlur": "Linear, zoom or radial directional blur.",
        "DropShadow": "Casts a blurred, offset, tinted shadow from the image's alpha.",
        "EdgeBlur": "Blurs only along the alpha's edge.",
        "EdgeExtend": "Pushes edge colour outward to kill dark fringes.",
        "VectorBlur": "Blurs along a motion-vector field.",
    },
    "Keyer": {
        "Keyer": "Keys alpha from a chosen channel through a four-point range.",
        "HueKeyer": "Keys alpha from a hue range with softness.",
        "ChromaKeyer": "A colour-distance key against a chosen screen colour.",
        "IBKColor": "Builds a clean screen plate for image-based keying.",
        "IBKGizmo": "Keys against an IBKColor clean plate.",
        "ScreenKeyer": "A Keylight-style screen-difference keyer.",
        "Cryptomatte": "Extracts an ID matte from a Cryptomatte layer.",
        "Difference": "Keys alpha from the colour difference between two inputs.",
    },
    "Merge": {
        "Merge": "Composites two inputs with a chosen operation (over, plus, screen, and more).",
        "Premult": "Multiplies colour by alpha.",
        "Unpremult": "Divides colour by alpha.",
        "Switch": "Passes one of two inputs.",
        "Dissolve": "Cross-fades linearly between two inputs.",
        "Keymix": "Copies one input over another wherever a mask is non-zero.",
        "AddMix": "Premultiplies A, then merges it over B.",
        "Blend": "A weighted average of up to eight inputs.",
        "CopyRectangle": "Copies a rectangular area from one input onto another.",
    },
    "Transform": {
        "Transform": "Translate, rotate and scale, with sub-pixel filtering.",
        "Crop": "Shrinks the data window to a box.",
        "Tracker": "Applies a solved match-move or stabilise transform.",
        "Reformat": "Changes the image's format: a named preset, a scale or a box.",
        "CornerPin": "A four-point projective warp.",
        "Mirror": "Flips the image horizontally or vertically.",
        "Position": "Moves pixels and the data window by whole pixels.",
        "AdjustBBox": "Grows or shrinks the data window without moving pixels.",
        "BlackOutside": "Adds a one-pixel black border outside the data window.",
        "STMap": "Remaps pixels using an absolute UV map.",
        "IDistort": "Offsets pixels using a relative UV displacement map.",
    },
    "3D": {
        "Axis3D": "Parents whatever geometry, light or scene is wired into it.",
        "TransformGeo3D": "Bakes a transform into geometry's own vertices and normals.",
        "MergeGeo3D": "Merges up to eight geometries into one, baking their transforms in.",
        "Normals3D": "Recomputes, flips or unifies a geometry's normals.",
        "DisplaceGeo3D": "Moves vertices along their normals by an image channel.",
        "Shrinkwrap3D": "Fits UV'd proxy geometry onto the surface of a target mesh.",
        "Card3D": "A flat, texturable card, subdividable into a grid.",
        "Cube3D": "A textured cube.",
        "Sphere3D": "A smooth-shaded latitude/longitude sphere.",
        "Cylinder3D": "A cylinder with optional end caps.",
        "ReadGeo3D": "A Wavefront OBJ mesh from disk.",
        "ReadSplat3D": "A 3D Gaussian splat cloud from a .ply file, with relighting.",
        "ReadAlembic3D": "Polygon meshes from an Alembic .abc file.",
        "ReadAlembicCamera3D": "A camera from an Alembic .abc file.",
        "ReadUSD3D": "A USD stage as a scene.",
        "ReadUSDCamera3D": "A camera from a USD stage.",
        "ReadGLTF3D": "Meshes from a glTF .glb or .gltf file.",
        "Light3D": "A directional, point, spot or environment light.",
        "Camera3D": "A camera with position, target and film-back lens knobs.",
        "Project3D": "Projects an image through a camera onto geometry.",
        "WriteGeo3D": "Exports the scene to a Wavefront OBJ on request.",
        "WriteSplat3D": "Exports the scene's splats to a 3DGS .ply on request.",
        "Scene3D": "Groups up to eight geometry, light or scene inputs under one transform.",
        "Relight": "Recombines a Render3D relight bundle with new light colour and intensity.",
        "Render3D": "Renders a scene through a camera to an image.",
    },
    "Particles": {
        "ParticleEmitter3D": "Emits particles from a point or from input geometry.",
        "ParticleCache3D": "Caches solved particles to disk so scrubbing never re-solves.",
        "ParticleGravity3D": "Accelerates particles toward a direction.",
        "ParticleDrag3D": "Slows particles down over time.",
        "ParticleWind3D": "Pushes particles with a gusting wind.",
        "ParticleTurbulence3D": "Adds curling turbulence to particle motion.",
        "ParticleBounce3D": "Collides particles against geometry.",
        "ParticleRender3D": "Chooses how particles are drawn: points, spheres or cards.",
        "Instance3D": "Copies a mesh (or up to eight, by variant) onto every particle or point.",
    },
    "Fluids": {
        "Plume3D": "A deterministic analytic smoke plume, for demos and tests.",
        "ReadVDB3D": "A Houdini Pyro or Blender OpenVDB smoke cache from disk.",
        "FluidSource3D": "Where a fluid solve gets its smoke, fire or liquid.",
        "FluidForce3D": "Adds buoyancy, gravity, wind, turbulence or drag to a fluid.",
        "FluidCollide3D": "Makes geometry a solid obstacle for a fluid solve.",
        "FluidSolver3D": "Solves smoke and fire on a grid.",
        "FluidLiquidSolver3D": "Solves a liquid with a FLIP/PIC particle solver.",
        "FluidSurface3D": "Meshes a liquid's particles into a closed surface.",
        "FluidFoam3D": "Generates foam and spray particles from a fast-moving liquid.",
        "FluidCache3D": "Caches solved fluid volumes to disk so scrubbing never re-solves.",
    },
    "Metadata": {
        "ViewMetaData": "Lists the image's metadata keys and values.",
        "ModifyMetaData": "Sets, removes or renames metadata keys.",
        "CopyMetaData": "Copies metadata keys from a second input onto the image.",
        "CompareMetaData": "Lists metadata keys that differ from a second input.",
        "AddTimeCode": "Writes an SMPTE timecode into the image's metadata.",
        "BurnIn": "Draws frame, timecode or metadata text into the corners of the image.",
    },
    "Other": {
        "Dot": "A neutral reroute on a wire.",
        "NoOp": "A passthrough node with a properties panel and a note.",
        "Backdrop": "A labelled, coloured box behind nodes on the graph, for organising.",
        "PostageStamp": "Shows a live thumbnail of its input on the graph.",
        "Group": "A node graph of its own, with Input and Output nodes inside.",
        "Input": "One of a group's numbered input slots.",
        "Output": "What a group produces.",
    },
}

# Reverse index built once: node type -> category, so a caller never has to scan every category.
NODE_CATEGORY_OF = {kind: category for category, kinds in NODE_CATEGORIES.items() for kind in kinds}


def node_category(kind):
    """The category a node type is filed under, or "Other" for one somehow not in the catalog
    (kept as a fallback so a stale catalog degrades to an extra Other row rather than a KeyError)."""
    return NODE_CATEGORY_OF.get(kind, "Other")


def node_description(kind):
    """The node's one-line description, or "" when it is not in the catalog."""
    category = NODE_CATEGORY_OF.get(kind)
    return NODE_CATEGORIES[category][kind] if category else ""
