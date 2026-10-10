"""Neutral charcoal surfaces; limited, legible accents identify node families."""
COLORS = {"Read": "#d9b879", "ReadBundle": "#d9b879", "ConditionedRead": "#d9b879", "Generate": "#d9b879", "Checker": "#d9b879", "Constant": "#d9b879",
          "Grade": "#83cbb7", "Vectorfield": "#80c8b0", "GenerateLUT": "#e06f6f", "ColorCorrect": "#6fc9b0", "Blur": "#79c7d9",
          "OCIOColorspace": "#80c8b0", "OCIODisplay": "#80c8b0", "OCIOFileTransform": "#80c8b0", "OCIOLookTransform": "#80c8b0", "OCIOLogConvert": "#80c8b0",
          "Transform": "#89aff0", "Crop": "#6f9be0", "Shuffle": "#a889d9",
          "ChannelShuffle": "#9d80d4", "ShuffleCopy": "#a889d9", "Roto": "#e2937f", "RotoPaint": "#d87862", "Tracker": "#e0b06a", "Stabilize": "#e0b06a",
          "Merge": "#bd9ee3", "Dot": "#b8a6db", "Switch": "#c6a5db",
          # Premult/Unpremult are the other Merge-toolbar single-input nodes; not filed here was
          # a `theme.COLORS` gap that crashed `NodeItem` the moment either landed on the graph
          # (`nodebased/app.py`'s R/G/M/T/.../P/U hotkeys and NODES dock both create them).
          "Premult": "#a689d9", "Unpremult": "#9a7bcf",
          # Invert/Clamp/Multiply/Add/Gamma/Saturation split a single Grade/ColorCorrect knob out
          # onto its own node, so they stay in the same teal-green Color family.
          "Invert": "#6fbfa8", "Clamp": "#6fbfa8", "Multiply": "#7ecab3", "Add": "#7ecab3",
          "Gamma": "#83cbb7", "Saturation": "#6fc9b0",
          # Erode/Dilate/Median/Sharpen/Glow are Filter-menu siblings of Blur, so they stay in the
          # same sky-blue family, each a step around Blur's own hue; Mirror is a Transform-menu
          # node and stays in Transform's blue family instead.
          "Erode": "#6bb8cc", "Dilate": "#6bb8cc", "Median": "#72c2d4", "Sharpen": "#8fd0de",
          "Matrix": "#82c9d8", "Laplacian": "#79c1d2", "EdgeDetect": "#82c9d8",
          "Emboss": "#82c9d8", "BumpBoss": "#82c9d8", "ErodeFilter": "#6bb8cc",
          "Glow": "#a3d8e2", "Soften": "#7cc4d6", "Defocus": "#6fb4d0", "Bilateral": "#64abc3", "Denoise": "#58a2bc", "DegrainSimple": "#4b99b5", "ZDefocus": "#408fab", "Inpaint": "#377f99", "ZMerge": "#377f99", "ZSlice": "#33758e", "Remove": "#6f9caa", "DirBlur": "#68acc8", "DropShadow": "#5fa3c0",
          "GodRays": "#6fb4d0", "VolumeRays": "#64abc3", "LevelSet": "#609ab5", "ScannedGrain": "#58a2bc",
          # EdgeBlur/EdgeExtend/LightWrap/Dither (step 5a) are Filter-menu matte and finishing nodes.
          "EdgeBlur": "#7dbad0", "EdgeExtend": "#89c2d6", "LightWrap": "#96cadc", "Dither": "#6aaac4",
          # Grain (Draw menu), Posterize/SoftClip/HSVTool (Color menu), AddMix/Blend/CopyRectangle (Merge menu), step 5b.
          "Grain": "#a3d8a3", "Flare": "#a3d8a3", "Glint": "#77c9a8", "Sparkles": "#83cbb7", "Posterize": "#77c9a8", "SoftClip": "#83cbb7", "HSVTool": "#8fcfbd",
          # Colorspace (Color menu) and Convolve (Filter menu) shipped in SPECS and the node
          # catalog with no COLORS entry, which crashed the panel the moment either was selected.
          "Colorspace": "#6fc9b0", "Convolve": "#82c9d8",
          "AddMix": "#a081d6", "Blend": "#b399de", "CopyRectangle": "#ab8fdb", "CopyBBox": "#9e82cf",
          # Position/BlackOutside/AdjustBBox are Transform-menu window utilities, Transform's family.
          "Position": "#7fa7ea", "BlackOutside": "#6c97dc", "AdjustBBox": "#6390d8",
          # TVIScale (2D parity plan 15, step F2) is a Transform-menu scaler, Transform's family.
          "TVIScale": "#84a3e6",
          "Exposure": "#83cbb7", "HueCorrect": "#77c9a8", "ColorLookup": "#76cbb4", "ColorMatrix": "#8fcfbd", "Log2Lin": "#82cbb7", "PLogLin": "#76c2ab", "CrossTalk": "#8bc6a9", "Toe": "#90d0ad", "Expression": "#73bfa4", "Histogram": "#77c9a8", "HistEQ": "#77c9a8", "MinColor": "#77c9a8", "Sampler": "#77c9a8", "MatchGrade": "#77c9a8", "Mirror": "#7ba3e8",
          # Dissolve/Keymix/Copy/ChannelMerge are the other Merge-toolbar two-input nodes.
          "Dissolve": "#c7a8e6", "Keymix": "#b591de", "Copy": "#a889d9", "ChannelMerge": "#9d80d4",
          # Ramp/Radial/Rectangle/Noise/Text are Draw-menu siblings of Roto, so they stay in the
          # same warm orange family, each a step around Roto's own hue.
          "Ramp": "#e29b7f", "Radial": "#e2a37f", "Rectangle": "#e2ab7f", "Noise": "#e2b37f",
          "Text": "#e28f7f", "Grid": "#e2b97f",
          # Metadata-menu nodes and BurnIn (step S3): a teal family, BurnIn one step warmer.
          "ViewMetaData": "#6fc3c9", "ModifyMetaData": "#66bcc6", "CopyMetaData": "#5eb5c3",
          "CompareMetaData": "#56aec0", "AddTimeCode": "#4ea7bd", "BurnIn": "#6fc9b4",
          "NoOp": "#b8a6db", "Profile": "#b8a6db", "PostageStamp": "#b8a6db", "Backdrop": "#6a7f99",
          # Precomp/Assert (2D parity plan 15, step F2) are the Other-menu's script-management and
          # QA nodes, one step around NoOp/Profile's own grey-violet hue.
          "Precomp": "#a6a6db", "Assert": "#c9a6db",
          "Group": "#7a8fa8", "Input": "#8ba0b8", "Output": "#8ba0b8",
          # Keyer/HueKeyer/Difference are the lane's group (c3) Keyer-menu nodes, a yellow-green
          # family distinct from every other node group.
          "Keyer": "#c3cf6e", "HueKeyer": "#b6cf6e", "Difference": "#a9cf6e",
          # ChromaKeyer/IBKColor/IBKGizmo (step K1) continue the Keyer-menu yellow-green run.
          "ChromaKeyer": "#9ccf6e", "IBKColor": "#8fcf6e", "IBKGizmo": "#82cf6e",
          # ScreenKeyer (step K2) ends the Keyer-menu run.
          "ScreenKeyer": "#75cf6e",
          # Cryptomatte (step K3) closes the Keyer-menu run; Encryptomatte (step D2 finish) is its writer twin.
          "Cryptomatte": "#68cf70", "Encryptomatte": "#5bcf72",
          # TimeOffset/FrameHold/Retime are the lane's group (c4) Time-menu nodes, a violet family
          # distinct from every other node group.
          "TimeOffset": "#c48fe0", "FrameHold": "#b881e0", "Retime": "#ac74e0",
          "TimeBlur": "#a36de0", "TimeEcho": "#9966d8", "TimeDissolve": "#bd95e3",
          "TimeWarp": "#c07de0",
          "VectorGenerator": "#8063d4", "Kronos": "#8568d7", "OFlow": "#8568d7", "VectorToMotion": "#62a6c2", "MotionBlur": "#906fdc", "MotionBlur2D": "#a878dc", "MotionBlur3D": "#936bcf", "CurveTool": "#b079dc",
          "SmartVector": "#8063d4",
          # TimeClip/FrameRange/AppendClip (step 4b) continue the same violet Time-menu family.
          "TimeClip": "#a067e0", "FrameRange": "#945ae0", "AppendClip": "#884de0",
          # Reformat/CornerPin are the lane's step 2c5 Transform-menu siblings of Transform/Crop/
          # Mirror, so they stay in that same blue family, each a step around it.
          "Reformat": "#6390e0", "CornerPin": "#5683e0", "VectorDistort": "#4c76d8", "VectorCornerPin": "#4269d0",
          # STMap/IDistort are Transform-menu warps; VectorBlur is the Filter-menu motion blur.
          "STMap": "#4f78d8", "IDistort": "#4970d0", "SplineWarp": "#467bd5", "GridWarp": "#5288e0", "GridWarpTracker": "#5886dc", "VectorBlur": "#62a6c2", "Tile": "#7ba3e8", "ContactSheet": "#6f9be0",
          # A Viewer is deliberately the most muted card in the graph and a Write the most
          # emphatic: one is a place you look from, the other is the only node that writes to disk.
          "Viewer": "#a2a2ac", "Write": "#e06f6f",
          "Card3D": "#e0a96d", "Cube3D": "#d98f63", "Sphere3D": "#d9a263", "Cylinder3D": "#d9b06d", "ReadGeo3D": "#d97f63", "ReadSplat3D": "#d97f63", "ReadAlembic3D": "#d97f63", "ReadAlembicCamera3D": "#b69be6", "ReadUSD3D": "#d97f63", "ReadUSDCamera3D": "#b69be6", "ReadGLTF3D": "#d97f63",
          "Light3D": "#e8d98d", "Camera3D": "#8db8e8",
          "WriteGeo3D": "#e06f6f", "WriteSplat3D": "#e06f6f", "WriteVDB3D": "#e06f6f", "Project3D": "#b69be6", "Scene3D": "#a99be6", "Render3D": "#78c9c0",
          # Relight recombines a Render3D bundle, so it stays a close cousin of Render3D's teal
          # (also filling a `theme.COLORS` gap that crashed `NodeItem` for this kind).
          "Relight": "#6fbdb4",
          "LightMixer": "#74c3ba",
          # Axis3D is a pure parenting transform, so it reads as a paler cousin of Scene3D's
          # hierarchy purple; TransformGeo3D bakes vertices, so it stays in the geometry orange family.
          "Axis3D": "#a08ee3", "TransformGeo3D": "#d9895f",
          "PointsTo3D": "#a08ee3", "Reconcile3D": "#a08ee3",
          # MergeGeo3D, Normals3D and DisplaceGeo3D edit geometry, so they stay in the same orange family.
          "MergeGeo3D": "#d9946a", "Normals3D": "#d98f78", "DisplaceGeo3D": "#d9a56f",
          "Shrinkwrap3D": "#d9b085",
          # Simulation nodes get their own teal-green so a particle stream reads apart from geometry.
          "ParticleEmitter3D": "#8fd9a8", "ParticleCache3D": "#78c9a0",
          "ParticleGravity3D": "#9fe0b0", "ParticleDrag3D": "#a3dfa0", "ParticleWind3D": "#8fe0c0",
          "ParticleTurbulence3D": "#85d6b4",
          "ParticleBounce3D": "#9be0a0", "ParticleCollide3D": "#8ad4ac",
          "ParticleRender3D": "#7fd0b8", "Instance3D": "#79d9a4",
          "Plume3D": "#8fc7e8", "ReadVDB3D": "#7fb6d9",
          "FluidSource3D": "#e8b07f", "FluidForce3D": "#e0a070", "FluidCollide3D": "#d99a7f",
          "FluidSolver3D": "#e89a5f", "FluidCache3D": "#d98c5c", "FluidUpres3D": "#d98772",
          "FluidLiquidSolver3D": "#e8a06a", "FluidSurface3D": "#e6b07f", "FluidFoam3D": "#e0b98a",
          "FluidWhitewater3D": "#a9d8d0",
          "RigidBody3D": "#b7a1e8", "RigidSolver3D": "#9a83d1"}

# ---- Design tokens of the new look (approved 10/8; values from the main-window mockup's :root) ----
# Everything the stylesheet and the chrome widgets paint with comes from here, so a colour, a
# radius or a type size changes in exactly one place. Widget code reads TOKENS / FAMILY_COLORS /
# RADIUS / TYPE and never carries a colour literal of its own.
TOKENS = {
    "bg0": "#0b0d10", "bg1": "#111418", "bg2": "#161a1f", "bg3": "#1c2127",     # surfaces, deepest first
    "line": "#232930", "line2": "#2c333b",                                      # 1 px hairlines
    "tx0": "#e9edf1", "tx1": "#b4bcc6", "tx2": "#7d8793", "tx3": "#56606b",     # text, strongest first
    "acc": "#5ee0b5", "acc2": "#3fb894", "acc_ink": "#05231a",                  # accent, deeper accent, text on it
    "accbg": "rgba(94, 224, 181, 0.10)",                                        # accent wash (hover, selection)
    "mark_b": "#56c2ff", "mark_c": "#9aa8ff",                                   # logo rim colours
    "bar_top": "#12161a", "bar_bottom": "#0f1215",                              # top bar gradient
    "danger": "#ff6b6b", "warn": "#f39a4c", "shadow": "rgba(0, 0, 0, 0.4)",
    "glass": "rgba(17, 20, 24, 0.85)",                                          # panels floating over the graph
    "acc_ring": "rgba(94, 224, 181, 0.35)",                                     # accent outline on a checked chip
    "hud": "rgba(17, 20, 24, 0.78)",                                            # the viewer's floating strip
    "ch_r": "#ff6b6b", "ch_g": "#7be07b", "ch_b": "#6cc4ff",                    # the R, G and B channel buttons
}

# The thirteen node families (UI-SPEC.md). No yellow: dark yellow reads brown and dirty.
FAMILY_COLORS = {
    "Image": "#a3acb7", "Draw": "#7cc8f2", "Time": "#ff8a65", "Channel": "#f2849e",
    "Color": "#34d399", "Filter": "#f39a4c", "Keyer": "#79d36b", "Merge": "#5b86f0",
    "Transform": "#a681f2", "3D": "#e8585f", "Particles": "#e676d6", "Fluids": "#3fd0d0",
    "Metadata": "#6c7682",
}

def family_color(kind):
    """The colour of a node type's family (a type's name, not a family name)."""
    from .nodecatalog import node_family
    return FAMILY_COLORS[node_family(kind)]


# Corner radii in pixels. Panels and the larger controls sit between 9 and 12; small chips and
# keycaps are tighter, the status pill is a full capsule.
RADIUS = {"control": 9, "field": 8, "panel": 12, "popup": 12, "chip": 6, "kbd": 5, "pill": 14, "flyout": 14}

# Type scale. Sizes are pixels at 100% scaling and every widget sizes itself from font metrics,
# never from a pixel width that only fits one platform's fonts.
TYPE = {
    "family": "'Noto Sans', 'Segoe UI', 'Inter', system-ui, sans-serif",
    "mono": "'Noto Sans Mono', 'Cascadia Mono', 'DejaVu Sans Mono', monospace",
    "base": 13, "small": 12, "caption": 11, "title": 16, "value": 12,
}

# Padding, in pixels, that every control shares.
SPACING = {"hair": 1, "xs": 3, "sm": 6, "md": 10, "lg": 14}

# Interface themes. Only surface and accent values vary -- the node-family colours above stay
# fixed, because they carry meaning an artist learns once and should not have to relearn per
# theme. A theme is a *user* preference, stored per machine (see app.Preferences) rather than in
# the document: a comp handed to another artist must not drag this one's colour scheme with it.
THEMES = {
    "NodeBased": {"window": TOKENS["bg1"], "panel": TOKENS["bg1"], "title": TOKENS["bg2"],
                  "field": TOKENS["bg0"], "border": TOKENS["line"], "button": TOKENS["bg2"],
                  "button_border": TOKENS["line2"], "hover": TOKENS["bg3"], "text": TOKENS["tx0"],
                  "muted": TOKENS["tx2"], "accent": TOKENS["acc"], "grid": "#171b20",
                  "status": TOKENS["bg1"], "raised": TOKENS["bg3"], "faint": TOKENS["tx3"],
                  "accent_ink": TOKENS["acc_ink"]},
    "Charcoal": {"window": "#242426", "panel": "#29292c", "title": "#2e2e31", "field": "#1b1b1d",
                 "border": "#414146", "button": "#343438", "button_border": "#49494f",
                 "hover": "#414146", "text": "#e4e4e7", "muted": "#a1a1aa",
                 "accent": "#83cbb7", "grid": "#313135", "status": "#1c1c1e"},
    "Graphite": {"window": "#1a1a1c", "panel": "#202023", "title": "#26262a", "field": "#121214",
                 "border": "#37373d", "button": "#2a2a2e", "button_border": "#3d3d44",
                 "hover": "#36363c", "text": "#dedee2", "muted": "#94949d",
                 "accent": "#7fc4d9", "grid": "#28282c", "status": "#131315"},
    "Slate": {"window": "#22262c", "panel": "#272c34", "title": "#2c323b", "field": "#171a20",
              "border": "#3b434f", "button": "#2f3641", "button_border": "#434c59",
              "hover": "#3b4451", "text": "#e1e6ee", "muted": "#9aa3b2",
              "accent": "#89aff0", "grid": "#2e343d", "status": "#191d23"},
    "Ash": {"window": "#32323a", "panel": "#393942", "title": "#3f3f49", "field": "#25252b",
            "border": "#53535f", "button": "#42424d", "button_border": "#5a5a68",
            "hover": "#4e4e5b", "text": "#ececef", "muted": "#b0b0bb",
            "accent": "#e0b06a", "grid": "#40404a", "status": "#28282e"},
}
DEFAULT_THEME = "NodeBased"

# Accent choices layered over any theme. "Theme default" (None) keeps the theme's own accent.
ACCENTS = {"Theme default": None, "Teal": "#83cbb7", "Sky": "#7fc4d9", "Blue": "#89aff0",
           "Violet": "#b59cf0", "Pink": "#e592c0", "Red": "#e67c7c", "Orange": "#e89a5b",
           "Amber": "#e0b06a", "Lime": "#a9cf6e"}


def valid_accent(value):
    """A #rrggbb string, or None. Anything else (a stale or hand-edited preference) is None."""
    if isinstance(value, str) and len(value) == 7 and value[0] == "#":
        try:
            int(value[1:], 16)
            return value.lower()
        except ValueError:
            return None
    return None


def theme_colors(theme=DEFAULT_THEME, accent=None):
    colors = dict(THEMES.get(theme) or THEMES[DEFAULT_THEME])
    # Roles the older themes predate: a raised surface, the faintest text, and the ink that sits
    # on an accent-filled button.
    colors.setdefault("raised", colors["hover"])
    colors.setdefault("faint", colors["muted"])
    colors.setdefault("accent_ink", TOKENS["acc_ink"])
    if valid_accent(accent):
        colors["accent"] = valid_accent(accent)
    return colors


def _rgba(color, alpha):
    """"#rrggbb" as a stylesheet `rgba(...)` with `alpha` (0..1)."""
    value = int(color.lstrip("#"), 16)
    return f"rgba({value >> 16}, {(value >> 8) & 255}, {value & 255}, {alpha})"


def build_style(theme=DEFAULT_THEME, accent=None):
    """The application stylesheet for one named theme and optional accent override. Unknown names
    fall back to the default rather than raising: a preferences file from a newer build must not
    stop the app opening. This is the one stylesheet for the whole application; widgets name
    themselves with an objectName here and carry no colours of their own."""
    c = theme_colors(theme, accent)
    r, t, sp = RADIUS, TYPE, SPACING
    return f"""
QMainWindow, QWidget {{ background: {c['window']}; color: {c['text']}; font: {t['base']}px {t['family']}; }}
QMenuBar, QMenu, QToolBar {{ background: {c['panel']}; border: 0; }}
QMenu {{ border: 1px solid {c['button_border']}; border-radius: {r['popup']}px; padding: {sp['xs'] + 1}px; }}
QMenu::item {{ padding: 6px 24px 6px 20px; border-radius: {r['chip'] + 1}px; }}
QMenu::item:selected {{ background: {c['hover']}; }}
QMenu::separator {{ height: 1px; background: {c['border']}; margin: 4px 8px; }}
QToolTip {{ background: {c['title']}; color: {c['text']}; border: 1px solid {c['button_border']}; border-radius: {r['chip']}px; padding: 4px 8px; }}
/* Dock title bars (new look, step 5): slim and quiet like the mockup's panel labels. The docks still
   float, close and rearrange exactly as before; only the bar's size and colour change. */
QDockWidget {{ font-size: {t['caption']}px; font-weight: 600; color: {TOKENS['tx3']}; }}
QDockWidget::title {{ background: {c['panel']}; padding: {sp['xs'] + 1}px {sp['md']}px; border-bottom: 1px solid {c['border']}; text-align: left; }}
QDockWidget::close-button, QDockWidget::float-button {{ background: transparent; border: 0; border-radius: {r['chip'] - 2}px; padding: 0px; }}
QDockWidget::close-button:hover, QDockWidget::float-button:hover {{ background: {c['hover']}; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ background: {c['field']}; border: 1px solid {c['border']}; border-radius: {r['field']}px; padding: 5px {sp['sm'] + 2}px; selection-background-color: {c['accent']}; selection-color: {c['accent_ink']}; }}
QSpinBox, QDoubleSpinBox {{ font-family: {t['mono']}; font-size: {t['value']}px; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border: 1px solid {c['accent']}; }}
QPushButton {{ background: {c['button']}; border: 1px solid {c['button_border']}; border-radius: {r['control']}px; padding: 6px {sp['lg']}px; }}
QDoubleSpinBox#knob-value {{ background: {c['field']}; border-radius: {r['field']}px; padding: 3px 6px; }}
QSlider#knob-slider::groove:horizontal {{ height: 6px; background: {c['border']}; border-radius: 3px; }}
QSlider#knob-slider::sub-page:horizontal {{ background: {c['accent']}; border-radius: 3px; }}
QSlider#knob-slider::handle:horizontal {{ width: 14px; margin: -4px 0; border-radius: 7px; background: {c['text']}; }}
QPushButton:hover {{ background: {c['hover']}; border-color: {c['faint']}; }}
QPushButton:pressed {{ background: {c['field']}; }}
QPushButton:disabled {{ color: {c['muted']}; }}
QPushButton#update {{ background: {c['accent']}; color: {c['accent_ink']}; border: 1px solid {c['accent']}; font-weight: 600; font-size: {t['base']}px; }}
QPushButton#update:hover {{ background: {c['accent']}; border-color: {c['text']}; }}
QPushButton#update:pressed {{ background: {c['panel']}; color: {c['accent']}; }}
QPushButton#update:disabled {{ background: {c['raised']}; color: {c['muted']}; border-color: {c['border']}; }}
QToolButton {{ padding: 7px; border-radius: {r['field']}px; }}
QToolButton:hover {{ background: {c['hover']}; }}
QToolButton#viewer-mode-2d:checked, QToolButton#viewer-mode-3d:checked {{ background: {c['title']}; color: {c['accent']}; border: 1px solid {c['accent']}; border-radius: {r['chip'] - 2}px; }}
QSplitter::handle {{ background: {c['border']}; height: {sp['hair'] + 3}px; width: {sp['hair'] + 3}px; }}
QStatusBar {{ background: {c['status']}; color: {c['muted']}; border-top: 1px solid {c['border']}; }}
QStatusBar QLabel {{ color: {c['muted']}; font-size: {t['caption']}px; padding: 0 9px; }}
QLabel#muted {{ color: {c['muted']}; }}
QLabel#brand {{ color: {c['accent']}; font-size: {t['title']}px; font-weight: 700; padding: 6px; }}
QCheckBox::indicator {{ width: 13px; height: 13px; background: {c['field']}; border: 1px solid {c['button_border']}; border-radius: 3px; }}
QCheckBox::indicator:checked {{ background: {c['accent']}; border: 1px solid {c['accent']}; border-radius: 3px; }}
QTabBar::tab {{ background: {c['panel']}; padding: 6px {sp['lg']}px; border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {c['accent']}; border-bottom: 2px solid {c['accent']}; }}

/* ---- node graph corner: wire mode toggle, zoom control ---- */
QFrame#graph-panel {{ background: {TOKENS['glass']}; border: 1px solid {TOKENS['line2']}; border-radius: {r['control']}px; }}
QFrame#graph-panel QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: {r['chip']}px; padding: 3px {sp['md'] - 1}px; color: {TOKENS['tx2']}; font-size: {t['small']}px; }}
QFrame#graph-panel QToolButton:hover {{ color: {TOKENS['tx0']}; }}
QFrame#graph-panel QToolButton:checked {{ background: {TOKENS['bg3']}; color: {TOKENS['tx0']}; border: 1px solid {TOKENS['acc_ring']}; }}

/* ---- top bar: logo, project, workspace tabs, search, GPU pill, menu, update ---- */
QToolBar#workspace-toolbar {{ background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 {TOKENS['bar_top']}, stop: 1 {TOKENS['bar_bottom']}); border: 0; border-bottom: 1px solid {c['border']}; padding: 0; spacing: 0; }}
QWidget#topbar, QWidget#topbar QLabel, QWidget#topbar-brand {{ background: transparent; }}
QLabel#topbar-project {{ color: {TOKENS['tx1']}; font-size: {t['base']}px; }}
QLabel#topbar-project-name {{ color: {c['text']}; font-weight: 600; }}
QFrame#segmented {{ background: {c['title']}; border: 1px solid {c['border']}; border-radius: {r['control']}px; }}
QPushButton#workspace-tab {{ background: transparent; border: 0; border-radius: {r['chip']}px; color: {c['muted']}; padding: 4px {sp['lg']}px; font-size: {t['base']}px; }}
QPushButton#workspace-tab:hover {{ color: {c['text']}; background: transparent; }}
QPushButton#workspace-tab:checked {{ background: {c['raised']}; color: {c['text']}; }}
QPushButton#topbar-search {{ background: {c['title']}; border: 1px solid {c['border']}; border-radius: {r['control']}px; padding: 0; }}
QPushButton#topbar-search:hover {{ border-color: {c['button_border']}; background: {c['title']}; }}
QLabel#topbar-search-text {{ color: {c['faint']}; }}
QPushButton#topbar-search:hover QLabel#topbar-search-text {{ color: {c['muted']}; }}
QLabel#topbar-kbd {{ color: {c['muted']}; font-size: {t['caption']}px; background: {c['panel']}; border: 1px solid {c['button_border']}; border-radius: {r['kbd']}px; padding: 0 {sp['sm'] - 1}px; }}
QFrame#gpu-pill {{ background: {c['title']}; border: 1px solid {c['border']}; border-radius: {r['pill']}px; }}
QLabel#gpu-pill-text {{ color: {TOKENS['tx1']}; font-size: {t['small']}px; }}
QToolButton#topbar-menu {{ color: {c['muted']}; border: 0; border-radius: {r['field']}px; padding: 5px; }}
QToolButton#topbar-menu::menu-indicator {{ image: none; width: 0px; }}
QToolButton#topbar-menu:hover, QToolButton#topbar-menu:checked {{ background: {c['hover']}; color: {c['text']}; }}

/* ---- the viewer strip and its corner readouts (new look, step 5) ---- */
QFrame#viewer-strip {{ background: {TOKENS['hud']}; border: 1px solid {TOKENS['line2']}; border-radius: {r['panel']}px; }}
QFrame#viewer-strip QWidget {{ background: transparent; }}
QFrame#viewer-strip QToolButton {{ background: transparent; border: 0; border-radius: {r['chip'] + 1}px; padding: 4px {sp['md'] - 1}px; color: {TOKENS['tx1']}; font-size: {t['small']}px; }}
QFrame#viewer-strip QToolButton:hover {{ background: {TOKENS['bg2']}; color: {TOKENS['tx0']}; }}
QFrame#viewer-strip QToolButton:checked {{ background: {TOKENS['bg3']}; color: {TOKENS['tx0']}; }}
QFrame#viewer-strip QToolButton::menu-indicator {{ image: none; width: 0px; }}
QFrame#viewer-strip QToolButton#strip-channel {{ padding: 3px {sp['sm'] - 1}px; font-size: {t['caption']}px; font-weight: 700; }}
QFrame#viewer-strip QToolButton#strip-channel[channel="R"] {{ color: {TOKENS['ch_r']}; }}
QFrame#viewer-strip QToolButton#strip-channel[channel="G"] {{ color: {TOKENS['ch_g']}; }}
QFrame#viewer-strip QToolButton#strip-channel[channel="B"] {{ color: {TOKENS['ch_b']}; }}
QFrame#viewer-strip QToolButton#strip-channel[channel="A"] {{ color: {TOKENS['tx2']}; }}
QFrame#viewer-strip QToolButton#strip-channel[channel="RGB"]:checked {{ color: {TOKENS['tx0']}; }}
QFrame#viewer-strip QToolButton#viewer-mode-2d:checked, QFrame#viewer-strip QToolButton#viewer-mode-3d:checked {{ background: {TOKENS['bg3']}; color: {TOKENS['tx0']}; border: 0; border-radius: {r['chip'] + 1}px; }}
QFrame#viewer-strip QFrame#strip-separator {{ background: {TOKENS['line2']}; border: 0; }}
QFrame#viewer-strip QLabel {{ background: transparent; color: {TOKENS['tx2']}; font-size: {t['small']}px; }}
QFrame#viewer-strip QDoubleSpinBox {{ background: transparent; border: 1px solid transparent; border-radius: {r['chip'] + 1}px; padding: 2px {sp['xs']}px; color: {TOKENS['tx1']}; font-family: {t['mono']}; font-size: {t['small']}px; }}
QFrame#viewer-strip QDoubleSpinBox:hover {{ background: {TOKENS['bg2']}; }}
QFrame#viewer-strip QDoubleSpinBox:focus {{ background: {TOKENS['bg0']}; border: 1px solid {TOKENS['acc_ring']}; color: {TOKENS['tx0']}; }}
QWidget#viewer-corner, QWidget#pixel-readout {{ background: transparent; }}
QWidget#viewer-corner QLabel, QWidget#pixel-readout QLabel {{ background: transparent; color: {TOKENS['tx2']}; font-size: {t['caption'] + 0.5}px; }}
QLabel#corner-strong {{ color: {TOKENS['tx1']}; font-weight: 500; }}
QLabel#corner-buffer {{ color: #f4ce63; font-weight: 700; }}
QWidget#pixel-readout QLabel#corner-strong {{ font-family: {t['mono']}; }}
QLabel#corner-swatch {{ border: 1px solid {TOKENS['line2']}; border-radius: 3px; }}

/* ---- the slim time row (new look, step 5) ---- */
QWidget#time-row {{ background: {TOKENS['bg1']}; border-top: 1px solid {TOKENS['line']}; border-bottom: 1px solid {TOKENS['line']}; }}
QWidget#time-row QLabel {{ background: transparent; color: {TOKENS['tx3']}; font-size: {t['caption']}px; }}
QWidget#time-row QLabel#muted {{ color: {TOKENS['tx3']}; }}
QWidget#time-row QLabel#time-realtime {{ color: {TOKENS['acc']}; font-size: {t['small']}px; }}
QWidget#time-row QLabel#time-realtime[behind="true"] {{ color: {TOKENS['warn']}; }}
QWidget#transport {{ background: transparent; }}
QWidget#time-row QToolButton {{ background: transparent; border: 0; border-radius: {r['field']}px; padding: 6px; }}
QWidget#time-row QToolButton:hover {{ background: {TOKENS['bg3']}; }}
QWidget#time-row QToolButton#transport-play {{ background: {TOKENS['acc']}; }}
QWidget#time-row QToolButton#transport-play:hover {{ background: {TOKENS['acc2']}; }}
QWidget#time-row QSpinBox#frame-current {{ background: transparent; border: 1px solid transparent; color: {TOKENS['tx0']}; font-family: {t['mono']}; font-size: 15px; font-weight: 600; padding: 3px {sp['xs']}px; }}
QWidget#time-row QSpinBox#frame-current:hover {{ background: {TOKENS['bg2']}; }}
QWidget#time-row QSpinBox#frame-current:focus {{ background: {TOKENS['bg0']}; border: 1px solid {TOKENS['acc_ring']}; }}
QWidget#time-row QSpinBox#time-field, QWidget#time-row QDoubleSpinBox#time-field {{ background: transparent; border: 1px solid transparent; color: {TOKENS['tx1']}; padding: 3px {sp['xs']}px; }}
QWidget#time-row QSpinBox#time-field:hover, QWidget#time-row QDoubleSpinBox#time-field:hover {{ background: {TOKENS['bg2']}; }}
QWidget#time-row QSpinBox#time-field:focus, QWidget#time-row QDoubleSpinBox#time-field:focus {{ background: {TOKENS['bg0']}; border: 1px solid {TOKENS['acc_ring']}; }}
QWidget#time-row QComboBox#time-presets {{ background: transparent; border: 1px solid transparent; color: {TOKENS['tx2']}; padding: 3px {sp['xs'] + 2}px; }}
QWidget#time-row QComboBox#time-presets:hover {{ background: {TOKENS['bg2']}; }}

/* ---- the graph's "?" shortcut overlay (new look, step 5) ---- */
QFrame#shortcut-overlay {{ background: {_rgba(c['title'], 0.96)}; border: 1px solid {c['button_border']}; border-radius: {r['flyout']}px; }}
QFrame#shortcut-overlay QLabel {{ background: transparent; }}
QLabel#shortcut-title {{ color: {c['text']}; font-size: {t['base']}px; font-weight: 600; }}
QLabel#shortcut-hint {{ color: {c['faint']}; font-size: {t['caption']}px; }}
QLabel#shortcut-keys {{ color: {TOKENS['tx1']}; font-size: {t['caption']}px; background: {c['panel']}; border: 1px solid {c['button_border']}; border-radius: {r['kbd']}px; padding: 1px {sp['sm']}px; }}
QLabel#shortcut-action {{ color: {c['muted']}; font-size: {t['small']}px; }}

/* ---- the left column of node families and the floating node panel (see nodeshelf.py) ---- */
QToolBar#node-rail-toolbar {{ background: {c['panel']}; border: 0; padding: 0; spacing: 0; }}
QFrame#node-panel {{ background: {_rgba(c['title'], 0.92)}; border: 1px solid {c['button_border']}; border-radius: {r['flyout']}px; }}
QFrame#node-panel QLabel {{ background: transparent; }}
QLabel#node-panel-title {{ color: {c['text']}; font-size: {t['base']}px; font-weight: 600; }}
QLabel#node-panel-count {{ color: {c['faint']}; font-size: {t['base']}px; }}
QLineEdit#node-panel-filter {{ background: {c['field']}; border: 1px solid {c['border']}; border-radius: {r['field']}px; color: {c['text']}; padding: 0 {sp['md'] - 1}px; font-size: {t['small']}px; }}
QLineEdit#node-panel-filter:focus {{ border-color: {TOKENS['acc_ring']}; }}
QListWidget#node-panel-list {{ background: transparent; border: 0; outline: 0; font-size: {t['small']}px; }}
QLabel#node-panel-more {{ color: {c['faint']}; font-size: {t['caption']}px; }}
QLabel#node-panel-more:hover {{ color: {c['muted']}; }}
QLabel#node-panel-hint {{ color: {c['faint']}; font-size: {t['caption']}px; }}
QLabel#node-panel-kbd {{ color: {c['muted']}; font-size: {t['caption']}px; background: {c['panel']}; border: 1px solid {c['button_border']}; border-radius: {r['kbd']}px; padding: 0 {sp['sm'] - 1}px; }}
QScrollBar:vertical {{ background: {c['status']}; width: 10px; }}
QScrollBar::handle:vertical {{ background: {c['button_border']}; min-height: 25px; }}
/* Styling QScrollBar:vertical's background switches Qt to fully custom rendering: an
   add-line/sub-line left undefined still reserves its usual arrow-button box, but blank,
   drawing as a stray empty square at each end of the track (QA pass 1, finding 5,
   2026-09-30). Collapsing both to zero height removes the boxes instead of filling them
   with an arrow glyph nobody asked for. */
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; border: none; background: none; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}
QScrollBar:horizontal {{ background: {c['status']}; height: 10px; }}
QScrollBar::handle:horizontal {{ background: {c['button_border']}; min-width: 25px; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0px; border: none; background: none; }}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{ background: none; }}
"""


def grid_color(theme=DEFAULT_THEME):
    """Node-graph dot grid for one theme, so the graph background follows the interface."""
    return (THEMES.get(theme) or THEMES[DEFAULT_THEME])["grid"]


# Kept as a module-level name because tests and the packaged entry point both import it.
STYLE = build_style()
