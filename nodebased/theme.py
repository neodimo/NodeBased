"""Neutral charcoal surfaces; limited, legible accents identify node families."""
COLORS = {"Read": "#d9b879", "Checker": "#d9b879", "Constant": "#d9b879",
          "Grade": "#83cbb7", "ColorCorrect": "#6fc9b0", "Blur": "#79c7d9",
          "Transform": "#89aff0", "Crop": "#6f9be0", "Shuffle": "#a889d9",
          "ChannelShuffle": "#9d80d4", "Roto": "#e2937f", "Tracker": "#e0b06a",
          "Merge": "#bd9ee3", "Dot": "#b8a6db", "Switch": "#c6a5db",
          # Invert/Clamp/Multiply/Add/Gamma/Saturation split a single Grade/ColorCorrect knob out
          # onto its own node, so they stay in the same teal-green Color family.
          "Invert": "#6fbfa8", "Clamp": "#6fbfa8", "Multiply": "#7ecab3", "Add": "#7ecab3",
          "Gamma": "#83cbb7", "Saturation": "#6fc9b0",
          # Erode/Dilate/Median/Sharpen/Glow are Filter-menu siblings of Blur, so they stay in the
          # same sky-blue family, each a step around Blur's own hue; Mirror is a Transform-menu
          # node and stays in Transform's blue family instead.
          "Erode": "#6bb8cc", "Dilate": "#6bb8cc", "Median": "#72c2d4", "Sharpen": "#8fd0de",
          "Glow": "#a3d8e2", "Mirror": "#7ba3e8",
          # Dissolve/Keymix/Copy/ChannelMerge are the other Merge-toolbar two-input nodes.
          "Dissolve": "#c7a8e6", "Keymix": "#b591de", "Copy": "#a889d9", "ChannelMerge": "#9d80d4",
          # Ramp/Radial/Rectangle/Noise/Text are Draw-menu siblings of Roto, so they stay in the
          # same warm orange family, each a step around Roto's own hue.
          "Ramp": "#e29b7f", "Radial": "#e2a37f", "Rectangle": "#e2ab7f", "Noise": "#e2b37f",
          "Text": "#e28f7f",
          # Keyer/HueKeyer/Difference are the lane's group (c3) Keyer-menu nodes, a yellow-green
          # family distinct from every other node group.
          "Keyer": "#c3cf6e", "HueKeyer": "#b6cf6e", "Difference": "#a9cf6e",
          # TimeOffset/FrameHold/Retime are the lane's group (c4) Time-menu nodes, a violet family
          # distinct from every other node group.
          "TimeOffset": "#c48fe0", "FrameHold": "#b881e0", "Retime": "#ac74e0",
          # Reformat/CornerPin are the lane's step 2c5 Transform-menu siblings of Transform/Crop/
          # Mirror, so they stay in that same blue family, each a step around it.
          "Reformat": "#6390e0", "CornerPin": "#5683e0",
          # A Viewer is deliberately the most muted card in the graph and a Write the most
          # emphatic: one is a place you look from, the other is the only node that writes to disk.
          "Viewer": "#a2a2ac", "Write": "#e06f6f",
          "Card3D": "#e0a96d", "Cube3D": "#d98f63", "Sphere3D": "#d9a263", "ReadGeo3D": "#d97f63", "ReadSplat3D": "#d97f63", "ReadAlembic3D": "#d97f63", "ReadAlembicCamera3D": "#b69be6", "ReadUSD3D": "#d97f63", "ReadUSDCamera3D": "#b69be6", "ReadGLTF3D": "#d97f63",
          "Light3D": "#e8d98d", "Camera3D": "#8db8e8",
          "WriteGeo3D": "#e06f6f", "Project3D": "#b69be6", "Scene3D": "#a99be6", "Render3D": "#78c9c0",
          # Axis3D is a pure parenting transform, so it reads as a paler cousin of Scene3D's
          # hierarchy purple; TransformGeo3D bakes vertices, so it stays in the geometry orange family.
          "Axis3D": "#a08ee3", "TransformGeo3D": "#d9895f"}

# Interface themes. Only surface and accent values vary -- the node-family colours above stay
# fixed, because they carry meaning an artist learns once and should not have to relearn per
# theme. A theme is a *user* preference, stored per machine (see app.Preferences) rather than in
# the document: a comp handed to another artist must not drag this one's colour scheme with it.
THEMES = {
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
DEFAULT_THEME = "Charcoal"

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
    if valid_accent(accent):
        colors["accent"] = valid_accent(accent)
    return colors


def build_style(theme=DEFAULT_THEME, accent=None):
    """The application stylesheet for one named theme and optional accent override. Unknown names
    fall back to the default rather than raising: a preferences file from a newer build must not
    stop the app opening."""
    c = theme_colors(theme, accent)
    return f"""
QMainWindow, QWidget {{ background: {c['window']}; color: {c['text']}; font: 12px 'Inter', 'Segoe UI', sans-serif; }}
QMenuBar, QMenu, QToolBar {{ background: {c['panel']}; border: 0; }}
QMenu {{ border: 1px solid {c['border']}; padding: 4px; }}
QMenu::item {{ padding: 5px 24px 5px 20px; }}
QMenu::item:selected {{ background: {c['hover']}; }}
QMenu::separator {{ height: 1px; background: {c['border']}; margin: 4px 8px; }}
QDockWidget {{ font-weight: 600; }}
QDockWidget::title {{ background: {c['title']}; padding: 9px; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ background: {c['field']}; border: 1px solid {c['border']}; border-radius: 5px; padding: 5px; selection-background-color: {c['accent']}; selection-color: {c['field']}; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border: 1px solid {c['accent']}; }}
QPushButton {{ background: {c['button']}; border: 1px solid {c['button_border']}; border-radius: 5px; padding: 6px 12px; }}
QPushButton:hover {{ background: {c['hover']}; border-color: #777780; }}
QPushButton:pressed {{ background: {c['field']}; }}
QPushButton:disabled {{ color: {c['muted']}; }}
QPushButton#update {{ color: {c['accent']}; border-color: {c['border']}; background: {c['title']}; }}
QPushButton#update:hover {{ background: {c['hover']}; }}
QToolButton {{ padding: 7px; }}
QToolButton:hover {{ background: {c['hover']}; }}
QSplitter::handle {{ background: {c['border']}; height: 4px; width: 4px; }}
QStatusBar {{ background: {c['status']}; color: {c['muted']}; }}
QLabel#muted {{ color: {c['muted']}; }}
QLabel#brand {{ color: {c['accent']}; font-size: 16px; font-weight: 700; padding: 6px; }}
QCheckBox::indicator {{ width: 13px; height: 13px; background: {c['field']}; border: 1px solid {c['button_border']}; border-radius: 3px; }}
QCheckBox::indicator:checked {{ background: {c['accent']}; border: 1px solid {c['accent']}; border-radius: 3px; }}
QTabBar::tab {{ background: {c['panel']}; padding: 6px 14px; border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {c['accent']}; border-bottom: 2px solid {c['accent']}; }}
QScrollBar:vertical {{ background: {c['status']}; width: 10px; }}
QScrollBar::handle:vertical {{ background: {c['button_border']}; min-height: 25px; }}
"""


def grid_color(theme=DEFAULT_THEME):
    """Node-graph dot grid for one theme, so the graph background follows the interface."""
    return (THEMES.get(theme) or THEMES[DEFAULT_THEME])["grid"]


# Kept as a module-level name because tests and the packaged entry point both import it.
STYLE = build_style()
