"""Native Qt desktop workbench; UI writes only through Dispatcher commands."""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, CancelledError
import copy
import json
import math
import numpy as np
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import uuid

from PySide6.QtCore import Qt, QLineF, QPoint, QPointF, QRect, QRectF, QTimer, Signal, QObject, QEvent, QEventLoop, QSettings, QSize, QByteArray, QMimeData
from PySide6.QtGui import (QAction, QColor, QCursor, QImage, QPainter, QPainterPath, QPen, QPixmap,
                           QKeySequence, QPolygonF, QIcon, QOffscreenSurface, QFont, QFontMetrics,
                           QShortcut, QTextCursor, QTextFormat)
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsSimpleTextItem,
    QGraphicsPathItem, QGraphicsPixmapItem, QGraphicsItem, QDockWidget, QLabel, QComboBox, QDoubleSpinBox,
    QSpinBox, QLineEdit, QPushButton, QFormLayout, QFileDialog, QMessageBox, QToolBar,
    QInputDialog, QSplitter, QScrollArea, QDialog, QListWidget, QListWidgetItem, QStyle, QSlider,
    QCheckBox, QMenu, QSizePolicy, QProgressDialog, QProgressBar, QTabWidget, QPlainTextEdit, QFrame,
    QColorDialog, QAbstractSpinBox, QAbstractItemView, QTextEdit, QToolButton, QDialogButtonBox)

from . import __version__
from .updater import Updater
from .core import (Dispatcher, SPECS, LIMITS, TIME_LIMITS, demo_document, load_document,
                   MASK_MIX_KINDS, artifact_type, node_label, node_thumbnail,
                   DEFAULT_THUMBNAIL_TYPES, bypass_slot, GEOMETRY_TYPES)
from .nodecatalog import NODE_CATEGORIES, node_category, node_description, doc_for_kind, find_doc_row
from . import radialcommands
from . import presets as preset_model
from . import fluidshelf
from . import fluidknobpresets
from .knowledge import read_doc
from .color import viewer_displays
from .imaging import Evaluator, Cancelled, ZEBRA_HIGH, ZEBRA_LOW, to_qimage, write_png, curve_tool_metrics
from .dustbust import detect_specks, dustbust_items_for_specks
from .renderprogress import ThreadProgress, progress_text
from .playback import PlaybackQueue, DisplayCache

from .theme import (COLORS, STYLE, THEMES, DEFAULT_THEME, ACCENTS, build_style, grid_color,
                    valid_accent)
from .color import VIEWS
from . import gpudisplay
from .core import (CHOICES, COMPARE_MODES, VIEWER_GAIN_RANGE, VIEWER_GAMMA_RANGE, VIEWER_INPUT_COUNT,
                   VIEWER_MASK_MODES, VIEWER_MASKS, VIEWER_PROXY_TIERS, VIEWER_ROI_DEFAULT,
                   viewer_look, viewer_masks, viewer_proxy, viewer_roi, viewer_state)
from . import viewframe
from . import compare as compare_model
from .bundle import write_frame as write_bundle_frame
from . import metadata as metadata_module
from .media import (write_exr, raster_layer_arrays, group_directory, IMAGE_EXTENSIONS, is_sequence, sequence_path)
from .cachetier import DiskCache
from .decodepool import DecodeAheadPool
from .tileexec import TileExecutor
from .tiles import TileRegion
from .tiers import PROXY_TIERS, auto_playback_tier
from .artisttools import CacheInspectorPanel, SliceView
from . import cachecontext
from .timeline import TimelineBar, KEY_COLOR
from .animation import CURVE_INTERPOLATIONS, resolve_document
from .groups import scope_document
from . import shapes as shape_model
from . import handles2d
from . import tracker as tracker_model
from .agentpanel import AgentPanel
from .knobs import knob_layout
from . import materialpreview
from .curveeditor import CurveEditorDialog
from .viewport3d import Viewport3D
from .radialmenu import RadialMenu, DEAD_ZONE_RADIUS as RADIAL_DEAD_ZONE_DEFAULT
from .radialrules import commands_for, context_for_selection, SLOT_COUNT as RADIAL_SLOT_COUNT, \
                         builtin_ids_for_context

# Delivery rates an artist actually asks for, offered next to the free-form rate box. 24 leads
# because it is the document default; the rest are the rates a comp gets handed in practice.
FPS_PRESETS = (("24", 24.0), ("23.976", 24000.0 / 1001.0), ("25", 25.0), ("29.97", 30000.0 / 1001.0),
               ("30", 30.0), ("48", 48.0), ("50", 50.0), ("59.94", 60000.0 / 1001.0), ("60", 60.0))


# A Viewer's noodle is deliberately the quietest line in the graph: see Graph.rebuild.
VIEW_EDGE_COLOR = "#5f5f6b"


# The on-screen reference for every artist-facing keyboard shortcut. Menu shortcuts below use the
# File/Edit/Time rows directly; graph and viewer bindings remain owned by their key handlers.
SHORTCUT_SECTIONS = (
    ("File", (("Ctrl+I", "Read image"), ("Ctrl+O", "Open project"),
              ("Ctrl+S", "Save"), ("Ctrl+Shift+S", "Save as"),
              ("Ctrl+E", "Export image"))),
    ("Edit", (("Ctrl+Z", "Undo"), ("Ctrl+Shift+Z", "Redo"), ("S", "Settings"))),
    ("Time", (("Left", "previous frame"), ("Right", "next frame"),
               ("Home", "first frame"), ("End", "last frame"), ("Space", "play/stop"))),
    ("Node graph", (("Tab", "node search"), ("R/G/M/T/B/C/S/O/P/U/W", "create node (Read/Grade/Merge/Transform/Blur/ColorCorrect/Shuffle/Roto/Premult/Unpremult/Write)"),
                    ("Ctrl+F", "focus the NODES dock's search box"),
                    ("Down/Up in that box", "move through its results"),
                    ("Return in that box", "add the selected (or first) result"),
                    ("Right-click a node", "graph: What is this? · NODES dock: star it or What is this?"),
                    ("Period", "create Dot"), ("1", "view selected node (viewer input 1)"),
                    ("2-9", "connect selected node to viewer input 2-9 and show it"), ("D", "toggle bypass"),
                    ("F", "frame"), ("Delete/Backspace", "delete selected"),
                    ("Ctrl+A", "select all"), ("Ctrl+C/Ctrl+X/Ctrl+V", "copy/cut/paste"),
                    ("Alt+C", "duplicate"), ("Ctrl+G", "group the selected nodes"),
                    ("Ctrl+Shift+G", "ungroup the selected group"),
                    ("Double-click a group", "enter it (the Root > Group bar above the graph goes back)"),
                    ("Hold Q", "radial menu: flick to a slice and release to run it, or tap to keep it open and click"),
                    ("MMB / Alt+LMB", "pan"), ("Scroll / Alt+Scroll", "zoom"))),
    ("Viewer", (("R/G/B/A", "channel solo (press again for RGB)"), ("F/H", "fit"),
                ("Ctrl+= / Ctrl+-", "zoom"), ("Ctrl+1", "1:1 zoom"),
                ("J", "step back/stop"), ("K", "stop"), ("L", "play"),
                ("1-9", "show viewer input 1-9 (the A buffer; empty inputs are ignored)"),
                ("Alt+1-9", "set viewer input 1-9 as the B buffer (Alt+0 clears B)"),
                ("Shift+W", "reset the wipe to the centre, vertical"),
                ("Ctrl+P", "toggle the viewer proxy (off / the last proxy chosen)"),
                ("Drag ROI box / edges / Shift-drag", "region of interest: move / resize / draw a new one (ROI button on)"),
                ("Drag wipe centre / rotation handle", "wipe compare: move the split / turn it · Ctrl-click resets"),
                ("Drag box / ring / handles", "Transform: translate / rotate / scale · Ctrl-drag centre moves the pivot"),
                ("MMB / Alt+LMB", "pan"), ("Scroll / Alt+Scroll", "zoom"), ("Escape", "cancel roto/tracker edit"))),
    ("3D viewport", (("LMB drag", "orbit"), ("MMB drag", "pan"), ("Scroll", "dolly"),
                     ("F", "frame selection or scene"), ("C", "toggle camera view"),
                     ("W / E / R", "gizmo mode: translate / rotate / scale"),
                     ("Drag arrow / square", "translate mode: move along a world axis / in a world plane"),
                     ("Drag ring", "rotate mode: turn about that world axis"),
                     ("Drag axis cube / centre cube", "scale mode: scale along that axis / uniformly"),
                     ("Q", "toggle pivot mode (the gizmo moves the pivot, not the object)"),
                     ("Escape", "cancel a gizmo drag"))),
)


def _choose_paint_color(viewer):
    chosen = QColorDialog.getColor(QColor.fromRgbF(*viewer.paint_color), viewer, "RotoPaint colour")
    if chosen.isValid():
        viewer.paint_color = [chosen.redF(), chosen.greenF(), chosen.blueF(), chosen.alphaF()]


def _lifetime_for_mode(mode, current, frame):
    """Rebuild a paint item's lifetime dict for a new mode, keeping its existing frame numbers
    where they still apply so switching modes in the layer editor never invents new ones."""
    if mode == "range":
        first = int(current.get("first", frame))
        return {"mode": "range", "first": first, "last": max(first, int(current.get("last", first)))}
    if mode in ("single", "from_current"):
        return {"mode": mode, "first": int(current.get("first", frame))}
    return {"mode": "all"}


class PropertiesIconButton(QToolButton):
    """Compact, high-DPI vector icons for the properties-panel action cluster."""

    def __init__(self, icon_name, parent=None):
        super().__init__(parent)
        self.icon_name = icon_name
        self.setObjectName(icon_name)
        self.setFixedSize(28, 28)
        self.setCheckable(icon_name == "reference-node")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = QColor("#f2f2f5" if (self.isChecked() or self.underMouse()) else "#a8a8b2")
        painter.setPen(QPen(color, 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                            Qt.PenJoinStyle.RoundJoin))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        r = QRectF(5, 5, self.width() - 10, self.height() - 10)
        if self.icon_name == "view-node":
            painter.drawEllipse(QRectF(r.left(), r.center().y() - 5, r.width(), 10))
            painter.drawEllipse(QRectF(r.center().x() - 2.2, r.center().y() - 2.2, 4.4, 4.4))
        elif self.icon_name == "reference-node":
            painter.drawLine(8, 4, 8, 23)
            painter.drawPolyline([QPointF(9, 5), QPointF(20, 7), QPointF(17, 11),
                                  QPointF(20, 15), QPointF(9, 13)])
        elif self.icon_name == "help-node":
            painter.drawEllipse(r)
            painter.drawText(r, Qt.AlignmentFlag.AlignCenter, "?")
        elif self.icon_name == "revert-knobs":
            painter.drawArc(QRectF(6, 6, 16, 16), 35 * 16, 285 * 16)
            painter.drawLine(5, 9, 5, 15)
            painter.drawLine(5, 9, 11, 9)
        elif self.icon_name == "close-knobs":
            painter.drawLine(8, 8, 20, 20)
            painter.drawLine(20, 8, 8, 20)
        if self.hasFocus():
            painter.setPen(QPen(QColor("#65b9ff"), 1.0, Qt.PenStyle.DashLine))
            painter.drawRoundedRect(QRectF(1.5, 1.5, self.width() - 3, self.height() - 3), 3, 3)
        painter.end()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.click()
            event.accept()
            return
        super().keyPressEvent(event)


PANEL_NODE_HELP = {
    "Merge": "A over B · scene-linear, premultiplied. Inputs must have matching dimensions. An optional mask gates the merge: where mask.a is 0, the result is B.",
    "Crop": "Masks to a rectangle in fixed image bounds; the canvas is not resized.",
    "Blur": "Separable box blur with radius measured in pixels.",
    "ColorCorrect": "Lift, gamma, gain and saturation operate on unpremultiplied colour.",
    "Shuffle": "Remaps output channels from any input channel or the constants 0 and 1.",
    "Premult": "Multiplies RGB by alpha to premultiply a straight-alpha input.",
    "Unpremult": "Divides RGB by alpha. At alpha 0, RGB remains untouched without NaN or infinity.",
    "Dot": "Graph reroute and passthrough; pixel data stays unchanged.",
    "Switch": "Selects an input through which (0 or 1). It does not resample; pixel formats must match.",
    "ChannelShuffle": "Routes each output channel from A, B or a constant. A B channel requires a connected B input.",
    "Roto": "Animatable bezier and polygon shapes produce a premultiplied matte. View this node to drag existing points. Drawing and point edits commit through one validated set_shapes command.",
    "RotoPaint": "Drag in this node's viewer to add one undoable stroke. Connect input2 for reveal; clone samples the plate at the previous frame for DustBust.",
    "Tracker": "Pixel NCC tracking uses float scene-linear pixels. One validated set_tracks command commits after completion; failure or cancellation leaves the document unchanged.",
    "Viewer": "Shows the current view target. Its input follows the view target and appears as a faint tap, never a processing connection. Pixels pass through unchanged.",
    "WriteGeo3D": "OBJ exports world-space geometry, UVs and vertex normals. USD also exports colours. USD ranges use one time-sampled file; patterns write per frame. OBJ ranges need a padded pattern such as geo.%04d.obj. Lights, textures and projections are not exported. The scene passes through unchanged, including when disabled.",
    "GenerateLUT": "Samples the upstream colour graph on an identity lattice and writes the selected 3D LUT size.",
    "WriteSplat3D": "Writes a 3DGS PLY with transforms baked into the splats. Ranges need a padded pattern such as splats.%04d.ply. Existing files require Overwrite. Per-splat visibility and shadow knobs are not stored in PLY. The scene passes through unchanged, including when disabled.",
    "WriteVDB3D": "Writes the scene's one fluid Volume or one liquid surface to VDB. A Volume writes density, temperature, vel and flame; a liquid writes a narrow-band level set named surface. A scene with both is refused. Ranges need a padded pattern such as smoke.%04d.vdb. Existing files require Overwrite. The scene passes through unchanged, including when disabled.",
    "Write": "Writes at full resolution through the reference evaluator. Viewer proxy, exposure and channel controls are display-only. Pixels pass through unchanged, so a Write mid-branch is inert.",
}


class RulerSlider(QSlider):
    """Integer-backed slider with a compact float ruler beneath its groove."""

    def __init__(self, soft_min, soft_max, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.soft_min = float(soft_min)
        self.soft_max = float(soft_max)
        self.setRange(0, 1000)
        self.setFixedHeight(38)
        self.setToolTip(f"Soft range: {self.soft_min:g} to {self.soft_max:g}")

    def float_value(self):
        if self.soft_max == self.soft_min:
            return self.soft_min
        return self.soft_min + (self.value() / 1000.0) * (self.soft_max - self.soft_min)

    def set_float_value(self, value):
        if self.soft_max == self.soft_min:
            position = 0
        else:
            value = max(self.soft_min, min(self.soft_max, float(value)))
            position = round((value - self.soft_min) / (self.soft_max - self.soft_min) * 1000)
        self.setValue(position)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        painter.setPen(self.palette().color(self.foregroundRole()))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        left, right = 8, max(8, self.width() - 8)
        for fraction in (0.0, 0.5, 1.0):
            x = left + fraction * (right - left)
            painter.drawLine(QPointF(x, 24), QPointF(x, 28))
            label = f"{self.soft_min + fraction * (self.soft_max - self.soft_min):g}"
            bounds = painter.fontMetrics().boundingRect(label)
            painter.drawText(QRectF(x - bounds.width() / 2, 27, bounds.width(), 11),
                             Qt.AlignmentFlag.AlignCenter, label)


class FloatSliderControl(QWidget):
    """A float spin box paired with a soft-range ruler slider."""

    def __init__(self, hard_range, soft_range, value, parent=None):
        super().__init__(parent)
        self.spin = QDoubleSpinBox()
        self.spin.setRange(*hard_range)
        self.spin.setDecimals(3)
        self.spin.setSingleStep(0.1)
        self.spin.setKeyboardTracking(False)
        self.slider = RulerSlider(*(soft_range or hard_range))
        self.slider.set_float_value(value)
        self.spin.setValue(value)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.spin)
        layout.addWidget(self.slider, 1)
        self._syncing = False
        self.spin.valueChanged.connect(self._spin_changed)
        self.slider.valueChanged.connect(self._slider_changed)
        self.spin.customContextMenuRequested.connect(self.customContextMenuRequested)
        self.slider.customContextMenuRequested.connect(self.customContextMenuRequested)

    def _spin_changed(self, value):
        # The slider only has 1000 steps across its soft range, so mirroring its quantized
        # position back into the spin box on every slider move would round away whatever
        # precision the artist typed. The guard lets the slider follow the spin without the
        # spin ever following the slider's own rounding.
        if self._syncing:
            return
        self._syncing = True
        self.slider.set_float_value(value)
        self._syncing = False

    def _slider_changed(self, value):
        if self._syncing:
            return
        self._syncing = True
        self.spin.setValue(self.slider.float_value())
        self._syncing = False

    def value(self):
        return self.spin.value()

    def setValue(self, value):
        self.spin.setValue(value)
        self.slider.set_float_value(value)

    def setEnabled(self, enabled):
        super().setEnabled(enabled)
        self.spin.setEnabled(enabled)
        self.slider.setEnabled(enabled)

    def setContextMenuPolicy(self, policy):
        super().setContextMenuPolicy(policy)
        self.spin.setContextMenuPolicy(policy)
        self.slider.setContextMenuPolicy(policy)


class ClickableColorSwatch(QFrame):
    clicked = Signal()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


def resource_path(relative: str) -> Path:
    """Locate a source asset both from a checkout and a PyInstaller bundle."""
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return root / relative


class Preferences:
    """Per-machine interface preferences, stored outside the document.

    A theme belongs to the artist looking at the screen, not to the comp: a .nbcomp handed to
    someone else must not repaint their application. QSettings is used rather than a file in the
    project so the two can never be confused for one another.
    """
    THEME = "interface/theme"
    THUMBNAILS = "interface/node_thumbnails"
    ACCENT = "interface/accent"
    MAX_PANELS = "interface/max_properties_panels"
    DISPLAY_CACHE_FLOAT32 = "interface/display_cache_float32"
    WORKSPACE = "workspace"

    def __init__(self):
        self._store = QSettings("NodeBased", "NodeBased")

    def theme(self):
        name = self._store.value(self.THEME, DEFAULT_THEME)
        return name if name in THEMES else DEFAULT_THEME

    def set_theme(self, name):
        if name in THEMES:
            self._store.setValue(self.THEME, name)
            self._store.sync()

    def thumbnails(self):
        value = self._store.value(self.THUMBNAILS, True)
        # QSettings hands booleans back as strings from some backends (INI on Linux).
        return value not in (False, "false", "0", 0)

    def accent(self):
        return valid_accent(self._store.value(self.ACCENT, None))

    def set_accent(self, value):
        value = valid_accent(value)
        if value is None:
            self._store.remove(self.ACCENT)
        else:
            self._store.setValue(self.ACCENT, value)
        self._store.sync()

    def set_thumbnails(self, enabled):
        self._store.setValue(self.THUMBNAILS, bool(enabled))
        self._store.sync()

    def display_cache_float32(self):
        value = self._store.value(self.DISPLAY_CACHE_FLOAT32, False)
        return value not in (False, "false", "0", 0)

    def set_display_cache_float32(self, enabled):
        self._store.setValue(self.DISPLAY_CACHE_FLOAT32, bool(enabled))
        self._store.sync()

    def max_properties_panels(self):
        value = self._store.value(self.MAX_PANELS, 5)
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return 5

    def set_max_properties_panels(self, value):
        self._store.setValue(self.MAX_PANELS, max(1, int(value)))
        self._store.sync()

    # Bump when the window's dock/toolbar layout changes shape, so an older saved layout is
    # dropped for the default instead of restored into widgets it no longer describes.
    WORKSPACE_VERSION = 2

    def workspace(self):
        """The window layout saved at the last close, or None when there isn't a usable one."""
        try:
            version = int(self._store.value(self.WORKSPACE + "/version", 0))
        except (TypeError, ValueError):
            return None
        if version != self.WORKSPACE_VERSION:
            return None
        saved = {name: self._store.value(f"{self.WORKSPACE}/{name}")
                 for name in ("geometry", "state")}
        if not all(isinstance(value, QByteArray) and not value.isEmpty() for value in saved.values()):
            return None
        return saved

    def set_workspace(self, geometry, state):
        self._store.setValue(self.WORKSPACE + "/version", self.WORKSPACE_VERSION)
        self._store.setValue(self.WORKSPACE + "/geometry", geometry)
        self._store.setValue(self.WORKSPACE + "/state", state)
        self._store.sync()

    def clear_workspace(self):
        self._store.remove(self.WORKSPACE)
        self._store.sync()

    FAVOURITE_KINDS = "interface/node_favourites"
    RECENT_KINDS = "interface/node_recents"
    RECENT_KINDS_CAP = 10
    NODE_TOOLBAR_COMPACT = "interface/node_toolbar_compact"

    def _kind_list(self, key):
        try:
            kinds = json.loads(self._store.value(key, "[]"))
        except (TypeError, ValueError):
            return []
        return [kind for kind in kinds if isinstance(kind, str)]

    def favourite_kinds(self):
        """Starred node kinds, oldest-starred first, kept across sessions like the theme."""
        return self._kind_list(self.FAVOURITE_KINDS)

    def set_favourite(self, kind, favourite):
        """Star or unstar `kind`. Returns whether it ends up starred."""
        kinds = self.favourite_kinds()
        if favourite and kind not in kinds:
            kinds.append(kind)
        elif not favourite and kind in kinds:
            kinds.remove(kind)
        self._store.setValue(self.FAVOURITE_KINDS, json.dumps(kinds))
        self._store.sync()
        return kind in kinds

    def recent_kinds(self):
        """Node kinds added to the graph, most recent first, capped at RECENT_KINDS_CAP."""
        return self._kind_list(self.RECENT_KINDS)

    def add_recent_kind(self, kind):
        """Record `kind` as just added, moving it to the front of the recent list."""
        kinds = [k for k in self.recent_kinds() if k != kind]
        kinds.insert(0, kind)
        del kinds[self.RECENT_KINDS_CAP:]
        self._store.setValue(self.RECENT_KINDS, json.dumps(kinds))
        self._store.sync()

    def node_toolbar_compact(self):
        value = self._store.value(self.NODE_TOOLBAR_COMPACT, False)
        return value not in (False, "false", "0", 0)

    def set_node_toolbar_compact(self, enabled):
        self._store.setValue(self.NODE_TOOLBAR_COMPACT, bool(enabled))
        self._store.sync()

    # ---- radial menu: trigger key, dead zone, and local-usage learning (plan "Radial menu", ----
    # ---- DiMo 9/27, deliverable R3) --------------------------------------------------------------

    RADIAL_TRIGGER_KEY = "interface/radial_trigger_key"
    RADIAL_DEAD_ZONE = "interface/radial_dead_zone"
    RADIAL_LEARNING_ENABLED = "interface/radial_learning_enabled"
    RADIAL_USAGE = "interface/radial_usage"
    RADIAL_LEARNED = "interface/radial_learned_slots"
    RADIAL_PINS = "interface/radial_pinned_slots"
    RADIAL_LEARN_THRESHOLD = 3   # picks in one context+type bucket before a favourite is promoted

    def radial_trigger_key(self):
        try:
            return int(self._store.value(self.RADIAL_TRIGGER_KEY, int(Qt.Key.Key_Q)))
        except (TypeError, ValueError):
            return int(Qt.Key.Key_Q)

    def set_radial_trigger_key(self, key):
        self._store.setValue(self.RADIAL_TRIGGER_KEY, int(key))
        self._store.sync()

    def radial_dead_zone(self):
        try:
            value = float(self._store.value(self.RADIAL_DEAD_ZONE, RADIAL_DEAD_ZONE_DEFAULT))
        except (TypeError, ValueError):
            return RADIAL_DEAD_ZONE_DEFAULT
        return value if value > 0 else RADIAL_DEAD_ZONE_DEFAULT

    def set_radial_dead_zone(self, value):
        self._store.setValue(self.RADIAL_DEAD_ZONE, float(value))
        self._store.sync()

    def radial_learning_enabled(self):
        value = self._store.value(self.RADIAL_LEARNING_ENABLED, True)
        return value not in (False, "false", "0", 0)

    def set_radial_learning_enabled(self, enabled):
        self._store.setValue(self.RADIAL_LEARNING_ENABLED, bool(enabled))
        self._store.sync()

    def _radial_json(self, key):
        try:
            data = json.loads(self._store.value(key, "{}"))
        except (TypeError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _set_radial_json(self, key, value):
        self._store.setValue(key, json.dumps(value))
        self._store.sync()

    def radial_usage(self):
        """`{bucket: {command_id: pick_count}}`, never sent anywhere -- read by nothing but
        `record_radial_usage` below and the tests."""
        return self._radial_json(self.RADIAL_USAGE)

    def radial_learned(self):
        """`{bucket: {slot: command_id}}` -- the slots local-usage learning has promoted a
        favourite into. `radialrules.commands_for` overlays this before the ordinary rule table
        and user-command fill, then overlays `radial_pins` on top of that."""
        return self._radial_json(self.RADIAL_LEARNED)

    def radial_pins(self):
        """`{bucket: {slot: command_id}}` -- slices pinned by a right-click; never touched by
        `reset_radial_learning`."""
        return self._radial_json(self.RADIAL_PINS)

    def radial_bucket(self, context_kind, node_type):
        """Local-usage learning's key: the ring's context plus, when exactly one node is
        selected, its type -- DiMo's "context kind and selected node type". A multi-node
        selection (or none) only ever has the context to go on."""
        return f"{context_kind}|{node_type or ''}"

    def record_radial_usage(self, context_kind, node_type, command_id):
        """One more pick of `command_id` in this bucket. Once it reaches `RADIAL_LEARN_THRESHOLD`
        picks and does not already have a promoted slot, it claims this bucket's lowest-priority
        slot that is not already pinned or already promoted to something else -- and keeps that
        slot from then on, however usage shifts later, so flick muscle memory survives. A command
        that already has a fixed slot of its own whenever it applies (`builtin_ids_for_context`)
        is never promoted: there is nothing to win it that it does not already have."""
        bucket = self.radial_bucket(context_kind, node_type)
        usage = self.radial_usage()
        counts = usage.setdefault(bucket, {})
        counts[command_id] = counts.get(command_id, 0) + 1
        self._set_radial_json(self.RADIAL_USAGE, usage)
        if not self.radial_learning_enabled() or counts[command_id] < self.RADIAL_LEARN_THRESHOLD:
            return
        learned = self.radial_learned()
        bucket_learned = learned.setdefault(bucket, {})
        if command_id in bucket_learned.values() or command_id in builtin_ids_for_context(context_kind):
            return
        taken = set(bucket_learned) | set(self.radial_pins().get(bucket, {}))
        for slot in range(RADIAL_SLOT_COUNT - 1, -1, -1):   # lowest priority (highest index) first
            if str(slot) not in taken:
                bucket_learned[str(slot)] = command_id
                self._set_radial_json(self.RADIAL_LEARNED, learned)
                return

    def reset_radial_learning(self):
        """Preferences -> Radial settings -> "Reset learned slots": forgets every promotion and
        the usage counts behind them, putting every context back to its rule defaults. Pins are a
        deliberate right-click choice, not something learning did, so they are untouched."""
        self._store.remove(self.RADIAL_USAGE)
        self._store.remove(self.RADIAL_LEARNED)
        self._store.sync()

    def toggle_radial_pin(self, context_kind, node_type, slot, command_id):
        """Right-click a slice: pin `command_id` there, or unpin it if it is already the pin at
        that slot. Returns whether the slot ends up pinned."""
        bucket = self.radial_bucket(context_kind, node_type)
        pins = self.radial_pins()
        bucket_pins = pins.setdefault(bucket, {})
        slot_key = str(slot)
        if bucket_pins.get(slot_key) == command_id:
            del bucket_pins[slot_key]
            pinned = False
        else:
            bucket_pins[slot_key] = command_id
            pinned = True
        if not bucket_pins:
            del pins[bucket]
        self._set_radial_json(self.RADIAL_PINS, pins)
        return pinned


def apply_theme(name, accent=None):
    """Restyle the whole running application. Returns the theme actually applied."""
    name = name if name in THEMES else DEFAULT_THEME
    application = QApplication.instance()
    if application is not None:
        application.setStyleSheet(build_style(name, accent))
    return name


class ElidedLabel(QLabel):
    """A status label whose text can never widen the layout it sits in.

    A QLabel's size hint *is* its text, so a status line that grows mid-playback -- the viewer
    gains "ahead 4/8 · dropped 12" the moment the transport starts -- raises the window's minimum
    width and visibly resizes the panels around it. That is the "viewer enlarges randomly during
    playback" bug: nothing about the image changed, only the length of a string describing it.

    `text()` deliberately still returns the whole status. The agent bridge and the tests read it
    as the real status of the last render, so truncating the stored value would trade a layout bug
    for a lying one; only the painted copy is elided, with the full text on hover.
    """
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setToolTip(text)

    def setText(self, text):
        super().setText(text)
        self.setToolTip(text)

    def sizeHint(self):
        return QSize(0, super().sizeHint().height())

    def minimumSizeHint(self):
        return QSize(0, super().minimumSizeHint().height())

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(self.palette().color(self.foregroundRole()))
        elided = QFontMetrics(self.font()).elidedText(
            self.text(), Qt.TextElideMode.ElideRight, self.width())
        painter.drawText(self.rect(), int(self.alignment()) | int(Qt.AlignmentFlag.AlignVCenter),
                         elided)


# The default workspace: what a first launch gets, and what Workspace → Default workspace restores.
DEFAULT_WINDOW_SIZE = (1440, 920)
DEFAULT_PROPERTIES_WIDTH = 400

# Narrowest a single knob field may be squeezed to. Enough for a short number; anything longer
# stays readable by scrolling inside the field, which beats the field running off the dock.
FLUID_FIELD_MIN_WIDTH = 48


class FluidRoot(QWidget):
    """Top of a properties panel: its floor is zero, so the dock width (not font metrics) wins."""

    def minimumSizeHint(self):
        return QSize(0, super().minimumSizeHint().height())


def make_fluid(root):
    """Let a properties panel reflow to whatever width its dock has, instead of setting it.

    Every stock control's minimum size hint is its content: a spin box is as wide as the longest
    number its range allows, a note is as wide as its longest line, a button as wide as its text.
    Summed across a row, those hints put the panel's floor above the dock's width, and the scroll
    area then paints the panel wider than the dock and clips the right edge off every row. This
    relaxes those floors so the dock width wins: form labels wrap above their field when a row no
    longer fits beside it, notes word-wrap, and numeric fields share whatever width is left.
    """
    for form in root.findChildren(QFormLayout):
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    for field in root.findChildren(QAbstractSpinBox):
        field.setMinimumWidth(FLUID_FIELD_MIN_WIDTH)
        field.setSizePolicy(QSizePolicy.Policy.Preferred, field.sizePolicy().verticalPolicy())
    for combo in root.findChildren(QComboBox):
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(6)
    for button in root.findChildren(QPushButton):
        if button.minimumWidth() == 0 and button.maximumWidth() > 1000:
            button.setMinimumWidth(min(button.minimumSizeHint().width(), 2 * FLUID_FIELD_MIN_WIDTH))
            button.setSizePolicy(QSizePolicy.Policy.Preferred, button.sizePolicy().verticalPolicy())
            if not button.toolTip():
                button.setToolTip(button.text())
    for box in root.findChildren(QCheckBox):
        # A check box can't wrap its text, so its hint is the whole sentence. Let a narrow dock
        # clip the tail (the full text stays in the tooltip) rather than the whole panel's edge.
        if box.minimumWidth() == 0 and box.text():
            box.setMinimumWidth(min(box.minimumSizeHint().width(), 2 * FLUID_FIELD_MIN_WIDTH + 24))
            if not box.toolTip():
                box.setToolTip(box.text())
    for label in root.findChildren(QLabel):
        if not isinstance(label, ElidedLabel) and label.text():
            label.setWordWrap(True)
    for slider in root.findChildren(QSlider):
        slider.setMinimumWidth(FLUID_FIELD_MIN_WIDTH)
    return root


class PanZoomView(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        scene.setParent(self)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Image pixels are the data the artist is inspecting.  Keep display scaling nearest-
        # neighbour even if another painter hint or platform style changes the defaults.
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QColor("#19191b"))
        self.pan = None
        self.pan_button = None

    def wheelEvent(self, event):
        delta = event.angleDelta()
        # Qt hands an Alt+wheel to the horizontal axis on X11 and Windows, so read whichever
        # axis moved. The zoom is proportional to the distance scrolled: one mouse notch (120)
        # is one 1.15 step, and a touchpad's stream of small deltas zooms smoothly.
        steps = (delta.y() or delta.x()) / 120
        if steps:
            factor = 1.15 ** steps
            if 0.05 < self.transform().m11() * factor < 20:
                self.scale(factor, factor)
        event.accept()

    def starts_pan(self, event):
        """Middle-drag pans, and so does Nuke's Alt+left-drag."""
        return (event.button() == Qt.MouseButton.MiddleButton
                or (event.button() == Qt.MouseButton.LeftButton
                    and event.modifiers() == Qt.KeyboardModifier.AltModifier))

    def mousePressEvent(self, event):
        if self.pan is None and self.starts_pan(event):
            self.pan = event.position()
            self.pan_button = event.button()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.pan is not None:
            delta = event.position() - self.pan
            self.pan = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - round(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - round(delta.y()))
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        # The pan ends with the button that started it; letting go of Alt mid-drag does not.
        if self.pan is not None and event.button() == self.pan_button:
            self.pan = None
            self.pan_button = None
            self.unsetCursor()
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def fit(self):
        rect = self.scene().itemsBoundingRect()
        if not rect.isEmpty():
            self.fitInView(rect.adjusted(-24, -24, 24, 24), Qt.AspectRatioMode.KeepAspectRatio)


class PixelReadout(QWidget):
    """A fixed-size viewer overlay; changing pixel text must not affect the window layout."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(272, 28)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("QWidget { background: rgba(25, 25, 27, 220); border: 1px solid #5f5f6b; }")
        self.label = QLabel(self)
        self.label.setGeometry(8, 0, 214, 27)
        self.label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self.label.setStyleSheet("border: 0; color: #e8e8eb; background: transparent;")
        self.swatch = QLabel(self)
        self.swatch.setGeometry(226, 6, 16, 16)
        # Which buffer the numbers come from while an A/B compare is on; empty otherwise.
        self.buffer_tag = QLabel(self)
        self.buffer_tag.setGeometry(246, 0, 24, 27)
        self.buffer_tag.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.buffer_tag.setStyleSheet("border: 0; color: #f4ce63; background: transparent; font-weight: bold;")
        self.swatch.setStyleSheet("border: 1px solid #e8e8eb; background: transparent;")
        self.setToolTip("Pixel values are raw scene-linear floats; y=0 is the bottom row, Nuke-style.")

    def set_value(self, text, color, buffer=""):
        self.label.setText(text)
        self.buffer_tag.setText(buffer)
        self.swatch.setStyleSheet(
            f"border: 1px solid #e8e8eb; background: rgb({color.red()}, {color.green()}, {color.blue()});")


class ViewerInputStrip(QWidget):
    """Nuke-style viewer inputs: nine numbered buttons (bright = active A, plain = wired, dim = empty,
    amber outline = B), the B selector and the compare mode. A fixed overlay on the viewport, like the
    pixel readout, so changing its text never moves the layout."""
    STYLES = {"active": "background: #4c7ddc; color: white;", "wired": "background: #34343c; color: #e8e8eb;",
              "empty": "background: #222226; color: #6d6d78;"}

    def __init__(self, viewer):
        super().__init__(viewer.viewport())
        self.viewer = viewer
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("ViewerInputStrip { background: rgba(25, 25, 27, 220); border: 1px solid #5f5f6b; }"
                           "QComboBox { background: #2a2a30; color: #e8e8eb; border: 1px solid #5f5f6b; padding: 0 4px; }")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 3, 4, 3)
        layout.setSpacing(2)
        self.buttons = []
        self._states = ["empty"] * VIEWER_INPUT_COUNT
        for number in range(1, VIEWER_INPUT_COUNT + 1):
            button = QPushButton(str(number))
            button.setFixedSize(22, 22)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.clicked.connect(lambda _=False, n=number: self.viewer.show_input(n))
            layout.addWidget(button)
            self.buttons.append(button)
        self.b_combo = QComboBox()
        self.b_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.b_combo.addItem("B: none", None)
        for number in range(1, VIEWER_INPUT_COUNT + 1):
            self.b_combo.addItem(f"B: {number}", number)
        self.b_combo.setToolTip("The B buffer of the compare (Alt+1-9)")
        self.b_combo.activated.connect(lambda _index: self.viewer.set_b(self.b_combo.currentData()))
        self.mode_combo = QComboBox()
        self.mode_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.mode_combo.addItems(COMPARE_MODES)
        self.mode_combo.setToolTip("How A and B are compared")
        self.mode_combo.activated.connect(lambda _index: self.viewer.set_compare_mode(self.mode_combo.currentText()))
        layout.addSpacing(4)
        layout.addWidget(self.b_combo)
        layout.addWidget(self.mode_combo)
        self.adjustSize()
        self.setFixedSize(self.sizeHint())
        self.setToolTip("Viewer inputs: press 1-9 in the graph to connect the selected node, "
                        "1-9 in the viewer to switch")

    def states(self):
        """One word per button: active, wired or empty (the tests and the tooltips read this)."""
        return [words.split()[0] for words in self._states]

    def refresh(self, state, names):
        for index, button in enumerate(self.buttons):
            number = index + 1
            node = state["inputs"][index]
            words = "active" if number == state["active"] else "wired" if node else "empty"
            if number == state["b"]:
                words += " b"
            self._states[index] = words
            style = self.STYLES["active" if number == state["active"] else "wired" if node else "empty"]
            border = "2px solid #f4ce63" if number == state["b"] else "1px solid #44444c"
            button.setStyleSheet(f"QPushButton {{ {style} border: {border}; padding: 0; }}")
            button.setToolTip(f"Input {number}: {names.get(node, 'empty')}"
                              + ("  (A)" if number == state["active"] else "")
                              + ("  (B)" if number == state["b"] else ""))
        for combo, value in ((self.b_combo, state["b"]), (self.mode_combo, state["compare"])):
            combo.blockSignals(True)
            combo.setCurrentIndex(combo.findData(value) if combo is self.b_combo else combo.findText(value))
            combo.blockSignals(False)


class Viewer(PanZoomView):
    """Image viewer shortcuts are active only while the pointer/focus is in the viewer."""
    def __init__(self, window):
        self.window = window
        self.format_rect = None
        # Roto editing is a viewer overlay, not a scene item.  Keeping it out of the scene is
        # important: itemsBoundingRect() is the rendered display window and must not grow when
        # a point or a control path is painted (especially for EXR overscan and tiled previews).
        self.roto_key = None
        self.roto_drawing = False
        self.roto_draw_points = []
        self.roto_draw_cursor = None
        self.roto_drag = None
        self.warp_drawing = False
        self.warp_draw_side = None
        self.warp_draw_points = []
        self.warp_draw_cursor = None
        self.warp_drag = None
        self.warp_pending = {}
        self.paint_drag = None
        self.paint_tool = "paint"
        self.paint_size = 12.0
        self.paint_hardness = 0.8
        self.paint_brush_opacity = 1.0
        self.paint_spacing = 0.2
        self.paint_strength = 0.2
        self.paint_layer_opacity = 1.0
        self.paint_color = [1.0, 0.0, 0.0, 1.0]
        self.paint_blend = "over"
        self.paint_visible = True
        self.paint_lifetime = "single"
        self.paint_range_first = 1
        self.paint_range_last = 1
        self.paint_source_offset = [0.0, 0.0]
        self.paint_source_frame = "relative"
        self.absolute_source_frame = 1
        self.paint_follow_track = None
        self.paint_selected_item_index = None
        self.dustbust_preset = False
        self.transform_drag = None
        self.analysis_drag = None
        self.flare_drag = None
        self.tracker_picking = False
        self.tracker_drag = None
        self.crypto_picking = None   # the Cryptomatte node whose matte list a click adds to
        self.zdefocus_picking = None
        self.focus_picking = None   # the Camera3D node whose focus distance a click sets
        # A/B compare. The wipe geometry is display state only (not in the document): the split
        # is a fraction of the format rectangle plus an angle, see compare.py. `wipe_pixmap` is
        # the already-evaluated B picture, painted over A in drawForeground through a half-plane
        # clip, so dragging the wipe never asks the graph for anything.
        self.wipe = dict(compare_model.WIPE_DEFAULT)
        self.wipe_drag = None
        self.wipe_pixmap = None
        self.wipe_pos = QPointF(0, 0)
        # Region of interest: `roi_drag` is the drag in progress (the document only changes on
        # release); `backdrop` is the last full-canvas picture, kept so the outside of the ROI can
        # show it dimmed instead of going blank.
        self.roi_drag = None
        self.backdrop = None
        self.readout_buffer = ""
        self._initial_fit_pending = True
        super().__init__(QGraphicsScene())
        self.last_scale = 1
        self.last_render_region = None
        self.last_frame_size = None
        self.pixel_readout = PixelReadout(self.viewport())
        self.pixel_readout.hide()
        self.input_strip = ViewerInputStrip(self)
        self.input_strip.move(8, 8)
        self.input_strip.show()
        self._pixel_readout_active = False
        self._handling_mouse_move = False
        self.viewport().setMouseTracking(True)
        self.setMouseTracking(True)

    def _place_pixel_readout(self):
        margin = 8
        viewport_rect = self.viewport().rect()
        image_poly = self.mapFromScene(self.sceneRect())
        image_rect = image_poly.boundingRect().intersected(viewport_rect)
        strip = getattr(self, "input_strip", None)
        if strip is not None:
            overlaps = image_rect.intersects(strip.geometry())
            if strip.isVisible() == overlaps:
                strip.setVisible(not overlaps)
        overlays = [image_rect]
        if strip is not None and strip.isVisible():
            overlays.append(strip.geometry())
        width, height = self.pixel_readout.width(), self.pixel_readout.height()
        candidates = (
            QPoint(max(margin, viewport_rect.right() - width - margin),
                   max(margin, viewport_rect.bottom() - height - margin)),
            QPoint(max(margin, viewport_rect.right() - width - margin), margin),
            QPoint(margin, max(margin, viewport_rect.bottom() - height - margin)),
            QPoint(margin, margin),
        )
        for point in candidates:
            rect = QRect(point, self.pixel_readout.size()).intersected(viewport_rect)
            if rect.width() == width and rect.height() == height and not any(rect.intersects(item) for item in overlays):
                self.pixel_readout.move(point)
                self.pixel_readout.raise_()
                return True
        # A HUD over the picture corrupts the pixels the viewer is meant to show.  Small images
        # get a free corner; when a zoomed/full-frame picture fills the viewport, keep it clear.
        self.pixel_readout.hide()
        return False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_initial_image()
        self._place_pixel_readout()

    def paintEvent(self, event):
        # Zooming or resizing a dock can move the frame under the fixed HUD widgets.  Reflow
        # before each paint so pixel grabs never sample the controls themselves.
        self._place_pixel_readout()
        super().paintEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        self._fit_initial_image()
        self._place_pixel_readout()
        self.sync_inputs()

    def _fit_initial_image(self):
        """Fit the first picture after its dock has a real, visible viewport size."""
        if (not self._initial_fit_pending or not self.isVisible()
                or self.viewport().width() <= 0 or self.viewport().height() <= 0
                or self.scene().itemsBoundingRect().isEmpty()):
            return
        self.fit()
        self._initial_fit_pending = False

    def _hide_pixel_readout(self):
        self._pixel_readout_active = False
        self.pixel_readout.hide()

    def _update_pixel_readout(self, event):
        frame = self.window.frame
        if frame is None or self.last_frame_size is None or not hasattr(frame, "shape"):
            self._hide_pixel_readout()
            return
        scene_pos = self._event_scene_pos(event)
        scale = max(float(self.last_scale), 1.0)
        render_region = self.last_render_region
        if render_region is None:
            image_x, image_y = 0.0, 0.0
            full_width = self.last_frame_size[0] * scale
            full_height = self.last_frame_size[1] * scale
        else:
            image_x, image_y = render_region.x * scale, render_region.y * scale
            full_width = render_region.full_width * scale
            full_height = render_region.full_height * scale
        image_width = self.last_frame_size[0] * scale
        image_height = self.last_frame_size[1] * scale
        local_x = scene_pos.x() - image_x
        local_y = scene_pos.y() - image_y
        if not (0 <= local_x < image_width and 0 <= local_y < image_height):
            self._hide_pixel_readout()
            return
        full_x = math.floor(scene_pos.x())
        full_y = math.floor(scene_pos.y())
        array_x = math.floor(local_x / scale)
        array_y = math.floor(local_y / scale)
        buffer = ""
        try:
            if array_y < 0 or array_x < 0 or array_y >= frame.shape[0] or array_x >= frame.shape[1]:
                raise IndexError
            values = frame[array_y, array_x]
            frame_b = getattr(self.window, "frame_b", None)
            mode = self.compare_mode()
            if frame_b is not None and mode != "A only" and frame_b.shape == frame.shape:
                # The readout reports the buffer under the pointer: B on B's side of the wipe,
                # the composite where the mode combines the two.
                values, buffer = compare_model.pixel_at(mode, frame, frame_b, array_x, array_y,
                                                        self._wipe_side_at(scene_pos))
                buffer = {"A": "A", "B": "B"}.get(buffer, "A/B")
            if len(values) < 4:
                raise IndexError
            red, green, blue, alpha = (float(values[index]) for index in range(4))
        except (IndexError, TypeError, ValueError):
            self._hide_pixel_readout()
            return
        nuke_y = math.floor(full_height) - 1 - full_y
        color = QColor.fromRgbF(min(max(red, 0.0), 1.0), min(max(green, 0.0), 1.0),
                                min(max(blue, 0.0), 1.0), 1.0)
        self.pixel_readout.set_value(
            f"{full_x}, {nuke_y}  {red:.5f} {green:.5f} {blue:.5f} {alpha:.5f}", color,
            buffer if self.window.frame_b is not None and self.compare_mode() != "A only" else "")
        self.readout_buffer = buffer
        self._pixel_readout_active = True
        if self._place_pixel_readout():
            self.pixel_readout.show()
            self.pixel_readout.raise_()

    # ---- viewer inputs and the A/B compare -------------------------------------------------
    def input_state(self):
        return viewer_state(self.window.dispatcher.document)

    def compare_mode(self):
        """The mode in force: "A only" whenever there is no usable B to compare against."""
        state = self.input_state()
        if state["b"] is None or state["inputs"][state["b"] - 1] is None:
            return "A only"
        return state["compare"]

    def sync_inputs(self):
        document = self.window.dispatcher.document
        names = {key: node["name"] for key, node in document["nodes"].items()}
        self.input_strip.refresh(self.input_state(), names)
        self.viewport().update()

    def show_input(self, number):
        """Make input `number` the A buffer. An empty input is ignored, like Nuke's."""
        node = self.input_state()["inputs"][number - 1]
        if node is None:
            return False
        self.window.command({"op": "viewer_input", "slot": number, "id": node})
        return True

    def set_b(self, number):
        """Choose the B input (None clears it). Picking a B while the mode is "A only" turns the
        compare on as a wipe, which is what reaching for B means."""
        commands = [{"op": "viewer_compare", "b": number}]
        if number is not None and self.input_state()["compare"] == "A only":
            commands[0]["mode"] = "wipe"
        self.window.command(commands[0])

    def set_compare_mode(self, mode):
        self.window.command({"op": "viewer_compare", "mode": mode})

    def reset_wipe(self):
        self.wipe = dict(compare_model.WIPE_DEFAULT)
        self.wipe_drag = None
        self.viewport().update()

    def set_compare_image(self, image, scale, render_region):
        """Keep the B picture for the wipe, placed exactly where the A pixmap is."""
        if image is None:
            self.wipe_pixmap = None
            return
        if scale != 1:
            image = image.scaled(image.width() * scale, image.height() * scale,
                                 Qt.AspectRatioMode.IgnoreAspectRatio,
                                 Qt.TransformationMode.FastTransformation)
        self.wipe_pixmap = QPixmap.fromImage(image)
        self.wipe_pos = (QPointF(render_region.x * scale, render_region.y * scale)
                         if render_region is not None else QPointF(0, 0))

    def _wipe_active(self):
        return (self.compare_mode() == "wipe" and self.wipe_pixmap is not None
                and self.format_rect is not None)

    def _wipe_scene_geometry(self):
        """Centre, unit line direction, unit B-side normal and rotation-handle point, in scene units."""
        rect = self.format_rect
        centre = QPointF(rect.left() + self.wipe["x"] * rect.width(),
                         rect.top() + self.wipe["y"] * rect.height())
        dx, dy = compare_model.wipe_direction(self.wipe["angle"])
        nx, ny = compare_model.wipe_normal(self.wipe["angle"])
        reach = 70.0 / max(abs(self.transform().m11()), 0.05)
        return centre, (dx, dy), (nx, ny), QPointF(centre.x() + dx * reach, centre.y() + dy * reach)

    def _wipe_side_at(self, scene_pos):
        if self.format_rect is None:
            return "A"
        centre = self._wipe_scene_geometry()[0]
        return compare_model.wipe_side(scene_pos.x(), scene_pos.y(), centre.x(), centre.y(),
                                       self.wipe["angle"])

    def _wipe_hit(self, scene_pos):
        centre, (dx, dy), _, handle = self._wipe_scene_geometry()
        zoom = max(abs(self.transform().m11()), 0.05)
        if math.hypot(scene_pos.x() - handle.x(), scene_pos.y() - handle.y()) <= 12.0 / zoom:
            return "rotate"
        if math.hypot(scene_pos.x() - centre.x(), scene_pos.y() - centre.y()) <= 12.0 / zoom:
            return "centre"
        # Anywhere else along the line grabs it too, and moves it the way the centre handle does.
        along = (scene_pos.x() - centre.x()) * dx + (scene_pos.y() - centre.y()) * dy
        across = abs(-(scene_pos.x() - centre.x()) * dy + (scene_pos.y() - centre.y()) * dx)
        if across <= 6.0 / zoom and abs(along) <= max(self.format_rect.width(), self.format_rect.height()):
            return "line"
        return None

    def _wipe_drag_to(self, scene_pos):
        rect = self.format_rect
        drag = self.wipe_drag
        if drag["kind"] == "rotate":
            centre = self._wipe_scene_geometry()[0]
            self.wipe["angle"] = compare_model.wipe_angle_from_drag(
                centre.x(), centre.y(), scene_pos.x(), scene_pos.y())
        else:
            self.wipe["x"] = min(max((scene_pos.x() - drag["offset"][0] - rect.left()) / rect.width(), 0.0), 1.0)
            self.wipe["y"] = min(max((scene_pos.y() - drag["offset"][1] - rect.top()) / rect.height(), 0.0), 1.0)
        self.viewport().update()

    def _draw_wipe_b(self, painter):
        """B's half of the picture: the already-evaluated B pixmap clipped to the far side of the line."""
        centre, (dx, dy), (nx, ny), _ = self._wipe_scene_geometry()
        big = 1e6
        a = QPointF(centre.x() - dx * big, centre.y() - dy * big)
        b = QPointF(centre.x() + dx * big, centre.y() + dy * big)
        clip = QPainterPath()
        clip.addPolygon(QPolygonF([a, b, QPointF(b.x() + nx * big, b.y() + ny * big),
                                   QPointF(a.x() + nx * big, a.y() + ny * big)]))
        painter.save()
        painter.setClipPath(clip, Qt.ClipOperation.IntersectClip)
        painter.drawPixmap(self.wipe_pos, self.wipe_pixmap)
        painter.restore()

    def _draw_wipe_handles(self, painter):
        centre, (dx, dy), _, handle = self._wipe_scene_geometry()
        zoom = max(abs(self.transform().m11()), 0.05)
        reach = 1e6
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(QColor("#f4ce63"), 1.5)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawLine(QPointF(centre.x() - dx * reach, centre.y() - dy * reach),
                         QPointF(centre.x() + dx * reach, centre.y() + dy * reach))
        radius = 6.0 / zoom
        painter.setBrush(QColor("#202127"))
        painter.drawEllipse(centre, radius, radius)
        painter.drawEllipse(handle, radius, radius)
        painter.setBrush(QColor("#f4ce63"))
        painter.drawEllipse(centre, radius * 0.35, radius * 0.35)
        painter.drawEllipse(handle, radius * 0.35, radius * 0.35)
        painter.restore()

    def leaveEvent(self, event):
        if not self._handling_mouse_move:
            self._hide_pixel_readout()
        super().leaveEvent(event)

    def _roto_context(self):
        """Return the selected Roto payload and display scale when it is safe to edit it.

        Coordinates in a payload are in the Roto node's own output format.  An overlay is only
        editable while that exact node is being viewed; drawing over a downstream Transform or
        Tracker would make a screen gesture mean something different from the stored coordinates.
        """
        graph = getattr(self.window, "graph", None)
        if graph is None:
            return None
        key = graph.selected_id()
        document = self.window.dispatcher.document
        node = document["nodes"].get(key) if key else None
        if node is None or node["type"] not in ("Roto", "RotoPaint") or document.get("view") != key:
            return None
        payload = document.get("node_data", {}).get(key, {"shapes": []})
        if node["type"] == "RotoPaint":
            payload = {"shapes": [{k: item[k] for k in ("name", "mode", "opacity", "feather", "points")}
                                  for item in payload.get("items", []) if item.get("kind") == "shape"]}
        tier = getattr(getattr(self.window, "proxy", None), "currentData", lambda: 1)()
        try:
            tier = max(1, int(tier))
        except (TypeError, ValueError):
            tier = 1
        return key, node, payload, tier

    def _warp_context(self):
        graph = getattr(self.window, "graph", None)
        if graph is None or self.format_rect is None:
            return None
        key = graph.selected_id()
        document = self.window.dispatcher.document
        node = document["nodes"].get(key) if key else None
        if node is None or node["type"] not in ("SplineWarp", "GridWarp") or document.get("view") != key:
            return None
        payload = copy.deepcopy(document.get("node_data", {}).get(key))
        if node["type"] == "SplineWarp":
            payload = payload or {"pairs": []}
        elif payload is None:
            from . import warps
            grid = warps.default_grid(self.format_rect.width(), self.format_rect.height(),
                                      node["params"]["rows"], node["params"]["columns"])
            points = [[{"x": float(point[0]), "y": float(point[1]), "in_x": 0.0, "in_y": 0.0,
                        "out_x": 0.0, "out_y": 0.0} for point in row] for row in grid]
            payload = {"source": copy.deepcopy(points), "destination": copy.deepcopy(points)}
        tier = max(1, int(getattr(getattr(self.window, "proxy", None), "currentData", lambda: 1)() or 1))
        return key, node, payload, tier

    def begin_warp_draw(self, key, side):
        context = self._warp_context()
        if context is None or context[0] != key or context[1]["type"] != "SplineWarp":
            return False
        if side not in ("source", "destination"):
            return False
        self.warp_drawing, self.warp_draw_side, self.warp_draw_points = True, side, []
        self.warp_drag = None
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.window.statusBar().showMessage(f"SplineWarp: draw the {side} curve, then press Enter; Esc cancels")
        self.viewport().update()
        return True

    def cancel_warp_edit(self):
        self.warp_drawing = False
        self.warp_draw_side = None
        self.warp_draw_points = []
        self.warp_draw_cursor = None
        self.warp_drag = None
        self.unsetCursor()
        self.viewport().update()

    def finish_warp_draw(self):
        if not self.warp_drawing or len(self.warp_draw_points) < 2:
            self.window.statusBar().showMessage("SplineWarp curves need at least two points", 4000)
            return False
        context = self._warp_context()
        if context is None:
            self.cancel_warp_edit()
            return False
        key, _, payload, tier = context
        side = self.warp_draw_side
        pending = self.warp_pending.setdefault(key, {})
        pending[side] = [self._roto_data_point(p, context[1], tier) for p in self.warp_draw_points]
        self.warp_drawing = False
        self.warp_draw_side = None
        self.warp_draw_points = []
        self.warp_draw_cursor = None
        self.unsetCursor()
        if "source" in pending and "destination" in pending:
            source, destination = pending["source"], pending["destination"]
            n = max(2, max(len(source), len(destination)))
            def resample(points):
                values = np.asarray(points, dtype=float)
                old = np.linspace(0.0, 1.0, len(values))
                new = np.linspace(0.0, 1.0, n)
                return np.stack([np.interp(new, old, values[:, axis]) for axis in range(2)], axis=1)
            source, destination = resample(source), resample(destination)
            def controls(points):
                return [{"x": float(x), "y": float(y), "in_x": 0.0, "in_y": 0.0,
                         "out_x": 0.0, "out_y": 0.0} for x, y in points]
            pairs = copy.deepcopy(payload.get("pairs", []))
            names = {pair["name"] for pair in pairs}
            i = 1
            while f"pair{i}" in names:
                i += 1
            pairs.append({"name": f"pair{i}", "source": controls(source), "destination": controls(destination)})
            self.warp_pending.pop(key, None)
            self.window.command({"op": "set_warp_data", "id": key, "data": {"pairs": pairs}})
        self.viewport().update()
        return True

    def _warp_drag_to(self, scene_pos):
        if self.warp_drag is None:
            return
        self.warp_drag["scene"] = scene_pos
        self.warp_drag["moved"] = (scene_pos - self.warp_drag["start"]).manhattanLength() > 2
        self.viewport().update()

    def _commit_warp_drag(self):
        drag = self.warp_drag
        context = self._warp_context()
        if drag is None or context is None or context[0] != drag["key"]:
            self.warp_drag = None
            return False
        key, node, payload, tier = context
        x, y = self._roto_data_point(drag["scene"], node, tier)
        frame = int(self.window.dispatcher.document["time"]["current"])
        if node["type"] == "SplineWarp":
            pair = payload["pairs"][drag["pair"]]
            point = pair[drag["side"]][drag["index"]]
            if drag.get("handle"):
                rx, ry = (shape_model.resolve_scalar(point[field], frame, field)
                          for field in ("x", "y"))
                hx, hy = drag["handle"]
                point[hx] = self._roto_scalar_at_frame(point[hx], frame, x-rx)
                point[hy] = self._roto_scalar_at_frame(point[hy], frame, y-ry)
            else:
                point["x"] = self._roto_scalar_at_frame(point["x"], frame, x)
                point["y"] = self._roto_scalar_at_frame(point["y"], frame, y)
        else:
            point = payload[drag["side"]][drag["row"]][drag["column"]]
            if drag.get("handle"):
                rx, ry = (shape_model.resolve_scalar(point[field], frame, field)
                          for field in ("x", "y"))
                hx, hy = drag["handle"]
                point[hx] = self._roto_scalar_at_frame(point.get(hx, 0.0), frame, x-rx)
                point[hy] = self._roto_scalar_at_frame(point.get(hy, 0.0), frame, y-ry)
            else:
                point["x"] = self._roto_scalar_at_frame(point["x"], frame, x)
                point["y"] = self._roto_scalar_at_frame(point["y"], frame, y)
        self.warp_drag = None
        self.window.command({"op": "set_warp_data", "id": key, "data": payload})
        return True

    def _warp_hit(self, context, scene_pos):
        key, node, payload, tier = context
        radius = 12.0 / max(abs(self.transform().m11()), 0.05)
        best, best_distance = None, radius
        if node["type"] == "SplineWarp":
            resolved = shape_model.resolve_warp_data("SplineWarp", payload,
                                                       self.window.dispatcher.document["time"]["current"])
            for pair_index, pair in enumerate(resolved):
                for side in ("source", "destination"):
                    for point_index, point in enumerate(pair[side]):
                        x, y = point[:2]
                        scene = self._roto_scene_point({"x": x, "y": y}, tier)
                        distance = math.hypot(scene.x() - scene_pos.x(), scene.y() - scene_pos.y())
                        if distance < best_distance:
                            best, best_distance = (pair_index, side, point_index), distance
                        for label, indices in (("in", (2, 3)), ("out", (4, 5))):
                            hx, hy = point[0] + point[indices[0]], point[1] + point[indices[1]]
                            if math.hypot(point[indices[0]], point[indices[1]]) <= 1e-6:
                                continue
                            handle = self._roto_scene_point({"x": hx, "y": hy}, tier)
                            distance = math.hypot(handle.x() - scene_pos.x(), handle.y() - scene_pos.y())
                            if distance < best_distance:
                                best, best_distance = (pair_index, side, point_index, label), distance
        else:
            resolved = shape_model.resolve_warp_data("GridWarp", payload,
                                                       self.window.dispatcher.document["time"]["current"])
            for side in ("source", "destination"):
                for row, points in enumerate(resolved[side]):
                    for column, point in enumerate(points):
                        x, y = point[:2]
                        scene = self._roto_scene_point({"x": x, "y": y}, tier)
                        distance = math.hypot(scene.x() - scene_pos.x(), scene.y() - scene_pos.y())
                        if distance < best_distance:
                            best, best_distance = (side, row, column), distance
                        for label, indices in (("in", (2, 3)), ("out", (4, 5))):
                            if math.hypot(point[indices[0]], point[indices[1]]) <= 1e-6:
                                continue
                            handle = self._roto_scene_point({"x": point[0]+point[indices[0]],
                                                             "y": point[1]+point[indices[1]]}, tier)
                            distance = math.hypot(handle.x() - scene_pos.x(), handle.y() - scene_pos.y())
                            if distance < best_distance:
                                best, best_distance = (side, row, column, label), distance
        if best is None:
            return None
        if node["type"] == "SplineWarp":
            pair, side, index, *handle = best
            return {"key": key, "pair": pair, "side": side, "index": index,
                    **({"handle": ("in_x", "in_y")} if handle and handle[0] == "in" else
                       {"handle": ("out_x", "out_y")} if handle else {})}
        side, row, column, *handle = best
        return {"key": key, "side": side, "row": row, "column": column,
                **({"handle": ("in_x", "in_y")} if handle and handle[0] == "in" else
                   {"handle": ("out_x", "out_y")} if handle else {})}

    def _draw_warp_overlay(self, painter):
        context = self._warp_context()
        if context is None:
            return
        key, node, payload, tier = context
        frame = self.window.dispatcher.document["time"]["current"]
        resolved = shape_model.resolve_warp_data(node["type"], payload, frame)
        if self.warp_drag is not None and self.warp_drag.get("moved"):
            x, y = self._roto_data_point(self.warp_drag["scene"], node, tier)
            if node["type"] == "SplineWarp":
                point = resolved[self.warp_drag["pair"]][self.warp_drag["side"]][self.warp_drag["index"]]
            else:
                point = resolved[self.warp_drag["side"]][self.warp_drag["row"]][self.warp_drag["column"]]
            if self.warp_drag.get("handle"):
                names = self.warp_drag["handle"]
                indexes = (2, 3) if names[0] == "in_x" else (4, 5)
                point[indexes[0]], point[indexes[1]] = x-point[0], y-point[1]
            else:
                point[0], point[1] = x, y
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        zoom = max(abs(self.transform().m11()), 0.05)
        radius = 5.0 / zoom
        def draw_bezier(points):
            if not points:
                return
            path = QPainterPath(self._roto_scene_point({"x": points[0][0], "y": points[0][1]}, tier))
            for a, b in zip(points, points[1:]):
                c1 = self._roto_scene_point({"x": a[0]+a[4], "y": a[1]+a[5]}, tier)
                c2 = self._roto_scene_point({"x": b[0]+b[2], "y": b[1]+b[3]}, tier)
                end = self._roto_scene_point({"x": b[0], "y": b[1]}, tier)
                path.cubicTo(c1, c2, end)
            painter.drawPath(path)
        def draw_tangents(points, color):
            painter.setPen(QPen(QColor(color), 1)); painter.setBrush(QColor("#202127"))
            for data in points:
                anchor = self._roto_scene_point({"x": data[0], "y": data[1]}, tier)
                for ix, iy in ((2, 3), (4, 5)):
                    if math.hypot(data[ix], data[iy]) <= 1e-6:
                        continue
                    end = self._roto_scene_point({"x": data[0]+data[ix], "y": data[1]+data[iy]}, tier)
                    painter.drawLine(anchor, end)
                    painter.drawEllipse(end, radius*0.7, radius*0.7)
        if node["type"] == "SplineWarp":
            for pair in resolved:
                source = [self._roto_scene_point({"x": point[0], "y": point[1]}, tier) for point in pair["source"]]
                destination = [self._roto_scene_point({"x": point[0], "y": point[1]}, tier) for point in pair["destination"]]
                painter.setPen(QPen(QColor("#58d7ff"), 2)); painter.setBrush(Qt.BrushStyle.NoBrush)
                draw_bezier(pair["source"])
                draw_bezier(pair["destination"])
                join_pen = QPen(QColor("#9494a0"), 1); join_pen.setStyle(Qt.PenStyle.DashLine)
                join_pen.setCosmetic(True); painter.setPen(join_pen)
                for a, b in zip(source, destination): painter.drawLine(a, b)
                for points, color, data in ((source, "#58d7ff", pair["source"]),
                                             (destination, "#f4ce63", pair["destination"])):
                    draw_tangents(data, color)
                    painter.setPen(QPen(QColor(color), 1)); painter.setBrush(QColor("#202127"))
                    for point in points: painter.drawEllipse(point, radius, radius)
        else:
            source, destination = resolved["source"], resolved["destination"]
            for grid, color in ((source, "#58d7ff"), (destination, "#f4ce63")):
                painter.setPen(QPen(QColor(color), 1.5)); painter.setBrush(Qt.BrushStyle.NoBrush)
                for row in grid:
                    draw_bezier(row)
                for column in range(len(grid[0]) if grid else 0):
                    draw_bezier([row[column] for row in grid])
                for row in grid:
                    draw_tangents(row, color)
            join_pen = QPen(QColor("#9494a0"), 1); join_pen.setStyle(Qt.PenStyle.DashLine)
            join_pen.setCosmetic(True); painter.setPen(join_pen)
            for sr, dr in zip(source, destination):
                for s, d in zip(sr, dr):
                    painter.drawLine(self._roto_scene_point({"x": s[0], "y": s[1]}, tier),
                                     self._roto_scene_point({"x": d[0], "y": d[1]}, tier))
            for grid, color in ((source, "#58d7ff"), (destination, "#f4ce63")):
                painter.setPen(QPen(QColor(color), 1)); painter.setBrush(QColor("#202127"))
                for row in grid:
                    for point_data in row:
                        x, y = point_data[:2]
                        point = self._roto_scene_point({"x": x, "y": y}, tier)
                        painter.drawRect(QRectF(point.x()-radius, point.y()-radius, radius*2, radius*2))
        if self.warp_drawing:
            points = [QPointF(self.format_rect.left()+x, self.format_rect.top()+y) for x, y in self.warp_draw_points]
            if self.warp_draw_cursor is not None: points.append(self.warp_draw_cursor)
            if points:
                path = QPainterPath(points[0])
                for point in points[1:]: path.lineTo(point)
                pen = QPen(QColor("#f4ce63"), 2); pen.setStyle(Qt.PenStyle.DashLine); pen.setCosmetic(True)
                painter.setPen(pen); painter.drawPath(path)
        painter.restore()

    def _paint_context(self):
        graph = getattr(self.window, "graph", None)
        key = graph.selected_id() if graph is not None else None
        document = self.window.dispatcher.document
        node = document["nodes"].get(key) if key else None
        if node is None or node["type"] != "RotoPaint" or document.get("view") != key:
            return None
        return key, node, document.get("node_data", {}).get(key, {"items": []})

    def _paint_sample(self, scene_pos, event, start=False):
        context = self._paint_context()
        if context is None: return False
        value = self._roto_data_point(scene_pos, context[1], 1)
        pressure = float(event.pressure()) if callable(getattr(event, "pressure", None)) else 1.0
        sample = {"x": value[0], "y": value[1], "pressure": min(1.0, max(0.0, pressure))}
        if start: self.paint_drag = {"key": context[0], "points": [sample]}
        elif self.paint_drag is not None: self.paint_drag["points"].append(sample)
        self.viewport().update()
        return True

    def _finish_paint(self):
        drag = self.paint_drag
        self.paint_drag = None
        context = self._paint_context()
        if drag is None or context is None or context[0] != drag["key"]: return False
        key, _, payload = context
        items = copy.deepcopy(payload.get("items", []))
        frame = int(self.window.dispatcher.document["time"]["current"])
        lifetime = ({"mode": "single", "first": frame} if self.dustbust_preset or self.paint_lifetime == "single"
                    else {"mode": "all"} if self.paint_lifetime == "all"
                    else {"mode": "range", "first": int(self.paint_range_first), "last": int(self.paint_range_last)}
                    if self.paint_lifetime == "range"
                    else {"mode": "from_current", "first": frame})
        names = {item.get("name") for item in items}
        index = 1
        while f"stroke{index}" in names: index += 1
        tool = "clone" if self.dustbust_preset else self.paint_tool
        follow = None if self.dustbust_preset else self.paint_follow_track
        items.append({"kind": "stroke", "name": f"stroke{index}", "points": drag["points"],
                      "brush": {"size": self.paint_size, "hardness": self.paint_hardness,
                                "opacity": self.paint_brush_opacity, "spacing": self.paint_spacing,
                                "strength": self.paint_strength},
                      "tool": tool, "lifetime": lifetime, "color": list(self.paint_color),
                      "source_offset": list(self.paint_source_offset),
                      "source_frame": "relative" if self.dustbust_preset else
                                      (self.paint_source_frame if tool == "clone" else frame),
                      "opacity": self.paint_layer_opacity, "blend": self.paint_blend,
                      "visible": self.paint_visible, "follow_track": follow})
        self.dustbust_preset = False
        self.window.command({"op": "set_paint_items", "id": key, "items": items})
        return True

    def _transform_context(self):
        """The selected Transform node when it is safe to draw its handle: it is selected in the
        properties panel (the graph selection) and the viewer is showing it or something
        downstream of it, so the overlay's coordinates line up with what is on screen."""
        graph = getattr(self.window, "graph", None)
        if graph is None or self.format_rect is None:
            return None
        key = graph.selected_id()
        document = self.window.dispatcher.document
        node = document["nodes"].get(key) if key else None
        if node is None or node["type"] != "Transform":
            return None
        view_key = document.get("view")
        if view_key is None:
            return None
        nodes = document["nodes"]
        stack, seen = [view_key], set()
        while stack:
            current = stack.pop()
            if current is None or current in seen or current not in nodes:
                continue
            if current == key:
                return key, node
            seen.add(current)
            stack.extend(nodes[current]["inputs"].values())
        return None

    def _flare_context(self):
        """Selected Flare over its own output or a downstream viewer target."""
        graph = getattr(self.window, "graph", None)
        if graph is None or self.format_rect is None:
            return None
        key = graph.selected_id()
        document = self.window.dispatcher.document
        node = document["nodes"].get(key) if key else None
        if node is None or node["type"] != "Flare":
            return None
        view_key = document.get("view")
        if view_key is None:
            return None
        nodes = document["nodes"]
        stack, seen = [view_key], set()
        while stack:
            current = stack.pop()
            if current is None or current in seen or current not in nodes:
                continue
            if current == key:
                return key, node
            seen.add(current)
            stack.extend(nodes[current]["inputs"].values())
        return None

    def _analysis_context(self):
        """Selected MinColor/Sampler whose properties panel is actually open."""
        graph = getattr(self.window, "graph", None)
        if graph is None or self.format_rect is None:
            return None
        key = graph.selected_id()
        document = self.window.dispatcher.document
        node = document["nodes"].get(key) if key else None
        if node is None or node["type"] not in ("MinColor", "Sampler"):
            return None
        dock = getattr(self.window, "properties_dock", None)
        if dock is not None and not dock.isVisible():
            return None
        pinned = getattr(self.window, "pinned_panels", [])
        if pinned and key not in pinned:
            return None
        view_key = document.get("view")
        if view_key is None:
            return None
        nodes = document["nodes"]
        stack, seen = [view_key], set()
        while stack:
            current = stack.pop()
            if current is None or current in seen or current not in nodes:
                continue
            if current == key:
                return key, node
            seen.add(current)
            stack.extend(nodes[current]["inputs"].values())
        return None

    def _analysis_values(self):
        context = self._analysis_context()
        if context is None:
            return None
        key, node = context
        values = dict(node["params"])
        drag = self.analysis_drag
        if drag is None or drag["key"] != key:
            return values
        sx, sy = self._transform_data_point(drag["start"])
        cx, cy = self._transform_data_point(drag["scene"])
        dx, dy = cx - sx, cy - sy
        start = drag["original"]
        kind = node["type"]
        if kind == "Sampler":
            if drag["part"] == "line":
                for suffix in ("0", "1"):
                    values[f"sample_x{suffix}"] = start[f"sample_x{suffix}"] + dx
                    values[f"sample_y{suffix}"] = start[f"sample_y{suffix}"] + dy
            else:
                index = drag["part"]
                values[f"sample_x{index}"] = start[f"sample_x{index}"] + dx
                values[f"sample_y{index}"] = start[f"sample_y{index}"] + dy
        else:
            x, y = start["box_x"], start["box_y"]
            w, h = start["box_width"], start["box_height"]
            part = drag["part"]
            if part == "body": x += dx; y += dy
            else:
                if "w" in part: x += dx; w -= dx
                if "e" in part: w += dx
                if "n" in part: y += dy; h -= dy
                if "s" in part: h += dy
            values.update(box_x=x, box_y=y, box_width=max(1.0, w), box_height=max(1.0, h))
        return values

    def _analysis_commands(self, key, values):
        return [{"op": "set", "id": key, "param": name, "value": float(value)}
                for name, value in values.items()]

    def _flare_values(self, drag=None):
        context = self._flare_context()
        if context is None:
            return None
        key, node = context
        document = self.window.dispatcher.document
        frame = int(document["time"]["current"])
        resolved = resolve_document(document, frame)
        params = dict(resolved["nodes"][key]["params"])
        dx, dy = self._flare_link_delta(node, frame)
        params["position_x"] += dx
        params["position_y"] += dy
        if drag is not None:
            start = self._transform_data_point(drag["start"])
            current = self._transform_data_point(drag["scene"])
            params["position_x"] = drag["position"][0] + current[0] - start[0]
            params["position_y"] = drag["position"][1] + current[1] - start[1]
        return params

    def _flare_link_delta(self, node, frame=None):
        document = self.window.dispatcher.document
        params = node["params"]
        tracker_id = params.get("tracker_id", "")
        if not tracker_id:
            return 0.0, 0.0
        frame = int(document["time"]["current"] if frame is None else frame)
        tracker_node = document["nodes"][tracker_id]
        track = document["node_data"][tracker_id]["tracks"][params["track_index"]]
        reference = int(tracker_node["params"].get("reference_frame", 1))
        return (shape_model.resolve_scalar(track["x"], frame, "x")
                - shape_model.resolve_scalar(track["x"], reference, "x"),
                shape_model.resolve_scalar(track["y"], frame, "y")
                - shape_model.resolve_scalar(track["y"], reference, "y"))

    def _event_scene_pos(self, event):
        """Map a viewport mouse event into scene coordinates (Qt sends QMouseEvent here)."""
        return self.viewportTransform().inverted()[0].map(event.position())

    def _roto_scene_point(self, point, tier):
        # preview_ready upscales a proxy pixmap back to the full format rectangle.  The scene
        # therefore remains in full-resolution display coordinates and the payload's pixel
        # coordinates are not divided by tier here.
        rect = self.format_rect
        origin_x = rect.left() if rect is not None else 0.0
        origin_y = rect.top() if rect is not None else 0.0
        return QPointF(origin_x + point["x"], origin_y + point["y"])

    def _roto_data_point(self, scene_pos, node, tier):
        rect = self.format_rect
        origin_x = rect.left() if rect is not None else 0.0
        origin_y = rect.top() if rect is not None else 0.0
        width = node["params"].get("width", self.format_rect.width() if self.format_rect else 0)
        height = node["params"].get("height", self.format_rect.height() if self.format_rect else 0)
        # See _roto_scene_point: the upscaled proxy still occupies the full format scene rect.
        x = scene_pos.x() - origin_x
        y = scene_pos.y() - origin_y
        return [min(max(x, 0.0), float(width)), min(max(y, 0.0), float(height))]

    def _transform_scene_point(self, x, y):
        rect = self.format_rect
        origin_x = rect.left() if rect is not None else 0.0
        origin_y = rect.top() if rect is not None else 0.0
        return QPointF(origin_x + x, origin_y + y)

    def _transform_data_point(self, scene_pos):
        rect = self.format_rect
        origin_x = rect.left() if rect is not None else 0.0
        origin_y = rect.top() if rect is not None else 0.0
        return scene_pos.x() - origin_x, scene_pos.y() - origin_y

    def _transform_values(self, drag=None):
        """Return committed or live values for the selected Transform overlay."""
        if drag is None:
            drag = self.transform_drag
        context = self._transform_context()
        if context is None:
            return None
        _, node = context
        values = dict(node["params"])
        if drag is None:
            return values
        original = drag["original"]
        start = self._transform_data_point(drag["start"])
        current = self._transform_data_point(drag["scene"])
        if drag["kind"] == "translate":
            values.update(translate_x=original["translate_x"] + current[0] - start[0],
                          translate_y=original["translate_y"] + current[1] - start[1])
        elif drag["kind"] == "pivot":
            cx, cy, tx, ty = handles2d.pivot_drag_result(
                original["translate_x"], original["translate_y"], values["rotate"], values["scale"],
                original["center_x"], original["center_y"], current[0] - start[0], current[1] - start[1])
            values.update(center_x=cx, center_y=cy, translate_x=tx, translate_y=ty)
        elif drag["kind"] == "rotate":
            values["rotate"] = handles2d.rotate_from_drag(
                original["rotate"], drag["pivot"], start, current)
        elif drag["kind"] == "scale":
            values["scale"] = handles2d.scale_from_drag(
                original["scale"], drag["pivot"], start, current)
        return values

    def _transform_commands(self, key, values):
        frame = int(self.window.dispatcher.document["time"]["current"])
        commands = []
        for param, value in values.items():
            if self.window.node_curve(key, param) is not None:
                commands.append({"op": "set_key", "id": key, "param": param, "frame": frame,
                                 "value": float(value)})
            else:
                commands.append({"op": "set", "id": key, "param": param, "value": float(value)})
        return commands

    def _tracker_data_point(self, scene_pos):
        rect = self.format_rect
        return [scene_pos.x() - (rect.left() if rect is not None else 0.0),
                scene_pos.y() - (rect.top() if rect is not None else 0.0)]

    def _tracker_hit(self, context, scene_pos):
        key, node, payload, _ = context
        frame = int(self.window.dispatcher.document["time"]["current"])
        tracks = shape_model.resolve_tracks(payload, frame)
        point = self._tracker_data_point(scene_pos)
        zoom = max(abs(self.transform().m11()), 0.05)
        hit_radius = 10.0 / zoom
        pattern = float(node["params"].get("pattern_radius", 8))
        search = float(node["params"].get("search_radius", 24))
        best = None
        distance = hit_radius
        for index, track in enumerate(tracks):
            x, y = track["x"], track["y"]
            if abs(abs(point[0]-x)-pattern) <= hit_radius and abs(point[1]-y) <= pattern+hit_radius:
                return {"kind": "pattern", "center": (x, y)}
            if abs(abs(point[1]-y)-pattern) <= hit_radius and abs(point[0]-x) <= pattern+hit_radius:
                return {"kind": "pattern", "center": (x, y)}
            if abs(abs(point[0]-x)-search) <= hit_radius and abs(point[1]-y) <= search+hit_radius:
                return {"kind": "search", "center": (x, y)}
            if abs(abs(point[1]-y)-search) <= hit_radius and abs(point[0]-x) <= search+hit_radius:
                return {"kind": "search", "center": (x, y)}
            d = math.hypot(point[0]-x, point[1]-y)
            if d < distance:
                best, distance = {"kind": "point", "index": index}, d
        return best

    def _draw_tracker_overlay(self, painter):
        context = self.window._tracker_context()
        if context is None:
            return
        key, node, payload, _ = context
        params = node["params"]
        frame = int(self.window.dispatcher.document["time"]["current"])
        tracks = shape_model.resolve_tracks(payload, frame)
        first = int(self.window.dispatcher.document["time"]["first"])
        last = int(self.window.dispatcher.document["time"]["last"])
        zoom = max(abs(self.transform().m11()), 0.05)
        marker = 4.0 / zoom
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for index, raw in enumerate(payload.get("tracks", [])):
            xkeys = raw["x"].get("curve", {}).get("keys", []) if isinstance(raw["x"], dict) else []
            ykeys = raw["y"].get("curve", {}).get("keys", []) if isinstance(raw["y"], dict) else []
            frames = sorted({int(k["frame"]) for k in xkeys + ykeys if first <= int(k["frame"]) <= last})
            path_points = []
            for path_frame in frames:
                resolved = shape_model.resolve_tracks({"tracks": [raw]}, path_frame)[0]
                path_points.append((path_frame, self._roto_scene_point(resolved, 1), resolved["error"]))
            for a, b in zip(path_points, path_points[1:]):
                err = max(a[2], b[2])
                color = QColor.fromRgbF(min(1.0, 0.25 + 1.5*err), max(0.0, 0.85-1.5*err), 0.2, 0.9)
                pen = QPen(color, 2); pen.setCosmetic(True); painter.setPen(pen)
                painter.drawLine(a[1], b[1])
            if index >= len(tracks) or not tracks[index]["enabled"]:
                continue
            track = tracks[index]
            center = self._roto_scene_point(track, 1)
            for radius, color in ((float(params.get("search_radius", 24)), "#f4ce63"),
                                  (float(params.get("pattern_radius", 8)), "#58d7ff")):
                side = 2 * radius
                pen = QPen(QColor(color), 1); pen.setCosmetic(True); pen.setStyle(Qt.PenStyle.DashLine)
                painter.setPen(pen); painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(QRectF(center.x()-radius, center.y()-radius, side, side))
            painter.setPen(QPen(QColor("#ff625c"), 2)); painter.setBrush(QColor("#202127"))
            painter.drawEllipse(center, marker, marker)
        painter.restore()

    def _commit_tracker_drag(self):
        drag = self.tracker_drag
        context = self.window._tracker_context()
        self.tracker_drag = None
        if drag is None or context is None or not drag.get("moved"):
            return False
        key, node, payload, _ = context
        frame = int(self.window.dispatcher.document["time"]["current"])
        if drag["kind"] == "point":
            x, y = self._tracker_data_point(drag["scene"])
            tracks = copy.deepcopy(payload.get("tracks", []))
            track = tracks[drag["index"]]
            for field, value in (("x", x), ("y", y)):
                track[field] = self._roto_scalar_at_frame(track[field], frame, value)
            self.window.command({"op": "set_tracks", "id": key, "tracks": tracks})
        else:
            center = drag["center"]
            point = self._tracker_data_point(drag["scene"])
            maximum = 128 if drag["kind"] == "pattern" else 256
            radius = min(maximum, max(1, int(round(max(abs(point[0]-center[0]),
                                                       abs(point[1]-center[1]))))))
            params = node["params"]
            param = "pattern_radius" if drag["kind"] == "pattern" else "search_radius"
            edits = [{"op": "set", "id": key, "param": param, "value": radius}]
            if param == "pattern_radius" and radius > int(params.get("search_radius", 24)):
                edits.append({"op": "set", "id": key, "param": "search_radius", "value": radius})
            elif param == "search_radius" and radius < int(params.get("pattern_radius", 8)):
                edits.append({"op": "set", "id": key, "param": "pattern_radius", "value": radius})
            self.window.command({"op": "batch", "commands": edits})
        return True

    def _roto_hit_point(self, scene_pos, resolved, tier):
        # A constant physical hit target remains usable at any viewer zoom.
        radius = 12.0 / max(abs(self.transform().m11()), 0.05)
        best = None
        best_distance = radius
        for shape_index, shape in enumerate(resolved):
            for point_index, point in enumerate(shape["points"]):
                scene_point = self._roto_scene_point(point, tier)
                distance = math.hypot(scene_point.x() - scene_pos.x(), scene_point.y() - scene_pos.y())
                if distance <= best_distance:
                    best = (shape_index, point_index)
                    best_distance = distance
        return best

    def _roto_scalar_at_frame(self, value, frame, replacement):
        """Change a point coordinate while retaining its v8 animated scalar envelope."""
        if not isinstance(value, dict) or "curve" not in value or value.get("curve") is None:
            return float(replacement)
        updated = copy.deepcopy(value)
        keys = updated["curve"]["keys"]
        for key in keys:
            if key["frame"] == frame:
                key["value"] = float(replacement)
                return updated
        keys.append({"frame": int(frame), "value": float(replacement)})
        keys.sort(key=lambda key: key["frame"])
        return updated

    def _roto_resolved(self, context):
        key, _, payload, _ = context
        return shape_model.resolve_shapes(payload, self.window.dispatcher.document["time"]["current"])

    def begin_roto_draw(self, key=None):
        context = self._roto_context()
        if context is None or (key is not None and context[0] != key):
            return False
        self.roto_key = context[0]
        self.roto_drawing = True
        self.roto_draw_points = []
        self.roto_draw_cursor = None
        self.roto_drag = None
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.viewport().update()
        self.window.statusBar().showMessage("Roto: click points · Enter closes the shape · Esc cancels")
        return True

    def cancel_roto_edit(self):
        self.roto_drawing = False
        self.roto_draw_points = []
        self.roto_draw_cursor = None
        self.roto_drag = None
        self.roto_key = None
        self.unsetCursor()
        self.viewport().update()

    def finish_roto_draw(self):
        if not self.roto_drawing:
            return False
        context = self._roto_context()
        if context is None or len(self.roto_draw_points) < shape_model.MINIMUM_SHAPE_POINTS:
            self.window.statusBar().showMessage("Roto: at least 3 points are required", 4000)
            return False
        key, _, payload, _ = context
        frame = int(self.window.dispatcher.document["time"]["current"])
        points = [{"x": x, "y": y, "in_x": 0.0, "in_y": 0.0, "out_x": 0.0, "out_y": 0.0}
                  for x, y in self.roto_draw_points]
        node = self.window.dispatcher.document["nodes"][key]
        if node["type"] == "RotoPaint":
            items = copy.deepcopy(self.window.dispatcher.document.get("node_data", {}).get(key, {}).get("items", []))
            names = {item.get("name") for item in items}
        else:
            items = None
            names = {shape.get("name") for shape in payload.get("shapes", [])}
        index = 1
        while f"shape{index}" in names:
            index += 1
        shapes = copy.deepcopy(payload.get("shapes", []))
        shape = {"name": f"shape{index}", "mode": "union", "opacity": 1.0,
                 "feather": 0.0, "points": points}
        shapes.append(shape)
        self.cancel_roto_edit()
        # Whole-payload replacement is the Dispatcher validation/undo boundary.  The local frame
        # variable documents that this draw is static; future point drags key animated scalars.
        if items is None:
            self.window.command({"op": "set_shapes", "id": key, "shapes": shapes})
        else:
            items.append({"kind": "shape", **shape, "blend": "over", "visible": True})
            self.window.command({"op": "set_paint_items", "id": key, "items": items})
        return True

    def _commit_roto_drag(self):
        drag = self.roto_drag
        if drag is None:
            return False
        context = self._roto_context()
        if context is None:
            self.roto_drag = None
            return False
        key, node, payload, tier = context
        shape_index, point_index = drag["point"]
        shapes = copy.deepcopy(payload.get("shapes", []))
        if shape_index >= len(shapes) or point_index >= len(shapes[shape_index]["points"]):
            self.roto_drag = None
            return False
        x, y = self._roto_data_point(drag["scene"], context[1], tier)
        frame = int(self.window.dispatcher.document["time"]["current"])
        point = shapes[shape_index]["points"][point_index]
        point["x"] = self._roto_scalar_at_frame(point["x"], frame, x)
        point["y"] = self._roto_scalar_at_frame(point["y"], frame, y)
        self.roto_drag = None
        if node["type"] == "RotoPaint":
            items = copy.deepcopy(self.window.dispatcher.document.get("node_data", {}).get(key, {}).get("items", []))
            shape_items = [i for i, item in enumerate(items) if item.get("kind") == "shape"]
            if shape_index >= len(shape_items): return False
            target = items[shape_items[shape_index]]
            target.update(shapes[shape_index])
            self.window.command({"op": "set_paint_items", "id": key, "items": items})
        else:
            self.window.command({"op": "set_shapes", "id": key, "shapes": shapes})
        return True

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape and (self.crypto_picking or self.zdefocus_picking or self.focus_picking):
            what = "Focus" if self.focus_picking else "Cryptomatte"
            self.zdefocus_picking = None
            self.crypto_picking = None
            self.focus_picking = None
            self.unsetCursor()
            self.window.statusBar().showMessage(f"{what} picking finished")
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape and self.tracker_picking:
            self.tracker_picking = False
            self.unsetCursor()
            self.window.statusBar().showMessage("Tracker point picking cancelled")
            event.accept()
            return
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self.roto_drawing:
            self.finish_roto_draw()
            event.accept()
            return
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self.warp_drawing:
            self.finish_warp_draw()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape and (self.roto_drawing or self.roto_drag is not None
                                                  or self.transform_drag is not None or self.analysis_drag is not None or self.flare_drag is not None or self.warp_drawing
                                                  or self.warp_drag is not None or self.tracker_drag is not None):
            if self.roto_drawing or self.roto_drag is not None:
                self.cancel_roto_edit()
            if self.warp_drawing or self.warp_drag is not None:
                self.cancel_warp_edit()
            if self.analysis_drag is not None:
                key = self.analysis_drag["key"]
                node = self.window.dispatcher.document["nodes"].get(key)
                if node is not None:
                    self.window.preview_knobs(key, node["params"])
            self.transform_drag = None
            self.analysis_drag = None
            self.flare_drag = None
            self.tracker_drag = None
            self.unsetCursor()
            self.viewport().update()
            event.accept()
            return
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier and event.key() in (
                Qt.Key.Key_Equal, Qt.Key.Key_Plus, Qt.Key.Key_Minus, Qt.Key.Key_1):
            if event.key() == Qt.Key.Key_1:
                self.resetTransform()
            else:
                factor = 1.15 if event.key() in (Qt.Key.Key_Equal, Qt.Key.Key_Plus) else 1 / 1.15
                if 0.05 < self.transform().m11() * factor < 20:
                    self.scale(factor, factor)
            event.accept()
            return
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier and event.key() == Qt.Key.Key_P:
            self.window.toggle_viewer_proxy()
            event.accept()
            return
        if not event.modifiers() and Qt.Key.Key_1 <= event.key() <= Qt.Key.Key_9:
            self.show_input(event.key() - Qt.Key.Key_0)
            event.accept()
            return
        if event.modifiers() == Qt.KeyboardModifier.AltModifier and Qt.Key.Key_0 <= event.key() <= Qt.Key.Key_9:
            self.set_b(event.key() - Qt.Key.Key_0 or None)
            event.accept()
            return
        if event.modifiers() == Qt.KeyboardModifier.ShiftModifier and event.key() == Qt.Key.Key_W:
            self.reset_wipe()
            event.accept()
            return
        channel_for_key = {Qt.Key.Key_R: "R", Qt.Key.Key_G: "G", Qt.Key.Key_B: "B", Qt.Key.Key_A: "A"}
        if event.key() in channel_for_key and not event.modifiers():
            channel = channel_for_key[event.key()]
            # Nuke-style solo behavior: pressing an already-soloed channel returns to RGB.
            self.window.channels.setCurrentText("RGB" if self.window.channels.currentText() == channel else channel)
            event.accept()
        elif event.key() == Qt.Key.Key_L and not event.modifiers():
            self.window.toggle_playback(True)
            event.accept()
        elif event.key() == Qt.Key.Key_K and not event.modifiers():
            self.window.toggle_playback(False)
            event.accept()
        elif event.key() == Qt.Key.Key_J and not event.modifiers():
            self.window.step_frame(-1)
            event.accept()
        elif event.key() in (Qt.Key.Key_F, Qt.Key.Key_H) and not event.modifiers():
            # F is the direct fit command. H is the familiar home/frame alias: with a
            # single 2D image there is no separate selected-object extent to frame.
            self.fit()
            event.accept()
        else:
            super().keyPressEvent(event)

    def mouseReleaseEvent(self, event):
        if self.pan is not None:
            PanZoomView.mouseReleaseEvent(self, event)
            return
        scene_pos = self._event_scene_pos(event)
        if event.button() == Qt.MouseButton.LeftButton and self.paint_drag is not None:
            self._paint_sample(scene_pos, event)
            self._finish_paint()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self.wipe_drag is not None:
            self._wipe_drag_to(scene_pos)
            self.wipe_drag = None
            self.unsetCursor()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self.roi_drag is not None:
            self._roi_drag_to(scene_pos)
            rect = self.roi_drag["rect"]
            self.roi_drag = None
            self.unsetCursor()
            self.window.set_viewer_roi(rect=rect)
            self.viewport().update()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self.roto_drawing:
            if self.roto_draw_cursor is not None:
                context = self._roto_context()
                if context is not None:
                    self.roto_draw_points.append(self._roto_data_point(scene_pos, context[1], context[3]))
                    self.roto_draw_cursor = None
                    self.viewport().update()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self.warp_drawing:
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self.warp_drag is not None:
            self._warp_drag_to(scene_pos)
            self._commit_warp_drag()
            self.unsetCursor()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self.roto_drag is not None:
            self.roto_drag["scene"] = scene_pos
            self.roto_drag["moved"] = (scene_pos - self.roto_drag["start"]).manhattanLength() > 2
            if not self.roto_drag["moved"]:
                self.roto_drag = None
                self.unsetCursor()
                event.accept()
                return
            self._commit_roto_drag()
            self.unsetCursor()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self.tracker_drag is not None:
            self.tracker_drag["scene"] = scene_pos
            self.tracker_drag["moved"] = (scene_pos-self.tracker_drag["start"]).manhattanLength() > 2
            self._commit_tracker_drag()
            self.unsetCursor()
            self.viewport().update()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self.analysis_drag is not None:
            drag = self.analysis_drag
            drag["scene"] = scene_pos
            drag["moved"] = (scene_pos - drag["start"]).manhattanLength() > 2
            if drag["moved"]:
                values = self._analysis_values()
                context = self._analysis_context()
                if values is not None and context is not None:
                    names = (("sample_x0", "sample_y0", "sample_x1", "sample_y1")
                             if context[1]["type"] == "Sampler"
                             else ("box_x", "box_y", "box_width", "box_height"))
                    self.window.command({"op": "batch", "commands": self._analysis_commands(
                        drag["key"], {name: values[name] for name in names})})
            else:
                node = self.window.dispatcher.document["nodes"].get(drag["key"])
                if node is not None:
                    self.window.preview_knobs(drag["key"], node["params"])
            self.analysis_drag = None
            self.unsetCursor(); self.viewport().update(); event.accept(); return
        if event.button() == Qt.MouseButton.LeftButton and self.flare_drag is not None:
            drag = self.flare_drag
            drag["scene"] = scene_pos
            drag["moved"] = (scene_pos - drag["start"]).manhattanLength() > 2
            if drag["moved"]:
                start = self._transform_data_point(drag["start"])
                current = self._transform_data_point(scene_pos)
                values = {"position_x": drag["base"][0] + current[0] - start[0],
                          "position_y": drag["base"][1] + current[1] - start[1]}
                commands = self._transform_commands(drag["key"], values)
                self.window.command({"op": "batch", "commands": commands})
            self.flare_drag = None
            self.unsetCursor()
            self.viewport().update()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self.transform_drag is not None:
            drag = self.transform_drag
            drag["scene"] = scene_pos
            drag["moved"] = (scene_pos - drag["start"]).manhattanLength() > 2
            if not drag["moved"]:
                self.transform_drag = None
                self.unsetCursor()
                event.accept()
                return
            values = self._transform_values(drag)
            if values is not None:
                touched = {param: values[param] for param in drag["original"]}
                self.window.command({"op": "batch", "commands": self._transform_commands(drag["key"], touched)})
            self.transform_drag = None
            self.unsetCursor()
            event.accept()
            return
        was_panning = self.pan is not None
        super().mouseReleaseEvent(event)
        if was_panning:
            # The visible scene window changed; request only the newly exposed bounding box.
            self.window.request_preview()

    # ---- region of interest, proxy badge and format masks ---------------------------------------
    def _roi_active(self):
        return self.format_rect is not None and viewer_roi(self.window.dispatcher.document)["on"]

    def _roi_fractions(self):
        if self.roi_drag is not None:
            return self.roi_drag["rect"]
        return viewer_roi(self.window.dispatcher.document)["rect"]

    def roi_scene_rect(self):
        """The ROI box in scene units (the format rectangle is the canvas at full resolution)."""
        x0, y0, x1, y1 = self._roi_fractions()
        rect = self.format_rect
        return QRectF(rect.left() + x0 * rect.width(), rect.top() + y0 * rect.height(),
                      (x1 - x0) * rect.width(), (y1 - y0) * rect.height())

    def _roi_hit(self, scene_pos):
        """Which part of the ROI box is under the pointer: a corner or edge name, "move" for the
        inside, None for outside."""
        box = self.roi_scene_rect()
        tolerance = 8.0 / max(abs(self.transform().m11()), 0.05)
        x, y = scene_pos.x(), scene_pos.y()
        if not (box.left() - tolerance <= x <= box.right() + tolerance
                and box.top() - tolerance <= y <= box.bottom() + tolerance):
            return None
        vertical = "t" if abs(y - box.top()) <= tolerance else "b" if abs(y - box.bottom()) <= tolerance else ""
        horizontal = "l" if abs(x - box.left()) <= tolerance else "r" if abs(x - box.right()) <= tolerance else ""
        return (vertical + horizontal) or "move"

    def _roi_start(self, kind, scene_pos):
        rect = list(viewer_roi(self.window.dispatcher.document)["rect"])
        return {"kind": kind, "start": scene_pos, "orig": rect, "rect": list(rect)}

    def _roi_drag_to(self, scene_pos):
        drag, canvas = self.roi_drag, self.format_rect
        x0, y0, x1, y1 = drag["orig"]
        fx = (scene_pos.x() - canvas.left()) / canvas.width()
        fy = (scene_pos.y() - canvas.top()) / canvas.height()
        dx = (scene_pos.x() - drag["start"].x()) / canvas.width()
        dy = (scene_pos.y() - drag["start"].y()) / canvas.height()
        clamp = lambda value: min(1.0, max(0.0, value))
        least = 0.01
        kind = drag["kind"]
        if kind == "new":
            ax, ay = clamp((drag["start"].x() - canvas.left()) / canvas.width()), \
                     clamp((drag["start"].y() - canvas.top()) / canvas.height())
            bx, by = clamp(fx), clamp(fy)
            x0, x1 = sorted((ax, bx))
            y0, y1 = sorted((ay, by))
        elif kind == "move":
            dx = min(1.0 - x1, max(-x0, dx))
            dy = min(1.0 - y1, max(-y0, dy))
            x0, x1, y0, y1 = x0 + dx, x1 + dx, y0 + dy, y1 + dy
        else:
            if "l" in kind:
                x0 = min(x1 - least, clamp(fx))
            if "r" in kind:
                x1 = max(x0 + least, clamp(fx))
            if "t" in kind:
                y0 = min(y1 - least, clamp(fy))
            if "b" in kind:
                y1 = max(y0 + least, clamp(fy))
        if x1 - x0 < least or y1 - y0 < least:
            return
        drag["rect"] = [x0, y0, x1, y1]
        self.viewport().update()

    def proxy_badge_text(self):
        """The tier of the picture on screen, shown in the corner of the viewer."""
        tier = int(self.last_scale) if self.last_frame_size is not None else 1
        return "" if tier <= 1 else f"proxy 1/{tier}"

    def _paint_mask(self, painter, rect, mask, mode):
        """Paint the format mask over `rect` (scene units): the outside of the chosen aspect is
        darkened (`half`, `full`) or bounded by a line (`lines`). Display only."""
        if mode == "none":
            return
        width, height = rect.width(), rect.height()
        painter.save()
        if mode == "lines":
            pen = QPen(QColor("#e8e8f0"))
            pen.setCosmetic(True)
            painter.setPen(pen)
            x0, y0, x1, y1 = viewframe.mask_rect(mask, width, height)
            for edge in (y0, y1):
                if 0.0 < edge < height:
                    painter.drawLine(QLineF(rect.left(), rect.top() + edge, rect.right(), rect.top() + edge))
            for edge in (x0, x1):
                if 0.0 < edge < width:
                    painter.drawLine(QLineF(rect.left() + edge, rect.top(), rect.left() + edge, rect.bottom()))
        else:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, round(255 * viewframe.MASK_DARKEN[mode])))
            for bx0, by0, bx1, by1 in viewframe.mask_bars(mask, round(width), round(height)):
                painter.drawRect(QRectF(rect.left() + bx0, rect.top() + by0, bx1 - bx0, by1 - by0))
        painter.restore()

    def _draw_roi_and_masks(self, painter):
        document = self.window.dispatcher.document
        masks = viewer_masks(document)
        self._paint_mask(painter, self.format_rect, masks["mask"], masks["mode"])
        if self._roi_active():
            box = self.roi_scene_rect()
            painter.save()
            pen = QPen(QColor("#f4c542"), 1)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(box)
            size = 4.0 / max(abs(self.transform().m11()), 0.05)
            painter.setBrush(QColor("#f4c542"))
            for x in (box.left(), box.center().x(), box.right()):
                for y in (box.top(), box.center().y(), box.bottom()):
                    if (x, y) != (box.center().x(), box.center().y()):
                        painter.drawRect(QRectF(x - size, y - size, size * 2, size * 2))
            painter.restore()
        labels = [text for text in (self.proxy_badge_text(), "ROI" if self._roi_active() else "") if text]
        if labels:
            painter.save()
            painter.resetTransform()
            painter.setPen(QColor("#f4c542"))
            painter.drawText(self.viewport().width() - 100, 18, "  ".join(labels))
            painter.restore()

    def draw_format_overlay(self, scene_rect):
        """Record the display window so drawForeground can paint Nuke-style format guides
        around it. Deliberately not scene items: itemsBoundingRect() is used elsewhere
        (see the canvas-resize regression test) to mean "the rendered image", and scene
        items would perturb that with zoom-dependent, overlay-shaped geometry."""
        self.format_rect = scene_rect
        self.viewport().update()

    def drawForeground(self, painter, rect):
        super().drawForeground(painter, rect)
        if self.paint_drag is not None and self.paint_drag["points"]:
            painter.save()
            pen = QPen(QColor("#ff5c48"), max(1.0, self.paint_size))
            pen.setCapStyle(Qt.PenCapStyle.RoundCap); pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            mapped = [self._roto_scene_point(point, 1) for point in self.paint_drag["points"]]
            if len(mapped) == 1: painter.drawPoint(mapped[0])
            else:
                path = QPainterPath(mapped[0])
                for point in mapped[1:]: path.lineTo(point)
                painter.drawPath(path)
            painter.restore()
        if self._wipe_active():
            self._draw_wipe_b(painter)
            self._draw_wipe_handles(painter)
        context = self._roto_context()
        if context is not None:
            _, _, payload, tier = context
            resolved = self._roto_resolved(context)
            if self.roto_drag is not None:
                # Keep the wireframe responsive while the document remains unchanged.  The
                # payload is only replaced on release, so a cancelled drag cannot leave a partial
                # command behind.
                shape_index, point_index = self.roto_drag["point"]
                if shape_index < len(resolved) and point_index < len(resolved[shape_index]["points"]):
                    x, y = self._roto_data_point(self.roto_drag["scene"], context[1], tier)
                    resolved[shape_index]["points"][point_index]["x"] = x
                    resolved[shape_index]["points"][point_index]["y"] = y
            painter.save()
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            for shape_index, shape in enumerate(resolved):
                points = shape["points"]
                if len(points) < 3:
                    continue
                path = QPainterPath(self._roto_scene_point(points[0], tier))
                for index in range(len(points)):
                    start, end = points[index], points[(index + 1) % len(points)]
                    path.cubicTo(self._roto_scene_point({"x": start["x"] + start["out_x"],
                                                          "y": start["y"] + start["out_y"]}, tier),
                                 self._roto_scene_point({"x": end["x"] + end["in_x"],
                                                         "y": end["y"] + end["in_y"]}, tier),
                                 self._roto_scene_point(end, tier))
                pen = QPen(QColor("#ff986f" if shape["mode"] == "subtract" else "#58d7ff"), 2)
                pen.setCosmetic(True)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawPath(path)
                for point in points:
                    center = self._roto_scene_point(point, tier)
                    radius = 5.0 / max(abs(self.transform().m11()), 0.05)
                    painter.setBrush(QColor("#202127"))
                    painter.drawEllipse(center, radius, radius)
            if self.roto_drawing:
                origin_x = self.format_rect.left() if self.format_rect is not None else 0.0
                origin_y = self.format_rect.top() if self.format_rect is not None else 0.0
                draw_points = [QPointF(origin_x + x, origin_y + y)
                               for x, y in self.roto_draw_points]
                if self.roto_draw_cursor is not None:
                    draw_points.append(self.roto_draw_cursor)
                if draw_points:
                    path = QPainterPath(draw_points[0])
                    for point in draw_points[1:]:
                        path.lineTo(point)
                    if len(draw_points) >= 3:
                        path.lineTo(draw_points[0])
                    pen = QPen(QColor("#f4ce63"), 2)
                    pen.setStyle(Qt.PenStyle.DashLine)
                    pen.setCosmetic(True)
                    painter.setPen(pen)
                    painter.drawPath(path)
                    painter.setBrush(QColor("#f4ce63"))
                    for point in draw_points:
                        radius = 4.0 / max(abs(self.transform().m11()), 0.05)
                        painter.drawEllipse(point, radius, radius)
            if self.roto_drag is not None:
                point = self.roto_drag["scene"]
                radius = 6.0 / max(abs(self.transform().m11()), 0.05)
                painter.setBrush(QColor("#f4ce63"))
                painter.drawEllipse(point, radius, radius)
            painter.restore()
        self._draw_warp_overlay(painter)
        self._draw_tracker_overlay(painter)
        transform_context = self._transform_context()
        if transform_context is not None:
            _, node = transform_context
            values = self._transform_values()
            if values is not None:
                w, h = self.format_rect.width(), self.format_rect.height()
                corners = handles2d.box_corners(w, h, values["translate_x"], values["translate_y"],
                                                values["rotate"], values["scale"], values["center_x"], values["center_y"])
                pivot = handles2d.pivot_point(values["translate_x"], values["translate_y"],
                                               values["center_x"], values["center_y"])
                radius = 1.15 * math.hypot(corners[0][0] - pivot[0], corners[0][1] - pivot[1])
                painter.save()
                painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
                pen = QPen(QColor("#c9e26a"), 2)
                pen.setCosmetic(True)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                polygon = QPolygonF([self._transform_scene_point(x, y) for x, y in corners])
                painter.drawPolygon(polygon)
                ring_pen = QPen(QColor("#c9e26a"), 1)
                ring_pen.setStyle(Qt.PenStyle.DashLine)
                ring_pen.setCosmetic(True)
                painter.setPen(ring_pen)
                pivot_scene = self._transform_scene_point(*pivot)
                painter.drawEllipse(pivot_scene, radius, radius)
                handle_radius = 5.0 / max(abs(self.transform().m11()), 0.05)
                painter.setPen(pen)
                painter.setBrush(QColor("#202127"))
                for x, y in corners + handles2d.edge_midpoints(corners):
                    point = self._transform_scene_point(x, y)
                    painter.drawRect(QRectF(point.x() - handle_radius, point.y() - handle_radius,
                                             handle_radius * 2, handle_radius * 2))
                painter.setBrush(QColor("#c9e26a"))
                painter.drawEllipse(pivot_scene, handle_radius, handle_radius)
                painter.restore()
                if self.transform_drag is not None:
                    kind = self.transform_drag["kind"]
                    label = (f"tx {values['translate_x']:.1f}  ty {values['translate_y']:.1f}" if kind == "translate" else
                             f"pivot {values['center_x']:.1f}, {values['center_y']:.1f}" if kind == "pivot" else
                             f"rotate {values['rotate']:.2f}°" if kind == "rotate" else
                             f"scale {values['scale']:.4f}")
                    painter.save()
                    painter.resetTransform()
                    point = self.mapFromScene(self.transform_drag["scene"])
                    painter.setPen(QColor("#c9e26a"))
                    painter.drawText(point.x() + 12, point.y() - 12, label)
                    painter.restore()
        analysis_context = self._analysis_context()
        if analysis_context is not None:
            key, node = analysis_context
            values = self._analysis_values() or node["params"]
            zoom = max(abs(self.transform().m11()), 0.05)
            radius = 6.0 / zoom
            painter.save()
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            pen = QPen(QColor("#65d9e8"), 2); pen.setCosmetic(True)
            painter.setPen(pen); painter.setBrush(Qt.BrushStyle.NoBrush)
            if node["type"] == "Sampler":
                a = self._transform_scene_point(values["sample_x0"], values["sample_y0"])
                b = self._transform_scene_point(values["sample_x1"], values["sample_y1"])
                painter.drawLine(a, b)
                painter.setBrush(QColor("#202127"))
                painter.drawEllipse(a, radius, radius); painter.drawEllipse(b, radius, radius)
            else:
                x, y = values["box_x"], values["box_y"]
                w = values["box_width"] if values["box_width"] > 0 else 64.0
                h = values["box_height"] if values["box_height"] > 0 else 64.0
                corners = ((x, y), (x+w, y), (x+w, y+h), (x, y+h))
                rect = QRectF(self._transform_scene_point(x, y), self._transform_scene_point(x+w, y+h)).normalized()
                painter.drawRect(rect)
                grips = ("nw", "n", "ne", "e", "se", "s", "sw", "w")
                for part, point in zip(grips, ((x,y),(x+w/2,y),(x+w,y),(x+w,y+h/2),
                                                (x+w,y+h),(x+w/2,y+h),(x,y+h),(x,y+h/2))):
                    center = self._transform_scene_point(*point)
                    painter.setBrush(QColor("#202127")); painter.drawRect(QRectF(center.x()-radius, center.y()-radius, radius*2, radius*2))
            painter.restore()
        flare_context = self._flare_context()
        if flare_context is not None:
            values = self._flare_values(self.flare_drag)
            if values is not None:
                centre = self._transform_scene_point(values["position_x"], values["position_y"])
                radius = 8.0 / max(abs(self.transform().m11()), 0.05)
                painter.save()
                painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
                pen = QPen(QColor("#ffd26b"), 2); pen.setCosmetic(True)
                painter.setPen(pen); painter.setBrush(QColor("#202127"))
                painter.drawEllipse(centre, radius, radius)
                painter.drawLine(QPointF(centre.x() - radius * 1.6, centre.y()),
                                 QPointF(centre.x() + radius * 1.6, centre.y()))
                painter.drawLine(QPointF(centre.x(), centre.y() - radius * 1.6),
                                 QPointF(centre.x(), centre.y() + radius * 1.6))
                painter.restore()
                if self.flare_drag is not None:
                    painter.save(); painter.resetTransform()
                    point = self.mapFromScene(self.flare_drag["scene"])
                    painter.setPen(QColor("#ffd26b"))
                    painter.drawText(point.x() + 12, point.y() - 12,
                                     f"position {values['position_x']:.1f}, {values['position_y']:.1f}")
                    painter.restore()
        if self.format_rect is None:
            return
        painter.save()
        pen = QPen(QColor("#6d6d78"))
        pen.setStyle(Qt.PenStyle.DotLine)
        pen.setCosmetic(True)  # a constant 1px dashed line regardless of zoom
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(self.format_rect)
        painter.restore()
        painter.save()
        painter.resetTransform()  # switch to raw viewport pixels for constant-size text
        corner = self.mapFromScene(self.format_rect.bottomRight())
        painter.setPen(QColor("#9a9aa4"))
        painter.drawText(corner.x() + 4, corner.y() + 14,
                         f"{int(self.format_rect.width())} x {int(self.format_rect.height())}")
        painter.restore()
        self._draw_roi_and_masks(painter)

    def mousePressEvent(self, event):
        if self.starts_pan(event):
            # A pan never places a tracker, a roto point or a roto drag.
            PanZoomView.mousePressEvent(self, event)
            return
        scene_pos = self._event_scene_pos(event)
        if event.button() == Qt.MouseButton.LeftButton and self.zdefocus_picking:
            self.window.zdefocus_pick(self.zdefocus_picking, math.floor(scene_pos.x()), math.floor(scene_pos.y()))
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self.focus_picking:
            self.window.focus_pick(self.focus_picking, math.floor(scene_pos.x()), math.floor(scene_pos.y()))
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self.crypto_picking:
            # Each click adds the object under the cursor to the matte list; picking stays on until Esc.
            self.window.crypto_pick(self.crypto_picking, math.floor(scene_pos.x()), math.floor(scene_pos.y()))
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self.tracker_picking:
            context = self.window._tracker_context()
            if context is not None:
                point = self._tracker_data_point(scene_pos)
                self.tracker_picking = False
                self.unsetCursor()
                self.window.add_tracker_point(point)
                event.accept()
                return
        tracker_context = self.window._tracker_context()
        if event.button() == Qt.MouseButton.LeftButton and tracker_context is not None:
            hit = self._tracker_hit(tracker_context, scene_pos)
            if hit is not None:
                if hit["kind"] == "point":
                    self.window._tracker_selected_index = hit["index"]
                hit.update(start=scene_pos, scene=scene_pos, moved=False,
                           frame=int(self.window.dispatcher.document["time"]["current"]))
                self.tracker_drag = hit
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
                return
        context = self._roto_context()
        if event.button() == Qt.MouseButton.LeftButton and context is not None:
            if self.roto_drawing:
                self.roto_draw_cursor = scene_pos
                event.accept()
                return
            resolved = self._roto_resolved(context)
            hit = self._roto_hit_point(scene_pos, resolved, context[3])
            if hit is not None:
                self.roto_key = context[0]
                self.roto_drag = {"point": hit, "scene": scene_pos, "start": scene_pos,
                                  "moved": False}
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
                return
        warp_context = self._warp_context()
        if event.button() == Qt.MouseButton.LeftButton and warp_context is not None:
            if self.warp_drawing:
                point = self._roto_data_point(scene_pos, warp_context[1], warp_context[3])
                self.warp_draw_points.append(QPointF(*point))
                self.warp_draw_cursor = scene_pos
                self.viewport().update()
                event.accept()
                return
            hit = self._warp_hit(warp_context, scene_pos)
            if hit is not None:
                hit.update(start=scene_pos, scene=scene_pos, moved=False)
                self.warp_drag = hit
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
                return
        if event.button() == Qt.MouseButton.LeftButton and self._paint_context() is not None:
            self._paint_sample(scene_pos, event, start=True)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            context = self._analysis_context()
            if context is not None:
                key, node = context; p = node["params"]
                point = self._transform_data_point(scene_pos)
                hit = 12.0 / max(abs(self.transform().m11()), 0.05)
                part = None
                if node["type"] == "Sampler":
                    endpoints = [(p["sample_x0"], p["sample_y0"]), (p["sample_x1"], p["sample_y1"])]
                    distances = [math.hypot(point[0]-x, point[1]-y) for x, y in endpoints]
                    if min(distances) <= hit:
                        part = str(distances.index(min(distances)))
                    else:
                        (x0,y0),(x1,y1) = endpoints; vx,vy=x1-x0,y1-y0
                        t=max(0.0,min(1.0,((point[0]-x0)*vx+(point[1]-y0)*vy)/max(vx*vx+vy*vy,1e-12)))
                        if math.hypot(point[0]-(x0+t*vx),point[1]-(y0+t*vy)) <= hit:
                            part = "line"
                else:
                    x,y=p["box_x"],p["box_y"]
                    w=p["box_width"] if p["box_width"] > 0 else 64.0
                    h=p["box_height"] if p["box_height"] > 0 else 64.0
                    grips = (("nw",x,y),("n",x+w/2,y),("ne",x+w,y),("e",x+w,y+h/2),
                             ("se",x+w,y+h),("s",x+w/2,y+h),("sw",x,y+h),("w",x,y+h/2))
                    candidates=[(name,math.hypot(point[0]-gx,point[1]-gy)) for name,gx,gy in grips]
                    name,distance=min(candidates,key=lambda item:item[1])
                    if distance <= hit: part=name
                    elif x-hit <= point[0] <= x+w+hit and y-hit <= point[1] <= y+h+hit: part="body"
                if part is not None:
                    original=dict(p)
                    if node["type"] == "MinColor" and original["box_width"] <= 0 and original["box_height"] <= 0:
                        original["box_width"] = original["box_height"] = 64.0
                    self.analysis_drag={"key":key,"part":part,"start":scene_pos,"scene":scene_pos,
                                        "moved":False,"original":original}
                    self.setCursor(Qt.CursorShape.ClosedHandCursor)
                    event.accept(); return
        if event.button() == Qt.MouseButton.LeftButton:
            context = self._transform_context()
            if context is not None:
                key, node = context
                p = node["params"]
                w, h = self.format_rect.width(), self.format_rect.height()
                corners = handles2d.box_corners(w, h, p["translate_x"], p["translate_y"], p["rotate"],
                                                p["scale"], p["center_x"], p["center_y"])
                pivot = handles2d.pivot_point(p["translate_x"], p["translate_y"], p["center_x"], p["center_y"])
                point = self._transform_data_point(scene_pos)
                hit_radius = 12.0 / max(abs(self.transform().m11()), 0.05)
                kind = None
                if any(math.hypot(point[0] - x, point[1] - y) <= hit_radius for x, y in corners):
                    kind = "scale"
                elif any(math.hypot(point[0] - x, point[1] - y) <= hit_radius
                         for x, y in handles2d.edge_midpoints(corners)):
                    kind = "scale"
                else:
                    ring_radius = 1.15 * math.hypot(corners[0][0] - pivot[0], corners[0][1] - pivot[1])
                    distance = math.hypot(point[0] - pivot[0], point[1] - pivot[1])
                    if abs(distance - ring_radius) < 10.0 / max(abs(self.transform().m11()), 0.05):
                        kind = "rotate"
                    elif (handles2d.point_in_quad(point[0], point[1], corners)
                          or distance <= hit_radius):
                        kind = "pivot" if event.modifiers() & Qt.KeyboardModifier.ControlModifier else "translate"
                if kind is not None:
                    touched = {"translate": ("translate_x", "translate_y"),
                               "pivot": ("center_x", "center_y", "translate_x", "translate_y"),
                               "rotate": ("rotate",), "scale": ("scale",)}[kind]
                    self.transform_drag = {"kind": kind, "key": key, "start": scene_pos, "scene": scene_pos,
                                           "moved": False, "pivot": pivot,
                                           "original": {param: p[param] for param in touched}}
                    self.setCursor(Qt.CursorShape.ClosedHandCursor)
                    event.accept()
                    return
        if event.button() == Qt.MouseButton.LeftButton:
            context = self._flare_context()
            values = self._flare_values() if context is not None else None
            if values is not None:
                point = self._transform_data_point(scene_pos)
                position = (values["position_x"], values["position_y"])
                hit_radius = 14.0 / max(abs(self.transform().m11()), 0.05)
                if math.hypot(point[0] - position[0], point[1] - position[1]) <= hit_radius:
                    dx, dy = self._flare_link_delta(context[1])
                    self.flare_drag = {"key": context[0], "start": scene_pos, "scene": scene_pos,
                                       "moved": False, "position": position,
                                       "base": (values["position_x"] - dx, values["position_y"] - dy)}
                    self.setCursor(Qt.CursorShape.ClosedHandCursor)
                    event.accept()
                    return
        if event.button() == Qt.MouseButton.LeftButton and self._roi_active():
            shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            kind = "new" if shift else self._roi_hit(scene_pos)
            if kind is not None:
                self.roi_drag = self._roi_start(kind, scene_pos)
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
                return
        # The wipe comes last on purpose: a roto point, a tracker pick or a Transform handle under
        # the pointer has already taken the click above, so the compare never steals one.
        if event.button() == Qt.MouseButton.LeftButton and self._wipe_active():
            hit = self._wipe_hit(scene_pos)
            if hit is not None:
                if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                    self.reset_wipe()
                else:
                    centre = self._wipe_scene_geometry()[0]
                    offset = ((scene_pos.x() - centre.x(), scene_pos.y() - centre.y())
                              if hit == "line" else (0.0, 0.0))
                    self.wipe_drag = {"kind": hit, "offset": offset}
                    self.setCursor(Qt.CursorShape.ClosedHandCursor)
                    self._wipe_drag_to(scene_pos)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        # The pixel readout follows every move, including roto edits: the point of the numbers
        # is to read the picture under the pointer, which is exactly what an artist placing a
        # shape wants. This is the viewer's only mouse-move handler; a second definition would
        # silently replace it (the readout was dead for exactly that reason once).
        scene_pos = self._event_scene_pos(event)
        if self.wipe_drag is not None and self.pan is None:
            self._wipe_drag_to(scene_pos)
            self._update_pixel_readout(event)
            event.accept()
            return
        if self.roi_drag is not None and self.pan is None:
            self._roi_drag_to(scene_pos)
            self._update_pixel_readout(event)
            event.accept()
            return
        if self.roto_drawing and self.pan is None:
            self.roto_draw_cursor = scene_pos
            self.viewport().update()
            self._update_pixel_readout(event)
            event.accept()
            return
        if self.warp_drawing and self.pan is None:
            self.warp_draw_cursor = scene_pos
            self.viewport().update()
            self._update_pixel_readout(event)
            event.accept()
            return
        if self.warp_drag is not None and self.pan is None:
            self._warp_drag_to(scene_pos)
            self._update_pixel_readout(event)
            event.accept()
            return
        if self.roto_drag is not None and self.pan is None:
            self.roto_drag["scene"] = scene_pos
            self.roto_drag["moved"] = (scene_pos - self.roto_drag["start"]).manhattanLength() > 2
            self.viewport().update()
            self._update_pixel_readout(event)
            event.accept()
            return
        if self.tracker_drag is not None and self.pan is None:
            self.tracker_drag["scene"] = scene_pos
            self.tracker_drag["moved"] = (scene_pos-self.tracker_drag["start"]).manhattanLength() > 2
            self.viewport().update()
            self._update_pixel_readout(event)
            event.accept()
            return
        if self.flare_drag is not None and self.pan is None:
            self.flare_drag["scene"] = scene_pos
            self.flare_drag["moved"] = (scene_pos - self.flare_drag["start"]).manhattanLength() > 2
            self.viewport().update()
            self._update_pixel_readout(event)
            event.accept()
            return
        if self.analysis_drag is not None and self.pan is None:
            self.analysis_drag["scene"] = scene_pos
            self.analysis_drag["moved"] = (scene_pos - self.analysis_drag["start"]).manhattanLength() > 2
            values = self._analysis_values()
            context = self._analysis_context()
            if values is not None and context is not None:
                names = (("sample_x0", "sample_y0", "sample_x1", "sample_y1")
                         if context[1]["type"] == "Sampler"
                         else ("box_x", "box_y", "box_width", "box_height"))
                self.window.preview_knobs(self.analysis_drag["key"], {name: values[name] for name in names})
            self.viewport().update(); self._update_pixel_readout(event); event.accept(); return
        if self.transform_drag is not None and self.pan is None:
            self.transform_drag["scene"] = scene_pos
            self.transform_drag["moved"] = (scene_pos - self.transform_drag["start"]).manhattanLength() > 2
            self.viewport().update()
            self._update_pixel_readout(event)
            event.accept()
            return
        if self.paint_drag is not None and self.pan is None:
            self._paint_sample(scene_pos, event)
            self._update_pixel_readout(event)
            event.accept()
            return
        self._handling_mouse_move = True
        try:
            super().mouseMoveEvent(event)
        finally:
            self._handling_mouse_move = False
        self._update_pixel_readout(event)

    def tabletEvent(self, event):
        context = self._paint_context()
        if context is None and self.paint_drag is None:
            event.ignore()
            return
        pos = self.mapToScene(event.position().toPoint())
        kind = event.type()
        if kind == QEvent.Type.TabletPress:
            self._paint_sample(pos, event, start=True)
        elif kind == QEvent.Type.TabletMove and self.paint_drag is not None:
            self._paint_sample(pos, event)
        elif kind == QEvent.Type.TabletRelease and self.paint_drag is not None:
            self._paint_sample(pos, event)
            self._finish_paint()
        event.accept()


def dot_grab_radius(graph):
    """Scene radius around a Dot centre that grabs the Dot: its own size, or ~8 screen px."""
    zoom = graph.transform().m11() if graph is not None else 1.0
    return max(10.0, 8.0 / max(zoom, 1e-3))


class Port(QGraphicsEllipseItem):
    def __init__(self, node, slot, x, y, label_text=None, label_center=None, output_name="rgba"):
        # The hit target is intentionally much larger than the visible socket.
        # A 12 px drawn port was too easy to miss while the label/noodle occupied
        # nearby pixels, making wiring feel randomly broken.
        radius = 8 if node.is_dot else 13
        super().__init__(-radius, -radius, radius * 2, radius * 2, node)
        self.node, self.slot, self.output_name = node, slot, output_name
        self.setPos(x, y)
        self.setBrush(Qt.BrushStyle.NoBrush)
        self.setPen(QPen(Qt.PenStyle.NoPen))
        self.setZValue(3)
        visible = QGraphicsEllipseItem(-6, -6, 12, 12, self)
        visible.setBrush(QColor("#1b1b1d"))
        visible.setPen(QPen(QColor("#a4a4ae"), 1.5))
        visible.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._press_scene = None
        self._dragging = False
        self._rewire_input = None
        self.setToolTip("Output: drag to an input, or click then click an input" if slot is None else
                         f"Input {slot}: drag to an output; drag an output here; click to pick up and rewire; right-click disconnects")
        if (slot or label_text) and not node.is_dot:
            label = QGraphicsSimpleTextItem(label_text or slot, node)
            label.setBrush(QColor("#b4b4bd"))
            if label_center is None:
                label.setPos(x + 9, y - 18)
            else:
                bounds = label.boundingRect()
                label.setPos(label_center[0] - bounds.width() / 2, label_center[1] - bounds.height() / 2)

    def shape(self):
        path = super().shape()
        if self.node.is_dot:
            # A Dot's two sockets sit on its rim; zoomed out, their generous hit circles swallow
            # the whole Dot. The Dot body always wins: carve it out of both sockets.
            radius = dot_grab_radius(self.node.graph)
            body = QPainterPath()
            body.addEllipse(self.mapFromItem(self.node, QPointF(10, 10)), radius, radius)
            path = path.subtracted(body)
        return path

    def mousePressEvent(self, event):
        graph = self.node.graph
        self._press_scene = event.scenePos()
        self._dragging = False
        self._rewire_input = None
        if event.button() == Qt.MouseButton.RightButton and self.slot is not None:
            graph.cancel_wire()
            graph.window.defer_command({"op": "connect", "id": self.node.key, "input": self.slot, "source": None})
        elif self.slot is None:
            if graph.wire_input:
                graph.finish_input_wire(self.node.key, self.output_name)
            else:
                graph.start_wire(self.node.key, output=self.output_name)
                graph.window.statusBar().showMessage("Connect: drag to an input port · Esc cancels")
        elif graph.wire_source:
            graph.finish_wire_at(event.scenePos())
        else:
            current_source = graph.window.graph_nodes()[self.node.key]["inputs"][self.slot]
            if current_source:
                # An input is an endpoint: dragging it always seeks a new output and
                # replaces *this* input only after a valid drop.  Keep the existing
                # connection visible until then so a missed gesture cannot damage a comp.
                graph.start_input_wire(self.node.key, self.slot)
                graph.window.statusBar().showMessage("Input picked up: drag to an output · Esc or empty space keeps the original")
            else:
                graph.start_input_wire(self.node.key, self.slot)
                graph.window.statusBar().showMessage("Connect: drag to an output port · Esc cancels")
        event.accept()

    def mouseMoveEvent(self, event):
        if self._press_scene is not None and (event.scenePos() - self._press_scene).manhattanLength() > 4:
            self._dragging = True
        if graph := self.node.graph:
            if graph.wire_source or graph.wire_input:
                graph.update_pending_edge(event.scenePos())
        event.accept()

    def mouseReleaseEvent(self, event):
        # Ports own the mouse during a drag, so the destination never receives its
        # own press event. Resolve it here for the direct Nuke-style drag gesture.
        if self._dragging:
            graph = self.node.graph
            if graph.wire_source:
                graph.finish_wire_at(event.scenePos())
            elif graph.wire_input:
                graph.finish_input_wire_at(event.scenePos())
        self._press_scene = None
        self._dragging = False
        self._rewire_input = None
        event.accept()


NODE_WIDTH, NODE_HEIGHT = 190, 52
# A thumbnail band sits under the title. Nuke shows a postage stamp on the node itself; the band
# keeps the title and sockets where they always were and simply makes the card taller.
THUMB_WIDTH, THUMB_HEIGHT = 174, 72
NO_THUMBNAIL_TYPES = ("Dot", "Viewer", "Backdrop")
BACKDROP_TITLE_HEIGHT = 30
BACKDROP_MIN_SIZE = 80


def wants_thumbnail(node, thumbnails=True):
    """A node shows a stamp when the machine allows it and the node's own Node-tab switch is on."""
    return bool(thumbnails) and node["type"] not in NO_THUMBNAIL_TYPES and node_thumbnail(node)


# 3D nodes read as a family at a glance, as in Nuke: everything that lives in 3D space has rounded
# ends, and the nodes that are a point in that space rather than a process (scene, light, camera)
# are full circles.
CIRCLE_TYPES = ("Scene3D", "Light3D", "Camera3D", "ReadUSDCamera3D", "ReadAlembicCamera3D")
# Every node type that carries a transform (the 2026-09-19 3D UX direction): the properties
# panel shows read-only local/world matrix readouts for these (see `scene3d.local_and_world_matrix`).
MATRIX_READOUT_TYPES = ("Card3D", "Cube3D", "Sphere3D", "Cylinder3D", "Scene3D", "Axis3D",
                        "TransformGeo3D", "MergeGeo3D", "ParticleEmitter3D", "Plume3D", "ReadVDB3D", "Camera3D", "Light3D")
CIRCLE_DIAMETER = 112
CIRCLE_PORT_STEP = 24.5  # degrees between neighbouring input sockets on the rim


def node_form(node):
    kind = node["type"]
    if kind == "Dot":
        return "dot"
    if kind == "Backdrop":
        return "backdrop"
    if kind in CIRCLE_TYPES:
        return "circle"
    return "round" if kind.endswith("3D") else "card"


def node_size(node, thumbnails):
    form = node_form(node)
    if form == "dot":
        return 20, 20
    if form == "backdrop":
        params = {**SPECS["Backdrop"]["params"], **node.get("params", {})}
        return params["width"], params["height"]
    if form == "circle":  # a circle has no band to hang a postage stamp in
        return CIRCLE_DIAMETER, CIRCLE_DIAMETER
    return NODE_WIDTH, NODE_HEIGHT + (THUMB_HEIGHT + 6 if wants_thumbnail(node, thumbnails) else 0)


def node_height(node, thumbnails):
    return node_size(node, thumbnails)[1]


def circle_port_positions(count, diameter=CIRCLE_DIAMETER):
    """Rim positions for `count` input sockets, fanned symmetrically about the top of a circle.

    Returns (x, y, angle) per socket with the angle in degrees clockwise from straight up, so a
    label can be pushed outward along the same ray.
    """
    radius = diameter / 2
    step = min(CIRCLE_PORT_STEP, 170.0 / max(1, count - 1))
    angles = [(index - (count - 1) / 2) * step for index in range(count)]
    return [(radius + radius * math.sin(math.radians(angle)),
             radius - radius * math.cos(math.radians(angle)), angle) for angle in angles]


def thumbnail_key(document, target, frame, view):
    """What decides a node's thumbnail: its upstream graph, the frame and the view.

    Deliberately narrower than the display-cache key. Node positions, the viewed node and every
    unrelated branch are left out, so dragging a node or editing a sibling never re-renders a
    thumbnail whose picture cannot have changed.
    """
    nodes = document["nodes"]
    expressions = document.get("expressions") or {}
    # A formula may read any node's knob, so with expressions present every node can matter.
    upstream, stack = {}, (list(nodes) if expressions else [target])
    while stack:
        key = stack.pop()
        if key in upstream or key not in nodes:
            continue
        node = nodes[key]
        # Only what changes pixels: position, name, label and the stamp switch never do.
        upstream[key] = {name: value for name, value in node.items()
                         if name not in ("pos", "name", "label", "thumbnail")}
        stack.extend(source for source in node["inputs"].values() if source)
    curves = document.get("animation", {}).get("curves", {})
    node_data = document.get("node_data") or {}
    payload = {"nodes": upstream, "curves": {key: curves[key] for key in upstream if key in curves},
               "node_data": {key: node_data[key] for key in upstream if key in node_data},
               "expressions": expressions, "frame": int(frame), "view": view, "target": target}
    return json.dumps(payload, sort_keys=True, default=str)


def top_aligned(page):
    """Keep a form's rows packed at the top of a tab instead of spread over its height."""
    holder = QWidget()
    layout = QVBoxLayout(holder)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(page)
    layout.addStretch(1)
    return holder


class LabelEdit(QPlainTextEdit):
    """Multi-line label knob. QPlainTextEdit has no editingFinished, so focus-out stands in for it,
    matching how the single-line knobs commit."""
    finished = Signal()

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        self.finished.emit()


class BackdropGrip(QGraphicsRectItem):
    """The corner handle that resizes a Backdrop; the new size is written when the drag ends."""
    SIZE = 18

    def __init__(self, backdrop):
        super().__init__(0, 0, self.SIZE, self.SIZE, backdrop)
        self.backdrop = backdrop
        self.setPen(QPen(Qt.PenStyle.NoPen))
        self.setBrush(Qt.BrushStyle.NoBrush)
        self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        self.setToolTip("Drag to resize the backdrop")
        self.place()

    def place(self):
        rect = self.backdrop.rect()
        self.setPos(rect.width() - self.SIZE, rect.height() - self.SIZE)

    def paint(self, painter, option, widget=None):
        painter.save()
        painter.setPen(QPen(QColor(255, 255, 255, 150), 1.5))
        for step in (5, 10, 15):
            painter.drawLine(self.SIZE - step, self.SIZE - 2, self.SIZE - 2, self.SIZE - step)
        painter.restore()

    def mousePressEvent(self, event):
        self.grab_offset = event.pos()   # where in the grip it was taken, so the corner does not jump
        event.accept()

    def mouseMoveEvent(self, event):
        corner = self.backdrop.mapFromScene(event.scenePos()) + (QPointF(self.SIZE, self.SIZE) - self.grab_offset)
        self.backdrop.setRect(0, 0, max(BACKDROP_MIN_SIZE, corner.x()), max(BACKDROP_MIN_SIZE, corner.y()))
        self.place()

    def mouseReleaseEvent(self, event):
        backdrop = self.backdrop
        rect = backdrop.rect()
        commands = [{"op": "set", "id": backdrop.key, "param": name, "value": int(round(value))}
                    for name, value in (("width", rect.width()), ("height", rect.height()))]
        # Defer: the resulting rebuild deletes this very item, mid-event.
        QTimer.singleShot(0, lambda: backdrop.graph.window.command({"op": "batch", "commands": commands}, render=False))
        event.accept()


class NodeItem(QGraphicsRectItem):
    def __init__(self, graph, key, node):
        self.form = node_form(node)
        self.is_dot = self.form == "dot"
        self.is_backdrop = self.form == "backdrop"
        self.thumbnail = None
        width, height = node_size(node, graph.window.show_thumbnails)
        super().__init__(0, 0, width, height)
        self.graph, self.key = graph, key
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setPos(*node["pos"])
        self.setBrush(QColor("#303033" if not self.is_dot else "#23242a"))
        self.setPen(QPen(QColor(COLORS[node["type"]]), 1.5))
        self.disabled = bool(node.get("disabled", False))
        if self.disabled:
            # Keep the graph readable while making bypassed processing unmistakable.  The
            # opacity applies to the card, title, and sockets; paint() adds the persistent X.
            self.setOpacity(0.52)
        if self.is_backdrop:
            # A backdrop sits behind every node and has no sockets: it only frames and labels.
            self.setZValue(-10)
            self.inputs = {}
            self.output = None
            params = node["params"]
            self.tint = QColor.fromRgbF(*(min(1.0, max(0.0, params[c])) for c in ("red", "green", "blue")))
            self.caption = (node_label(node) or node["name"]).splitlines()[0]
            self.setPen(QPen(self.tint.lighter(150), 1.5))
            self.setToolTip(f"{node['name']} (Backdrop)\nDrag the title to move it with the nodes inside")
            self.grip = BackdropGrip(self)
            return
        if self.is_dot:
            # Dots are graph routing points, not miniature processing cards.
            # Keep them compact and put their sockets on the vertical noodle path.
            self.setRect(0, 0, 20, 20)
            # Noodles live below nodes.  A Dot is deliberately above them so its small circular
            # control point remains visible where a Ctrl-drag inserts it into a connection.
            self.setZValue(5)
            self.inputs = {"input": Port(self, "input", 10, 0)}
            self.output = Port(self, None, 10, 20)
            self.outputs = {"rgba": self.output}
            return
        title = QGraphicsSimpleTextItem(node["name"][:26], self)
        title.setBrush(QColor("#eeeef2"))
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        if self.form == "circle":
            # The name has to fit the chord it sits on, so it shrinks rather than spilling past the rim.
            while title.boundingRect().width() > width - 14 and title_font.pointSize() > 9:
                title_font.setPointSize(title_font.pointSize() - 1)
                title.setFont(title_font)
            text = title.text()
            while title.boundingRect().width() > width - 14 and len(text) > 4:
                # Still too long at the smallest readable size: keep the end, which carries the
                # number that tells two nodes apart. The tooltip has the whole name.
                text = text[:-1]
                title.setText(text[:len(text) - 2] + "…" + node["name"][-2:])
            title.setPos((width - title.boundingRect().width()) / 2,
                         height / 2 - title.boundingRect().height() + 4)
        else:
            title.setPos((width - title.boundingRect().width()) / 2, 5)
        # A Node-tab label draws under the name, as in Nuke. Only the first line fits the card;
        # the full text stays in the tooltip.
        label = node_label(node)
        # No type line: a node's name already says what it is, and repeating it underneath was
        # noise. The line carries only a label and state.
        caption = label.splitlines()[0][:30] if label else ""
        parts = [part for part in ("BYPASSED" if node["disabled"] else "", caption,
                                   "viewing" if graph.window.dispatcher.document["view"] == key else "")
                 if part]
        subtitle = QGraphicsSimpleTextItem("  ·  ".join(parts), self)
        self.setToolTip(f"{node['name']} ({node['type']})" + (f"\n{label}" if label else ""))
        subtitle.setBrush(QColor("#a6a6b0"))
        if self.form == "circle":
            if subtitle.boundingRect().width() > width - 24:
                subtitle.setText("  ·  ".join(part[:12] for part in parts[:2]))
            subtitle.setPos((width - subtitle.boundingRect().width()) / 2, height / 2 + 6)
        else:
            subtitle.setPos((width - subtitle.boundingRect().width()) / 2, 32)
        # Inputs default to the top edge, which is where B lives: in Nuke the B stream is the
        # trunk flowing straight down through a Merge. Semantic side ports follow compositor
        # convention: A joins from the left and an optional mask from the right. Keep every
        # declared socket present even for a disabled Merge, so bypassing never hides B.
        slots = list(node["inputs"])
        spacing = 34
        top_slots = [slot for slot in slots if slot not in ("A", "mask")]
        self.inputs = {}
        rim = circle_port_positions(len(top_slots), width) if self.form == "circle" else ()
        for i, slot in enumerate(slots):
            if slot == "A":
                x, y = 0, height / 2 if self.form == "circle" else 26
            elif slot == "mask":
                x, y = width, height / 2 if self.form == "circle" else 26
            elif rim:
                # Sockets ride the rim. Their labels sit outside it on the same ray, and a numbered
                # run such as object0..object7 shows only its number: eight full words do not fit.
                x, y, angle = rim[top_slots.index(slot)]
                reach = width / 2 + 17
                center = (width / 2 + reach * math.sin(math.radians(angle)),
                          height / 2 - reach * math.cos(math.radians(angle)))
                short = slot[len("object"):] if slot.startswith("object") and slot[6:].isdigit() else slot
                self.inputs[slot] = Port(self, slot, x, y, label_text=short, label_center=center)
                continue
            else:
                top_i = top_slots.index(slot)
                x, y = width / 2 + (top_i - (len(top_slots) - 1) / 2) * spacing, 0
            self.inputs[slot] = Port(self, slot, x, y)
        if self.form != "circle" and height > NODE_HEIGHT:
            self.thumbnail = QGraphicsPixmapItem(self)
            self.thumbnail.setPos((NODE_WIDTH - THUMB_WIDTH) / 2, NODE_HEIGHT)
            cached = graph.window.thumbnails.get(key)
            if cached is not None:
                self.set_thumbnail(cached[1])
        if node["type"] == "ShuffleCopy":
            self.outputs = {"out1": Port(self, None, width / 2 - 18, height, "out1", output_name="out1"),
                            "out2": Port(self, None, width / 2 + 18, height, "out2", output_name="out2")}
            self.output = self.outputs["out1"]
        else:
            self.output = Port(self, None, width / 2, height)
            self.outputs = {"rgba": self.output}

    def set_thumbnail(self, image):
        if self.thumbnail is None:
            return
        pixmap = QPixmap.fromImage(image)
        # Centre the picture in its band; the band is fixed so cards never resize as stamps land.
        self.thumbnail.setPixmap(pixmap)
        self.thumbnail.setPos((NODE_WIDTH - pixmap.width()) / 2,
                              NODE_HEIGHT + (THUMB_HEIGHT - pixmap.height()) / 2)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged and hasattr(self, "output"):
            self.graph.update_edges()
        return super().itemChange(change, value)

    def outline(self):
        """The node's silhouette: what is painted, and what a click has to land inside."""
        path = QPainterPath()
        if self.form in ("dot", "circle"):
            path.addEllipse(self.rect())
        elif self.form == "round":
            path.addRoundedRect(self.rect(), NODE_HEIGHT / 2, NODE_HEIGHT / 2)
        else:
            path.addRect(self.rect())
        return path

    def shape(self):
        if self.is_backdrop:
            # Only the title strip and the grip take clicks: the body has to leave nodes above it,
            # rubber-band selection and empty-space clicks to the graph.
            path = QPainterPath()
            path.addRect(QRectF(0, 0, self.rect().width(), BACKDROP_TITLE_HEIGHT))
            path.addRect(self.grip.mapRectToParent(self.grip.rect()))
            return path
        if self.form in ("circle", "round"):
            # Without this the empty corners of the bounding box would still grab clicks and drags.
            return self.outline()
        return super().shape()

    def enclosed_items(self):
        """The other graph items whose centre lies inside this backdrop."""
        area = self.sceneBoundingRect()
        return [item for item in self.graph.items_by_id.values()
                if item is not self and area.contains(item.sceneBoundingRect().center())]

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if self.is_backdrop and event.button() == Qt.MouseButton.LeftButton:
            self.graph.primary_id = self.key
            # Qt moves every selected item together, so selecting the enclosed nodes makes the
            # title drag carry them. The selection signal is held back so the properties panel
            # stays on the backdrop rather than jumping to one of its nodes.
            scene = self.graph.scene()
            scene.blockSignals(True)
            for item in self.enclosed_items():
                item.setSelected(True)
            scene.blockSignals(False)

    def contextMenuEvent(self, event):
        """Right-click a node on the graph: "What is this?" (`Window.show_node_help`), the same
        docs lookup the NODES dock's row context menu offers."""
        kind = self.graph.window.graph_nodes()[self.key]["type"]
        menu = QMenu(self.graph.window)
        menu.addAction("What is this?", lambda: self.graph.window.show_node_help(kind))
        menu.addAction("Save selection as preset…", self._save_selection_as_preset)
        menu.exec(event.screenPos())
        event.accept()

    def _save_selection_as_preset(self):
        if not self.isSelected():
            self.graph.scene().clearSelection()
            self.setSelected(True)
        self.graph.window.node_toolbar.save_selection_as_preset()

    def paint_backdrop(self, painter):
        rect = self.rect()
        selected = self.isSelected()
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        body = QColor(self.tint)
        body.setAlpha(70)
        pen = QPen(self.pen())
        if selected:
            pen.setWidthF(3)
        painter.setPen(pen)
        painter.setBrush(body)
        painter.drawRoundedRect(rect, 6, 6)
        bar = QColor(self.tint)
        bar.setAlpha(200)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(bar)
        painter.drawRoundedRect(QRectF(0, 0, rect.width(), BACKDROP_TITLE_HEIGHT), 6, 6)
        font = QFont()
        font.setPointSize(13)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor("#f4f4f8"))
        painter.drawText(QRectF(10, 0, rect.width() - 20, BACKDROP_TITLE_HEIGHT),
                         int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                         QFontMetrics(font).elidedText(self.caption, Qt.TextElideMode.ElideRight,
                                                       int(rect.width() - 20)))
        painter.restore()

    def paint(self, painter, option, widget=None):
        if self.is_backdrop:
            self.paint_backdrop(painter)
            return
        if not self.is_dot:
            if self.form == "card":
                super().paint(painter, option, widget)
            else:
                painter.save()
                painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
                pen = QPen(self.pen())
                if option.state & QStyle.StateFlag.State_Selected:
                    pen.setWidthF(3)
                    pen.setColor(pen.color().lighter(135))
                painter.setPen(pen)
                painter.setBrush(self.brush())
                painter.drawPath(self.outline())
                painter.restore()
            if self.disabled:
                painter.save()
                painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
                pen = QPen(QColor("#f08a8a"), 5.0, Qt.PenStyle.SolidLine,
                           Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                # On a circle the cross has to stay inside the rim, so it is drawn on the inscribed square.
                inset = self.rect().width() * 0.22 if self.form == "circle" else 12
                rect = self.rect().adjusted(inset, inset, -inset, -inset)
                painter.drawLine(rect.topLeft(), rect.bottomRight())
                painter.drawLine(rect.topRight(), rect.bottomLeft())
                painter.restore()
            return
        painter.save()
        pen = QPen(self.pen())
        if option.state & QStyle.StateFlag.State_Selected:
            pen.setWidthF(3)
        painter.setPen(pen)
        painter.setBrush(self.brush())
        painter.drawEllipse(self.rect())
        painter.restore()


class Edge(QGraphicsPathItem):
    """A readable noodle with a small arrow showing output -> input direction."""
    def __init__(self, color="#898995", dashed=False, arrow=True, width=3.25):
        super().__init__()
        self.color = QColor(color)
        self.setPen(QPen(self.color, width, Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine,
                         Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        self.setBrush(Qt.BrushStyle.NoBrush)
        self.arrow = arrow
        self.arrowhead = QPolygonF()

    def set_curve(self, start, end):
        distance = max(40, abs(end.y() - start.y()) * 0.5)
        control = end - QPointF(0, distance)
        path = QPainterPath(start)
        path.cubicTo(start + QPointF(0, distance), control, end)
        tangent = end - control
        length = math.hypot(tangent.x(), tangent.y()) or 1.0
        unit = QPointF(tangent.x() / length, tangent.y() / length)
        normal = QPointF(-unit.y(), unit.x())
        base = end - unit * 10
        self.arrowhead = QPolygonF([end, base + normal * 4, base - normal * 4])
        self.setPath(path)
        self.handle = path.pointAtPercent(0.5)

    def paint(self, painter, option, widget=None):
        # Keep the curve stroked. Combining a closed arrow polygon with the curve
        # in one path causes Qt to fill the implied region as a ribbon.
        painter.save()
        painter.setPen(self.pen())
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(self.path())
        if self.arrow and not self.arrowhead.isEmpty():
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(self.color)
            painter.drawPolygon(self.arrowhead)
        painter.restore()


# The mime type a node kind travels under when dragged out of the NODES dock onto the graph.
NODE_KIND_MIME_TYPE = "application/x-nodebased-kind"


def _node_chip_icon(kind):
    """A small flat colour swatch for a node kind, the same colour its NodeItem is drawn in
    (theme.COLORS), so the dock's list rows and the graph agree on what a family looks like."""
    pixmap = QPixmap(12, 12)
    pixmap.fill(QColor(COLORS.get(kind, "#7a8fa8")))
    return QIcon(pixmap)


class NodeListWidget(QListWidget):
    """A list of node kinds an artist can click or drag onto the graph. Dragging embeds only the
    kind name (`NODE_KIND_MIME_TYPE`), never the row's display text, which may carry a category
    suffix during a search."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

    def mimeData(self, items):
        data = QMimeData()
        if items:
            kind = items[0].data(Qt.ItemDataRole.UserRole)
            if kind:
                data.setData(NODE_KIND_MIME_TYPE, kind.encode("utf-8"))
        return data


# The two pinned rows above every real `nodecatalog.NODE_CATEGORIES` group in the NODES dock.
FAVOURITES_CATEGORY = "Favourites"
RECENT_CATEGORY = "Recent"


def _glyph_icon(glyph, color="#e6c15c"):
    """A small text glyph as an icon, for the dock's pinned rows (no node kind of their own to
    take a colour swatch from, unlike `_node_chip_icon`)."""
    pixmap = QPixmap(14, 14)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QColor(color))
    font = painter.font()
    font.setPointSize(9)
    painter.setFont(font)
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, glyph)
    painter.end()
    return QIcon(pixmap)


class NodeToolbar(QWidget):
    """The NODES dock: every `nodecatalog.NODE_CATEGORIES` type, browsable by category or by a
    search across names and descriptions -- so an artist can find a node without already knowing
    its name, the gap Tab search and the context menu both leave (both require the exact name).

    Two pinned rows sit above the real categories: Favourites (starred kinds, right-click a row
    to toggle) and Recent (the last `Preferences.RECENT_KINDS_CAP` kinds added, most recent
    first), both stored in `Preferences` so they outlive the session like the theme does.
    """
    EXPANDED_CATEGORY_WIDTH = 110
    COMPACT_CATEGORY_WIDTH = 34

    def __init__(self, window):
        super().__init__()
        self.window = window
        self._compact = False
        self._categories_hovered = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        header = QHBoxLayout()
        header.setSpacing(6)
        self.search = QLineEdit()
        self.search.setObjectName("nodeToolbarSearch")
        self.search.setPlaceholderText("Search nodes…")
        header.addWidget(self.search, 1)
        self.compact_toggle = QCheckBox("Compact")
        self.compact_toggle.setObjectName("nodeToolbarCompactToggle")
        self.compact_toggle.setToolTip(
            "Icon-only category column for small screens; hover it to see the names again")
        self.compact_toggle.toggled.connect(self.set_compact_mode)
        header.addWidget(self.compact_toggle)
        layout.addLayout(header)
        self.split = QSplitter(Qt.Orientation.Horizontal)
        self.categories = QListWidget()
        self.categories.setObjectName("nodeToolbarCategories")
        favourites_item = QListWidgetItem(FAVOURITES_CATEGORY)
        favourites_item.setIcon(_glyph_icon("★"))
        self.categories.addItem(favourites_item)
        recent_item = QListWidgetItem(RECENT_CATEGORY)
        recent_item.setIcon(_glyph_icon("↻"))
        self.categories.addItem(recent_item)
        for name, kinds in NODE_CATEGORIES.items():
            item = QListWidgetItem(name)
            item.setIcon(_node_chip_icon(next(iter(kinds))))
            self.categories.addItem(item)
        self.split.addWidget(self.categories)
        self.nodes = NodeListWidget()
        self.nodes.setObjectName("nodeToolbarNodes")
        self.nodes.setToolTipDuration(20000)
        self.nodes.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.nodes.customContextMenuRequested.connect(self._node_context_menu)
        self.split.addWidget(self.nodes)
        self.split.setSizes([self.EXPANDED_CATEGORY_WIDTH, 210])
        self._build_presets_browser(layout)
        self.categories.currentTextChanged.connect(self._show_category)
        self.search.textChanged.connect(self._search_changed)
        self.nodes.itemClicked.connect(self._add_clicked)
        self.categories.setCurrentRow(2 if NODE_CATEGORIES else 0)
        self.set_compact_mode(self.window.preferences.node_toolbar_compact(), persist=False)
        # Installed last, once every attribute either handler touches exists: Qt can deliver
        # events to `search`/`categories` (e.g. a layout pass) while this widget tree is still
        # being built, and the handlers below read `self.categories`/`self.nodes`.
        self.search.installEventFilter(self)
        self.categories.installEventFilter(self)

    def _build_presets_browser(self, layout):
        """Preset browser launched from the NODES dock; JSON remains the source of truth."""
        self.presets_dialog = QDialog(self, Qt.WindowType.Window)
        self.presets_dialog.setWindowTitle("Presets")
        self.presets_dialog.setObjectName("presetBrowser")
        self.presets_dialog.resize(440, 430)
        presets_layout = QVBoxLayout(self.presets_dialog)
        presets_layout.setContentsMargins(8, 8, 8, 8)
        self.preset_search = QLineEdit()
        self.preset_search.setObjectName("presetBrowserSearch")
        self.preset_search.setPlaceholderText("Search presets…")
        self.preset_category = QComboBox()
        self.preset_category.setObjectName("presetBrowserCategory")
        self.preset_list = QListWidget()
        self.preset_list.setObjectName("presetBrowserList")
        self.preset_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.preset_list.setIconSize(QSize(64, 48))
        self.preset_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.save_preset_button = QPushButton("Save selection as preset…")
        self.save_preset_button.setObjectName("saveSelectionAsPreset")
        launch = QPushButton("Presets…")
        launch.setObjectName("openPresetBrowser")
        launch.clicked.connect(self.open_presets_browser)
        layout.addWidget(launch)
        layout.addWidget(self.split, 1)
        presets_layout.addWidget(self.preset_search)
        presets_layout.addWidget(self.preset_category)
        presets_layout.addWidget(self.preset_list, 1)
        presets_layout.addWidget(self.save_preset_button)
        self.preset_search.textChanged.connect(self._refresh_presets)
        self.preset_category.currentIndexChanged.connect(self._refresh_presets)
        self.preset_list.itemDoubleClicked.connect(self._place_preset_item)
        self.preset_list.itemActivated.connect(self._place_preset_item)
        self.save_preset_button.clicked.connect(self.save_selection_as_preset)
        self.refresh_presets()

        shelf = QToolButton()
        shelf.setObjectName("fluidShelfGroup")
        shelf.setText("Fluids")
        shelf.setToolTip("One-click fluid setups for the selected 3D geometry")
        shelf.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        shelf_menu = QMenu(shelf)
        self.fluid_shelf_actions = {}
        for tool, label in fluidshelf.FLUID_SHELF_TOOLS:
            action = shelf_menu.addAction(label)
            action.setObjectName("fluidShelf_" + tool)
            action.triggered.connect(lambda checked=False, name=tool:
                                     self.window.run_fluid_shelf_tool(name))
            self.fluid_shelf_actions[tool] = action
        shelf.setMenu(shelf_menu)
        layout.addWidget(shelf)

    def refresh_presets(self):
        self._presets, self.preset_errors = preset_model.load_all()
        current = self.preset_category.currentData()
        self.preset_category.blockSignals(True)
        self.preset_category.clear()
        self.preset_category.addItem("All categories", "")
        for category in preset_model.categories(self._presets):
            self.preset_category.addItem(category, category)
        index = self.preset_category.findData(current)
        self.preset_category.setCurrentIndex(max(index, 0))
        self.preset_category.blockSignals(False)
        self._refresh_presets()

    def _refresh_presets(self, *_):
        if not hasattr(self, "preset_list"):
            return
        self.preset_list.clear()
        category = self.preset_category.currentData() or None
        for preset in preset_model.search(self._presets, self.preset_search.text(), category):
            item = QListWidgetItem(preset.name)
            item.setData(Qt.ItemDataRole.UserRole, (preset.user, preset.id))
            item.setToolTip(preset.description)
            if preset.thumbnail and preset.thumbnail.is_file():
                item.setIcon(QIcon(str(preset.thumbnail)))
            else:
                item.setIcon(_glyph_icon("✦", "#83c8ee"))
            self.preset_list.addItem(item)

    def _place_preset_item(self, item):
        identity = item.data(Qt.ItemDataRole.UserRole)
        preset = next((p for p in self._presets if (p.user, p.id) == identity), None)
        if preset is None:
            self.refresh_presets()
            return
        nodes = self.window.graph_nodes()
        selected = [obj.key for obj in self.window.graph.scene().selectedItems()
                    if isinstance(obj, NodeItem)]
        ops = preset_model.build_ops(preset, selected, nodes, self.window.graph_center())
        if ops:
            self.window.command({"op": "batch", "commands": ops})

    def open_presets_browser(self, selected_ids=None):
        if selected_ids:
            keys = set(selected_ids)
            for item in self.window.graph.scene().selectedItems():
                item.setSelected(getattr(item, "key", None) in keys)
        self.refresh_presets()
        self.presets_dialog.show()
        self.presets_dialog.raise_()
        self.preset_search.setFocus()

    def save_selection_as_preset(self):
        nodes = self.window.graph_nodes()
        ids = [obj.key for obj in self.window.graph.scene().selectedItems()
               if isinstance(obj, NodeItem) and obj.key in nodes]
        if not ids:
            self.window.statusBar().showMessage("Select nodes to save them as a preset", 3000)
            return
        name, accepted = QInputDialog.getText(self, "Save preset", "Preset name")
        if not accepted or not name.strip():
            return
        slug = "".join(c.lower() if c.isalnum() else "_" for c in name.strip()).strip("_") or "preset"
        path = preset_model.save_selection_as_preset(
            preset_model.user_presets_directory(), slug, name.strip(), "User",
            nodes, ids)
        self.refresh_presets()
        self.window.statusBar().showMessage(f"Saved preset: {path.name}", 4000)

    def save_fluid_knob_preset(self, key):
        node = self.window.graph_nodes().get(key)
        if node is None or node["type"] not in fluidknobpresets.FLUID_NODE_TYPES:
            return
        name, accepted = QInputDialog.getText(self, "Save knob preset", "Preset name")
        if not accepted or not name.strip():
            return
        try:
            path = fluidknobpresets.save(None, node["type"], name,
                                         copy.deepcopy(node["params"]))
        except (OSError, TypeError, ValueError) as error:
            QMessageBox.warning(self, "Could not save knob preset", str(error))
            return
        self.window.statusBar().showMessage(f"Saved {node['type']} knob preset: {path.stem}", 4000)

    def load_fluid_knob_preset(self, key):
        node = self.window.graph_nodes().get(key)
        if node is None or node["type"] not in fluidknobpresets.FLUID_NODE_TYPES:
            return
        presets = fluidknobpresets.list_presets(None, node["type"])
        if not presets:
            self.window.statusBar().showMessage(f"No saved presets for {node['type']}", 3500)
            return
        menu = QMenu(self)
        for name, path in presets:
            menu.addAction(name, lambda checked=False, p=path, k=key, kind=node["type"]:
                           self._apply_fluid_knob_preset(k, kind, p))
        menu.exec(QCursor.pos())

    def _apply_fluid_knob_preset(self, key, node_type, path):
        try:
            params = fluidknobpresets.load(path, node_type)
            commands = [{"op": "set", "id": key, "param": param, "value": value}
                        for param, value in params.items()]
            self.window.command({"op": "batch", "commands": commands})
        except (OSError, ValueError, json.JSONDecodeError) as error:
            QMessageBox.warning(self, "Could not load knob preset", str(error))

    def _category_kinds(self, name):
        if name == FAVOURITES_CATEGORY:
            return {kind: node_description(kind) for kind in self.window.preferences.favourite_kinds()}
        if name == RECENT_CATEGORY:
            return {kind: node_description(kind) for kind in self.window.preferences.recent_kinds()}
        return NODE_CATEGORIES.get(name, {})

    def _show_category(self, name):
        if self.search.text().strip():
            return
        self._populate(self._category_kinds(name),
                       with_category=name in (FAVOURITES_CATEGORY, RECENT_CATEGORY))

    def _search_changed(self, text):
        needle = text.casefold().strip()
        self.categories.setEnabled(not needle)
        if not needle:
            current = self.categories.currentItem()
            self._show_category(current.text() if current else "")
            return
        matches = {kind: description for kinds in NODE_CATEGORIES.values()
                  for kind, description in kinds.items()
                  if needle in kind.casefold() or needle in description.casefold()}
        self._populate(matches, with_category=True)

    def _populate(self, kinds, with_category):
        self.nodes.clear()
        favourites = set(self.window.preferences.favourite_kinds())
        for kind, description in kinds.items():
            star = "★ " if kind in favourites else ""
            label = f"{star}{kind}  ·  {node_category(kind)}" if with_category else f"{star}{kind}"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, kind)
            item.setIcon(_node_chip_icon(kind))
            hint = "Right-click to remove from Favourites" if kind in favourites \
                else "Right-click to add to Favourites"
            item.setToolTip(f"{description}\n{hint}" if description else hint)
            self.nodes.addItem(item)
        if self.nodes.count():
            self.nodes.setCurrentRow(0)

    def _refresh_current(self):
        if self.search.text().strip():
            self._search_changed(self.search.text())
        else:
            current = self.categories.currentItem()
            self._show_category(current.text() if current else "")

    def note_added(self, kind):
        """Record `kind` as just added to the graph (`Window.add_node`'s one choke point),
        refreshing the Recent category if it is what is on screen."""
        self.window.preferences.add_recent_kind(kind)
        self._refresh_current()

    def toggle_favourite(self, kind):
        """Star or unstar `kind`, refresh whatever category is on screen, and return the new
        starred state."""
        favourite = self.window.preferences.set_favourite(
            kind, kind not in self.window.preferences.favourite_kinds())
        self._refresh_current()
        return favourite

    def _node_context_menu(self, point):
        item = self.nodes.itemAt(point)
        kind = item.data(Qt.ItemDataRole.UserRole) if item else None
        if not kind:
            return
        is_favourite = kind in self.window.preferences.favourite_kinds()
        menu = QMenu(self)
        menu.addAction("Remove from Favourites" if is_favourite else "Add to Favourites",
                       lambda: self.toggle_favourite(kind))
        menu.addAction("What is this?", lambda: self.window.show_node_help(kind))
        menu.exec(self.nodes.mapToGlobal(point))

    def _step_selection(self, delta):
        count = self.nodes.count()
        if not count:
            return
        row = self.nodes.currentRow()
        row = 0 if row < 0 else max(0, min(count - 1, row + delta))
        self.nodes.setCurrentRow(row)

    def _add_first_match(self):
        item = self.nodes.currentItem() or self.nodes.item(0)
        if item:
            self._add_clicked(item)

    def set_compact_mode(self, enabled, persist=True):
        """Icons-only category column, for a small screen; hovering it still shows the names."""
        self._compact = bool(enabled)
        if self.compact_toggle.isChecked() != self._compact:
            self.compact_toggle.setChecked(self._compact)
        if persist:
            self.window.preferences.set_node_toolbar_compact(self._compact)
        self._apply_category_width()

    def _apply_category_width(self):
        if self._compact and not self._categories_hovered:
            # A fixed width is a hard constraint the splitter must honour; `setSizes` is only a
            # ratio hint that a narrow dock (or the other pane's own minimum) can override, which
            # is exactly why it is not used for the resizable, non-compact width below.
            self.categories.setFixedWidth(self.COMPACT_CATEGORY_WIDTH)
            return
        self.categories.setMinimumWidth(0)
        self.categories.setMaximumWidth(16_777_215)   # Qt's QWIDGETSIZE_MAX: "no maximum"
        if self._compact:   # hovered: nudge back toward the usual width, still freely resizable
            self.split.setSizes([self.EXPANDED_CATEGORY_WIDTH, 10_000])

    def _set_categories_hovered(self, hovered):
        self._categories_hovered = hovered
        # Never touch the column's width from a hover while not compact: it stays whatever the
        # artist last dragged it to, exactly as it did before compact mode existed.
        if self._compact:
            self._apply_category_width()

    def eventFilter(self, obj, event):
        if obj is self.categories:
            if event.type() == QEvent.Type.Enter:
                self._set_categories_hovered(True)
            elif event.type() == QEvent.Type.Leave:
                self._set_categories_hovered(False)
        elif obj is self.search and event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Down:
                self._step_selection(1)
                return True
            if event.key() == Qt.Key.Key_Up:
                self._step_selection(-1)
                return True
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self._add_first_match()
                return True
        return super().eventFilter(obj, event)

    def _add_clicked(self, item):
        kind = item.data(Qt.ItemDataRole.UserRole)
        if kind:
            self.window.add_node(kind, position=self.window.graph_center())


class NodeSearch(QDialog):
    """Small keyboard-first node picker, intentionally close to Nuke's Tab menu."""
    def __init__(self, parent, choices, global_pos):
        super().__init__(parent, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.setObjectName("nodeSearch")
        self.setWindowTitle("Create node")
        self.setMinimumWidth(260)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        self.query = QLineEdit()
        self.query.setPlaceholderText("Search nodes…")
        self.list = QListWidget()
        self.list.setMaximumHeight(280)
        layout.addWidget(self.query)
        layout.addWidget(self.list)
        self.choices = list(choices)
        self.update_matches("")
        self.query.textChanged.connect(self.update_matches)
        self.query.returnPressed.connect(self.choose_current)
        self.list.itemActivated.connect(lambda _: self.choose_current())
        self.move(global_pos)
        self.query.setFocus()

    def update_matches(self, query):
        # Matches the description too (node_description), not just the name, so "blur" finds
        # Defocus and DirBlur alongside Blur -- the same search the NODES dock's box runs.
        needle = query.casefold().strip()
        matches = [name for name in self.choices
                  if not needle or needle in name.casefold() or needle in node_description(name).casefold()]
        self.list.clear()
        for name in matches:
            item = QListWidgetItem(f"{name}  ·  {node_category(name)}")
            item.setData(Qt.ItemDataRole.UserRole, name)
            self.list.addItem(item)
        if matches:
            self.list.setCurrentRow(0)

    def choose_current(self):
        item = self.list.currentItem()
        if item:
            self.selected_kind = item.data(Qt.ItemDataRole.UserRole)
            self.accept()

    @classmethod
    def choose(cls, parent, choices, global_pos):
        picker = cls(parent, choices, global_pos)
        return picker.selected_kind if picker.exec() == QDialog.DialogCode.Accepted else None


class KeyboardShortcutsDialog(QDialog):
    """Compact, read-only reference for the application's keyboard shortcuts."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard shortcuts")
        self.setMinimumSize(560, 520)
        layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        for section, shortcuts in SHORTCUT_SECTIONS:
            heading = QLabel(section)
            heading.setObjectName("brand")
            content_layout.addWidget(heading)
            for keys, description in shortcuts:
                content_layout.addWidget(QLabel(f"{keys} — {description}"))
        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        layout.addWidget(close)


class NodeHelpDialog(QDialog):
    """"What is this?" from a node's right-click menu: the node's row in its bundled docs table
    (`docs/PARITY_2D.md` for a 2D kind, `docs/3D_FOUNDATION.md` for a 3D one, `nodecatalog.doc_for_kind`)
    scrolled to and highlighted, or -- for a kind with no row there yet, like a particle or fluid
    node -- just its one-line `nodecatalog` description, since opening an unrelated doc with
    nothing to show for it would be worse than not opening one.
    """
    def __init__(self, kind, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{kind} — What is this?")
        self.setMinimumSize(720, 480)
        layout = QVBoxLayout(self)
        doc_name = doc_for_kind(kind)
        try:
            text = read_doc(doc_name)
        except FileNotFoundError:
            text = None
        found = find_doc_row(text, kind) if text is not None else None
        if found is None:
            description = node_description(kind) or "No description is filed for this node kind yet."
            layout.addWidget(QLabel(f"No row for {kind!r} was found in {doc_name}."))
            note = QPlainTextEdit(description)
            note.setReadOnly(True)
            layout.addWidget(note, 1)
        else:
            line_index, _ = found
            layout.addWidget(QLabel(f"{kind} in {doc_name}:"))
            viewer = QPlainTextEdit(text)
            viewer.setReadOnly(True)
            viewer.setFont(QFont("Monospace"))
            layout.addWidget(viewer, 1)
            cursor = QTextCursor(viewer.document().findBlockByNumber(line_index))
            cursor.select(QTextCursor.SelectionType.LineUnderCursor)
            viewer.setTextCursor(cursor)
            viewer.setExtraSelections([self._line_highlight(viewer)])
            viewer.centerCursor()
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        layout.addWidget(close)

    @staticmethod
    def _line_highlight(viewer):
        selection = QTextEdit.ExtraSelection()
        selection.format.setBackground(QColor("#4a3f1a"))
        selection.format.setProperty(QTextFormat.Property.FullWidthSelection, True)
        selection.cursor = viewer.textCursor()
        selection.cursor.clearSelection()
        return selection


class RadialCommandsDialog(QDialog):
    """Preferences -> Radial commands (deliverable R2, DiMo 9/27): list every command file in
    `radialcommands.user_commands_directory()`, edit its manifest as JSON, enable or disable it,
    and test-run it against whatever is selected on the graph right now. A malformed file shows
    up as an error row instead of a command row and never stops the others from loading."""

    def __init__(self, window, new_from_selection=None, parent=None):
        super().__init__(parent)
        self.window = window
        self.setWindowTitle("Radial commands")
        self.setMinimumSize(820, 520)
        radialcommands.install_default_commands()
        self.directory = radialcommands.user_commands_directory()

        layout = QHBoxLayout(self)
        self.list = QListWidget()
        self.list.setMinimumWidth(260)
        self.list.currentRowChanged.connect(self._select_row)
        layout.addWidget(self.list, 1)

        right = QVBoxLayout()
        layout.addLayout(right, 2)
        self.status_label = QLabel(f"Command files live in {self.directory}")
        self.status_label.setWordWrap(True)
        right.addWidget(self.status_label)
        self.editor = QPlainTextEdit()
        self.editor.setFont(QFont("Monospace"))
        right.addWidget(self.editor, 1)

        buttons = QHBoxLayout()
        self.new_button = QPushButton("New command…")
        self.new_button.clicked.connect(self._new_command)
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self._save_current)
        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self._delete_current)
        self.test_button = QPushButton("Test-run on current selection")
        self.test_button.clicked.connect(self._test_run_current)
        for button in (self.new_button, self.save_button, self.delete_button, self.test_button):
            buttons.addWidget(button)
        right.addLayout(buttons)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        right.addWidget(close)

        self._reload()
        if new_from_selection:
            self._new_command(seed_types=sorted({window.graph_nodes()[key]["type"]
                                                 for key in new_from_selection
                                                 if key in window.graph_nodes()}))

    def _reload(self, select_path=None):
        self.list.clear()
        self._entries = []   # parallel to self.list's rows: (path, is_error)
        commands, errors = radialcommands.load_all(self.directory)
        for command in commands:
            text = command.label
            if not command.enabled:
                text += "  [disabled]"
            if not command.available:
                text += f"  [unavailable: {command.unavailable_reason}]"
            self.list.addItem(text)
            self._entries.append((command.path, False))
        for error in errors:
            item = QListWidgetItem(f"⚠ {error.path.name}: {error.error}")
            item.setForeground(QColor("#d9534f"))
            self.list.addItem(item)
            self._entries.append((error.path, True))
        if select_path is not None:
            for row, (path, _is_error) in enumerate(self._entries):
                if path == select_path:
                    self.list.setCurrentRow(row)
                    return
        elif self._entries:
            self.list.setCurrentRow(0)

    def _current_path(self):
        row = self.list.currentRow()
        if row < 0 or row >= len(self._entries):
            return None
        return self._entries[row][0]

    def _select_row(self, row):
        if row < 0 or row >= len(self._entries):
            self.editor.setPlainText("")
            return
        path, _is_error = self._entries[row]
        self.editor.setPlainText(path.read_text())

    def _new_command(self, seed_types=None):
        name, ok = QInputDialog.getText(self, "New command", "File name (no extension, no spaces):")
        if not ok or not name:
            return
        path = self.directory / f"{name}.json"
        if path.exists():
            QMessageBox.warning(self, "Radial commands", f"{path.name} already exists")
            return
        when = {"types": seed_types} if seed_types else {}
        manifest = {"label": name, "when": when, "slot": None, "enabled": True, "ops": []}
        self.directory.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2))
        self._reload(select_path=path)

    def _save_current(self):
        path = self._current_path()
        if path is None:
            return
        path.write_text(self.editor.toPlainText())
        try:
            radialcommands._parse_manifest(path)
            self.status_label.setText(f"Saved {path.name}")
        except Exception as error:
            self.status_label.setText(f"Saved, but {path.name} does not parse: {error}")
        self._reload(select_path=path)

    def _delete_current(self):
        path = self._current_path()
        if path is None:
            return
        if QMessageBox.question(self, "Delete command", f"Delete {path.name}?",
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                                ) != QMessageBox.StandardButton.Yes:
            return
        path.unlink(missing_ok=True)
        self._reload()

    def _test_run_current(self):
        path = self._current_path()
        if path is None or path.suffix != ".json":
            self.status_label.setText("Select a command file to test-run")
            return
        try:
            command = radialcommands._parse_manifest(path)
        except Exception as error:
            self.status_label.setText(f"Cannot test-run: {error}")
            return
        if not command.available:
            self.status_label.setText(f"Cannot test-run: {command.unavailable_reason}")
            return
        ids = self.window.graph.selected_ids()
        pos = self.window.graph_center()
        try:
            radialcommands._run_command(command, self.window, ids, pos)
        except Exception as error:
            self.status_label.setText(f"Test-run failed: {error}")
            return
        self.status_label.setText(f"Ran {command.label!r} against the current selection "
                                  f"({len(ids)} node(s)) -- Edit > Undo reverts it.")


class RadialKeyCaptureButton(QPushButton):
    """A settings-dialog control that turns "click, then press a key" into a key value: one click
    starts listening, and the very next key press becomes the answer."""

    key_captured = Signal(int)

    def __init__(self, key, parent=None):
        super().__init__(parent)
        self._key = key
        self._listening = False
        self._refresh()
        self.clicked.connect(self._start_listening)

    def _refresh(self):
        self.setText("Press a key…" if self._listening
                     else f"Trigger key: {QKeySequence(self._key).toString() or '?'}")

    def _start_listening(self):
        self._listening = True
        self._refresh()
        self.setFocus(Qt.FocusReason.OtherFocusReason)

    def keyPressEvent(self, event):
        if self._listening and not event.isAutoRepeat():
            self._key = event.key()
            self._listening = False
            self._refresh()
            self.key_captured.emit(self._key)
            event.accept()
            return
        super().keyPressEvent(event)


class RadialSettingsDialog(QDialog):
    """Preferences -> Radial settings (deliverable R3, DiMo 9/27): the ring's trigger key, its
    flick dead-zone radius, whether local-usage learning is on, and a way to forget everything it
    has learned so far without touching any pinned slices."""

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.window = window
        self.setWindowTitle("Radial settings")
        layout = QFormLayout(self)

        self.key_button = RadialKeyCaptureButton(window.preferences.radial_trigger_key())
        self.key_button.key_captured.connect(window.preferences.set_radial_trigger_key)
        layout.addRow("Hold to open the ring", self.key_button)

        self.dead_zone = QDoubleSpinBox()
        self.dead_zone.setRange(8.0, 80.0)
        self.dead_zone.setSuffix(" px")
        self.dead_zone.setValue(window.preferences.radial_dead_zone())
        self.dead_zone.valueChanged.connect(window.preferences.set_radial_dead_zone)
        layout.addRow("Flick dead zone", self.dead_zone)

        self.learning = QCheckBox("Learn which commands I flick to most, per context")
        self.learning.setChecked(window.preferences.radial_learning_enabled())
        self.learning.toggled.connect(window.preferences.set_radial_learning_enabled)
        layout.addRow(self.learning)

        reset = QPushButton("Reset learned slots")
        reset.setToolTip("Forgets everything learning has promoted so far and puts every "
                         "context's ring back to its rule defaults. Pinned slices are untouched.")
        reset.clicked.connect(self._reset_learning)
        layout.addRow(reset)

        close = QPushButton("Close")
        close.clicked.connect(self.close)
        layout.addRow(close)

    def _reset_learning(self):
        self.window.preferences.reset_radial_learning()
        QMessageBox.information(self, "Radial settings", "Learned slots have been reset.")


class SequenceBrowser(QDialog):
    """A Read browser that understands image sequences.

    A generic file dialog shows 100 numbered EXRs as 100 rows, which is the wrong unit of work:
    the artist is loading one plate, and picking one member of it gives Read a still. This lists a
    directory with numbered frames batched into a single entry carrying its own range, the way
    Nuke's browser does, and hands back a printf pattern Read can map timeline frames onto.

    Grouping is a checkbox, enabled by default, because the one case it gets wrong is a folder of
    unrelated numbered stills -- and that case has to stay reachable.
    """
    def __init__(self, parent, directory=None):
        super().__init__(parent)
        self.setWindowTitle("Read image or sequence")
        self.setMinimumSize(720, 480)
        self.chosen = None
        layout = QVBoxLayout(self)
        path_row = QHBoxLayout()
        self.directory = QLineEdit(str(Path(directory or Path.home()).expanduser()))
        self.directory.setToolTip("Type a folder and press Return, or use Browse…")
        self.directory.returnPressed.connect(self.reload)
        up = QPushButton("Up")
        up.clicked.connect(self.go_up)
        pick = QPushButton("Browse…")
        pick.clicked.connect(self.pick_directory)
        path_row.addWidget(QLabel("Folder"))
        path_row.addWidget(self.directory, 1)
        path_row.addWidget(up)
        path_row.addWidget(pick)
        layout.addLayout(path_row)
        self.group = QCheckBox("Group image sequences into one entry")
        self.group.setChecked(True)
        self.group.setToolTip("Off lists every file separately, for a folder of unrelated stills")
        self.group.toggled.connect(self.reload)
        layout.addWidget(self.group)
        self.list = QListWidget()
        self.list.itemActivated.connect(lambda _: self.accept_current())
        layout.addWidget(self.list, 1)
        self.detail = QLabel("")
        self.detail.setObjectName("muted")
        self.detail.setWordWrap(True)
        self.list.currentItemChanged.connect(lambda *_: self.describe())
        layout.addWidget(self.detail)
        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        self.open_button = QPushButton("Open")
        self.open_button.setDefault(True)
        self.open_button.clicked.connect(self.accept_current)
        buttons.addWidget(cancel)
        buttons.addWidget(self.open_button)
        layout.addLayout(buttons)
        self.reload()

    def go_up(self):
        current = Path(self.directory.text()).expanduser()
        if current.parent != current:
            self.directory.setText(str(current.parent))
            self.reload()

    def pick_directory(self):
        chosen = QFileDialog.getExistingDirectory(self, "Choose folder", self.directory.text())
        if chosen:
            self.directory.setText(chosen)
            self.reload()

    def reload(self):
        self.list.clear()
        directory = Path(self.directory.text()).expanduser()
        if not directory.is_dir():
            self.detail.setText(f"Not a folder: {directory}")
            return
        # Subfolders first, so navigating a plate tree does not mean retyping paths.
        try:
            children = sorted((entry for entry in directory.iterdir() if entry.is_dir()),
                              key=lambda entry: entry.name.casefold())
            entries = group_directory(directory, IMAGE_EXTENSIONS, self.group.isChecked())
        except OSError as error:
            self.detail.setText(str(error))
            return
        for child in children:
            item = QListWidgetItem(f"[ {child.name} ]")
            item.setData(Qt.ItemDataRole.UserRole, {"directory": str(child)})
            self.list.addItem(item)
        for entry in entries:
            item = QListWidgetItem(entry["label"])
            item.setData(Qt.ItemDataRole.UserRole, entry)
            self.list.addItem(item)
        self.detail.setText(f"{len(entries)} image entr{'y' if len(entries) == 1 else 'ies'} · "
                            f"{len(children)} subfolder{'' if len(children) == 1 else 's'}")
        if self.list.count():
            self.list.setCurrentRow(0)

    def describe(self):
        entry = self.current_entry()
        if not entry or "directory" in entry:
            return
        if entry["sequence"]:
            gap = (f"  ·  missing {', '.join(str(f) for f in entry['missing'][:12])}"
                   f"{'…' if len(entry['missing']) > 12 else ''}" if entry["missing"] else "")
            self.detail.setText(f"{entry['path']}\n{entry['frames']} frames, "
                                f"{entry['first']}–{entry['last']}{gap}")
        else:
            self.detail.setText(f"{entry['path']}\nsingle image")

    def current_entry(self):
        item = self.list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def accept_current(self):
        entry = self.current_entry()
        if not entry:
            return
        if "directory" in entry:
            self.directory.setText(entry["directory"])
            self.reload()
            return
        self.chosen = entry
        self.accept()

    @classmethod
    def choose(cls, parent, directory=None):
        dialog = cls(parent, directory)
        return dialog.chosen if dialog.exec() == QDialog.DialogCode.Accepted else None


class Graph(PanZoomView):
    def __init__(self, window):
        self.window = window
        super().__init__(QGraphicsScene())
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setAcceptDrops(True)
        self.setSceneRect(-5000, -5000, 10000, 10000)
        self.viewport().setMouseTracking(True)
        self.items_by_id, self.edges = {}, []
        self.wire_source = None
        self.wire_output = "rgba"
        self.wire_input = None
        self.picked_input = None
        self.pending_edge = None
        self.inserting_edge = None
        self.dot_preview = None
        self.ctrl_handles_visible = False
        self.primary_id = None
        self.last_click_scene_pos = QPointF(0, 0)
        self.last_hover_scene_pos = QPointF(0, 0)
        self.radial_menu = RadialMenu(self.viewport())
        self.radial_menu.add_command_requested.connect(self.open_add_radial_command)
        self.radial_menu.presets_requested.connect(self.open_presets_browser)
        self._radial_selection = []
        self._radial_scene_pos = QPointF(0, 0)
        self._radial_context = "empty"
        self._radial_node_type = None
        self.scene().selectionChanged.connect(self.selection_changed)

    def _new_pending_edge(self):
        edge = Edge("#e3b18d", dashed=True, arrow=False)
        edge.setZValue(10)
        self.scene().addItem(edge)
        return edge

    def start_wire(self, source_key, picked_input=None, output="rgba"):
        self.cancel_wire()
        self.wire_source = source_key
        self.wire_output = output
        self.picked_input = picked_input
        self.pending_edge = self._new_pending_edge()

    def start_input_wire(self, node_key, slot):
        self.cancel_wire()
        self.wire_input = (node_key, slot)
        self.pending_edge = self._new_pending_edge()

    def cancel_wire(self):
        if self.pending_edge is not None:
            self.scene().removeItem(self.pending_edge)
        self.wire_source = None
        self.wire_output = "rgba"
        self.wire_input = None
        self.picked_input = None
        self.pending_edge = None

    def cancel_dot_insert(self):
        self.inserting_edge = None
        if self.dot_preview is not None:
            self.scene().removeItem(self.dot_preview)
            self.dot_preview = None
        self.unsetCursor()

    def start_dot_insert(self, edge_data, scene_pos):
        self.cancel_dot_insert()
        self.inserting_edge = edge_data
        # A live compact Dot follows the pointer.  The graph is untouched until
        # release, so cancelling this gesture can never erase an existing noodle.
        preview = QGraphicsEllipseItem(-10, -10, 20, 20)
        preview.setBrush(QColor("#23242a"))
        preview.setPen(QPen(QColor(COLORS["Dot"]), 2))
        preview.setZValue(20)
        self.scene().addItem(preview)
        self.dot_preview = preview
        self.update_dot_preview(scene_pos)
        self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def update_dot_preview(self, scene_pos):
        if self.dot_preview is not None:
            self.dot_preview.setPos(scene_pos)

    def update_pending_edge(self, scene_pos):
        if self.pending_edge is None:
            return
        if self.wire_source in self.items_by_id:
            self.pending_edge.set_curve(self.items_by_id[self.wire_source].outputs[self.wire_output].scenePos(), scene_pos)
        elif self.wire_input and self.wire_input[0] in self.items_by_id:
            key, slot = self.wire_input
            self.pending_edge.set_curve(scene_pos, self.items_by_id[key].inputs[slot].scenePos())

    def nearest_port(self, scene_pos, input_port):
        ports = [port for node in self.items_by_id.values() if not node.is_backdrop
                 for port in (node.inputs.values() if input_port else node.outputs.values())]
        return min(ports, key=lambda port: math.hypot(port.scenePos().x() - scene_pos.x(),
                                                       port.scenePos().y() - scene_pos.y()), default=None)

    def _snap_port(self, scene_pos, input_port):
        port = self.nearest_port(scene_pos, input_port)
        radius = 24 / max(self.transform().m11(), 0.05)  # 24 physical pixels
        if port and math.hypot(port.scenePos().x() - scene_pos.x(), port.scenePos().y() - scene_pos.y()) <= radius:
            return port
        return None

    def input_at(self, scene_pos):
        return self._snap_port(scene_pos, input_port=True)

    def output_at(self, scene_pos):
        return self._snap_port(scene_pos, input_port=False)

    def finish_wire_at(self, scene_pos):
        destination = self.input_at(scene_pos)
        if destination is None:
            self.cancel_wire()
            self.window.statusBar().showMessage("Wire dropped", 3000)
            return
        source, picked, output = self.wire_source, self.picked_input, self.wire_output
        self.cancel_wire()
        if picked == (destination.node.key, destination.slot):
            return
        commands = []
        if picked:
            commands.append({"op": "connect", "id": picked[0], "input": picked[1], "source": None})
        connection = {"op": "connect", "id": destination.node.key,
                      "input": destination.slot, "source": source}
        if output != "rgba":
            connection["output"] = output
        commands.append(connection)
        self.window.command({"op": "batch", "commands": commands})

    def finish_input_wire(self, source_key, output="rgba"):
        if self.wire_input:
            key, slot = self.wire_input
            self.cancel_wire()
            command = {"op": "connect", "id": key, "input": slot, "source": source_key}
            if output != "rgba": command["output"] = output
            self.window.defer_command(command)

    def finish_input_wire_at(self, scene_pos):
        output = self.output_at(scene_pos)
        if output:
                self.finish_input_wire(output.node.key, output.output_name)
        else:
            self.cancel_wire()
            self.window.statusBar().showMessage("Wire dropped", 3000)

    def rebuild(self):
        selected = self.selected_id()
        self.scene().blockSignals(True)
        self.edges = []
        self.items_by_id = {}
        self.scene().clear()
        self.pending_edge = None  # scene().clear() already deleted the previous item, if any.
        doc = self.window.graph_document()
        for key, node in doc["nodes"].items():
            item = NodeItem(self, key, node)
            self.items_by_id[key] = item
            self.scene().addItem(item)
            item.setSelected(key == selected)
        for key, node in doc["nodes"].items():
            # A PostageStamp with "hide input" keeps its connection but draws no noodle.
            if node["type"] == "PostageStamp" and node["params"]["hide_input"]:
                continue
            for slot, source in node["inputs"].items():
                if source:
                    # A Viewer's connection is a place the artist is looking from, not a stage in
                    # the comp. Drawing it like any other noodle makes the Viewer look like a
                    # consumer whose pixels matter downstream; it has none. Faint, dashed and
                    # arrowless says "this is a tap" without hiding where the view is pointed.
                    edge = (Edge(color=VIEW_EDGE_COLOR, dashed=True, arrow=False, width=1.6)
                            if node["type"] == "Viewer" else Edge())
                    edge.setZValue(-2 if node["type"] == "Viewer" else -1)
                    self.scene().addItem(edge)
                    out_name = node.get("input_outputs", {}).get(slot, "out1" if doc["nodes"][source]["type"] == "ShuffleCopy" else "rgba")
                    self.edges.append((edge, source, key, slot, out_name))
        self.update_edges()
        self.scene().blockSignals(False)
        self.window.sync_breadcrumbs()
        if self.wire_source or self.wire_input:
            # A picked-up wire's disconnect command rebuilds the scene mid-drag; keep the preview alive.
            if self.wire_source in self.items_by_id or (self.wire_input and self.wire_input[0] in self.items_by_id):
                self.pending_edge = self._new_pending_edge()
                self.update_pending_edge(self.mapToScene(self.viewport().mapFromGlobal(QCursor.pos())))
            else:
                self.cancel_wire()

    def update_edges(self):
        for edge, source, key, slot, out_name in self.edges:
            start = self.items_by_id[source].outputs[out_name].scenePos()
            end = self.items_by_id[key].inputs[slot].scenePos()
            edge.set_curve(start, end)

    def selected_id(self):
        keys = [i.key for i in self.scene().selectedItems() if isinstance(i, NodeItem)]
        # A backdrop drag selects the nodes it carries too; the backdrop that was grabbed stays
        # the one the properties panel shows.
        return self.primary_id if self.primary_id in keys else next(iter(keys), None)

    def selected_ids(self):
        """Every selected node, top to bottom (a stream's own reading order) so callers such as
        the radial menu and the Commands editor see a deterministic order regardless of the order
        Qt happens to report a rubber-band selection in."""
        items = sorted((item for item in self.scene().selectedItems() if isinstance(item, NodeItem)),
                       key=lambda item: (item.pos().y(), item.pos().x()))
        return [item.key for item in items]

    def selection_changed(self):
        self.window.inspect(self.selected_id())
        # The keyed-frame band is scoped to the selection, so it has to follow it.
        self.window.refresh_timeline_marks()

    def mousePressEvent(self, event):
        if self.radial_menu.is_open() and self.radial_menu.sustained:
            # A pinned-open menu (a tap, not a flick) takes every click until it resolves: a
            # slice runs it, the dead zone or an empty slot cancels -- either way it closes.
            # A right-click instead pins or unpins the slice under the pointer and stays open.
            dead_zone = self.window.preferences.radial_dead_zone()
            if event.button() == Qt.MouseButton.LeftButton:
                self.run_radial_command(self.radial_menu.command_at(event.position().toPoint(), dead_zone))
            elif event.button() == Qt.MouseButton.RightButton:
                self.toggle_radial_pin_at(event.position().toPoint())
            event.accept()
            return
        if self.starts_pan(event):
            # Panning leaves wires, the selection and the last click position alone.
            PanZoomView.mousePressEvent(self, event)
            return
        self.ctrl_handles_visible = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        self.primary_id = None   # only a backdrop press (NodeItem.mousePressEvent) sets it again
        self.viewport().update()
        if (event.button() == Qt.MouseButton.LeftButton
                and event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            scene_pos = self.mapToScene(event.position().toPoint())
            edge = self.edge_handle_at(scene_pos)
            # A Dot already sits on its noodle; Ctrl-clicking the Dot selects it, not the wire.
            if edge and self.dot_at(scene_pos) is None:
                self.start_dot_insert(edge, self.mapToScene(event.position().toPoint()))
                event.accept()
                return
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_click_scene_pos = self.mapToScene(event.position().toPoint())
        if (self.wire_source or self.wire_input) and event.button() == Qt.MouseButton.LeftButton and self.port_item_at(event.position().toPoint()) is None:
            self.cancel_wire()
            self.window.statusBar().showMessage("Wire dropped", 3000)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            item = self.itemAt(event.position().toPoint())
            while item is not None and not isinstance(item, NodeItem):
                item = item.parentItem()
            if item is not None:
                if self.window.graph_nodes()[item.key]["type"] == "Group":
                    self.window.enter_group(item.key)
                    event.accept()
                    return
                self.window.pin_panel(item.key)
        super().mouseDoubleClickEvent(event)

    def dot_at(self, scene_pos):
        radius = dot_grab_radius(self)
        for item in self.items_by_id.values():
            if item.is_dot and QLineF(item.sceneBoundingRect().center(), scene_pos).length() <= radius:
                return item
        return None

    def port_item_at(self, viewport_pos):
        """Resolve the visible socket child back to its large invisible Port target."""
        item = self.itemAt(viewport_pos)
        while item is not None:
            if isinstance(item, Port):
                return item
            item = item.parentItem()
        return None

    def mouseMoveEvent(self, event):
        self.last_hover_scene_pos = self.mapToScene(event.position().toPoint())
        if self.radial_menu.is_open() and not self.radial_menu.sustained:
            # Tracking a live flick: nothing else on the graph reacts to the pointer until Q
            # comes back up (see finish_radial_gesture).
            self.radial_menu.update_pointer(event.position().toPoint(),
                                            self.window.preferences.radial_dead_zone())
            event.accept()
            return
        visible = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        if visible != self.ctrl_handles_visible:
            self.ctrl_handles_visible = visible
            self.viewport().update()
        if self.inserting_edge is not None:
            self.update_dot_preview(self.mapToScene(event.position().toPoint()))
            event.accept()
            return
        super().mouseMoveEvent(event)
        if self.wire_source or self.wire_input:
            self.update_pending_edge(self.mapToScene(event.position().toPoint()))

    def mouseReleaseEvent(self, event):
        if self.pan is not None:
            PanZoomView.mouseReleaseEvent(self, event)
            return
        if self.inserting_edge is not None and event.button() == Qt.MouseButton.LeftButton:
            edge, source, destination, slot = self.inserting_edge
            pos = self.mapToScene(event.position().toPoint()) - QPointF(10, 10)
            self.cancel_dot_insert()
            dot_id = __import__("uuid").uuid4().hex[:12]
            # One atomic edit: a failed validation leaves the original connection untouched.
            self.window.command({"op": "batch", "commands": [
                {"op": "create", "id": dot_id, "type": "Dot", "pos": [pos.x(), pos.y()]},
                {"op": "connect", "id": dot_id, "input": "input", "source": source},
                {"op": "connect", "id": destination, "input": slot, "source": dot_id},
            ]})
            event.accept()
            return
        super().mouseReleaseEvent(event)
        edits = []
        for key, item in self.items_by_id.items():
            pos = [round(item.pos().x(), 2), round(item.pos().y(), 2)]
            if pos != self.window.graph_nodes()[key]["pos"]:
                edits.append({"op": "move", "id": key, "pos": pos})
        if edits:
            # Defer rebuild until QGraphicsScene has finished delivering this event.
            QTimer.singleShot(0, lambda: self.window.command({"op": "batch", "commands": edits}, render=False))

    def group_selection(self):
        """Ctrl+G: the selected nodes become one Group node (the S1 `group` op, one undo step)."""
        nodes = self.window.graph_nodes()
        ids = [item.key for item in self.scene().selectedItems()
               if isinstance(item, NodeItem) and nodes[item.key]["type"] not in ("Backdrop", "Input", "Output")]
        if not ids:
            self.window.statusBar().showMessage("Select the nodes to group first", 4000)
            return
        taken = {node["name"] for node in nodes.values()}
        number = 1
        while f"Group{number}" in taken:
            number += 1
        key = uuid.uuid4().hex[:12]
        if self.window.command({"op": "group", "ids": ids, "id": key, "name": f"Group{number}"}) is None:
            return
        self.scene().clearSelection()
        if key in self.items_by_id:
            self.items_by_id[key].setSelected(True)

    def ungroup_selection(self):
        """Ctrl+Shift+G: each selected Group node gives its nodes back to this graph."""
        nodes = self.window.graph_nodes()
        groups = [item.key for item in self.scene().selectedItems()
                  if isinstance(item, NodeItem) and nodes[item.key]["type"] == "Group"]
        if not groups:
            self.window.statusBar().showMessage("Select a Group to ungroup", 4000)
            return
        self.window.command({"op": "batch", "commands": [{"op": "ungroup", "id": key} for key in groups]})

    def align_selection(self):
        """Snap every selected node's x to the topmost selected node's x: one column, top to
        bottom, the direction a stream already reads in (see `Window.node_position`'s `below`).
        One batch, so it is one undo step."""
        items = [item for item in self.scene().selectedItems() if isinstance(item, NodeItem)]
        if len(items) < 2:
            self.window.statusBar().showMessage("Select two or more nodes to align", 4000)
            return
        anchor = min(items, key=lambda item: item.pos().y())
        edits = [{"op": "move", "id": item.key, "pos": [round(anchor.pos().x(), 2), round(item.pos().y(), 2)]}
                 for item in items if item is not anchor]
        if edits:
            self.window.command({"op": "batch", "commands": edits})

    # ---- radial menu (hold Q): geometry and gestures live in radialmenu.py, ------------------
    # ---- which slots hold which commands lives in radialrules.py ------------------------------

    def open_radial_menu(self):
        """The trigger key pressed (`Preferences.radial_trigger_key`, Q by default): open the
        ring at the last-known pointer position, filled for whatever is selected right now."""
        if self.radial_menu.is_open():
            return
        selected = self.selected_ids()
        self._radial_selection = selected
        self._radial_scene_pos = QPointF(self.last_hover_scene_pos)
        nodes = self.window.graph_nodes()
        self._radial_context = context_for_selection(nodes, selected)
        self._radial_node_type = (nodes[selected[0]]["type"]
                                  if len(selected) == 1 and selected[0] in nodes else None)
        commands = commands_for(nodes, selected, self.window.preferences)
        self.radial_menu.open_at(self.mapFromScene(self.last_hover_scene_pos), commands)

    def run_radial_command(self, command):
        """Execute (or, given `None`, simply cancel) the resolved slice, record the pick for
        local-usage learning, and close the menu."""
        ids, pos = self._radial_selection, self._radial_scene_pos
        context, node_type = self._radial_context, self._radial_node_type
        self.radial_menu.close_menu()
        if command is not None:
            command.run(self.window, ids, pos)
            self.window.preferences.record_radial_usage(context, node_type, command.id)

    def finish_radial_gesture(self):
        """Q released while the menu was still tracking the live flick: a flick past the dead
        zone runs the highlighted slice; a tap (never left the dead zone) pins the menu open."""
        if self.radial_menu.flicked():
            self.run_radial_command(self.radial_menu.command_at_highlight())
        else:
            self.radial_menu.sustain()

    def toggle_radial_pin_at(self, viewport_pos):
        """Right-click a slice on a sustained ring: pin it to this context+type bucket so it
        never moves again, or unpin it if it is already the pin there. The dead zone and an empty
        slot do nothing -- there is nothing to pin."""
        command = self.radial_menu.command_at(viewport_pos, self.window.preferences.radial_dead_zone())
        if command is None:
            return
        slot = self.radial_menu.commands.index(command)
        pinned = self.window.preferences.toggle_radial_pin(
            self._radial_context, self._radial_node_type, slot, command.id)
        self.window.statusBar().showMessage(
            f"Pinned {command.label!r} to this slice" if pinned else f"Unpinned {command.label!r}", 3000)
        commands = commands_for(self.window.graph_nodes(), self._radial_selection, self.window.preferences)
        self.radial_menu.refresh(commands)

    def open_add_radial_command(self):
        """The ring's own "+ Add command..." button (only visible once the ring is sustained):
        open the Commands editor to write a new user command, seeded from whatever is selected
        right now, then close the ring."""
        ids = list(self._radial_selection)
        self.radial_menu.close_menu()
        self.window.open_radial_commands_editor(new_from_selection=ids)

    def open_presets_browser(self):
        """The sustained radial ring's Presets button opens the shared preset browser."""
        ids = list(self._radial_selection)
        self.radial_menu.close_menu()
        self.window.node_toolbar.open_presets_browser(selected_ids=ids)

    def _selected_node_data(self):
        nodes = self.window.graph_nodes()
        return [{"type": nodes[item.key]["type"], "params": copy.deepcopy(nodes[item.key]["params"]),
                 "pos": list(nodes[item.key]["pos"])}
                for item in self.scene().selectedItems() if isinstance(item, NodeItem)]

    def _paste_nodes(self, node_data):
        commands, pasted = [], []
        for node in node_data:
            key = __import__("uuid").uuid4().hex[:12]
            x, y = node["pos"]
            commands.append({"op": "create", "id": key, "type": node["type"],
                             "params": copy.deepcopy(node["params"]), "pos": [x + 40, y + 40]})
            pasted.append(key)
        if not commands or self.window.command({"op": "batch", "commands": commands}) is None:
            return
        self.scene().clearSelection()
        for key in pasted:
            self.items_by_id[key].setSelected(True)

    def _clipboard_paste(self):
        try:
            payload = json.loads(QApplication.clipboard().text())
            if (not isinstance(payload, list)
                    or any(not isinstance(node, dict) for node in payload)):
                return
            for node in payload:
                if (node.get("type") not in SPECS or not isinstance(node.get("params"), dict)
                        or not isinstance(node.get("pos"), list) or len(node["pos"]) != 2
                        or not all(isinstance(value, (int, float)) for value in node["pos"])):
                    return
        except (json.JSONDecodeError, TypeError, ValueError):
            return
        self._paste_nodes(payload)

    def keyPressEvent(self, event):
        key = self.selected_id()
        modifiers = event.modifiers()
        if (event.key() == self.window.preferences.radial_trigger_key() and not modifiers
                and not event.isAutoRepeat()):
            self.open_radial_menu()
        elif event.key() == Qt.Key.Key_A and modifiers == Qt.KeyboardModifier.ControlModifier:
            self.scene().clearSelection()
            for item in self.items_by_id.values():
                item.setSelected(True)
        elif event.key() == Qt.Key.Key_C and modifiers == Qt.KeyboardModifier.ControlModifier:
            QApplication.clipboard().setText(json.dumps(self._selected_node_data()))
        elif event.key() == Qt.Key.Key_V and modifiers == Qt.KeyboardModifier.ControlModifier:
            self._clipboard_paste()
        elif event.key() == Qt.Key.Key_X and modifiers == Qt.KeyboardModifier.ControlModifier:
            QApplication.clipboard().setText(json.dumps(self._selected_node_data()))
            edits = [{"op": "delete", "id": item.key} for item in self.scene().selectedItems()
                     if isinstance(item, NodeItem)]
            self.window.command({"op": "batch", "commands": edits})
        elif event.key() == Qt.Key.Key_C and modifiers == Qt.KeyboardModifier.AltModifier:
            self._paste_nodes(self._selected_node_data())
        elif event.key() == Qt.Key.Key_G and modifiers == Qt.KeyboardModifier.ControlModifier:
            self.group_selection()
        elif event.key() == Qt.Key.Key_G and modifiers == (Qt.KeyboardModifier.ControlModifier
                                                           | Qt.KeyboardModifier.ShiftModifier):
            self.ungroup_selection()
        elif event.key() == Qt.Key.Key_Tab and not modifiers:
            self.window.node_search()
        elif event.key() == Qt.Key.Key_F and not modifiers:
            self.fit()
        elif event.key() == Qt.Key.Key_Escape and not modifiers:
            self.cancel_wire()
            self.cancel_dot_insert()
            if self.radial_menu.is_open():
                self.radial_menu.close_menu()
        elif (Qt.Key.Key_1 <= event.key() <= Qt.Key.Key_9 and not modifiers and key
              and self.window.graph_path):
            self.window.statusBar().showMessage("The viewer shows the top level: leave the group to view a node", 4000)
        elif (Qt.Key.Key_1 <= event.key() <= Qt.Key.Key_9 and not modifiers and key):
            # Nuke's viewer inputs: 1 is the classic "view this node"; 2-9 fill further inputs.
            self.window.command({"op": "viewer_input", "slot": event.key() - Qt.Key.Key_0, "id": key})
        elif event.key() == Qt.Key.Key_D and not modifiers and key:
            self.window.command({"op": "disable", "id": key, "value": not self.window.graph_nodes()[key]["disabled"]})
        elif event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and not modifiers:
            edits = [{"op": "delete", "id": item.key} for item in self.scene().selectedItems() if isinstance(item, NodeItem)]
            self.window.command({"op": "batch", "commands": edits})
        elif event.key() == Qt.Key.Key_Period and not modifiers:
            self.window.add_node("Dot")
        elif event.key() in (Qt.Key.Key_R, Qt.Key.Key_G, Qt.Key.Key_M, Qt.Key.Key_T, Qt.Key.Key_B, Qt.Key.Key_C, Qt.Key.Key_S, Qt.Key.Key_O,
                              Qt.Key.Key_P, Qt.Key.Key_U, Qt.Key.Key_W) and not modifiers:
            self.window.add_node({Qt.Key.Key_R: "Read", Qt.Key.Key_G: "Grade", Qt.Key.Key_M: "Merge", Qt.Key.Key_T: "Transform",
                                   Qt.Key.Key_B: "Blur", Qt.Key.Key_C: "ColorCorrect", Qt.Key.Key_S: "Shuffle", Qt.Key.Key_O: "Roto",
                                   Qt.Key.Key_P: "Premult", Qt.Key.Key_U: "Unpremult",
                                   Qt.Key.Key_W: "Write"}[event.key()])
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Control:
            self.ctrl_handles_visible = False
            self.viewport().update()
        elif (event.key() == self.window.preferences.radial_trigger_key() and not event.isAutoRepeat()
              and self.radial_menu.is_open() and not self.radial_menu.sustained):
            self.finish_radial_gesture()
        super().keyReleaseEvent(event)

    def edge_handle_at(self, scene_pos):
        radius = 14 / max(self.transform().m11(), 0.05)
        for edge, source, key, slot, _out in self.edges:
            handle = getattr(edge, "handle", None)
            if handle and math.hypot(handle.x() - scene_pos.x(), handle.y() - scene_pos.y()) <= radius:
                return edge, source, key, slot
        return None

    def drawBackground(self, painter, rect):
        super().drawBackground(painter, rect)
        if self.transform().m11() < 0.25:
            return
        painter.setPen(getattr(self, "grid_pen", None) or QPen(QColor(grid_color()), 1))
        left, top = math.floor(rect.left() / 32) * 32, math.floor(rect.top() / 32) * 32
        points = [QPointF(x, y) for x in range(left, int(rect.right()), 32) for y in range(top, int(rect.bottom()), 32)]
        painter.drawPoints(points)

    def drawForeground(self, painter, rect):
        super().drawForeground(painter, rect)
        if not self.ctrl_handles_visible:
            return
        scale = max(self.transform().m11(), 0.05)
        radius = 8 / scale
        painter.setPen(QPen(QColor("#f0c39d"), max(1.5 / scale, 0.75)))
        painter.setBrush(QColor("#242428"))
        for edge, *_ in self.edges:
            if hasattr(edge, "handle"):
                painter.drawEllipse(edge.handle, radius, radius)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor("#f0c39d"))
                painter.drawEllipse(edge.handle, radius * 0.35, radius * 0.35)
                painter.setPen(QPen(QColor("#f0c39d"), max(1.5 / scale, 0.75)))
                painter.setBrush(QColor("#242428"))

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(NODE_KIND_MIME_TYPE):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(NODE_KIND_MIME_TYPE):
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        # A node dropped from the NODES dock lands exactly where it was dropped, unlike a click
        # (which goes to add_node's own selected-node/graph-centre rule) or Tab search (last click).
        if event.mimeData().hasFormat(NODE_KIND_MIME_TYPE):
            kind = bytes(event.mimeData().data(NODE_KIND_MIME_TYPE)).decode("utf-8")
            self.window.add_node(kind, position=self.mapToScene(event.position().toPoint()))
            event.acceptProposedAction()
        else:
            super().dropEvent(event)


class PreviewSignals(QObject):
    finished = Signal(object, object, object, str, object)
    # A stand-in picture shown while the real one renders: (payload, image, status, scale, region).
    interim = Signal(object, object, str, int, object)
    # A finished node thumbnail: (node id, thumbnail key, image).
    thumbnail = Signal(str, str, object)
    # Progress inside one slow frame: (payload, stage, fraction, info). See renderprogress.py.
    progress = Signal(object, str, float, object)


class ProjectSettingsDialog(QDialog):
    """Small project-settings surface modelled after a compositor's project settings.

    The bundled ACES config, display, and ACEScg processing space are explicit rather than
    magical constants. They are read-only until external OCIO configs are supported; the artist
    can choose the saved default view and the viewer background today.
    """
    CUSTOM_ACCENT = "Custom…"

    def __init__(self, settings, parent=None, theme=DEFAULT_THEME, thumbnails=True, accent=None,
                 max_panels=5, cache_float32=False):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(480)
        layout = QVBoxLayout(self)
        interface_heading = QLabel("INTERFACE")
        interface_heading.setObjectName("brand")
        layout.addWidget(interface_heading)
        interface = QFormLayout()
        self.theme = QComboBox()
        self.theme.addItems(list(THEMES))
        self.theme.setCurrentText(theme if theme in THEMES else DEFAULT_THEME)
        self.theme.setToolTip("Applies immediately to the whole application")
        interface.addRow("Theme colour", self.theme)
        self.accent = QComboBox()
        self.accent.setObjectName("accent")
        for name, value in ACCENTS.items():
            self.accent.addItem(name, value)
        accent = valid_accent(accent)
        if accent is not None and self.accent.findData(accent) < 0:
            self.accent.addItem(f"Custom {accent}", accent)
        self.accent.addItem(self.CUSTOM_ACCENT, "custom")
        self.accent.setCurrentIndex(max(0, self.accent.findData(accent)))
        self._accent_index = self.accent.currentIndex()
        self.accent.setToolTip("Highlight colour for focus rings, selections and headings")
        self.accent.activated.connect(self._accent_activated)
        interface.addRow("Accent colour", self.accent)
        self.thumbnails = QCheckBox("Show thumbnails on nodes")
        self.thumbnails.setChecked(bool(thumbnails))
        self.thumbnails.setToolTip("Each node shows a small picture of its output at the current frame")
        interface.addRow("Node graph", self.thumbnails)
        self.cache_float32 = QCheckBox("Cache display frames as float32 (no half-float rounding)")
        self.cache_float32.setChecked(bool(cache_float32))
        self.cache_float32.setToolTip("Uses roughly 2x the display-cache memory per cached frame. "
                                      "Depth, position, motion-vector and UV/ST passes are always "
                                      "kept at float32 regardless of this setting.")
        interface.addRow("Display cache precision", self.cache_float32)
        self.max_panels = QSpinBox()
        self.max_panels.setRange(1, 20)
        self.max_panels.setValue(int(max_panels))
        self.max_panels.setObjectName("max-panels-spin")
        interface.addRow("Max properties panels", self.max_panels)
        layout.addLayout(interface)
        theme_note = QLabel("The theme is stored per machine, not in the project: a comp handed to "
                            "another artist keeps their colours, not yours. Node colours stay fixed "
                            "so a node family always reads the same.")
        theme_note.setWordWrap(True)
        theme_note.setObjectName("muted")
        layout.addWidget(theme_note)
        heading = QLabel("COLOR MANAGEMENT")
        heading.setObjectName("brand")
        layout.addWidget(heading)
        form = QFormLayout()
        color = settings["color"]
        config_name = QLabel("ACES CG Config v4.0 · ACES 2.0")
        config_name.setToolTip(color["config"])
        form.addRow("OCIO config", config_name)
        form.addRow("Working space", QLabel(color["working_space"] + " · scene-linear float32"))
        form.addRow("Display", QLabel(color["display"]))
        self.view = QComboBox()
        self.view.addItems(VIEWS)
        self.view.setCurrentText(color["view"])
        form.addRow("Default view", self.view)
        self.background = QComboBox()
        self.background.addItem("Pure black", "black")
        self.background.addItem("Checkerboard", "checker")
        self.background.setCurrentIndex(max(0, self.background.findData(settings["viewer"]["background"])))
        form.addRow("Viewer background", self.background)
        layout.addLayout(form)
        note = QLabel("Input nodes still own source color-space overrides. Auto-detected inputs are "
                      "converted into ACEScg before entering the graph.")
        note.setWordWrap(True)
        note.setObjectName("muted")
        layout.addWidget(note)
        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        apply = QPushButton("Apply")
        apply.setDefault(True)
        apply.clicked.connect(self.accept)
        buttons.addWidget(cancel)
        buttons.addWidget(apply)
        layout.addLayout(buttons)

    def changes(self):
        return {"color": {"view": self.view.currentText()},
                "viewer": {"background": self.background.currentData()}}

    def _accent_activated(self, index):
        if self.accent.itemData(index) != "custom":
            self._accent_index = index
            return
        from PySide6.QtWidgets import QColorDialog
        start = self.accent.itemData(self._accent_index) or "#83cbb7"
        color = QColorDialog.getColor(QColor(start), self, "Accent colour")
        if not color.isValid():
            self.accent.setCurrentIndex(self._accent_index)
            return
        value = color.name().lower()
        found = self.accent.findData(value)
        if found < 0:
            found = self.accent.count() - 1
            self.accent.insertItem(found, f"Custom {value}", value)
        self.accent.setCurrentIndex(found)
        self._accent_index = found

    def chosen_accent(self):
        value = self.accent.currentData()
        return None if value == "custom" else value

    def chosen_theme(self):
        """The interface theme, returned separately from `changes` because it is a machine
        preference rather than a document setting and must not travel inside the comp."""
        return self.theme.currentText()

    def chosen_max_panels(self):
        return self.max_panels.value()


def _is_data_target(document, target):
    # Intentionally narrow v1 rule: only a directly viewed, explicitly Raw Read is data;
    # passes viewed through Shuffle, Merge, or another node are still cached as half.
    node = (document.get("nodes") or {}).get(target)
    return bool(node and node.get("type") == "Read"
                and (node.get("params") or {}).get("colorspace") == "Raw")


class Window(QMainWindow):
    def _fit_workspace_toolbar(self):
        toolbar = getattr(self, "workspace_toolbar", None)
        if toolbar is None or not hasattr(self, "_toolbar_overflow"):
            return
        for action in self._toolbar_overflow:
            action.setVisible(True)
        self._toolbar_more_action.setVisible(False)
        if toolbar.sizeHint().width() <= toolbar.width():
            return
        self._toolbar_more_action.setVisible(True)
        for action in self._toolbar_overflow:
            action.setVisible(False)
            if toolbar.sizeHint().width() <= toolbar.width():
                break

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "_toolbar_reflow_timer"):
            self._toolbar_reflow_timer.start(0)

    def __init__(self, document=None, agent_name=None):
        super().__init__()
        self.dispatcher = Dispatcher(document or demo_document())
        self.saved_document = copy.deepcopy(self.dispatcher.document)
        self.preferences = Preferences()
        self.theme_name = self.preferences.theme()
        self.accent_color = self.preferences.accent()
        self.show_thumbnails = self.preferences.thumbnails()
        self.properties_tab = 0
        self.pinned_panels = []  # node keys, most-recent-first; independent of graph selection
        self.panel_cap = self.preferences.max_properties_panels()
        self.rendered_identity = None
        # node id -> (thumbnail_key, QImage). Lives on the window so a graph rebuild keeps them.
        self.thumbnails = {}
        self.thumbnail_cancel = threading.Event()
        self.thumbnails_closed = False
        self.thumbnail_timer = QTimer(self)
        self.thumbnail_timer.setSingleShot(True)
        self.thumbnail_timer.setInterval(250)
        self.thumbnail_timer.timeout.connect(self.schedule_thumbnails)
        # Where the sequence browser opens next. Kept on the window rather than in the document:
        # it is navigation history, not part of the comp.
        self.last_browse_directory = None
        self.project_path = None
        self.keyboard_shortcuts_dialog = None
        self.update_exit = False
        self.frame = None
        # The group ids leading to the graph the node graph shows; empty is the top level.
        self.graph_path = []
        # The B buffer of the viewer compare, aligned to `frame`, and the per-request extras the
        # render worker hands over (B's frame and picture), keyed by (generation, frame).
        self.frame_b = None
        self.compare_results = {}
        self.frame_generation = -1
        self.generation = 0
        # QStatusBar.currentMessage() is transient: an asynchronous preview can replace it while
        # a deferred document command is reporting an error. Keep command failures on a stable
        # surface as well, so the actionable validation error remains visible after the event loop
        # advances.
        self.last_command_error = None
        self.busy = False
        # Bounded live history of evaluation errors, oldest evicted first. Exists so an attached
        # agent can see every render failure through the "errors" op (see agent_command) instead
        # of only the single most recent one visible in the status bar -- read-ahead can error on
        # a frame that never becomes "current" and reaches viewer_info at all.
        self.render_errors = deque(maxlen=200)
        self.preview_queue = PlaybackQueue()
        self.display_cache = DisplayCache(force_float32=self.preferences.display_cache_float32())
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nodebased-preview")
        # QOffscreenSurface must be created on the GUI thread; the QOpenGLContext bound to
        # it is built lazily on nodebased-preview (the single worker thread above) on its
        # first display request and used only from that thread afterward. See the design
        # note at the top of gpudisplay.py.
        gpu_surface = QOffscreenSurface()
        gpu_surface.setFormat(gpudisplay.context_format())
        gpu_surface.create()
        gpudisplay.configure_surface(gpu_surface, owner_thread_prefix="nodebased-preview")
        self._gpu_surface = gpu_surface
        self._tracker_future = None
        self._tracker_cancel = None
        self._tracker_index = None
        self._tracker_seed = None
        self._tracker_selected_index = None
        self._tracker_key = None
        self._tracker_job = None
        # The desktop app is where the persistent disk tier is switched on: results evicted from
        # memory survive a restart, so reopening yesterday's comp does not recompute it.
        self.evaluator = Evaluator(disk=DiskCache.shared())
        # Installed once: it routes progress per thread, and its presence makes this window's
        # renders interactive, so a large CPU splat frame shows time left instead of being refused.
        self.render_progress_router = ThreadProgress()
        self.evaluator.progress = self.render_progress_router
        # Bounded, separately-threaded read-ahead for Read-node source decode only -- see
        # decodepool.py. Decode shares no state with the single-owner Evaluator/TileCache above,
        # so it is safe to run several of these in parallel while the preview worker stays single.
        self.decode_pool = DecodeAheadPool()
        # Preview takes the tile path when every upstream node supports it. The executor reports
        # an explicit fallback for unsupported graphs; export remains the reference evaluator.
        self.tile_executor = TileExecutor(evaluator=self.evaluator, decode_pool=self.decode_pool)
        self.signals = PreviewSignals()
        self.signals.finished.connect(self.preview_ready)
        self.signals.interim.connect(self.preview_interim)
        self.signals.progress.connect(self.preview_progress)
        self.signals.thumbnail.connect(self.thumbnail_ready)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(35)
        self.timer.timeout.connect(self.start_preview)
        self.playing = False
        self.playback_origin_frame = 0
        self.playback_origin_time = 0.0
        self.playback_elapsed_frames = 0
        self.playback_dropped_frames = 0
        # Nuke-style playback fallback: count of primary (display) frames actually completed
        # since playback started. playback_tick never lets the playhead advance more than one
        # frame past this, so a render pipeline that can't sustain real time degrades to slower,
        # strictly in-order playback instead of racing ahead on the wall clock and jumping to
        # wherever it landed -- including backward -- once a slow frame finally finishes.
        self.playback_frames_rendered = 0
        # Set only while playback has auto-switched the proxy combo away from an artist's
        # explicit "Full" choice; holds the index to restore on stop. See toggle_playback.
        self.playback_auto_proxy_index = None
        # Frame the properties panel was last built for. Animated knobs read their value at the
        # playhead, so the panel has to be rebuilt when that moves; this is how the rebuild is
        # kept to actual frame changes. See refresh_animated_panel.
        self.panel_frame = None
        self.playback_timer = QTimer(self)
        self.playback_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.playback_timer.timeout.connect(self.playback_tick)
        self.setWindowTitle("NodeBased · Untitled")
        self.setMinimumSize(800, 500)
        icon_path = resource_path("assets/nodebased-icon.png")
        if icon_path.is_file():
            self.setWindowIcon(QIcon(str(icon_path)))
        self.resize(*DEFAULT_WINDOW_SIZE)
        toolbar = QToolBar("Workspace")
        # saveState() identifies toolbars and docks by objectName; an unnamed one is skipped.
        toolbar.setObjectName("workspace-toolbar")
        toolbar.setMovable(True)
        self.addToolBar(toolbar)
        self.workspace_toolbar = toolbar
        brand = QLabel("◈  NODEBASED")
        brand.setObjectName("brand")
        toolbar.addWidget(brand)
        toolbar.addSeparator()
        primary_actions = []
        for name, callback in [("Open image", self.read_file), ("Add node", self.add_node),
                               ("Save project", self.save_project), ("Export image", self.export)]:
            action = toolbar.addAction(name)
            action.triggered.connect(lambda checked=False, fn=callback: fn())
            primary_actions.append(action)
        viewport_action = toolbar.addAction("3D viewport")
        viewport_action.setToolTip("Show the navigable 3D editor viewport")
        viewport_action.triggered.connect(lambda: self.viewport_dock.setVisible(not self.viewport_dock.isVisible()))
        slice_action = toolbar.addAction("Slice viewer")
        slice_action.setToolTip("Show an axis-aligned slice through the selected fluid node's volume")
        slice_action.triggered.connect(lambda: self.slice_dock.setVisible(not self.slice_dock.isVisible()))
        cache_action = toolbar.addAction("Cache inspector")
        cache_action.setToolTip("Show the frame list and stats behind the selected cache node")
        cache_action.triggered.connect(lambda: self.cache_inspector_dock.setVisible(not self.cache_inspector_dock.isVisible()))
        self.toolbar_more = QToolButton()
        self.toolbar_more.setText("More")
        self.toolbar_more.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.toolbar_more_menu = QMenu(self.toolbar_more)
        self.toolbar_more.setMenu(self.toolbar_more_menu)
        self._toolbar_more_action = toolbar.addWidget(self.toolbar_more)
        self._toolbar_overflow = [cache_action, slice_action, viewport_action,
                                  primary_actions[3], primary_actions[1], primary_actions[0]]
        for candidate in self._toolbar_overflow:
            menu_action = self.toolbar_more_menu.addAction(candidate.text())
            menu_action.triggered.connect(candidate.trigger)
        self._toolbar_more_action.setVisible(False)
        toolbar.addSeparator()
        info = QLabel("  2D WORKSPACE")
        info.setObjectName("muted")
        toolbar.addWidget(info)
        # An expanding spacer is how a QToolBar right-justifies: everything added after it is
        # pushed to the trailing edge and stays there as the window is resized.
        spacer = QWidget()
        spacer.setObjectName("toolbarSpacer")
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)
        self.update_button = QPushButton("Check for updates")
        self.update_button.setObjectName("update")
        self.update_button.setToolTip(f"NodeBased {__version__}")
        toolbar.addWidget(self.update_button)
        self._toolbar_reflow_timer = QTimer(self)
        self._toolbar_reflow_timer.setSingleShot(True)
        self._toolbar_reflow_timer.timeout.connect(self._fit_workspace_toolbar)
        self.updater = Updater(self)
        self.updater.changed.connect(self.update_status)
        self.update_button.clicked.connect(self.update_clicked)
        self.setDockOptions(QMainWindow.DockOption.AnimatedDocks |
                            QMainWindow.DockOption.AllowNestedDocks |
                            QMainWindow.DockOption.AllowTabbedDocks |
                            QMainWindow.DockOption.GroupedDragging)
        # A zero-minimum placeholder keeps QMainWindow's central-area contract without pinning
        # the layout. Every useful workspace surface lives in a movable dock.
        central = QWidget(self)
        central.setMinimumSize(0, 0)
        central.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setCentralWidget(central)
        viewer_panel = QWidget()
        vl = QVBoxLayout(viewer_panel)
        vl.setContentsMargins(0, 0, 0, 0)
        controls_widget = QWidget()
        controls = QHBoxLayout(controls_widget)
        controls.addWidget(QLabel("  VIEWER"))
        self.channels = QComboBox()
        self.channels.addItems(["RGB", "R", "G", "B", "A"])
        self.channels.currentTextChanged.connect(self.request_preview)
        controls.addWidget(self.channels)
        self.display_view = QComboBox()
        self.display_view.addItems(VIEWS)
        self.display_view.setCurrentText(self.dispatcher.document["settings"]["color"]["view"])
        self.display_view.setToolTip("Display transform only; exports stay independent of the viewer")
        self.display_view.currentTextChanged.connect(self.request_preview)
        controls.addWidget(self.display_view)
        # Proxy is viewer state, never document state: it is how one artist is looking at the comp
        # right now. Export and the agent's render op always evaluate at tier 1 (clause C3).
        self.proxy = QComboBox()
        for label, tier in (("Full", 1), ("1/2", 2), ("1/4", 4), ("1/8", 8)):
            self.proxy.addItem(label, tier)
        self.proxy.setToolTip("Proxy resolution for the viewer only. Sources generate at this "
                              "scale and pixel-unit parameters scale with them; exports are "
                              "always full resolution.")
        # currentIndexChanged renders (playback also switches the combo without the artist);
        # `activated` fires only for the artist's own pick, which is what gets saved with the file.
        self.proxy.currentIndexChanged.connect(self.request_preview)
        self.proxy.activated.connect(lambda _index: self.set_viewer_proxy(self.proxy.currentData()))
        self._proxy_seen = 1
        self._last_proxy = 2
        controls.addWidget(self.proxy)
        # Proxy-while-playing used to be unconditional. It is the right default -- native 4K
        # through ACES 2.0 cannot hit real time on the CPU -- but "the viewer silently changed
        # resolution when I pressed play" is a decision the artist gets to make, not one the app
        # makes for them. Unchecking it plays at whatever tier is selected, however slow that is.
        self.playback_proxy = QCheckBox("Proxy while playing")
        self.playback_proxy.setChecked(True)
        self.playback_proxy.setToolTip(
            "Drop to the smallest proxy tier that can keep up, for the duration of playback only, "
            "then restore the tier you had. Never overrides a proxy tier you chose yourself.\n"
            "Uncheck to always play at the selected tier.")
        controls.addWidget(self.playback_proxy)
        # Gain, gamma, the clipping warning and the display choice are viewer state saved in the
        # document (settings.viewer.look) and applied to the picture on screen only. The gain
        # spinbox is the old Exposure control under Nuke's name; `self.exposure` stays its name.
        controls.addWidget(QLabel("Gain"))
        self.exposure = QDoubleSpinBox()
        self.exposure.setRange(*VIEWER_GAIN_RANGE)
        self.exposure.setSingleStep(0.25)
        self.exposure.setToolTip("Viewer gain in f-stops, before the display transform. Display only: "
                                 "never written to the document's pixels or to a Write.")
        self.exposure.valueChanged.connect(lambda value: self.set_viewer_look(gain=value))
        controls.addWidget(self.exposure)
        controls.addWidget(self._look_reset("gain"))
        controls.addWidget(QLabel("Gamma"))
        self.gamma = QDoubleSpinBox()
        self.gamma.setRange(*VIEWER_GAMMA_RANGE)
        self.gamma.setDecimals(2)
        self.gamma.setSingleStep(0.1)
        self.gamma.setValue(1.0)
        self.gamma.setToolTip("Viewer gamma, after gain and before the display transform. Display only.")
        self.gamma.valueChanged.connect(lambda value: self.set_viewer_look(gamma=value))
        controls.addWidget(self.gamma)
        controls.addWidget(self._look_reset("gamma"))
        self.zebra = QCheckBox(f"Zebra >{ZEBRA_HIGH:g} <{ZEBRA_LOW:g}")
        self.zebra.setToolTip(f"Clipping warning: stripes pixels whose viewed scene-linear value is above "
                              f"{ZEBRA_HIGH:g} (red) or below {ZEBRA_LOW:g} (blue). Display only.")
        self.zebra.toggled.connect(lambda on: self.set_viewer_look(zebra=on))
        controls.addWidget(self.zebra)
        self.viewer_display = QComboBox()
        self.viewer_display.addItem("Project view")
        self.viewer_display.addItems(viewer_displays())
        self.viewer_display.setToolTip("Display transform for this viewer: the project's default view, or "
                                       "any view of the fixed ACES config ('Raw' shows scene-linear "
                                       "values untransformed). The pixel readout stays scene-linear.")
        self.viewer_display.activated.connect(
            lambda _index: self.set_viewer_look(display=self.viewer_display.currentText()))
        controls.addWidget(self.viewer_display)
        self.roi_button = QPushButton("ROI")
        self.roi_button.setCheckable(True)
        self.roi_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.roi_button.setToolTip("Region of interest: evaluate and show only the box in the viewer; the "
                                   "rest keeps the last full picture, dimmed. Drag the box or its edges; "
                                   "Shift-drag draws a new one. Display only.")
        self.roi_button.toggled.connect(self.toggle_viewer_roi)
        controls.addWidget(self.roi_button)
        self.mask_choice = QComboBox()
        self.mask_choice.addItems(VIEWER_MASKS)
        self.mask_choice.setToolTip("Format mask aspect ratio ('format' is the frame's own). Display only.")
        self.mask_choice.activated.connect(lambda _i: self.set_viewer_mask(mask=self.mask_choice.currentText()))
        controls.addWidget(self.mask_choice)
        self.mask_mode = QComboBox()
        self.mask_mode.addItems(VIEWER_MASK_MODES)
        self.mask_mode.setToolTip("Mask mode: none, lines at the mask edge, half-dark or black outside. Display only.")
        self.mask_mode.activated.connect(lambda _i: self.set_viewer_mask(mode=self.mask_mode.currentText()))
        controls.addWidget(self.mask_mode)
        fit = QPushButton("Fit")
        fit.clicked.connect(lambda: self.viewer.fit())
        controls.addWidget(fit)
        one = QPushButton("1:1")
        one.clicked.connect(lambda: self.viewer.resetTransform())
        controls.addWidget(one)
        controls.addStretch()
        # Elided, so the length of the playback status can never resize the layout around it.
        self.viewer_info = ElidedLabel("Waiting for image")
        self.viewer_info.setObjectName("muted")
        self.viewer_info.setMinimumWidth(180)
        controls.addWidget(self.viewer_info, 1)
        self.command_error_label = ElidedLabel("")
        self.command_error_label.setObjectName("command-error")
        self.command_error_label.setStyleSheet("color: #e3b18d")
        # Same rule for the status bar: a long validation message must report a problem, not
        # cause a second one by stretching the window it is reported in.
        self.statusBar().addPermanentWidget(self.command_error_label, 1)
        self.render_progress = QProgressBar()
        self.render_progress.setObjectName("render-progress")
        self.render_progress.setRange(0, 1000)
        self.render_progress.setFixedWidth(180)
        self.render_progress.setTextVisible(False)
        self.render_progress.hide()
        self.statusBar().addPermanentWidget(self.render_progress)
        controls_scroll = QScrollArea()
        controls_scroll.setObjectName("viewer-controls-scroll")
        controls_scroll.setWidgetResizable(False)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        controls_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        controls_scroll.setFrameShape(QFrame.Shape.NoFrame)
        controls_scroll.setMinimumWidth(0)
        controls_scroll.setSizeAdjustPolicy(QScrollArea.SizeAdjustPolicy.AdjustIgnored)
        controls_widget.adjustSize()
        controls_scroll.setWidget(controls_widget)
        controls_scroll.setFixedHeight(max(controls_widget.sizeHint().height() + 4, 32))
        vl.addWidget(controls_scroll)
        self.viewer = Viewer(self)
        self.viewer.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        # The viewer is a window onto an image of any size; its own footprint must not follow the
        # comp's format. AdjustIgnored is Qt's default and is stated here because it is load-bearing
        # for that promise, not incidental.
        self.viewer.setSizeAdjustPolicy(QGraphicsView.SizeAdjustPolicy.AdjustIgnored)
        vl.addWidget(self.viewer)
        vl.addLayout(self._timeline())
        self.viewer_panel = viewer_panel
        viewer_panel.setMinimumSize(0, 0)
        viewer_panel.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        graph_panel = QWidget()
        gl = QVBoxLayout(graph_panel)
        gl.setContentsMargins(0, 0, 0, 0)
        # Elided: one long single-line hint must not set the floor for the whole window's width.
        help_label = ElidedLabel("  NODE GRAPH     Tab search/add  ·  R/G/M/T/B/C/S/O/P/U/W create  ·  Period Dot  ·  1 view  ·  D bypass  ·  F frame  ·  Ctrl+A select all  ·  Ctrl+C/X/V copy/cut/paste  ·  Alt+C duplicate  ·  Ctrl+G group / Ctrl+Shift+G ungroup  ·  MMB or Alt+drag pan  ·  "
                                 "drag output ↔ input to wire  ·  Ctrl-drag noodle midpoint inserts Dot  ·  click a wired input to rewire")
        help_label.setObjectName("graph-shortcuts-hint")
        gl.addWidget(help_label)
        self.breadcrumbs = QWidget()
        self.breadcrumbs.setObjectName("graph-breadcrumbs")
        crumb_row = QHBoxLayout(self.breadcrumbs)
        crumb_row.setContentsMargins(8, 0, 8, 0)
        gl.addWidget(self.breadcrumbs)
        self.graph = Graph(self)
        self.graph.setSizeAdjustPolicy(QGraphicsView.SizeAdjustPolicy.AdjustIgnored)
        gl.addWidget(self.graph)
        # Tab is graph-contextual: it follows the mouse over the graph even if a
        # dock/editor owns Qt keyboard focus. QGraphicsView otherwise uses Tab for
        # focus traversal before Graph.keyPressEvent can see it.
        QApplication.instance().installEventFilter(self)
        self.graph_panel = graph_panel
        self.viewer_dock = QDockWidget("2D VIEWER", self)
        self.viewer_dock.setObjectName("viewer-dock")
        self.viewer_dock.setMinimumWidth(0)
        self.viewer_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable |
                                     QDockWidget.DockWidgetFeature.DockWidgetMovable |
                                     QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.viewer_dock.setWidget(viewer_panel)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.viewer_dock)
        self.graph_dock = QDockWidget("NODE GRAPH", self)
        self.graph_dock.setObjectName("node-graph-dock")
        self.graph_dock.setMinimumWidth(0)
        self.graph_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable |
                                    QDockWidget.DockWidgetFeature.DockWidgetMovable |
                                    QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.graph_dock.setWidget(graph_panel)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.graph_dock)
        self.splitDockWidget(self.viewer_dock, self.graph_dock, Qt.Orientation.Vertical)
        dock = QDockWidget("PROPERTIES", self)
        dock.setObjectName("properties-dock")
        dock.setMinimumWidth(0)
        self.properties = QScrollArea()
        self.properties.setWidgetResizable(True)
        # Panels reflow to the dock's width (see make_fluid), so sideways scrolling would only
        # ever hide content; widen the dock instead.
        self.properties.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        dock.setWidget(self.properties)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.properties_dock = dock
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable |
                         QDockWidget.DockWidgetFeature.DockWidgetMovable |
                         QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        nodes_dock = QDockWidget("NODES", self)
        nodes_dock.setObjectName("nodes-dock")
        self.node_toolbar = NodeToolbar(self)
        nodes_dock.setWidget(self.node_toolbar)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, nodes_dock)
        self.nodes_dock = nodes_dock
        nodes_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable |
                               QDockWidget.DockWidgetFeature.DockWidgetMovable |
                               QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.node_search_shortcut = QShortcut(QKeySequence("Ctrl+F"), self)
        self.node_search_shortcut.activated.connect(self.focus_node_search)
        self.viewport_dock = QDockWidget("3D VIEWPORT", self)
        self.viewport_dock.setMinimumWidth(0)
        self.viewport_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable |
                                       QDockWidget.DockWidgetFeature.DockWidgetMovable |
                                       QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.viewport_dock.setObjectName("viewport3d-dock")
        self.viewport = Viewport3D(self)
        self.viewport_dock.setWidget(self.viewport)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.viewport_dock)
        self.tabifyDockWidget(self.nodes_dock, self.viewport_dock)
        self.viewport_dock.hide()
        self.slice_dock = QDockWidget("SLICE VIEWER", self)
        self.slice_dock.setMinimumWidth(0)
        self.slice_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable |
                                    QDockWidget.DockWidgetFeature.DockWidgetMovable |
                                    QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.slice_dock.setObjectName("slice-viewer-dock")
        self.slice_view = SliceView(self)
        self.slice_dock.setWidget(self.slice_view)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.slice_dock)
        self.tabifyDockWidget(self.properties_dock, self.slice_dock)
        self.slice_dock.hide()
        self.cache_inspector_dock = QDockWidget("CACHE INSPECTOR", self)
        self.cache_inspector_dock.setMinimumWidth(0)
        self.cache_inspector_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable |
                                              QDockWidget.DockWidgetFeature.DockWidgetMovable |
                                              QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.cache_inspector_dock.setObjectName("cache-inspector-dock")
        self.cache_inspector = CacheInspectorPanel(self)
        self.cache_inspector.resolve_callback = self._resolve_cache_range
        self.cache_inspector_dock.setWidget(self.cache_inspector)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.cache_inspector_dock)
        self.tabifyDockWidget(self.properties_dock, self.cache_inspector_dock)
        self.cache_inspector_dock.hide()
        agent_dock = QDockWidget("AGENT", self)
        agent_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable |
                               QDockWidget.DockWidgetFeature.DockWidgetMovable |
                               QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        agent_dock.setObjectName("agent-dock")
        agent_dock.setMinimumWidth(0)
        self.agent_panel = AgentPanel(self)
        agent_dock.setWidget(self.agent_panel)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, agent_dock)
        self.tabifyDockWidget(self.properties_dock, agent_dock)
        agent_dock.hide()
        agent_dock.visibilityChanged.connect(self._agent_dock_shown)
        self.agent_dock = agent_dock
        self.curve_editor_dock = QDockWidget("CURVE EDITOR", self)
        self.curve_editor_dock.setObjectName("curve-editor-dock")
        self.curve_editor_dock.setMinimumWidth(0)
        self.curve_editor_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable |
                                           QDockWidget.DockWidgetFeature.DockWidgetMovable |
                                           QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.curve_editor_dock.setWidget(QLabel("Choose Edit curve… in a node's properties."))
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.curve_editor_dock)
        self.tabifyDockWidget(self.properties_dock, self.curve_editor_dock)
        self.curve_editor_dock.hide()
        self.workspace_docks = [self.viewer_dock, self.graph_dock, self.properties_dock,
                                self.nodes_dock, self.viewport_dock, self.slice_dock,
                                self.cache_inspector_dock, self.agent_dock, self.curve_editor_dock]
        # QMainWindow otherwise divides a new, three-dock left column almost evenly, leaving
        # the viewer's actual canvas shorter than its controls and timeline.  Give the image
        # surface the largest share of the default workspace; artists can resize it afterwards.
        self.resizeDocks([self.viewer_dock, self.graph_dock, self.nodes_dock],
                         [500, 300, 125], Qt.Orientation.Vertical)
        self._menus()
        # The layout as built above *is* the default workspace; keep it before anything saved
        # replaces it, so Workspace → Default workspace has something exact to return to.
        self.resizeDocks([dock], [DEFAULT_PROPERTIES_WIDTH], Qt.Orientation.Horizontal)
        self._default_workspace_state = self.saveState(Preferences.WORKSPACE_VERSION)
        self._default_workspace_geometry = self.saveGeometry()
        self.restore_workspace()
        # Restore the stored theme before the first paint, so the app never flashes the default.
        self.apply_theme_name(self.theme_name, self.accent_color)
        self.graph.rebuild()
        self.inspect(None)
        self.viewport.set_document(self.dispatcher.document)
        self.server = None
        if agent_name:
            from .agent import LocalBridge
            self.server = LocalBridge(agent_name, self.agent_command, self)
            self.agent_panel.set_endpoint(agent_name, agent_name)
        QTimer.singleShot(0, self.graph.fit)
        self.request_preview()

    def _agent_dock_shown(self, visible):
        # The panel's own "Start Claude/Codex" buttons need a live LocalBridge endpoint; opening
        # one only when the dock actually becomes visible keeps a plain, non-agent NodeBased
        # session free of an extra local socket, matching the opt-in --agent flag's intent.
        if not visible or self.server is not None:
            return
        from .agent import LocalBridge
        name = f"nodebased-agent-panel-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self.server = LocalBridge(name, self.agent_command, self)
        self.agent_panel.set_endpoint(name, name)

    def show_curve_editor(self, title, value, on_change):
        editor = CurveEditorDialog(title, value, on_change, self.curve_editor_dock)
        editor.setWindowFlags(Qt.WindowType.Widget)
        old = self.curve_editor_dock.widget()
        self.curve_editor_dock.setWidget(editor)
        if old is not None and old is not editor:
            old.deleteLater()
        self.curve_editor_dock.setWindowTitle("CURVE EDITOR · " + title)
        self.curve_editor_dock.show()

    def _timeline(self):
        """Playhead strip under the viewer. Scrubbing is a document edit, so it undoes like one."""
        row = QHBoxLayout()
        row.addWidget(QLabel("  TIME"))
        self.play_button = QPushButton("▶")
        self.play_button.setFixedWidth(34)
        self.play_button.setToolTip("Play / stop (Space)")
        self.play_button.clicked.connect(lambda: self.toggle_playback())
        row.addWidget(self.play_button)
        self.frame_first = QSpinBox()
        self.frame_first.setRange(*TIME_LIMITS["first"])
        self.frame_first.setToolTip("First frame of the comp's range")
        row.addWidget(self.frame_first)
        # TimelineBar keeps the QSlider surface this row was built against (setRange/setValue/
        # value/valueChanged), and adds the tick marks, frame numbers, and the cached/keyed
        # underlines. Everything below wires to it unchanged.
        self.frame_slider = TimelineBar()
        row.addWidget(self.frame_slider, 1)
        self.frame_last = QSpinBox()
        self.frame_last.setRange(*TIME_LIMITS["last"])
        self.frame_last.setToolTip("Last frame of the comp's range")
        row.addWidget(self.frame_last)
        self.frame_current = QSpinBox()
        self.frame_current.setRange(*TIME_LIMITS["current"])
        self.frame_current.setToolTip("Current frame")
        row.addWidget(self.frame_current)
        # Playback rate is a property of the comp, so it is an undoable document edit through the
        # same boundary as the range — an agent setting fps and an artist typing it share one path.
        self.frame_fps = QDoubleSpinBox()
        self.frame_fps.setRange(*TIME_LIMITS["fps"])
        self.frame_fps.setDecimals(3)
        self.frame_fps.setSingleStep(1.0)
        # Commit on enter/focus-out rather than per keystroke, so typing "29.97" is one undoable
        # edit instead of four intermediate rates.
        self.frame_fps.setKeyboardTracking(False)
        self.frame_fps.setSuffix(" fps")
        self.frame_fps.setToolTip("Playback rate. New comps start at 24 fps; the transport and the "
                                  "dropped-frame counter both follow this value.")
        row.addWidget(self.frame_fps)
        self.fps_presets = QComboBox()
        self.fps_presets.setToolTip("Common delivery rates")
        self.fps_presets.addItem("rate", None)
        for label, value in FPS_PRESETS:
            self.fps_presets.addItem(label, value)
        self.fps_presets.currentIndexChanged.connect(self.apply_fps_preset)
        row.addWidget(self.fps_presets)
        self.frame_info = QLabel("")
        self.frame_info.setObjectName("muted")
        row.addWidget(self.frame_info)
        for widget, name in ((self.frame_first, "first"), (self.frame_last, "last"),
                             (self.frame_current, "current")):
            widget.valueChanged.connect(lambda value, key=name: self.set_time(**{key: value}))
        self.frame_fps.valueChanged.connect(lambda value: self.set_time(fps=float(value)))
        self.frame_slider.valueChanged.connect(lambda value: self.set_time(current=value))
        self.sync_timeline()
        return row

    def sync_timeline(self):
        """Push document time into the widgets without re-emitting edits back into the dispatcher."""
        time_range = self.dispatcher.document["time"]
        widgets = (self.frame_first, self.frame_last, self.frame_current, self.frame_slider,
                   self.frame_fps, self.fps_presets)
        for widget in widgets:
            widget.blockSignals(True)
        self.frame_slider.setRange(time_range["first"], time_range["last"])
        self.frame_current.setRange(time_range["first"], time_range["last"])
        self.frame_first.setValue(time_range["first"])
        self.frame_last.setValue(time_range["last"])
        self.frame_current.setValue(time_range["current"])
        self.frame_slider.setValue(time_range["current"])
        self.frame_fps.setValue(float(time_range["fps"]))
        # The preset box follows the rate rather than leading it, so a document opened at 23.976
        # shows that name instead of leaving a stale selection from the previous comp.
        match = next((index for index in range(1, self.fps_presets.count())
                      if abs(self.fps_presets.itemData(index) - time_range["fps"]) < 1e-6), 0)
        self.fps_presets.setCurrentIndex(match)
        for widget in widgets:
            widget.blockSignals(False)
        span = time_range["last"] - time_range["first"] + 1
        seconds = span / float(time_range["fps"])
        self.frame_info.setText(f"{span} frame{'' if span == 1 else 's'} · {seconds:.2f} s")
        self.refresh_timeline_marks()
        self.refresh_animated_panel()

    def refresh_animated_panel(self):
        """Re-read the properties panel when the playhead lands on a new frame.

        Only when the selected node is actually animated, and never during playback. An animated
        knob displays its value *at the current frame*, and its key button says whether a key
        exists here -- both go stale the instant the playhead moves. Rebuilding unconditionally
        would destroy the widget under an artist mid-edit and cost a panel rebuild on every
        transport tick, so the narrow condition is the point rather than an optimisation.
        """
        frame = self.dispatcher.document["time"]["current"]
        if frame == self.panel_frame:
            return
        self.panel_frame = frame
        if self.playing or getattr(self, "graph", None) is None:
            return
        key = self.graph.selected_id()
        if key and ((self.graph_document().get("animation") or {}).get("curves", {}).get(key)
                    or (self.graph_document().get("expressions") or {}).get(key)):
            self.inspect(key)

    def refresh_timeline_marks(self):
        """Push the cached-frame and keyed-frame sets onto the strip.

        Cached frames are read straight out of the display cache under the current viewer
        identity, so the answer is exact: an LRU eviction stops showing immediately, and a graph
        edit changes the identity so the whole band clears without any invalidation hook to keep
        in sync with the edit operations.

        Keyed frames follow the selection, the way Nuke's does: with a node selected you see that
        node's keys, which is what you are actually animating. With nothing selected the strip
        shows every key in the comp, so an unselected animated graph is not silently blank.
        """
        document = self.dispatcher.document
        time_range = document["time"]
        frames = range(time_range["first"], time_range["last"] + 1)
        try:
            # A frame counts as cached at any proxy tier. Playback with "Proxy while playing"
            # fills a proxy tier; judging the band by the tier selected after stopping made it
            # collapse to the one or two frames that happened to be shown at full resolution.
            cached = set()
            for tier in PROXY_TIERS:
                identity = DisplayCache.identity(
                    document, document.get("view"), tier)
                cached |= self.display_cache.resident_frames(identity, frames)
        except Exception:
            # A malformed in-flight document must never take the timeline down with it; an empty
            # cache band is a truthful "we don't know" rather than a crash.
            cached = set()
        curves = (self.graph_document().get("animation") or {}).get("curves") or {}
        # _timeline() syncs itself while it is being built, which is before the graph view exists.
        selected = self.graph.selected_id() if getattr(self, "graph", None) is not None else None
        scope = {selected: curves[selected]} if selected in curves else ({} if selected else curves)
        keyed = {key["frame"] for node_curves in scope.values()
                 for curve in node_curves.values() for key in curve["keys"]}
        self.frame_slider.set_marks(cached, keyed)

    def apply_fps_preset(self, index):
        rate = self.fps_presets.itemData(index)
        if rate is not None:
            self.set_time(fps=float(rate))

    def set_time(self, transient=False, playhead_only=False, **changes):
        """Single funnel for every playhead/range edit — UI, keys and agents share this boundary."""
        current = self.dispatcher.document["time"]
        if all(current.get(key) == value for key, value in changes.items()):
            return
        # Changing the rate mid-play re-anchors the transport on the current frame. Without this the
        # wall-clock origin still belongs to the old rate and the playhead jumps to wherever the new
        # rate says the elapsed time landed.
        if "fps" in changes and self.playing:
            self.playback_origin_frame = current["current"]
            self.playback_origin_time = time.monotonic()
            self.playback_elapsed_frames = 0
            self.playback_timer.start(max(5, round(500 / float(changes["fps"]))))
        # A range edit that would strand the playhead clamps it rather than failing validation.
        if "first" in changes and changes["first"] > current["last"]:
            changes.setdefault("last", changes["first"])
        if "last" in changes and changes["last"] < current["first"]:
            changes.setdefault("first", changes["last"])
        if transient:
            try:
                self.dispatcher.execute({"op": "time", "transient": True, **changes})
                self.sync_timeline()
                self.update_title()
                self.request_preview(playhead_only=playhead_only)
            except (ValueError, KeyError, TypeError) as error:
                self._show_command_error(error)
        else:
            self.command({"op": "time", **changes})

    def step_frame(self, delta):
        if self.playing:
            self.toggle_playback(False)
        time_range = self.dispatcher.document["time"]
        target = min(max(time_range["current"] + delta, time_range["first"]), time_range["last"])
        self.set_time(current=target)

    def future_frames(self, frame, count=3):
        """Return bounded forward read-ahead order, looping inside the comp range."""
        time_range = self.dispatcher.document["time"]
        first, last = time_range["first"], time_range["last"]
        span = last - first + 1
        return [first + ((frame - first + offset) % span)
                for offset in range(1, min(count, max(0, span - 1)) + 1)]

    def toggle_playback(self, checked=None):
        start = (not self.playing) if checked is None else bool(checked)
        if start == self.playing:
            return
        self.playing = start
        self.play_button.setText("■" if start else "▶")
        if start:
            time_range = self.dispatcher.document["time"]
            self.playback_origin_frame = time_range["current"]
            self.playback_origin_time = time.monotonic()
            self.playback_elapsed_frames = 0
            self.playback_dropped_frames = 0
            self.playback_frames_rendered = 0
            # Standard proxy-resolution playback: a source above HD makes the ACES 2.0 CPU
            # transform too slow for real-time (measured ~2.3s at 4K), so drop to the smallest
            # downscale that brings it under budget for the duration of playback only. Never
            # touches a tier the artist already chose below Full themselves -- that is their
            # call, not an automatic override.
            if self.playback_proxy.isChecked() and self.proxy.currentData() == 1:
                try:
                    target = self.dispatcher.document.get("view")
                    if target and self.tile_executor.supports_tiled(self.dispatcher.document, target):
                        bounds = self.tile_executor.canvas_region(self.dispatcher.document, target,
                                                                   frame=time_range["current"], tier=1)
                        tier = auto_playback_tier(bounds.width, bounds.height)
                        if tier != 1:
                            for index in range(self.proxy.count()):
                                if self.proxy.itemData(index) == tier:
                                    self.playback_auto_proxy_index = self.proxy.currentIndex()
                                    self.proxy.setCurrentIndex(index)
                                    break
                except Exception:
                    pass
            self.playback_timer.start(max(5, round(500 / time_range["fps"])))
            self.request_preview()
        else:
            self.playback_timer.stop()
            self.preview_queue.cancel()
            if self.playback_auto_proxy_index is not None:
                self.proxy.setCurrentIndex(self.playback_auto_proxy_index)
                self.playback_auto_proxy_index = None

    def playback_tick(self):
        """Follow the wall clock when rendering can keep up. When it can't, Nuke-style fallback:
        the playhead may only sit at an offset from playback_origin_frame equal to how many
        primary frames have actually finished rendering since playback started -- offset 0 (stay
        put) until the origin frame's own render completes, then offset 1, and so on. A render
        pipeline too slow for real time (native 4K/ACES) degrades to slower, strictly in-order
        playback instead of racing ahead on the wall clock and later jumping to wherever that
        landed -- including backward -- once a slow frame finally finishes. When rendering keeps
        up, playback_frames_rendered climbs at least as fast as elapsed_frames, so this is
        identical to plain wall-clock-following."""
        time_range = self.dispatcher.document["time"]
        span = time_range["last"] - time_range["first"] + 1
        elapsed_frames = int((time.monotonic() - self.playback_origin_time) * time_range["fps"])
        advanced = elapsed_frames - self.playback_elapsed_frames
        if advanced > 1:
            self.playback_dropped_frames += advanced - 1
        self.playback_elapsed_frames = elapsed_frames
        bounded_frames = min(elapsed_frames, self.playback_frames_rendered)
        target = time_range["first"] + (
            (self.playback_origin_frame - time_range["first"] + bounded_frames) % span)
        if target != time_range["current"]:
            self.set_time(transient=True, playhead_only=True, current=target)

    def _menus(self):
        file = self.menuBar().addMenu("File")
        edit = self.menuBar().addMenu("Edit")
        # Transport keys match the compositing convention (Nuke/Resolve): bare arrows step, Home/End
        # jump to the range ends. Qt's ShortcutOverride lets a focused spin box or line edit keep
        # them for text navigation, so typing a frame number still behaves normally.
        time_menu = self.menuBar().addMenu("Time")
        agent_menu = self.menuBar().addMenu("Agent")
        agent_action = QAction("Show Agent panel", self, checkable=True)
        agent_action.toggled.connect(self.agent_dock.setVisible)
        self.agent_dock.visibilityChanged.connect(agent_action.setChecked)
        agent_menu.addAction(agent_action)
        shortcuts = dict(SHORTCUT_SECTIONS)
        shortcut = lambda section, description: next(key for key, text in shortcuts[section]
                                                      if text.casefold() == description.casefold())
        for menu, name, section, description, callback in [
            (file, "Read image…", "File", "Read image", self.read_file),
            (file, "Open project…", "File", "Open project", self.open_project),
            (file, "Save", "File", "Save", self.save_project),
            (file, "Save as…", "File", "Save as", lambda: self.save_project(True)),
            (file, "Export image…", "File", "Export image", self.export),
            (edit, "Undo", "Edit", "Undo", lambda: self.command({"op": "undo"})),
            (edit, "Redo", "Edit", "Redo", lambda: self.command({"op": "redo"})),
            (edit, "Settings…", "Edit", "Settings", self.project_settings),
            (time_menu, "Previous frame", "Time", "previous frame", lambda: self.step_frame(-1)),
            (time_menu, "Next frame", "Time", "next frame", lambda: self.step_frame(1)),
            (time_menu, "First frame", "Time", "first frame",
             lambda: self.set_time(current=self.dispatcher.document["time"]["first"])),
            (time_menu, "Last frame", "Time", "last frame",
             lambda: self.set_time(current=self.dispatcher.document["time"]["last"])),
            (time_menu, "Play / Stop", "Time", "play/stop", self.toggle_playback)]:
            action = QAction(name, self)
            action.setShortcut(QKeySequence(shortcut(section, description)))
            action.triggered.connect(lambda checked=False, fn=callback: fn())
            menu.addAction(action)
        workspace_menu = self.menuBar().addMenu("Workspace")
        workspace_menu.setObjectName("workspace-menu")
        nodes_action = QAction("Show Nodes panel", self, checkable=True)
        nodes_action.setChecked(self.nodes_dock.isVisible())
        nodes_action.toggled.connect(self.nodes_dock.setVisible)
        self.nodes_dock.visibilityChanged.connect(nodes_action.setChecked)
        workspace_menu.addAction(nodes_action)
        window_menu = self.menuBar().addMenu("Window")
        toolbar_action = self.workspace_toolbar.toggleViewAction()
        toolbar_action.setText("Workspace toolbar")
        window_menu.addAction(toolbar_action)
        window_menu.addSeparator()
        for dock in self.workspace_docks:
            action = QAction(dock.windowTitle(), self, checkable=True)
            action.setChecked(not dock.isHidden())
            action.toggled.connect(dock.setVisible)
            dock.visibilityChanged.connect(action.setChecked)
            window_menu.addAction(action)
        window_menu.addSeparator()
        workspaces_menu = window_menu.addMenu("Workspaces")
        workspaces_menu.addAction("Save as…", self.save_workspace_as)
        self.saved_workspaces_menu = workspaces_menu.addMenu("Open")
        workspaces_menu.addAction("Set current as default", self.set_default_workspace)
        workspaces_menu.addAction("Reset to default", self.reset_workspace)
        self._refresh_workspace_menu()
        self.maximise_panel_shortcut = QShortcut(QKeySequence("Ctrl+Space"), self)
        self.maximise_panel_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self.maximise_panel_shortcut.activated.connect(self.toggle_maximise_panel)
        workspace_menu.addSeparator()
        default_workspace = workspace_menu.addAction("Default workspace")
        default_workspace.setObjectName("default-workspace")
        default_workspace.setToolTip("Put the window, panels and dividers back where a fresh "
                                     "install has them")
        default_workspace.triggered.connect(lambda checked=False: self.reset_workspace())
        preferences_menu = self.menuBar().addMenu("Preferences")
        radial_commands_action = preferences_menu.addAction("Radial commands…")
        radial_commands_action.triggered.connect(lambda checked=False: self.open_radial_commands_editor())
        radial_settings_action = preferences_menu.addAction("Radial settings…")
        radial_settings_action.triggered.connect(lambda checked=False: self.open_radial_settings())
        help_menu = self.menuBar().addMenu("Help")
        action = help_menu.addAction("Keyboard shortcuts…")
        action.triggered.connect(self.show_keyboard_shortcuts)

    def save_workspace(self):
        """Remember window placement, dock layout and panel dividers for the next launch."""
        self.preferences.set_workspace(self.saveGeometry(),
                                       self.saveState(Preferences.WORKSPACE_VERSION))
        store = QSettings("NodeBased", "NodeBased")
        store.setValue("workspace/floating", self._floating_dock_geometry())
        store.setValue("workspace/metrics", {"size": [self.width(), self.height()],
                                              "docks": self._dock_size_metrics()})
        store.sync()

    def _refresh_workspace_menu(self):
        self.saved_workspaces_menu.clear()
        store = QSettings("NodeBased", "NodeBased")
        names = store.value("workspaces/names", [])
        if isinstance(names, str):
            names = [names]
        for name in names:
            action = self.saved_workspaces_menu.addAction(str(name))
            action.triggered.connect(lambda checked=False, key=str(name): self.load_named_workspace(key))
        self.saved_workspaces_menu.addAction("Default", self.restore_default_workspace)
        self.saved_workspaces_menu.addSeparator()
        self.saved_workspaces_menu.addAction("Two monitors: viewers left, graph and properties right",
                                             self.apply_two_monitor_workspace)

    def save_workspace_as(self):
        name, ok = QInputDialog.getText(self, "Save workspace", "Workspace name:")
        name = name.strip()
        if not ok or not name:
            return
        store = QSettings("NodeBased", "NodeBased")
        names = store.value("workspaces/names", [])
        if isinstance(names, str):
            names = [names]
        names = list(dict.fromkeys([*names, name]))
        store.setValue("workspaces/names", names)
        store.setValue(f"workspaces/{name}/geometry", self.saveGeometry())
        store.setValue(f"workspaces/{name}/state", self.saveState(Preferences.WORKSPACE_VERSION))
        store.setValue(f"workspaces/{name}/floating", self._floating_dock_geometry())
        store.setValue(f"workspaces/{name}/metrics", {"size": [self.width(), self.height()],
                                                       "docks": self._dock_size_metrics()})
        store.sync()
        self._refresh_workspace_menu()

    def _floating_dock_geometry(self):
        result = {}
        for dock in self.workspace_docks:
            if dock.isFloating():
                g = dock.frameGeometry()
                result[dock.objectName()] = [g.x(), g.y(), g.width(), g.height(),
                                             bool(dock.isMaximized()), bool(dock.isFullScreen())]
        return result

    def load_named_workspace(self, name):
        store = QSettings("NodeBased", "NodeBased")
        geometry = store.value(f"workspaces/{name}/geometry")
        state = store.value(f"workspaces/{name}/state")
        if not all(isinstance(v, QByteArray) and not v.isEmpty() for v in (geometry, state)):
            return False
        self.restoreState(state, Preferences.WORKSPACE_VERSION)
        self.restoreGeometry(geometry)
        metrics = store.value(f"workspaces/{name}/metrics", {})
        self._apply_workspace_metrics(geometry, metrics)
        self._restore_floating_metadata(store.value(f"workspaces/{name}/floating", {}))
        self._pending_geometry = geometry
        self._pending_workspace_metrics = metrics
        return True

    def _restore_floating_metadata(self, metadata):
        if not isinstance(metadata, dict):
            return
        docks = {dock.objectName(): dock for dock in self.workspace_docks}
        for name, values in metadata.items():
            dock = docks.get(name)
            if dock is None or not dock.isFloating() or not isinstance(values, (list, tuple)) or len(values) < 6:
                continue
            x, y, width, height, maximized, fullscreen = values[:6]
            self._place_on_available_screen(dock, int(x), int(y), int(width), int(height))
            if fullscreen:
                dock.showFullScreen()
            elif maximized:
                dock.showMaximized()

    def _place_on_available_screen(self, window, x, y, width, height):
        screens = QApplication.screens()
        rect = QRect(x, y, width, height)
        screen = next((s for s in screens if s.availableGeometry().contains(rect.center())), None)
        if screen is not None and screen.availableGeometry().contains(rect):
            return
        screen = screen or QApplication.primaryScreen()
        if screen is None:
            return
        area = screen.availableGeometry()
        rect.setSize(QSize(min(width, area.width()), min(height, area.height())))
        rect.moveCenter(area.center())
        window.setGeometry(rect)

    def set_default_workspace(self):
        store = QSettings("NodeBased", "NodeBased")
        store.setValue("workspaces/default/geometry", self.saveGeometry())
        store.setValue("workspaces/default/state", self.saveState(Preferences.WORKSPACE_VERSION))
        store.setValue("workspaces/default/floating", self._floating_dock_geometry())
        store.setValue("workspaces/default/metrics", {"size": [self.width(), self.height()],
                                                       "docks": self._dock_size_metrics()})
        store.sync()

    def restore_default_workspace(self):
        store = QSettings("NodeBased", "NodeBased")
        geometry = store.value("workspaces/default/geometry")
        state = store.value("workspaces/default/state")
        if isinstance(geometry, QByteArray) and not geometry.isEmpty() and \
                isinstance(state, QByteArray) and not state.isEmpty():
            self.restoreState(state, Preferences.WORKSPACE_VERSION)
            self.restoreGeometry(geometry)
            metrics = store.value("workspaces/default/metrics", {})
            self._apply_workspace_metrics(geometry, metrics)
            self._restore_floating_metadata(store.value("workspaces/default/floating", {}))
            self._pending_geometry = geometry
            self._pending_workspace_metrics = metrics
            return
        self.reset_workspace()

    def apply_two_monitor_workspace(self):
        screens = QApplication.screens()
        for dock in self.workspace_docks:
            if dock.isFloating():
                dock.setFloating(False)
        for dock in (self.viewer_dock, self.viewport_dock):
            if dock is self.viewport_dock and not self.dispatcher.document:
                continue
            dock.setFloating(True)
            dock.show()
        self.viewer_dock.showNormal()
        viewer_area = screens[0].availableGeometry()
        viewer_width = max(800, int(viewer_area.width() * 0.58))
        viewer_height = max(500, viewer_area.height())
        self.viewer_dock.resize(viewer_width, viewer_height)
        self._place_on_available_screen(self.viewer_dock, viewer_area.x(), viewer_area.y(),
                                        viewer_width, viewer_height)
        viewport_width = max(500, viewer_area.width() - viewer_width)
        self.viewport_dock.resize(viewport_width, viewer_height)
        self._place_on_available_screen(self.viewport_dock, viewer_area.x() + viewer_width,
                                        viewer_area.y(), viewport_width, viewer_height)
        if len(screens) > 1:
            area = screens[1].availableGeometry()
            self.resize(1100, max(650, area.height()))
            self.move(area.topLeft())

    def toggle_maximise_panel(self):
        if getattr(self, "_maximised_panel_dock", None) is not None:
            dock = self._maximised_panel_dock
            self._maximised_panel_dock = None
            dock.showNormal()
            dock.setFloating(False)
            return
        widget = QApplication.widgetAt(QCursor.pos())
        while widget is not None and not isinstance(widget, QDockWidget):
            widget = widget.parentWidget()
        if widget is None:
            return
        self._maximised_panel_dock = widget
        widget.setFloating(True)
        widget.showMaximized()

    def restore_workspace(self):
        """Reopen the way the app was last closed. Returns False when there was nothing to restore.

        A layout that fails to apply is discarded rather than half-applied: the default workspace
        is always a usable window, a partially restored one might not be.
        """
        saved = self.preferences.workspace()
        if saved is None:
            return False
        if not (self.restoreState(saved["state"], Preferences.WORKSPACE_VERSION)
                and self.restoreGeometry(saved["geometry"])):
            self.preferences.clear_workspace()
            self.reset_workspace()
            return False
        self._restore_floating_metadata(QSettings("NodeBased", "NodeBased").value("workspace/floating", {}))
        # Before the first show the unpolished widgets report a wider minimum than they end up
        # with, and showing enforces it: a saved 1200px window reopened 74px wider. Apply the
        # geometry again once the window is up.
        self._pending_geometry = saved["geometry"]
        self._pending_workspace_metrics = QSettings("NodeBased", "NodeBased").value("workspace/metrics", {})
        return True

    def showEvent(self, event):
        super().showEvent(event)
        geometry = getattr(self, "_pending_geometry", None)
        if geometry is not None:
            self._pending_geometry = None
            metrics = getattr(self, "_pending_workspace_metrics", {})
            self._pending_workspace_metrics = {}
            QTimer.singleShot(0, lambda: self._apply_workspace_metrics(geometry, metrics))

    def _dock_size_metrics(self):
        return {dock.objectName(): [dock.width(), dock.height()] for dock in self.workspace_docks
                if not dock.isFloating()}

    def _apply_workspace_metrics(self, geometry, metrics):
        self.restoreGeometry(geometry)
        if not isinstance(metrics, dict):
            return
        size = metrics.get("size")
        if isinstance(size, (list, tuple)) and len(size) == 2:
            self.resize(max(800, int(size[0])), max(500, int(size[1])))
        dock_sizes = metrics.get("docks", {})
        horizontal, widths, vertical, heights = [], [], [], []
        for dock in self.workspace_docks:
            values = dock_sizes.get(dock.objectName()) if isinstance(dock_sizes, dict) else None
            if dock.isFloating() or not isinstance(values, (list, tuple)) or len(values) != 2:
                continue
            area = self.dockWidgetArea(dock)
            if area in (Qt.DockWidgetArea.LeftDockWidgetArea, Qt.DockWidgetArea.RightDockWidgetArea):
                horizontal.append(dock); widths.append(max(1, int(values[0])))
            else:
                vertical.append(dock); heights.append(max(1, int(values[1])))
        if horizontal:
            self.resizeDocks(horizontal, widths, Qt.Orientation.Horizontal)
        if vertical:
            self.resizeDocks(vertical, heights, Qt.Orientation.Vertical)

    def minimumSizeHint(self):
        # Dock contents are intentionally allowed to clip/scroll while a workspace is narrow;
        # a content's natural width must never become a main-window hard floor.
        return QSize(800, 500)

    def reset_workspace(self):
        """Workspace → Default workspace: the layout a first launch gets, applied in place."""
        custom = QSettings("NodeBased", "NodeBased").value("workspaces/default/state")
        if isinstance(custom, QByteArray) and not custom.isEmpty():
            self.restore_default_workspace()
            return
        if self.isMaximized() or self.isFullScreen():
            self.showNormal()
        self.restoreState(self._default_workspace_state, Preferences.WORKSPACE_VERSION)
        self.restoreGeometry(self._default_workspace_geometry)
        for dock in self.workspace_docks:
            if dock.isFloating():
                dock.setFloating(False)
        # Settle the minimum size for the restored docks now; left to the next event loop pass,
        # the resize below is still clamped by the docks the reset just hid.
        self.layout().activate()
        self.resize(*DEFAULT_WINDOW_SIZE)
        screen = self.screen() or QApplication.primaryScreen()
        if screen is not None:
            frame = self.frameGeometry()
            frame.moveCenter(screen.availableGeometry().center())
            self.move(frame.topLeft())
        self.resizeDocks([self.properties_dock], [DEFAULT_PROPERTIES_WIDTH],
                         Qt.Orientation.Horizontal)

    def show_keyboard_shortcuts(self):
        if self.keyboard_shortcuts_dialog is None:
            self.keyboard_shortcuts_dialog = KeyboardShortcutsDialog(self)
        self.keyboard_shortcuts_dialog.show()
        self.keyboard_shortcuts_dialog.raise_()
        self.keyboard_shortcuts_dialog.activateWindow()

    def open_radial_commands_editor(self, new_from_selection=None):
        """Preferences -> Radial commands, or the ring's own "+ Add command..." button."""
        self.radial_commands_dialog = RadialCommandsDialog(self, new_from_selection=new_from_selection,
                                                           parent=self)
        self.radial_commands_dialog.show()
        self.radial_commands_dialog.raise_()
        self.radial_commands_dialog.activateWindow()

    def open_radial_settings(self):
        """Preferences -> Radial settings."""
        self.radial_settings_dialog = RadialSettingsDialog(self, parent=self)
        self.radial_settings_dialog.show()
        self.radial_settings_dialog.raise_()
        self.radial_settings_dialog.activateWindow()

    def show_node_help(self, kind):
        """"What is this?", from a node's right-click menu on the graph or in the NODES dock:
        its docs-table row, or its one-line description when no row exists for it yet."""
        self.node_help_dialog = NodeHelpDialog(kind, self)
        self.node_help_dialog.show()
        self.node_help_dialog.raise_()
        self.node_help_dialog.activateWindow()

    def focus_node_search(self):
        """Ctrl+F: jump straight to the NODES dock's search box, wherever focus currently is."""
        if not self.nodes_dock.isVisible():
            self.nodes_dock.setVisible(True)
        self.node_toolbar.search.setFocus()
        self.node_toolbar.search.selectAll()

    # -- the graph being edited: the top level, or the inside of a Group -------------------------

    def _scope_graph(self):
        """The graph dict of the current scope (the document itself, or a Group's graph). A path
        that no longer resolves (undo, load, ungroup) is cut back to the deepest group that does."""
        graph, valid = self.dispatcher.document, []
        for step in self.graph_path:
            node = graph["nodes"].get(step)
            if node is None or node["type"] != "Group":
                break
            graph = node["graph"]
            valid.append(step)
        self.graph_path = valid
        return graph

    def graph_nodes(self):
        return self._scope_graph()["nodes"]

    def graph_document(self):
        """A document-shaped view of the graph on screen: the document at the top level, a scope
        view (nodes, animation, node_data of the group) inside a group."""
        graph = self._scope_graph()
        if not self.graph_path:
            return self.dispatcher.document
        return scope_document(self.dispatcher.document, graph)

    def enter_group(self, key):
        node = self.graph_nodes().get(key)
        if node is None or node["type"] != "Group":
            return False
        self.graph_path = self.graph_path + [key]
        self._scope_changed()
        return True

    def go_to_depth(self, depth):
        """Show the graph `depth` groups deep on the current path (0 is the top level)."""
        self._scope_graph()
        if depth >= len(self.graph_path):
            return
        self.graph_path = self.graph_path[:depth]
        self._scope_changed()

    def _scope_changed(self):
        self.pinned_panels = []
        self.thumbnails.clear()
        self.graph.scene().clearSelection()
        self.graph.rebuild()
        self.inspect(None)
        self.refresh_timeline_marks()
        QTimer.singleShot(0, self.graph.fit)
        if self.show_thumbnails:
            self.thumbnail_timer.start()

    def sync_breadcrumbs(self):
        """Rebuild the Root > Group > ... bar to match the path."""
        layout = self.breadcrumbs.layout()
        while layout.count():
            widget = layout.takeAt(0).widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        nodes, names = self.dispatcher.document["nodes"], ["Root"]
        for step in self.graph_path:
            names.append(nodes[step]["name"])
            nodes = nodes[step]["graph"]["nodes"]
        for depth, name in enumerate(names):
            if depth:
                layout.addWidget(QLabel("›"))
            crumb = QPushButton(name)
            crumb.setObjectName("breadcrumb")
            crumb.setFlat(True)
            crumb.setEnabled(depth < len(names) - 1)
            crumb.setToolTip("Back to the top level" if depth == 0 else f"Back to {name}")
            crumb.clicked.connect(lambda checked=False, d=depth: self.go_to_depth(d))
            layout.addWidget(crumb)
        layout.addStretch(1)

    def scoped_command(self, cmd):
        """`cmd` addressed to the graph on screen: inside a group every graph edit (and each
        member of a batch) carries the group path."""
        if not self.graph_path or not isinstance(cmd, dict):
            return cmd
        if cmd.get("op") == "batch":
            return {**cmd, "commands": [self.scoped_command(inner) for inner in cmd.get("commands", [])]}
        if cmd.get("op") in Dispatcher.GRAPH_EDIT_OPS and "path" not in cmd:
            return {**cmd, "path": list(self.graph_path)}
        return cmd

    def command(self, cmd, render=True):
        try:
            before = self.dispatcher.revision
            self._scope_graph()
            cmd = self.scoped_command(cmd)
            result = self.dispatcher.execute(cmd)
            if hasattr(self, "viewport"):
                self.viewport.set_document(self.dispatcher.document)
            self.last_command_error = None
            self.command_error_label.clear()
            if cmd.get("op") in ("time", "settings"):
                self.sync_timeline()
                self.sync_project_settings()
                self.update_title()
                if render:
                    self.request_preview()
            else:
                self.after_command(render, sync_settings=cmd.get("op") in ("load", "undo", "redo"),
                                   revision=before)
            return result
        except (ValueError, KeyError, TypeError, OSError) as error:
            self._show_command_error(error)
            return None

    def _show_command_error(self, error):
        """Expose a command validation failure beyond the transient status-bar message."""
        self.last_command_error = str(error)
        self.command_error_label.setText(self.last_command_error)
        self.statusBar().showMessage(self.last_command_error, 10000)

    def agent_command(self, cmd):
        # Every LocalBridge request (a legacy --connect client, agentloop, or the MCP server)
        # passes through here, so this is the one place that can surface bridge activity in the
        # Agent panel regardless of which client is attached.
        if self.agent_panel is not None:
            op = cmd.get("op") if isinstance(cmd, dict) else None
            arguments = {k: v for k, v in cmd.items() if k != "op"} if isinstance(cmd, dict) else {}
            try:
                result = self._agent_command(cmd)
            except Exception as error:
                self.agent_panel.log_tool_call(op, arguments, f"error: {error}")
                raise
            self.agent_panel.log_tool_call(op, arguments, "ok")
            return result
        return self._agent_command(cmd)

    def _agent_command(self, cmd):
        # Return machine-readable errors through the bridge, not only the status bar.
        if cmd.get("op") == "errors":
            # Live render-error visibility for an attached agent: the Dispatcher only knows about
            # document edits, never about the async render pipeline, so this reaches into Window
            # state directly rather than through self.dispatcher.execute like every other op.
            since = cmd.get("since", 0)
            return {"errors": [e for e in self.render_errors if e["timestamp"] > since],
                    "current_frame": self.dispatcher.document["time"]["current"],
                    "displayed_frame": self.frame_generation, "generation": self.generation,
                    "playing": self.playing, "status": self.viewer_info.text()}
        if cmd.get("op") == "reference_context":
            return self.reference_context(cmd)
        if cmd.get("op") == "load" and self.dispatcher.document != self.saved_document:
            raise ValueError("Save current changes before agent load; human edits are unsaved")
        result = self.dispatcher.execute(cmd)
        if cmd.get("op") in ("save", "load"):
            self.project_path = str(Path(cmd["path"]).resolve())
            self.saved_document = copy.deepcopy(self.dispatcher.document)
        if cmd.get("op") not in ("describe", "inspect"):
            if cmd.get("op") == "time":
                self.sync_timeline()
                self.update_title()
                self.request_preview()
            else:
                self.after_command(sync_settings=cmd.get("op") in ("load", "undo", "redo"))
        return result

    def reference_context(self, cmd):
        """Capture the bounded display context an attached agent needs to inspect a graph.

        This is deliberately GUI-local: it reads the live viewer controls and uses the same
        evaluator/display conversion as the preview, while leaving the Dispatcher document and
        revision untouched. The resulting PNGs are display previews, never graph artifacts.
        """
        prompt = cmd.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 32768:
            raise ValueError("reference_context requires a non-empty prompt of at most 32768 characters")
        directory = cmd.get("directory")
        if not isinstance(directory, str) or not directory.strip():
            raise ValueError("reference_context requires an output directory")
        output = Path(directory).expanduser().resolve()
        if not output.exists() or not output.is_dir():
            raise ValueError(f"reference_context output directory does not exist: {output}")
        include_view = cmd.get("include_view", True)
        if type(include_view) is not bool:
            raise ValueError("reference_context include_view must be boolean")
        document = copy.deepcopy(self.dispatcher.document)
        view_id = document.get("view")
        reference_ids = list(document.get("references", []))
        ordered = []
        roles = {}
        if include_view and view_id is not None:
            ordered.append(view_id)
            roles[view_id] = ["view"]
        for node_id in reference_ids:
            if node_id not in roles:
                ordered.append(node_id)
                roles[node_id] = []
            roles[node_id].append("reference")
        if not ordered:
            raise ValueError("reference_context has no viewed or referenced node to capture")
        if len(ordered) > 8:
            raise ValueError("reference_context is limited to 8 distinct captures")
        frame = int(document["time"]["current"])
        view = self.effective_view(document)
        exposure = self.exposure.value()
        look = self.look_args(document)
        channel = self.channels.currentText()
        background = document["settings"]["viewer"]["background"]
        captures = []
        for index, node_id in enumerate(ordered, 1):
            if node_id not in document["nodes"]:
                raise ValueError(f"reference_context node {node_id!r} does not exist")
            pixels = self.evaluator.evaluate(document, node_id, frame=frame, tier=1)
            image = to_qimage(pixels, exposure, channel, background=background, view=view, look=look)
            captures.append((index, node_id, image))
        # A caller-owned directory may already contain earlier captures. Put every response in a
        # fresh unpredictable child directory so context collection never overwrites an unrelated
        # file or a prior response from the same document revision.
        capture_root = Path(tempfile.mkdtemp(
            prefix=f"nodebased-context-r{self.dispatcher.revision}-f{frame}-", dir=output))
        artifacts = []
        for index, node_id, image in captures:
            # Node IDs are document data and may contain path separators. Keep them out of the
            # filename entirely so an agent can never turn a context capture into path traversal.
            path = capture_root / f"capture-{index:02d}.png"
            if not image.save(str(path), "PNG"):
                raise ValueError(f"Cannot write reference_context PNG: {path}")
            artifacts.append({"id": node_id, "name": document["nodes"][node_id]["name"],
                              "roles": roles[node_id], "path": str(path.resolve()),
                              "width": image.width(), "height": image.height()})
        return {"revision": self.dispatcher.revision, "current_frame": frame,
                "prompt": prompt, "artifacts": artifacts}

    def _look_reset(self, name):
        button = QPushButton("↺")
        button.setFixedWidth(24)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.setToolTip(f"Reset viewer {name}")
        button.clicked.connect(lambda: self.set_viewer_look(**{name: {"gain": 0.0, "gamma": 1.0}[name]}))
        return button

    def set_viewer_look(self, **changes):
        """Store a display-only viewer adjustment in the document and redraw."""
        current = viewer_look(self.dispatcher.document)
        if all(current[name] == value for name, value in changes.items()):
            return
        self.command({"op": "viewer_look", **changes})

    def sync_viewer_look(self):
        """Bring the toolbar controls to the document's viewer state (undo, load, agent edits)."""
        look = viewer_look(self.dispatcher.document)
        for widget, value in ((self.exposure, look["gain"]), (self.gamma, look["gamma"])):
            if widget.value() != value:
                widget.blockSignals(True)
                widget.setValue(value)
                widget.blockSignals(False)
        if self.zebra.isChecked() != look["zebra"]:
            self.zebra.blockSignals(True)
            self.zebra.setChecked(look["zebra"])
            self.zebra.blockSignals(False)
        if self.viewer_display.currentText() != look["display"]:
            self.viewer_display.blockSignals(True)
            self.viewer_display.setCurrentText(look["display"])
            self.viewer_display.blockSignals(False)

    def set_viewer_roi(self, **changes):
        current = viewer_roi(self.dispatcher.document)
        if changes.get("on") and "rect" not in changes and current["rect"] == VIEWER_ROI_DEFAULT["rect"]:
            changes["rect"] = [0.25, 0.25, 0.75, 0.75]
        if all(current[name] == value for name, value in changes.items()):
            return
        self.command({"op": "viewer_roi", **changes})

    def toggle_viewer_roi(self, on):
        self.set_viewer_roi(on=on)

    def set_viewer_proxy(self, tier):
        """Save the artist's proxy pick with the document. The evaluation itself is already
        queued by the combo's own change signal, so this does not render again."""
        if tier is None or viewer_proxy(self.dispatcher.document) == tier:
            return
        if tier != 1:
            self._last_proxy = tier
        self.command({"op": "viewer_proxy", "tier": tier}, render=False)
        # The proxy is in the document's settings, which the identity check hashes, but the request
        # already queued by the combo carries the tier: it is not stale.
        self.rendered_identity = self.render_identity()

    def toggle_viewer_proxy(self):
        """Ctrl+P: proxy off, or back to the last proxy chosen."""
        now = self.proxy.currentData()
        target = 1 if now != 1 else self._last_proxy
        index = self.proxy.findData(target)
        if index >= 0:
            self.proxy.setCurrentIndex(index)
            self.set_viewer_proxy(target)

    def set_viewer_mask(self, **changes):
        current = viewer_masks(self.dispatcher.document)
        if all(current[name] == value for name, value in changes.items()):
            return
        self.command({"op": "viewer_mask", **changes})

    def sync_viewer_frame(self):
        """Bring the ROI button, proxy and mask controls to the document (undo, load, agent edits)."""
        document = self.dispatcher.document
        roi, masks, tier = viewer_roi(document), viewer_masks(document), viewer_proxy(document)
        if self.roi_button.isChecked() != roi["on"]:
            self.roi_button.blockSignals(True)
            self.roi_button.setChecked(roi["on"])
            self.roi_button.blockSignals(False)
        for widget, value in ((self.mask_choice, masks["mask"]), (self.mask_mode, masks["mode"])):
            if widget.currentText() != value:
                widget.blockSignals(True)
                widget.setCurrentText(value)
                widget.blockSignals(False)
        if tier != self._proxy_seen:
            # Only when the document's own value moved (undo, load, an agent): playback switches the
            # combo by itself and must not be undone by an unrelated edit.
            self._proxy_seen = tier
            if tier != 1:
                self._last_proxy = tier
            index = self.proxy.findData(tier)
            if index >= 0 and self.proxy.currentIndex() != index:
                self.proxy.blockSignals(True)
                self.proxy.setCurrentIndex(index)
                self.proxy.blockSignals(False)
        self.viewer.viewport().update()

    def effective_view(self, document=None):
        """The display view in force: the viewer's own choice, else the project's default view."""
        chosen = viewer_look(document or self.dispatcher.document)["display"]
        return self.display_view.currentText() if chosen == "Project view" else chosen

    def look_args(self, document=None):
        look = viewer_look(document or self.dispatcher.document)
        return {"gamma": look["gamma"], "zebra": look["zebra"]}

    def after_command(self, render=True, sync_settings=False, revision=None):
        key = self.graph.selected_id()
        # An edit that changed nothing must not rebuild the graph and the properties panel.
        # `editingFinished` fires on focus-out, so right-clicking a knob to open its context menu
        # re-submits the value the document already holds -- and the unconditional rebuild that
        # followed deleted the widget the menu was parented to, closing the menu a frame after it
        # appeared. That is the "menu flashes and vanishes" bug; the Dispatcher already treats
        # such an edit as a no-op and leaves its revision alone, so that is the signal to trust.
        changed = revision is None or self.dispatcher.revision != revision
        if changed:
            self.graph.rebuild()
            self.inspect(key)
        # Undo, project load and agent "time" edits all land here, so the strip follows the
        # document rather than only the widget that happened to be dragged.
        self.sync_timeline()
        self.viewer.sync_inputs()
        self.sync_viewer_look()
        self.sync_viewer_frame()
        if sync_settings:
            self.sync_project_settings()
        self.update_title()
        if changed and self.show_thumbnails:
            self.thumbnail_timer.start()
        # Only an edit that can change the viewed picture re-renders. Moving, renaming or
        # labelling a node -- or editing a branch the viewer does not see -- leaves it alone.
        if render and self.render_identity() != self.rendered_identity:
            self.request_preview()

    def render_identity(self):
        document = self.dispatcher.document
        target = document.get("view")
        _, b_target = compare_model.active_b(document)
        frame, view = document["time"]["current"], self.effective_view(document)
        return (thumbnail_key(document, target, frame, view) if target else None,
                json.dumps(document["settings"], sort_keys=True),
                thumbnail_key(document, b_target, frame, view) if b_target else None)

    def sync_project_settings(self):
        view = self.dispatcher.document["settings"]["color"]["view"]
        if self.display_view.currentText() != view:
            self.display_view.blockSignals(True)
            self.display_view.setCurrentText(view)
            self.display_view.blockSignals(False)

    def project_settings(self):
        dialog = ProjectSettingsDialog(copy.deepcopy(self.dispatcher.document["settings"]), self,
                                       theme=self.theme_name, thumbnails=self.show_thumbnails,
                                       accent=self.accent_color, max_panels=self.panel_cap,
                                       cache_float32=self.preferences.display_cache_float32())
        # Preview the theme live while the dialog is open: picking a colour scheme you cannot see
        # until you commit is a guess, not a choice. Cancel restores the one in force.
        original, original_accent = self.theme_name, self.accent_color
        preview = lambda *_: self.apply_theme_name(dialog.chosen_theme(), dialog.chosen_accent())
        dialog.theme.currentTextChanged.connect(preview)
        dialog.accent.currentIndexChanged.connect(preview)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.apply_theme_name(dialog.chosen_theme(), dialog.chosen_accent())
            self.preferences.set_theme(self.theme_name)
            self.preferences.set_accent(self.accent_color)
            self.set_show_thumbnails(dialog.thumbnails.isChecked())
            self.set_panel_cap(dialog.chosen_max_panels())
            new_cache_float32 = dialog.cache_float32.isChecked()
            if new_cache_float32 != self.preferences.display_cache_float32():
                self.preferences.set_display_cache_float32(new_cache_float32)
                self.display_cache = DisplayCache(force_float32=new_cache_float32)
                self.refresh_timeline_marks()
            self.command({"op": "settings", "settings": dialog.changes()})
        else:
            self.apply_theme_name(original, original_accent)

    def apply_theme_name(self, name, accent=None):
        """Restyle the application and repaint the graph background for one theme."""
        self.accent_color = valid_accent(accent)
        self.theme_name = apply_theme(name, self.accent_color)
        self.graph.grid_pen = QPen(QColor(grid_color(self.theme_name)), 1)
        self.graph.viewport().update()

    def update_title(self):
        dirty = self.dispatcher.document != self.saved_document
        self.setWindowTitle(f"NodeBased {__version__} · {Path(self.project_path).name if self.project_path else 'Untitled'}{' *' if dirty else ''}")

    def node_tab(self, key, node):
        page = QWidget()
        form = QFormLayout(page)
        form.setContentsMargins(16, 16, 16, 16)
        if node["type"] != "Group":   # a group keeps its note on the main tab
            label = LabelEdit(node_label(node))
            label.setObjectName("node-label")
            label.setPlaceholderText("Shown on the node under its name")
            label.setFixedHeight(72)

            def commit_label(widget=label, k=key):
                text = widget.toPlainText().strip()
                if text != node_label(self.graph_nodes()[k]):
                    self.defer_command({"op": "label", "id": k, "value": text})
            label.finished.connect(commit_label)
            form.addRow("Label", label)
        enabled = QCheckBox("Enabled")
        enabled.setObjectName("node-enabled")
        enabled.setChecked(not node["disabled"])
        if bypass_slot(node) is None:
            enabled.setEnabled(False)
            enabled.setToolTip("Source nodes have nothing to pass through, so they cannot be disabled")
        else:
            enabled.setToolTip("Disabled nodes pass their main input through unchanged  [D]")
        enabled.toggled.connect(lambda value, k=key: self.defer_command(
            {"op": "disable", "id": k, "value": not value}))
        form.addRow(enabled)
        stamp = QCheckBox("Postage stamp (thumbnail)")
        stamp.setObjectName("node-thumbnail")
        stamp.setChecked(node_thumbnail(node))
        if node["type"] in NO_THUMBNAIL_TYPES:
            stamp.setEnabled(False)
        elif not self.show_thumbnails:
            stamp.setToolTip("Thumbnails are switched off in Settings → Interface")
        else:
            default = "on" if node["type"] in DEFAULT_THUMBNAIL_TYPES else "off"
            stamp.setToolTip(f"Default for {node['type']} is {default}")
        stamp.toggled.connect(lambda value, k=key: self.defer_command(
            {"op": "thumbnail", "id": k, "value": value}))
        form.addRow(stamp)
        return page

    def build_node_panel(self, key):
        panel = QWidget()
        panel.setObjectName("node-properties-panel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(4)
        form = QFormLayout()
        form.setContentsMargins(16, 16, 16, 16)
        if key not in self.graph_nodes():
            label = QLabel("Select a node to edit its controls.\n\nLinear ACEScg · float RGBA\nPremultiplied alpha\nEXR / PNG / JPEG / TIFF input\n\n3D and AI generation are roadmap\nmilestones, not active tools yet.")
            label.setObjectName("muted")
            form.addRow(label)
            panel_layout.addWidget(label)
        else:
            node = self.graph_nodes()[key]
            # One reusable properties template owns the editable title and its compact action
            # cluster.  The graph's label is the only other place the node name is repeated.
            top = QWidget()
            top.setObjectName("node-panel-header")
            top_layout = QHBoxLayout(top)
            top_layout.setContentsMargins(8, 3, 8, 3)
            top_layout.setSpacing(3)
            collapse = QToolButton()
            collapse.setObjectName("panel-collapse-arrow")
            collapse.setText("▾")
            collapse.setToolTip("Collapse or expand this node's properties")
            collapse.setAccessibleName("Collapse properties")
            collapse.setVisible(key not in self.pinned_panels)
            title = QLineEdit(node["name"])
            title.setObjectName("node-name-header")
            title.setFrame(False)
            title.setStyleSheet(f"color: {COLORS[node['type']]}; font-weight: 700; font-size: 15px")
            title.editingFinished.connect(
                lambda k=key, w=title: w.text() != self.graph_nodes()[k]["name"]
                and self.defer_command({"op": "rename", "id": k, "name": w.text()}))
            title.installEventFilter(self)
            title.setToolTip("Node name · edit here; double-click to collapse or expand")
            top_layout.addWidget(collapse)
            top_layout.addWidget(title, 1)

            def icon_button(name, tooltip, callback):
                button = PropertiesIconButton(name)
                button.setToolTip(tooltip)
                button.setAccessibleName(tooltip)
                button.clicked.connect(lambda checked=False: callback(checked))
                top_layout.addWidget(button)
                return button

            if not self.graph_path:
                icon_button("view-node", "View this node (1)",
                            lambda _=False, k=key: self.command({"op": "view", "id": k}))
            reference_button = icon_button(
                "reference-node", "Reference for agent", lambda value, k=key:
                self.defer_command({"op": "reference", "id": k, "value": value}))
            reference_button.setChecked(key in self.dispatcher.document["references"])
            catalog_description = node_description(node["type"])
            long_description = ""
            try:
                doc_text = read_doc(doc_for_kind(node["type"]))
                doc_row = find_doc_row(doc_text, node["type"])
                if doc_row:
                    cells = [cell.strip() for cell in doc_row[1].strip().strip("|").split("|")]
                    long_description = cells[-1] if cells else ""
            except (OSError, ValueError):
                pass
            help_button = icon_button("help-node", "Node help", lambda _=False: None)
            help_text = catalog_description or ""
            if long_description and long_description.strip() != help_text.strip():
                help_text += "\n\n" + long_description.strip()
            extra_help = PANEL_NODE_HELP.get(node["type"], "")
            if extra_help and extra_help not in help_text:
                help_text += "\n\n" + extra_help
            help_button.setToolTip(help_text)
            icon_button("revert-knobs", "Revert all knobs to the node's defaults",
                        lambda _=False, k=key, defaults=SPECS[node["type"]]["params"]: self.command(
                            {"op": "batch", "commands": [
                                {"op": "set", "id": k, "param": param, "value": value}
                                for param, value in defaults.items()]}))
            icon_button("close-knobs", "Close this node's properties",
                        lambda _=False, k=key: self.close_panel(k) if k in self.pinned_panels
                        else self.inspect(None))
            content = QWidget()
            content.setObjectName("node-panel-content")
            content.setLayout(form)
            title._panel_content = content
            title._panel_collapse = collapse
            panel_layout.addWidget(top)
            collapse.clicked.connect(lambda: content.setVisible(not content.isVisible()))
            collapse.clicked.connect(lambda: collapse.setText("▸" if content.isVisible() else "▾"))
            curves = (self.graph_document().get("animation") or {}).get("curves", {}).get(key, {})
            expressions = (self.graph_document().get("expressions") or {}).get(key, {})
            # Curves and expressions are resolved through the same document boundary used by the
            # evaluator. This keeps the inspector honest: an expression-driven knob shows the
            # number the artist will actually render at the current frame, including a formula
            # that reads an animated parameter on another node.
            resolved_document = resolve_document(self.graph_document(),
                                                 self.dispatcher.document["time"]["current"])
            resolved = resolved_document["nodes"][key]["params"]
            if node["type"] == "Group":
                inner = node["graph"]["nodes"]
                count = sum(1 for member in inner.values() if member["type"] not in ("Input", "Output"))
                summary = QLabel(f"{count} node{'s' if count != 1 else ''} inside · "
                                 f"{len(node['inputs'])} input{'s' if len(node['inputs']) != 1 else ''}")
                summary.setObjectName("group-summary")
                form.addRow(summary)
                note = LabelEdit(node_label(node))
                note.setObjectName("group-note")
                note.setPlaceholderText("A note shown on the group under its name")
                note.setFixedHeight(72)
                note.finished.connect(
                    lambda k=key, w=note: w.toPlainText().strip() != node_label(self.graph_nodes()[k])
                    and self.defer_command({"op": "label", "id": k, "value": w.toPlainText().strip()}))
                form.addRow("Note", note)
                enter = QPushButton("Enter group   [double-click]")
                enter.setObjectName("enter-group")
                enter.clicked.connect(lambda checked=False, k=key: self.enter_group(k))
                form.addRow(enter)
            def add_legacy_param(param, value, kind=None, label=None):
                """Render one member of an unimplemented multi-param knob unchanged."""
                dynamic_ocio = {
                    "OCIOColorspace": {"src", "dst"},
                    "OCIODisplay": {"display", "view", "look"},
                    "OCIOLookTransform": {"src", "dst", "look"},
                }
                if param in dynamic_ocio.get(node["type"], set()):
                    from .ocio_nodes import parameter_choices
                    config_name = (node["params"].get("config") or
                                   self.graph_document().get("settings", {}).get("color", {}).get("config", ""))
                    display = node["params"].get("display", "")
                    try:
                        choices = parameter_choices(config_name, param, display)
                    except ValueError:
                        choices = []
                    if value and value not in choices:
                        choices.append(value)
                    control = QComboBox()
                    control.setObjectName(f"ocio-{param}-choice")
                    control.setEditable(True)
                    control.addItems(choices)
                    control.setCurrentText(value)
                    control.setToolTip(f"Choices from OCIO config: {config_name or 'document default'}")
                    control.activated.connect(
                        lambda index, k=key, p=param, w=control:
                        self.defer_command({"op": "set", "id": k, "param": p, "value": w.itemText(index)}))
                    control.lineEdit().editingFinished.connect(
                        lambda k=key, p=param, w=control:
                        w.currentText() != self.graph_nodes()[k]["params"][p]
                        and self.defer_command({"op": "set", "id": k, "param": p, "value": w.currentText()}))
                    form.addRow(label or param.replace("_", " ").title(), control)
                    return
                if param in CHOICES:
                    control = QComboBox()
                    control.addItems(CHOICES[param])
                    control.setCurrentText(value)
                    control.currentTextChanged.connect(lambda v, k=key, p=param: self.defer_command({"op": "set", "id": k, "param": p, "value": v}))
                    form.addRow({"colorspace": "Input space", "alpha_mode": "Alpha", "red_from": "Red", "green_from": "Green",
                                 "blue_from": "Blue", "alpha_from": "Alpha"}.get(param, label or param), control)
                elif node["type"] == "ReadVDB3D" and param in ("density_grid", "temperature_grid", "velocity_grid"):
                    # The grid names come from the file itself; "auto" and "none" are always offered, and
                    # a name typed by hand is kept (a sequence whose first frame lacks a grid stays usable).
                    from . import vdbio
                    control = QComboBox()
                    control.setEditable(True)
                    control.addItems(vdbio.grid_choices(node["params"]["vdb_path"]))
                    control.setCurrentText(value)
                    control.setToolTip("Grid to read: a name from the file, auto (usual names) or none")
                    control.lineEdit().editingFinished.connect(
                        lambda k=key, p=param, w=control:
                        w.currentText() != self.graph_nodes()[k]["params"][p]
                        and self.defer_command({"op": "set", "id": k, "param": p, "value": w.currentText()}))
                    control.activated.connect(
                        lambda index, k=key, p=param, w=control:
                        w.itemText(index) != self.graph_nodes()[k]["params"][p]
                        and self.defer_command({"op": "set", "id": k, "param": p, "value": w.itemText(index)}))
                    form.addRow({"density_grid": "Density grid", "temperature_grid": "Temperature grid",
                                 "velocity_grid": "Velocity grid"}[param], control)
                elif isinstance(value, str):
                    control = QLineEdit(value)
                    control.editingFinished.connect(
                        lambda k=key, p=param, w=control:
                        w.text() != self.graph_nodes()[k]["params"][p]
                        and self.defer_command({"op": "set", "id": k, "param": p, "value": w.text()}))
                    self.attach_text_menu(control, default=SPECS[node["type"]]["params"][param],
                                          commit=lambda text, k=key, p=param: self.defer_command(
                                              {"op": "set", "id": k, "param": p, "value": text}))
                    form.addRow({"splat_path": "Splat file", "abc_path": "Alembic file", "abc_root": "Root object",
                                 "abc_camera": "Camera object", "gltf_path": "glTF file", "vdb_path": "VDB file or sequence",
                                 "gltf_root": "Root node"}.get(param, param.title()), control)
                    if kind == "file_read" or (kind is None and param == "path" and node["type"] == "Read"):
                        if node["type"] == "Vectorfield" and param == "cube_path":
                            browse = QPushButton("Browse LUT…")
                            browse.setToolTip("Choose a .cube or Flame/Lustre .3dl colour lookup table")
                            browse.clicked.connect(lambda checked=False, k=key: self.browse_lut(k))
                        else:
                            browse = QPushButton("Browse image sequence…")
                            browse.setToolTip("Sequence-aware browser: numbered frames arrive as one entry")
                            browse.clicked.connect(lambda checked=False, k=key: self.browse_read(k))
                        form.addRow(browse)
                    elif kind == "file_write" or (kind is None and param == "path" and node["type"] == "Write"):
                        browse = QPushButton("Choose output…")
                        browse.setToolTip("Use a padded pattern (render.%04d.exr) to write a sequence")
                        browse.clicked.connect(lambda checked=False, k=key: self.browse_write(k))
                        form.addRow(browse)
                    elif param == "splat_path":
                        browse = QPushButton("Browse splat file…")
                        def browse_splat(checked=False, k=key):
                            path, _ = QFileDialog.getOpenFileName(self, "Splat file", "", "Gaussian splats (*.ply);;All files (*)")
                            if path:
                                self.defer_command({"op": "set", "id": k, "param": "splat_path", "value": path})
                        browse.clicked.connect(browse_splat)
                        form.addRow(browse)
                    elif param == "gltf_path":
                        browse = QPushButton("Browse glTF file…")
                        browse.setToolTip("glTF 2.0 (.glb or .gltf): meshes, base colours and textures are read")
                        def browse_gltf(checked=False, k=key):
                            path, _ = QFileDialog.getOpenFileName(self, "glTF file", self.last_browse_directory or "",
                                                                  "glTF 2.0 (*.glb *.gltf);;All files (*)")
                            if path:
                                self.last_browse_directory = str(Path(path).parent)
                                self.defer_command({"op": "set", "id": k, "param": "gltf_path", "value": path})
                        browse.clicked.connect(browse_gltf)
                        form.addRow(browse)
                    elif param == "geo_path":
                        browse = QPushButton("Browse geometry…")
                        browse.setToolTip("Wavefront OBJ: polygons, UVs and normals are read")
                        browse.clicked.connect(lambda checked=False, k=key: self.browse_geometry(k))
                        form.addRow(browse)
                    elif param == "layer":
                        control.setPlaceholderText("RGBA, or e.g. beauty / diffuse / Z")
                        control.setToolTip("Leave empty for root RGB; choose a named EXR layer or scalar channel")
                else:
                    control = QSpinBox() if type(value) is int else QDoubleSpinBox()
                    control.setRange(*( (2, 15) if node["type"] == "GridWarp" and param in ("rows", "columns")
                                        else LIMITS[param]))
                    if isinstance(control, QDoubleSpinBox):
                        control.setDecimals(3)
                        control.setSingleStep(0.1)
                    curve = curves.get(param)
                    control.setValue(resolved[param] if (curve or param in expressions) else value)
                    if param in expressions:
                        control.setEnabled(False)
                        control.setToolTip("Driven by an expression. Edit the formula below.")
                        control.setStyleSheet("color: #c58cff")
                    control.setKeyboardTracking(False)
                    control.editingFinished.connect(
                        lambda k=key, p=param, w=control: self.commit_param(k, p, w.value()))
                    self.install_expression_shortcut(control, key, param, control)
                    control.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
                    control.customContextMenuRequested.connect(
                        lambda point, k=key, p=param, w=control: self.curve_menu(k, p, w, w, point))
                    form.addRow(param.title(), self.animatable_row(
                        key, param, control, expression=expressions.get(param)))

            def numeric_field(param):
                control = QDoubleSpinBox()
                control.setObjectName(f"{param}-field")
                control.setProperty("nodebased_node_id", key)
                control.setProperty("nodebased_param", param)
                control.setRange(*LIMITS[param])
                control.setDecimals(3)
                control.setSingleStep(0.1)
                curve = curves.get(param)
                control.setValue(resolved[param] if (curve or param in expressions)
                                 else node["params"][param])
                if param in expressions:
                    control.setEnabled(False)
                    control.setToolTip("Driven by an expression. Edit the formula below.")
                    control.setStyleSheet("color: #c58cff")
                control.setKeyboardTracking(False)
                control.editingFinished.connect(
                    lambda k=key, p=param, w=control: self.commit_param(k, p, w.value()))
                self.install_expression_shortcut(control, key, param, control)
                control.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
                control.customContextMenuRequested.connect(
                    lambda point, k=key, p=param, w=control: self.curve_menu(k, p, w, w, point))
                return control

            def add_animation_button(row, param, control):
                button = QPushButton()
                button.setFixedWidth(26)
                button.setFlat(True)
                button.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
                curve = self.node_curve(key, param)
                frame = self.dispatcher.document["time"]["current"]
                keyed_here = curve is not None and any(k["frame"] == frame for k in curve["keys"])
                expression = expressions.get(param)
                if expression is not None:
                    button.setText("ƒ")
                    button.setEnabled(False)
                    button.setStyleSheet("color: #c58cff; border: none; font-size: 14px")
                    button.setToolTip("Expression-driven. Clear the expression before keying this knob.")
                elif keyed_here:
                    button.setText("◆")
                    button.setStyleSheet(f"color: {KEY_COLOR.name()}; border: none; font-size: 14px")
                    button.setToolTip(f"Key set at frame {frame}. Click to remove it.\nRight-click for "
                                      f"curve options.")
                elif curve is not None:
                    button.setText("◇")
                    button.setStyleSheet(f"color: {KEY_COLOR.name()}; border: none; font-size: 14px")
                    button.setToolTip(f"Animated ({len(curve['keys'])} keys, {curve['interpolation']}). "
                                      f"Click to key the current value at frame {frame}.\nRight-click for "
                                      f"curve options.")
                else:
                    button.setText("○")
                    button.setStyleSheet("color: #6d6d78; border: none; font-size: 14px")
                    button.setToolTip(f"Not animated. Click to set the first key at frame {frame}.")
                button.clicked.connect(
                    lambda checked=False, k=key, p=param, w=control, on=keyed_here:
                    self.toggle_key(k, p, w, on))
                button.customContextMenuRequested.connect(
                    lambda point, k=key, p=param, w=control, b=button: self.curve_menu(k, p, w, b, point))
                row.addWidget(button)

            for group in knob_layout(node["type"]):
                if group.kind == "legacy":
                    continue
                if node["type"] == "Flare" and set(group.params) & {"tracker_id", "track_index"}:
                    # The two stored link fields are presented as one validated track picker below.
                    continue
                if group.kind == "xyz":
                    # One row, three typed fields, as in Nuke. Each axis keys on its own, so every
                    # field carries its own diamond instead of the row sharing one.
                    row = QWidget()
                    row_layout = QHBoxLayout(row)
                    row_layout.setContentsMargins(0, 0, 0, 0)
                    row_layout.setSpacing(3)
                    for axis, param in zip("xyz", group.params):
                        field = numeric_field(param)
                        field.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
                        row_layout.addWidget(QLabel(axis))
                        row_layout.addWidget(field, 1)
                        add_animation_button(row_layout, param, field)
                        # Three diamonds share the row with three fields: keep them slim so the
                        # digits keep their room in a narrow dock. The theme's button padding is
                        # wider than a slim button, and clipped the glyph to nothing on a real display.
                        diamond = row_layout.itemAt(row_layout.count() - 1).widget()
                        diamond.setFixedWidth(18)
                        diamond.setStyleSheet(diamond.styleSheet() + "; padding: 0")
                    form.addRow(group.label, row)
                    continue
                if group.kind in ("xy", "xyz"):
                    fields = QWidget()
                    layout = QHBoxLayout(fields)
                    layout.setContentsMargins(0, 0, 0, 0)
                    layout.setSpacing(4)
                    axis_fields = [numeric_field(param) for param in group.params]
                    for axis, param, field in zip("xyz", group.params, axis_fields):
                        layout.addWidget(QLabel(axis))
                        layout.addWidget(self.animatable_row(key, param, field), 1)
                    row = QWidget()
                    row_layout = QHBoxLayout(row)
                    row_layout.setContentsMargins(0, 0, 0, 0)
                    row_layout.setSpacing(4)
                    row_layout.addWidget(fields, 1)
                    form.addRow(group.label, row)
                    continue
                if group.kind == "color":
                    fields = QWidget()
                    layout = QHBoxLayout(fields)
                    layout.setContentsMargins(0, 0, 0, 0)
                    layout.setSpacing(4)
                    color_fields = [numeric_field(param) for param in group.params]
                    for field in color_fields:
                        # Four stepper columns cost the digits their room in a narrow dock, and
                        # a colour is typed, scrubbed or picked from the swatch, never stepped.
                        field.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
                    swatch = ClickableColorSwatch()
                    swatch.setObjectName("color-swatch")
                    swatch.setFixedSize(24, 24)

                    def set_swatch():
                        rgb = [max(0.0, min(1.0, field.value())) for field in color_fields[:3]]
                        swatch.setStyleSheet("background-color: rgb(%d, %d, %d); border: 1px solid #777;" %
                                             tuple(int(value * 255) for value in rgb))

                    def pick_color():
                        current = [max(0.0, min(1.0, field.value())) for field in color_fields[:3]]
                        color = QColorDialog.getColor(
                            QColor(*(int(value * 255) for value in current)), self, "Choose color")
                        if color.isValid():
                            for param, value in zip(group.params[:3],
                                                    (color.redF(), color.greenF(), color.blueF())):
                                self.defer_command({"op": "set", "id": key, "param": param,
                                                    "value": value})
                            for field, value in zip(color_fields[:3],
                                                    (color.redF(), color.greenF(), color.blueF())):
                                field.setValue(value)
                            set_swatch()

                    swatch.clicked.connect(pick_color)
                    set_swatch()
                    layout.addWidget(swatch)
                    for field in color_fields:
                        layout.addWidget(field, 1)
                    form.addRow(group.label, fields)
                    continue
                param = group.params[0]
                value = node["params"][param]
                if group.kind == "curve":
                    row = QWidget(); line = QHBoxLayout(row); line.setContentsMargins(0, 0, 0, 0)
                    edit = QPushButton("Edit curve…")
                    edit.setObjectName(f"{param}-editor")
                    edit.clicked.connect(lambda checked=False, k=key, p=param, label=group.label:
                        self.show_curve_editor(f"{self.graph_nodes()[k]['type']} · {label}",
                            self.graph_nodes()[k]["params"][p],
                            lambda text, node_id=k, knob=p: self.defer_command(
                                {"op": "set", "id": node_id, "param": knob, "value": text})))
                    line.addWidget(edit, 1)
                    reset = QPushButton("Reset")
                    reset.setFixedWidth(54)
                    reset.clicked.connect(lambda checked=False, k=key, p=param:
                        self.defer_command({"op": "set", "id": k, "param": p,
                                            "value": SPECS[self.graph_nodes()[k]["type"]]["params"][p]}))
                    line.addWidget(reset)
                    form.addRow(group.label, row)
                elif group.kind == "bool":
                    control = QCheckBox()
                    control.setChecked(bool(value))
                    control.toggled.connect(lambda checked, k=key, p=param: self.defer_command(
                        {"op": "set", "id": k, "param": p, "value": 1 if checked else 0}))
                    form.addRow(group.label, control)
                elif group.kind in ("enum",):
                    # Only a label the layout spells out replaces the raw name; 2D enums that never
                    # declared one keep the row text they have always had.
                    # 3D panels always show the layout's label: "Shadows" title-cases to itself, so
                    # the comparison alone left Light3D showing the raw parameter name.
                    declared = (group.label != param.replace("_", " ").title()
                                or node["type"].endswith("3D"))
                    add_legacy_param(param, value, group.kind, label=group.label if declared else None)
                elif group.kind in ("string", "file_read", "file_write"):
                    add_legacy_param(param, value, group.kind)
                elif group.kind == "float":
                    control = numeric_field(param)
                    form.addRow(group.label, self.animatable_row(
                        key, param, control, expression=expressions.get(param)))
                elif group.kind == "float_slider":
                    curve = curves.get(param)
                    shown = resolved[param] if (curve or param in expressions) else value
                    control = FloatSliderControl(LIMITS[param], group.soft_range, shown)
                    if param in expressions:
                        control.setEnabled(False)
                        control.setToolTip("Driven by an expression. Edit the formula below.")
                        control.spin.setStyleSheet("color: #c58cff")
                    control.spin.editingFinished.connect(
                        lambda k=key, p=param, w=control: self.commit_param(k, p, w.value()))
                    control.slider.sliderReleased.connect(
                        lambda k=key, p=param, w=control: self.commit_param(k, p, w.value()))
                    self.install_expression_shortcut(control.spin, key, param, control)
                    control.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
                    control.customContextMenuRequested.connect(
                        lambda point, k=key, p=param, w=control: self.curve_menu(k, p, w, w, point))
                    form.addRow(group.label, self.animatable_row(
                        key, param, control, expression=expressions.get(param)))
                else:
                    add_legacy_param(param, value)
            if node["type"] == "CurveTool":
                progress = QProgressDialog("Analyzing frames…", "Cancel", 0, 1, self)
                progress.setWindowTitle("CurveTool analysis")
                progress.setWindowModality(Qt.WindowModality.WindowModal)
                progress.setMinimumDuration(0)

                def analyze_curve_tool(k=key, dialog=progress):
                    selected = self.graph_nodes()[k]
                    source_id = selected["inputs"].get("image")
                    if not source_id:
                        QMessageBox.information(self, "CurveTool", "Connect an image input first.")
                        dialog.close(); return
                    params = selected["params"]
                    first, last = int(params["frame_start"]), int(params["frame_end"])
                    if last < first:
                        QMessageBox.warning(self, "CurveTool", "The end frame must be at or after the start frame.")
                        dialog.close(); return
                    dialog.setRange(0, last-first+1)
                    dialog.show()
                    commands = []
                    names = ("average_r", "average_g", "average_b", "average_a", "average_luminance",
                             "crop_x", "crop_y", "crop_width", "crop_height", "max_x", "max_y", "max_value",
                             "min_x", "min_y", "min_value", "exposure_diff")
                    previous_luminance = None
                    for index, frame_number in enumerate(range(first, last+1)):
                        QApplication.processEvents()
                        if dialog.wasCanceled():
                            dialog.close(); return
                        try:
                            raster = self.evaluator.evaluate_raster(self.graph_document(), target=source_id,
                                                                    frame=frame_number).to_display()
                            measured = curve_tool_metrics(raster, (params["box_x"], params["box_y"],
                                                                   params["box_width"], params["box_height"]),
                                                          previous_luminance=previous_luminance)
                            previous_luminance = measured["average_luminance"]
                            commands.extend({"op":"set_key", "id":k, "param":name, "frame":frame_number,
                                             "value":value, "interpolation":"linear"}
                                            for name,value in measured.items())
                        except Exception as exc:
                            dialog.close(); QMessageBox.warning(self, "CurveTool", f"Analysis stopped: {exc}"); return
                        dialog.setValue(index+1)
                    if not dialog.wasCanceled():
                        self.command({"op":"batch", "commands":commands})
                    dialog.close()

                button = QPushButton("Analyze")
                button.clicked.connect(analyze_curve_tool)
                form.addRow(button)
            if node["type"] in ("Histogram", "Sampler", "MinColor", "MatchGrade"):
                readout = QLabel("Analyze the current frame")
                readout.setWordWrap(True)
                form.addRow(readout)

                def run_analysis(k=key, label=readout):
                    selected = self.graph_nodes()[k]
                    inputs = selected["inputs"]
                    if not inputs.get("image"):
                        label.setText("Connect an image input first.")
                        return
                    try:
                        doc = self.graph_document()
                        frame = int(doc["time"]["current"])
                        src = self.evaluator.evaluate_raster(doc, target=inputs["image"], frame=frame)
                        px = src.pixels
                        if selected["type"] == "Histogram":
                            hist, _ = np.histogram(px[..., :3], bins=64)
                            plot = QPixmap(320, 100)
                            plot.fill(QColor("#202027"))
                            painter = QPainter(plot)
                            painter.setPen(Qt.PenStyle.NoPen)
                            painter.setBrush(QColor("#77c9a8"))
                            peak = max(1, int(hist.max()))
                            for i, n in enumerate(hist):
                                height = int(92 * n / peak)
                                painter.drawRect(i * 5, 100 - height, 4, height)
                            painter.end()
                            label.setPixmap(plot)
                        elif selected["type"] == "MinColor":
                            p = selected["params"]
                            crop = px
                            if p["box_width"] > 0 and p["box_height"] > 0:
                                x, y = int(p["box_x"] - src.data.x), int(p["box_y"] - src.data.y)
                                crop = px[max(0, y):max(0, y) + int(p["box_height"]), max(0, x):max(0, x) + int(p["box_width"])]
                            if crop.size == 0:
                                label.setText("The selected box contains no pixels.")
                                return
                            rgba, _ = Evaluator._min_color(crop, p["mincolor_mode"])
                            changes = [{"op": "set", "id": k, "param": f"mincolor_{c}", "value": float(v)} for c, v in zip("rgba", rgba)]
                            label.setText("Result RGBA · " + " / ".join(f"{v:.5f}" for v in rgba))
                            self.command({"op": "batch", "commands": changes})
                        elif selected["type"] == "Sampler":
                            p = selected["params"]
                            values = Evaluator._sample_line(px, (p["sample_x0"], p["sample_y0"]), (p["sample_x1"], p["sample_y1"]), (src.data.x, src.data.y))
                            n = len(values)
                            plot = QPixmap(320, 100)
                            plot.fill(QColor("#202027"))
                            painter = QPainter(plot)
                            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                            low, high = float(values[:, :3].min()), float(values[:, :3].max())
                            high = high if high > low else low + 1.0
                            for channel, colour in enumerate(("#ff625e", "#76d67c", "#669bff")):
                                painter.setPen(QPen(QColor(colour), 1.5))
                                points = [QPointF(i * (plot.width() - 1) / max(1, n - 1),
                                                  plot.height() - 4 - (float(v[channel]) - low) / (high - low) * (plot.height() - 8))
                                          for i, v in enumerate(values)]
                                painter.drawPolyline(QPolygonF(points))
                            painter.end()
                            label.setPixmap(plot)
                            label.setToolTip(f"{n} line samples · first RGB " + ", ".join(f"{v:.4f}" for v in values[0, :3]) + " · last RGB " + ", ".join(f"{v:.4f}" for v in values[-1, :3]))
                        else:
                            if not inputs.get("reference"):
                                label.setText("Connect the target image first.")
                                return
                            dst = self.evaluator.evaluate_raster(doc, target=inputs["reference"], frame=frame).fit(src.data)[..., :3]
                            mask_id = inputs.get("mask")
                            weights = (self.evaluator.evaluate_raster(doc, target=mask_id, frame=frame).fit(src.data)[..., 3]
                                       if mask_id else None)
                            updates = []
                            for c, channel in enumerate("rgb"):
                                if weights is None:
                                    sm, tm = float(px[..., c].mean()), float(dst[..., c].mean())
                                    ss, ts = float(px[..., c].std()), float(dst[..., c].std())
                                else:
                                    total = max(float(weights.sum()), 1e-12)
                                    sm, tm = float((px[..., c] * weights).sum() / total), float((dst[..., c] * weights).sum() / total)
                                    ss = float(np.sqrt(((px[..., c] - sm) ** 2 * weights).sum() / total))
                                    ts = float(np.sqrt(((dst[..., c] - tm) ** 2 * weights).sum() / total))
                                gain = ts / ss if ss > 1e-12 else 1.0
                                updates.extend(({"op": "set", "id": k, "param": f"grade_gain_{channel}", "value": gain}, {"op": "set", "id": k, "param": f"grade_offset_{channel}", "value": tm - sm * gain}))
                            updates.append({"op": "set", "id": k, "param": "match_analyzed", "value": 1})
                            label.setText("Per-channel gain and offset measured from the current frames.")
                            self.command({"op": "batch", "commands": updates})
                    except Exception as exc:
                        label.setText(f"Analysis failed: {exc}")

                button = QPushButton("Analyze / refresh")
                button.clicked.connect(run_analysis)
                form.addRow(button)
            if node["type"] in MATRIX_READOUT_TYPES:
                from . import scene3d
                local, world = scene3d.local_and_world_matrix(resolved_document, key)

                def matrix_text(matrix):
                    return "\n".join("  ".join(f"{value:9.4f}" for value in row) for row in matrix)

                for label, matrix in (("Local matrix", local), ("World matrix", world)):
                    readout = QLabel(matrix_text(matrix))
                    readout.setObjectName("matrix-readout")
                    readout.setStyleSheet("font-family: monospace; color: #9a9aa4")
                    readout.setToolTip("Read-only: the node's own transform"
                                       if label == "Local matrix" else
                                       "Read-only: local matrix with every Scene3D/Axis3D "
                                       "ancestor's transform multiplied in")
                    form.addRow(label, readout)
            if node["type"] == "Camera3D":
                from . import filmback
                lens = resolved
                for label, aperture in (("Vertical FOV", lens["vaperture"]),
                                        ("Horizontal FOV", lens["haperture"])):
                    readout = QLabel(f"{filmback.fov_from_aperture(lens['focal'], aperture):.4f}°")
                    readout.setObjectName("fov-readout")
                    readout.setToolTip("Read-only: derived as 2 * atan(aperture / (2 * focal length))")
                    form.addRow(label, readout)
            if node["type"] == "FluidSolver3D":
                from . import fluid3d
                try:
                    nx, ny, nz = fluid3d.resolution(resolved)
                    text = f"{nx} x {ny} x {nz}  ({nx * ny * nz:,} cells)"
                except ValueError as error:
                    text = str(error)
                readout = QLabel(text)
                readout.setObjectName("resolution-readout")
                readout.setToolTip("Read-only: the bounds divided by the division size, rounded up")
                form.addRow("Resolution", readout)
            if node["type"] == "Roto":
                draw = QPushButton("Draw shape…")
                draw.setToolTip("Click points in the Roto viewer; press Enter to close, Esc to cancel")
                draw.clicked.connect(lambda checked=False, k=key: self.begin_roto_draw(k))
                form.addRow(draw)
            if node["type"] == "SplineWarp":
                for side in ("source", "destination"):
                    button = QPushButton(f"Draw {side} curve…")
                    button.setToolTip(f"Click curve points in the viewer; press Enter to finish, Esc to cancel")
                    button.clicked.connect(lambda checked=False, k=key, s=side: self.viewer.begin_warp_draw(k, s))
                    form.addRow(button)
                hint = QLabel("Drag curve points to animate them. Join lines pair source and destination controls.")
                hint.setWordWrap(True)
                form.addRow(hint)
            if node["type"] == "GridWarp":
                hint = QLabel("Drag yellow destination points; cyan points show the source grid.")
                hint.setWordWrap(True)
                form.addRow(hint)
            if node["type"] == "RotoPaint":
                draw_shape = QPushButton("Draw shape…")
                draw_shape.setToolTip("Click polygon points in the viewer, then Enter to commit · Esc cancels")
                draw_shape.clicked.connect(lambda checked=False, k=key: self.begin_roto_draw(k))
                form.addRow(draw_shape)
                tool = QComboBox()
                for label, value in (("Paint", "paint"), ("Eraser", "eraser"), ("Clone", "clone"),
                                     ("Reveal input 2", "reveal"), ("Blur", "blur"),
                                     ("Sharpen", "sharpen"), ("Smear", "smear"),
                                     ("Dodge", "dodge"), ("Burn", "burn")):
                    tool.addItem(label, value)
                tool.setCurrentIndex(tool.findData(self.viewer.paint_tool))
                tool.currentIndexChanged.connect(lambda index, box=tool: setattr(self.viewer, "paint_tool", box.itemData(index)))
                form.addRow("Tool", tool)
                brush_size = QDoubleSpinBox(); brush_size.setRange(0.1, 4096); brush_size.setValue(self.viewer.paint_size)
                brush_size.valueChanged.connect(lambda value: setattr(self.viewer, "paint_size", float(value)))
                form.addRow("Brush size", brush_size)
                lifetime = QComboBox()
                lifetime.addItem("Single frame", "single"); lifetime.addItem("All frames", "all")
                lifetime.addItem("Frame range", "range"); lifetime.addItem("From current frame", "from_current")
                lifetime.setCurrentIndex(lifetime.findData(self.viewer.paint_lifetime))
                lifetime.currentIndexChanged.connect(lambda index, box=lifetime: setattr(self.viewer, "paint_lifetime", box.itemData(index)))
                form.addRow("Lifetime", lifetime)
                for title, attr in (("First frame", "paint_range_first"), ("Last frame", "paint_range_last")):
                    field = QSpinBox(); field.setRange(-1000000, 1000000)
                    field.setValue(int(getattr(self.viewer, attr)))
                    field.valueChanged.connect(lambda value, name=attr: setattr(self.viewer, name, int(value)))
                    form.addRow(title, field)
                for title, attr, lo, hi, step in (
                    ("Hardness", "paint_hardness", 0.0, 1.0, 0.05),
                    ("Brush opacity", "paint_brush_opacity", 0.0, 1.0, 0.05),
                    ("Spacing", "paint_spacing", 0.01, 4.0, 0.05),
                    ("Dodge / burn strength", "paint_strength", 0.0, 1.0, 0.05),
                    ("Layer opacity", "paint_layer_opacity", 0.0, 1.0, 0.05)):
                    field = QDoubleSpinBox(); field.setRange(lo, hi); field.setSingleStep(step)
                    field.setValue(float(getattr(self.viewer, attr)))
                    field.valueChanged.connect(lambda value, name=attr: setattr(self.viewer, name, float(value)))
                    form.addRow(title, field)
                blend = QComboBox()
                for value in ("over", "add", "multiply", "screen"): blend.addItem(value.title(), value)
                blend.setCurrentIndex(blend.findData(self.viewer.paint_blend))
                blend.currentIndexChanged.connect(lambda index, box=blend: setattr(self.viewer, "paint_blend", box.itemData(index)))
                form.addRow("Layer blend", blend)
                visible = QCheckBox("Visible"); visible.setChecked(self.viewer.paint_visible)
                visible.toggled.connect(lambda value: setattr(self.viewer, "paint_visible", bool(value)))
                form.addRow(visible)
                color = QPushButton("Choose paint colour…")
                color.clicked.connect(lambda: _choose_paint_color(self.viewer))
                form.addRow(color)
                offsets = QHBoxLayout()
                for axis in range(2):
                    field = QDoubleSpinBox(); field.setRange(-8192, 8192); field.setSingleStep(1)
                    field.setValue(self.viewer.paint_source_offset[axis])
                    field.valueChanged.connect(lambda value, i=axis: self.viewer.paint_source_offset.__setitem__(i, float(value)))
                    offsets.addWidget(field)
                form.addRow("Clone source offset", offsets)
                source_frame = QComboBox(); source_frame.addItem("Previous frame", "relative"); source_frame.addItem("Absolute frame", "absolute")
                source_frame.currentIndexChanged.connect(lambda index, box=source_frame: setattr(self.viewer, "paint_source_frame", "relative" if box.itemData(index) == "relative" else int(self.viewer.absolute_source_frame)))
                source_frame.setCurrentIndex(0 if self.viewer.paint_source_frame == "relative" else 1)
                form.addRow("Clone source frame", source_frame)
                absolute_frame = QSpinBox(); absolute_frame.setRange(-1000000, 1000000)
                absolute_frame.setValue(int(self.viewer.absolute_source_frame if isinstance(self.viewer.paint_source_frame, int) else 1))
                absolute_frame.valueChanged.connect(lambda value: setattr(self.viewer, "absolute_source_frame", int(value)))
                form.addRow("Absolute source frame", absolute_frame)
                follow = QComboBox(); follow.addItem("No tracking", None)
                for tracker_id, tracker_node in self.dispatcher.document["nodes"].items():
                    if tracker_node["type"] != "Tracker": continue
                    tracks = self.dispatcher.document.get("node_data", {}).get(tracker_id, {}).get("tracks", [])
                    for track_index, track in enumerate(tracks):
                        follow.addItem(f"{tracker_node['name']} · {track['name']}",
                                       {"node_id": tracker_id, "track_index": track_index})
                selected_follow = follow.findData(self.viewer.paint_follow_track)
                follow.setCurrentIndex(max(0, selected_follow))
                follow.currentIndexChanged.connect(lambda index, box=follow: setattr(self.viewer, "paint_follow_track", box.itemData(index)))
                form.addRow("Follow track", follow)
                dust = QPushButton("DustBust · clone from previous frame")
                dust.clicked.connect(lambda: (setattr(self.viewer, "dustbust_preset", True), setattr(self.viewer, "paint_tool", "clone")))
                form.addRow(dust)

                def detect_dustbust_specks(k=key):
                    selected = self.graph_nodes()[k]
                    source_id = selected["inputs"].get("image")
                    if not source_id:
                        QMessageBox.information(self, "DustBust", "Connect an image input first.")
                        return
                    params = selected["params"]
                    first, last = int(params["dustbust_frame_start"]), int(params["dustbust_frame_end"])
                    if last < first:
                        QMessageBox.warning(self, "DustBust", "The end frame must be at or after the start frame.")
                        return
                    progress = QProgressDialog("Scanning frames…", "Cancel", 0, last - first + 1, self)
                    progress.setWindowTitle("DustBust speck detection")
                    progress.setWindowModality(Qt.WindowModality.WindowModal)
                    progress.setMinimumDuration(0)
                    progress.show()
                    frames = {}
                    for index, frame_number in enumerate(range(first, last + 1)):
                        QApplication.processEvents()
                        if progress.wasCanceled():
                            progress.close(); return
                        try:
                            frames[frame_number] = self.evaluator.evaluate_raster(
                                self.graph_document(), target=source_id, frame=frame_number).to_display()
                        except Exception as exc:
                            progress.close()
                            QMessageBox.warning(self, "DustBust", f"Detection stopped: {exc}")
                            return
                        progress.setValue(index + 1)
                    progress.close()
                    specks = detect_specks(frames, sensitivity=float(params["dustbust_sensitivity"]))
                    if not specks:
                        QMessageBox.information(self, "DustBust", "No specks found in that frame range.")
                        return
                    review = QDialog(self)
                    review.setWindowTitle("Review detected specks")
                    layout = QVBoxLayout(review)
                    layout.addWidget(QLabel(f"{len(specks)} candidate speck(s). Uncheck any you don't want cleaned."))
                    listing = QListWidget()
                    for speck in specks:
                        item = QListWidgetItem(f"Frame {speck['frame']}: {speck['width']:.0f}x{speck['height']:.0f} "
                                               f"at ({speck['x']:.0f}, {speck['y']:.0f}), strength {speck['magnitude']:.3f}")
                        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                        item.setCheckState(Qt.CheckState.Checked)
                        item.setData(Qt.ItemDataRole.UserRole, speck)
                        listing.addItem(item)
                    layout.addWidget(listing)
                    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
                    buttons.accepted.connect(review.accept)
                    buttons.rejected.connect(review.reject)
                    layout.addWidget(buttons)
                    if review.exec() != QDialog.DialogCode.Accepted:
                        return
                    accepted = [listing.item(i).data(Qt.ItemDataRole.UserRole) for i in range(listing.count())
                               if listing.item(i).checkState() == Qt.CheckState.Checked]
                    if not accepted:
                        return
                    existing_items = self.dispatcher.document.get("node_data", {}).get(k, {}).get("items", [])
                    items = dustbust_items_for_specks(existing_items, accepted)
                    self.command({"op": "set_paint_items", "id": k, "items": items})

                specks_button = QPushButton("Detect specks…")
                specks_button.setToolTip("Scans dustbust_frame_start..end for brief, small anomalies "
                                         "and lets you accept or reject each one before cloning it")
                specks_button.clicked.connect(detect_dustbust_specks)
                form.addRow(specks_button)

                # Dedicated per-layer editor: fixing an already-drawn shape or stroke no longer
                # means deleting it and redrawing, the one thing the brush controls above (which
                # only ever configure the *next* stroke) could never do.
                current_time = int(self.dispatcher.document["time"]["current"])
                layer_items = self.dispatcher.document.get("node_data", {}).get(key, {}).get("items", [])
                selected = self.viewer.paint_selected_item_index
                if selected is not None and not (0 <= selected < len(layer_items)):
                    selected = self.viewer.paint_selected_item_index = None

                def current_layer_items(k=key):
                    return copy.deepcopy(self.dispatcher.document.get("node_data", {}).get(k, {}).get("items", []))

                form.addRow(QLabel("Layers"))
                layer_list = QListWidget()
                layer_list.setObjectName("rotopaint-layer-list")
                for entry in layer_items:
                    label = (f"{entry['name']} · shape · {entry['mode']}" if entry["kind"] == "shape"
                             else f"{entry['name']} · stroke · {entry['tool']}")
                    layer_list.addItem(label)
                if selected is not None:
                    layer_list.setCurrentRow(selected)

                def select_layer(row, k=key):
                    self.viewer.paint_selected_item_index = row if row >= 0 else None
                    self.inspect(k)
                layer_list.currentRowChanged.connect(select_layer)
                form.addRow(layer_list)

                def move_layer(delta, k=key, idx=selected):
                    entries = current_layer_items(k)
                    target = None if idx is None else idx + delta
                    if idx is None or not (0 <= idx < len(entries)) or not (0 <= target < len(entries)):
                        return
                    entries[idx], entries[target] = entries[target], entries[idx]
                    self.viewer.paint_selected_item_index = target
                    self.command({"op": "set_paint_items", "id": k, "items": entries})

                def delete_layer(k=key, idx=selected):
                    entries = current_layer_items(k)
                    if idx is None or not (0 <= idx < len(entries)):
                        return
                    del entries[idx]
                    self.viewer.paint_selected_item_index = None
                    self.command({"op": "set_paint_items", "id": k, "items": entries})

                layer_buttons = QHBoxLayout()
                up_button = QPushButton("Move up")
                up_button.setObjectName("rotopaint-layer-move-up")
                up_button.setEnabled(selected is not None and selected > 0)
                up_button.clicked.connect(lambda: move_layer(-1))
                down_button = QPushButton("Move down")
                down_button.setObjectName("rotopaint-layer-move-down")
                down_button.setEnabled(selected is not None and selected < len(layer_items) - 1)
                down_button.clicked.connect(lambda: move_layer(1))
                delete_button = QPushButton("Delete layer")
                delete_button.setObjectName("rotopaint-layer-delete")
                delete_button.setEnabled(selected is not None)
                delete_button.clicked.connect(lambda: delete_layer())
                for button in (up_button, down_button, delete_button):
                    layer_buttons.addWidget(button)
                form.addRow(layer_buttons)

                if selected is not None:
                    edit_item = layer_items[selected]

                    def update_layer(path, value, k=key, idx=selected):
                        entries = current_layer_items(k)
                        if idx >= len(entries): return
                        target = entries[idx]
                        for step in path[:-1]:
                            target = target[step]
                        target[path[-1]] = value
                        self.command({"op": "set_paint_items", "id": k, "items": entries})

                    form.addRow(QLabel(f"Edit layer: {edit_item['name']}"))
                    name_field = QLineEdit(edit_item["name"])
                    name_field.setObjectName("rotopaint-layer-name")
                    name_field.editingFinished.connect(
                        lambda f=name_field, current=edit_item["name"]: f.text() != current
                        and update_layer(("name",), f.text()))
                    form.addRow("Name", name_field)
                    visible_field = QCheckBox("Visible")
                    visible_field.setObjectName("rotopaint-layer-visible")
                    visible_field.setChecked(edit_item["visible"])
                    visible_field.toggled.connect(lambda value: update_layer(("visible",), bool(value)))
                    form.addRow(visible_field)
                    layer_blend = QComboBox()
                    layer_blend.setObjectName("rotopaint-layer-blend")
                    for value in ("over", "add", "multiply", "screen"): layer_blend.addItem(value.title(), value)
                    layer_blend.setCurrentIndex(layer_blend.findData(edit_item["blend"]))
                    layer_blend.currentIndexChanged.connect(
                        lambda idx2, box=layer_blend: update_layer(("blend",), box.itemData(idx2)))
                    form.addRow("Blend", layer_blend)
                    layer_opacity_field = QDoubleSpinBox()
                    layer_opacity_field.setObjectName("rotopaint-layer-opacity")
                    layer_opacity_field.setRange(0.0, 1.0); layer_opacity_field.setSingleStep(0.05)
                    layer_opacity_field.setValue(edit_item["opacity"])
                    layer_opacity_field.valueChanged.connect(lambda value: update_layer(("opacity",), float(value)))
                    form.addRow("Opacity", layer_opacity_field)
                    if edit_item["kind"] == "shape":
                        mode_field = QComboBox()
                        mode_field.setObjectName("rotopaint-layer-mode")
                        for value in shape_model.SHAPE_MODES: mode_field.addItem(value.title(), value)
                        mode_field.setCurrentIndex(mode_field.findData(edit_item["mode"]))
                        mode_field.currentIndexChanged.connect(
                            lambda idx2, box=mode_field: update_layer(("mode",), box.itemData(idx2)))
                        form.addRow("Mode", mode_field)
                        feather_field = QDoubleSpinBox()
                        feather_field.setObjectName("rotopaint-layer-feather")
                        feather_field.setRange(0.0, 500.0); feather_field.setValue(edit_item["feather"])
                        feather_field.valueChanged.connect(lambda value: update_layer(("feather",), float(value)))
                        form.addRow("Feather", feather_field)
                    else:
                        form.addRow("Tool", QLabel(edit_item["tool"].title()))
                        for title, bkey, lo, hi, step in (
                            ("Brush size", "size", 0.1, 4096, 1.0), ("Hardness", "hardness", 0.0, 1.0, 0.05),
                            ("Brush opacity", "opacity", 0.0, 1.0, 0.05), ("Spacing", "spacing", 0.01, 4.0, 0.05),
                            ("Dodge / burn strength", "strength", 0.0, 1.0, 0.05)):
                            field = QDoubleSpinBox()
                            field.setObjectName(f"rotopaint-layer-brush-{bkey}")
                            field.setRange(lo, hi); field.setSingleStep(step)
                            field.setValue(edit_item["brush"][bkey])
                            field.valueChanged.connect(lambda value, bk=bkey: update_layer(("brush", bk), float(value)))
                            form.addRow(title, field)
                        if edit_item["tool"] == "paint":
                            def choose_layer_color(k=key, idx=selected):
                                entries = current_layer_items(k)
                                if idx >= len(entries): return
                                chosen = QColorDialog.getColor(QColor.fromRgbF(*entries[idx]["color"]),
                                                               self, "RotoPaint colour")
                                if chosen.isValid():
                                    entries[idx]["color"] = [chosen.redF(), chosen.greenF(),
                                                             chosen.blueF(), chosen.alphaF()]
                                    self.command({"op": "set_paint_items", "id": k, "items": entries})
                            color_button = QPushButton("Choose colour…")
                            color_button.setObjectName("rotopaint-layer-color")
                            color_button.clicked.connect(lambda: choose_layer_color())
                            form.addRow(color_button)
                        layer_lifetime = QComboBox()
                        layer_lifetime.setObjectName("rotopaint-layer-lifetime")
                        for label2, value in (("Single frame", "single"), ("All frames", "all"),
                                              ("Frame range", "range"), ("From current frame", "from_current")):
                            layer_lifetime.addItem(label2, value)
                        layer_lifetime.setCurrentIndex(layer_lifetime.findData(edit_item["lifetime"]["mode"]))
                        layer_lifetime.currentIndexChanged.connect(
                            lambda idx2, box=layer_lifetime, current=edit_item["lifetime"]: update_layer(
                                ("lifetime",), _lifetime_for_mode(box.itemData(idx2), current, current_time)))
                        form.addRow("Lifetime", layer_lifetime)
                        if edit_item["lifetime"]["mode"] == "range":
                            for title, lkey in (("First frame", "first"), ("Last frame", "last")):
                                range_field = QSpinBox()
                                range_field.setObjectName(f"rotopaint-layer-lifetime-{lkey}")
                                range_field.setRange(-1000000, 1000000)
                                range_field.setValue(int(edit_item["lifetime"][lkey]))
                                range_field.valueChanged.connect(
                                    lambda value, lk=lkey: update_layer(("lifetime", lk), int(value)))
                                form.addRow(title, range_field)
            if node["type"] == "Flare":
                link = QComboBox()
                link.setObjectName("flare-tracker-link")
                link.setToolTip("Follow a Tracker point's motion from its reference frame")
                link.addItem("No Tracker link", ("", -1))
                for tracker_id, tracker_node in self.dispatcher.document["nodes"].items():
                    if tracker_node["type"] not in ("Tracker", "Stabilize"):
                        continue
                    tracks = self.dispatcher.document.get("node_data", {}).get(tracker_id, {}).get("tracks", [])
                    for track_index, track in enumerate(tracks):
                        link.addItem(f"{tracker_node['name']} · {track['name']}", (tracker_id, track_index))
                current_link = (node["params"].get("tracker_id", ""),
                                int(node["params"].get("track_index", -1)))
                selected = link.findData(current_link)
                link.setCurrentIndex(max(0, selected))
                link.currentIndexChanged.connect(
                    lambda index, box=link, k=key: self.command({"op": "batch", "commands": [
                        {"op": "set", "id": k, "param": "tracker_id", "value": box.itemData(index)[0]},
                        {"op": "set", "id": k, "param": "track_index", "value": box.itemData(index)[1]}]}))
                form.addRow("Tracker link", link)
            if node["type"] == "ZDefocus":
                depth_pick = QPushButton("Pick focal plane from viewer…")
                depth_pick.clicked.connect(lambda checked=False, k=key: self.begin_zdefocus_pick(k))
                form.addRow(depth_pick)
            if node["type"] == "Camera3D":
                focus_pick = QPushButton("Pick focus from viewer…")
                focus_pick.setToolTip("View a Render3D that uses this camera, then click the surface to focus on · Esc ends")
                focus_pick.clicked.connect(lambda checked=False, k=key: self.begin_focus_pick(k))
                form.addRow(focus_pick)
            if node["type"] == "Cryptomatte":
                crypto_pick = QPushButton("Pick from viewer…")
                crypto_pick.setToolTip("Click objects in the viewer to add their names to the matte list · Esc ends")
                crypto_pick.clicked.connect(lambda checked=False, k=key: self.begin_crypto_pick(k))
                form.addRow(crypto_pick)
            if node["type"] in ("ViewMetaData", "CompareMetaData"):
                self.add_metadata_view(form, key, node["type"])
            metadata_hint = {
                "ModifyMetaData": "One edit per line: set <key> <value>, remove <key>,\nrename <old> <new>. Values take [frame] and [metadata key].",
                "CopyMetaData": "Lays keys from the second input over this image's.\nEmpty keys copies all of them.",
                "AddTimeCode": "Writes the timecode key: the start timecode shows at the\nstart frame and counts one frame per timeline frame.",
                "BurnIn": "Slots take text with [frame] and [metadata key],\nfor example [metadata input/filename].",
            }.get(node["type"])
            if metadata_hint:
                hint_label = QLabel(metadata_hint)
                hint_label.setWordWrap(True)
                form.addRow(hint_label)
            if node["type"] in ("Tracker", "Stabilize"):
                pick = QPushButton("Add track point at reference…")
                pick.setToolTip(f"View this {node['type']}, then click the reference point in the viewer")
                pick.clicked.connect(lambda checked=False, k=key: self.begin_tracker_pick(k))
                form.addRow(pick)
                range_first = QSpinBox(); range_first.setObjectName("tracker-range-first")
                range_last = QSpinBox(); range_last.setObjectName("tracker-range-last")
                timeline = self.dispatcher.document["time"]
                range_first.setRange(timeline["first"], timeline["last"])
                range_last.setRange(timeline["first"], timeline["last"])
                range_first.setValue(timeline["first"]); range_last.setValue(timeline["last"])
                analyse_button = QPushButton("Analyze forward")
                analyse_button.setToolTip("Analyze from the reference frame through the timeline end")
                analyse_button.clicked.connect(lambda checked=False, k=key, b=range_last:
                                               self.analyse_tracker(k, "forward", last_frame=b.value()))
                form.addRow(analyse_button)
                backward_button = QPushButton("Analyze backward")
                backward_button.setToolTip("Analyze from the reference frame back to the timeline start")
                backward_button.clicked.connect(lambda checked=False, k=key, a=range_first:
                                                self.analyse_tracker(k, "backward", first_frame=a.value()))
                form.addRow(backward_button)
                cancel_button = QPushButton("Cancel analysis")
                cancel_button.setEnabled(self._tracker_future is not None)
                cancel_button.clicked.connect(self.cancel_tracker_analysis)
                form.addRow(cancel_button)
                form.addRow("Track from frame", range_first)
                form.addRow("Track through frame", range_last)
                key_track = QPushButton("Key selected point at current frame")
                key_track.clicked.connect(lambda checked=False, k=key: self.key_tracker_point(k))
                form.addRow(key_track)
                clear_track = QPushButton("Clear selected point keys in range")
                clear_track.clicked.connect(lambda checked=False, k=key, a=range_first, b=range_last:
                                            self.clear_tracker_range(k, a.value(), b.value()))
                form.addRow(clear_track)
                export_transform = QPushButton("Export Transform")
                export_transform.clicked.connect(lambda checked=False, k=key: self.export_tracker_transform(k))
                form.addRow(export_transform)
                export_pin = QPushButton("Export CornerPin")
                export_pin.clicked.connect(lambda checked=False, k=key: self.export_tracker_cornerpin(k))
                form.addRow(export_pin)
            if node["type"] == "ReadAlembic3D" and not node["disabled"]:
                from . import alembicio
                # Inspector hints must never prevent opening an invalid node's panel.
                try:
                    counts = alembicio.unsupported_schemas(
                        node["params"]["abc_path"], node["params"]["abc_root"])
                except Exception:
                    counts = {}
                if counts:
                    schemas = ", ".join(f"{schema}: {count}" for schema, count in sorted(counts.items()))
                    hint = QLabel(f"Skipped {sum(counts.values())} unsupported objects: {schemas}")
                    hint.setWordWrap(True)
                    form.addRow(hint)
            if (node["type"] in ("ReadUSD3D", "ReadUSDCamera3D") or
                    (node["type"] == "WriteGeo3D" and
                     Path(node["params"]["geo_write_path"]).suffix.lower() in
                     (".usd", ".usda", ".usdc", ".usdz"))):
                from . import usdio
                if not usdio.available():
                    form.addRow(QLabel("USD support not installed: pip install nodebased[usd]"))
            if node["type"] == "WriteGeo3D":
                for label, single in (("Export current frame", True), ("Export frame range", False)):
                    button = QPushButton(label)
                    button.clicked.connect(lambda checked=False, k=key, s=single: self.export_geometry(k, s))
                    form.addRow(button)
            if node["type"] == "GenerateLUT":
                button = QPushButton("Generate .cube LUT")
                button.clicked.connect(lambda checked=False, k=key: self.export_lut(k))
                form.addRow(button)
            if node["type"] == "WriteSplat3D":
                for label, single in (("Export current frame", True), ("Export frame range", False)):
                    button = QPushButton(label)
                    button.clicked.connect(lambda checked=False, k=key, s=single: self.export_splats(k, s))
                    form.addRow(button)
            if node["type"] == "WriteVDB3D":
                for label, single in (("Export current frame", True), ("Export frame range", False)):
                    button = QPushButton(label)
                    button.clicked.connect(lambda checked=False, k=key, s=single: self.export_vdb(k, s))
                    form.addRow(button)
            if node["type"] in GEOMETRY_TYPES:
                # A material-ball preview (R6 "next", closed): a fixed sphere carrying this node's
                # own colour/specular/PBR knobs, not the node's real shape (the look-dev convention).
                # Built once with the rest of the panel; a slider drag does not refresh it until the
                # panel rebuilds (reselecting the node, most simply).
                try:
                    preview = materialpreview.render(node["params"])
                    preview_image = QImage(preview.data, preview.shape[1], preview.shape[0],
                                           preview.strides[0], QImage.Format.Format_RGBA8888).copy()
                    preview_label = QLabel()
                    preview_label.setPixmap(QPixmap.fromImage(preview_image))
                    preview_label.setToolTip("Material preview: a fixed sphere under one light and a "
                                             "soft dome, not this node's own shape")
                    form.addRow("Material", preview_label)
                except Exception:
                    pass  # a broken param must not take the properties panel down
            if node["type"] in fluidknobpresets.FLUID_NODE_TYPES:
                preset_row = QWidget()
                preset_layout = QHBoxLayout(preset_row)
                preset_layout.setContentsMargins(0, 0, 0, 0)
                save_knob_preset = QPushButton("Save knob preset")
                save_knob_preset.setObjectName("saveFluidKnobPreset")
                save_knob_preset.clicked.connect(
                    lambda checked=False, k=key: self.node_toolbar.save_fluid_knob_preset(k))
                load_knob_preset = QPushButton("Load knob preset")
                load_knob_preset.setObjectName("loadFluidKnobPreset")
                load_knob_preset.clicked.connect(
                    lambda checked=False, k=key: self.node_toolbar.load_fluid_knob_preset(k))
                preset_layout.addWidget(save_knob_preset)
                preset_layout.addWidget(load_knob_preset)
                form.addRow("Knob presets", preset_row)
            if node["type"] == "Write":
                render_frame = QPushButton("Render current frame")
                render_frame.setToolTip("Full-resolution reference render of the frame at the playhead")
                render_frame.clicked.connect(lambda checked=False, k=key: self.render_write(k, single=True))
                form.addRow(render_frame)
                render_range = QPushButton("Render frame range")
                render_range.setToolTip("Renders the project frame range; needs a padded path "
                                        "pattern such as render.%04d.exr")
                render_range.clicked.connect(lambda checked=False, k=key: self.render_write(k, single=False))
                form.addRow(render_range)
            # Nuke keeps presentation and bypass on a second "Node" tab, away from the knobs
            # that change pixels.
            tabs = QTabWidget()
            tabs.setObjectName("node-tabs")
            tabs.addTab(top_aligned(content), "Knobs")
            user_knobs = node.get("user_knobs") or (self.graph_document().get("node_data", {}).get(key, {}).get("user_knobs", []))
            if user_knobs:
                user_tab = QWidget()
                user_layout = QVBoxLayout(user_tab)
                for user_knob in user_knobs:
                    row = QLabel(str(user_knob))
                    user_layout.addWidget(row)
                tabs.addTab(top_aligned(user_tab), "User")
            tabs.addTab(top_aligned(self.node_tab(key, node)), "Node")
            tabs.setCurrentIndex(min(self.properties_tab, tabs.count() - 1))
            tabs.currentChanged.connect(lambda index: setattr(self, "properties_tab", index))
            panel_layout.addWidget(tabs, 1)
        return panel

    def inspect(self, key):
        if hasattr(self, "viewport"):
            self.viewport.refresh_sim_stats()
        if not self.pinned_panels:
            self.set_properties_widget(self.build_node_panel(key))
            self._refresh_artist_tools(key)
            return
        if key is not None and key in self.pinned_panels:
            self.pinned_panels.remove(key)
            self.pinned_panels.insert(0, key)
        self.rebuild_properties_dock()
        self._refresh_artist_tools(key)

    def _artist_tools_evaluator(self):
        if getattr(self, "_artist_tools_eval", None) is None:
            self._artist_tools_eval = Evaluator(cache_bytes=64 << 20)
        return self._artist_tools_eval

    def _refresh_artist_tools(self, key):
        """Feeds the slice viewer and cache inspector docks from the selected node, when it is a
        kind either understands. A broken or unwired graph must not take either panel down."""
        document = self.dispatcher.document if getattr(self, "dispatcher", None) is not None else None
        if not document or key is None or key not in document.get("nodes", {}):
            if self.slice_dock.isVisible():
                self.slice_view.set_volume(None)
            return
        evaluator = self._artist_tools_evaluator()
        if self.slice_dock.isVisible():
            try:
                volume = cachecontext.volume_for_node(evaluator, document, key)
            except Exception:
                volume = None
            self.slice_view.set_volume(volume)
        if self.cache_inspector_dock.isVisible():
            try:
                context = cachecontext.cache_context(evaluator, document, key)
            except Exception:
                context = None
            if context is None:
                self.cache_inspector.set_cache(None, None, 1, 1)
            else:
                store, run, start_frame, _substeps = context
                current = int((document.get("time") or {}).get("current", start_frame))
                self.cache_inspector.set_cache(store, run, start_frame, max(start_frame, current))

    def _resolve_cache_range(self, start_frame, end_frame):
        """"Re-solve range" on the cache inspector: walk the frames forward through the evaluator
        so each one is solved and banked, exactly like scrubbing the timeline across them."""
        key = self.graph.selected_id() if getattr(self, "graph", None) is not None else None
        document = self.dispatcher.document if getattr(self, "dispatcher", None) is not None else None
        if key is None or not document:
            return
        evaluator = self._artist_tools_evaluator()
        for frame in range(int(start_frame), int(end_frame) + 1):
            try:
                evaluator.evaluate_raster(document, key, frame=frame, typed=True)
            except Exception:
                break

    def pin_panel(self, key):
        if key is None or key not in self.graph_nodes():
            return
        if key in self.pinned_panels:
            self.pinned_panels.remove(key)
        self.pinned_panels.insert(0, key)
        self.pinned_panels = self.pinned_panels[:self.panel_cap]
        self.rebuild_properties_dock()

    def close_panel(self, key):
        if key in self.pinned_panels:
            self.pinned_panels.remove(key)
        self.rebuild_properties_dock()

    def clear_panels(self):
        self.pinned_panels = []
        self.rebuild_properties_dock()

    def set_panel_cap(self, value):
        value = max(1, int(value))
        self.panel_cap = value
        self.preferences.set_max_properties_panels(value)
        if len(self.pinned_panels) > value:
            self.pinned_panels = self.pinned_panels[:value]
        if self.pinned_panels:
            self.rebuild_properties_dock()

    def rebuild_properties_dock(self):
        if not self.pinned_panels:
            selected = self.graph.selected_id()
            self.set_properties_widget(self.build_node_panel(selected))
            return
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)
        header = QHBoxLayout()
        header.addWidget(ElidedLabel(f"{len(self.pinned_panels)} panel(s) open"), 1)
        header.addWidget(QLabel("Max panels"))
        cap_spin = QSpinBox()
        cap_spin.setObjectName("panel-cap-spin")
        cap_spin.setRange(1, 20)
        cap_spin.setValue(self.panel_cap)
        cap_spin.valueChanged.connect(self.set_panel_cap)
        header.addWidget(cap_spin)
        clear_all = QPushButton("Clear all")
        clear_all.setObjectName("clear-all-panels")
        clear_all.clicked.connect(self.clear_panels)
        header.addWidget(clear_all)
        layout.addLayout(header)
        for panel_key in self.pinned_panels:
            if panel_key not in self.graph_nodes():
                continue
            section = QWidget()
            section.setObjectName("stacked-panel-section")
            section_layout = QVBoxLayout(section)
            section_layout.setContentsMargins(0, 0, 0, 0)
            section_header = QHBoxLayout()
            node = self.graph_nodes()[panel_key]
            collapse = QPushButton("▾")
            collapse.setObjectName("panel-collapse")
            collapse.setCheckable(True)
            collapse.setChecked(True)
            collapse.setToolTip("Collapse or expand this node's panel")
            collapse.setFixedWidth(26)
            collapse.setStyleSheet("padding: 0px")
            section_header.addWidget(collapse, 1)
            section_layout.addLayout(section_header)
            body = self.build_node_panel(panel_key)
            section_layout.addWidget(body)
            collapse.toggled.connect(body.setVisible)
            collapse.toggled.connect(lambda expanded, w=collapse: w.setText("▾" if expanded else "▸"))
            layout.addWidget(section)
        layout.addStretch()
        self.set_properties_widget(container)

    def set_properties_widget(self, panel):
        """Swap the dock's contents, sized to the dock rather than to the panel's own wishes."""
        old = self.properties.takeWidget()
        if old:
            old.deleteLater()
        make_fluid(panel)
        if not isinstance(panel, FluidRoot):
            root = FluidRoot()
            box = QVBoxLayout(root)
            box.setContentsMargins(0, 0, 0, 0)
            box.addWidget(panel)
            panel = root
        self.properties.setWidget(panel)

    def preview_knobs(self, key, values):
        """Reflect a viewer drag in its open numeric controls without dispatching edits."""
        for control in self.properties_dock.findChildren(QDoubleSpinBox):
            if control.property("nodebased_node_id") != key:
                continue
            param = control.property("nodebased_param")
            if param not in values:
                continue
            control.blockSignals(True)
            control.setValue(float(values[param]))
            control.blockSignals(False)

    def attach_text_menu(self, editor, default=None, commit=None):
        """Give a text knob a context menu that outlives a panel rebuild.

        QLineEdit's own menu is parented to the line edit, so it is destroyed the moment the
        inspector is rebuilt -- which is what made a right-click look like a menu that flashed and
        vanished. This builds the same standard actions under a window-owned menu, and adds the
        "Set to default" entry a compositing knob is expected to have.
        """
        editor.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        def show(point):
            menu = QMenu(self)
            standard = editor.createStandardContextMenu()
            for action in standard.actions():
                menu.addAction(action)
            if commit is not None:
                menu.addSeparator()
                label = f"Set to default ({default!r})" if default else "Clear"
                menu.addAction(label, lambda: commit(default or ""))
            menu.exec(editor.mapToGlobal(point))
            standard.deleteLater()

        editor.customContextMenuRequested.connect(show)

    def defer_command(self, cmd):
        # Do not destroy an editor while it is emitting editingFinished.
        QTimer.singleShot(0, lambda: self.command(cmd))

    def begin_roto_draw(self, key):
        """Enter viewer drawing mode for a Roto node after making it the viewed format."""
        node = self.dispatcher.document["nodes"].get(key)
        if node is None or node["type"] != "Roto":
            self._show_command_error(ValueError("Roto drawing requires a Roto node"))
            return False
        if self.dispatcher.document.get("view") != key:
            self.command({"op": "view", "id": key})
        frame = int(self.dispatcher.document["time"]["current"])
        if int(node["params"].get("reference_frame", frame)) != frame:
            self.command({"op": "set", "id": key, "param": "reference_frame", "value": frame})
        return self.viewer.begin_roto_draw(key)

    def _tracker_context(self, key=None):
        selected = key or self.graph.selected_id()
        node = self.dispatcher.document["nodes"].get(selected) if selected else None
        if node is None or node["type"] not in ("Tracker", "Stabilize") or self.dispatcher.document.get("view") != selected:
            return None
        return selected, node, self.dispatcher.document.get("node_data", {}).get(selected, {"tracks": []}), 1

    def metadata_rows(self, key, kind):
        """`(columns, rows)` for the metadata inspector of node `key`: ViewMetaData lists every key and
        value of its input; CompareMetaData lists the keys whose values differ between its two inputs.
        Raises ValueError with the reason when the input cannot be evaluated."""
        document = self.dispatcher.document
        node = document["nodes"][key]
        frame = int(document["time"]["current"])

        def meta_of(slot):
            source = node["inputs"].get(slot)
            if source is None:
                return None
            raster = self.evaluator.evaluate_raster(document, source, frame=frame, tier=1)
            return raster.meta or {}
        first = meta_of("image")
        if first is None:
            raise ValueError("Connect an image to see its metadata")
        if kind == "ViewMetaData":
            return ("Key", "Value"), sorted(first.items())
        second = meta_of("other")
        if second is None:
            raise ValueError("Connect a second image to compare against")
        return ("Key", "Input", "Other"), [(k, a if a is not None else "", b if b is not None else "")
                                           for k, a, b in metadata_module.compare(first, second)]

    def add_metadata_view(self, form, key, kind):
        """The metadata inspector on a ViewMetaData or CompareMetaData panel: a key/value table with a
        search box that filters rows by key or value."""
        from PySide6.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView
        try:
            columns, rows = self.metadata_rows(key, kind)
            problem = ""
        except (ValueError, OSError, KeyError) as error:
            columns, rows, problem = ("Key", "Value"), [], str(error)
        search = QLineEdit()
        search.setObjectName("metadata-search")
        search.setPlaceholderText("Search keys and values")
        table = QTableWidget(len(rows), len(columns))
        table.setObjectName("metadata-table")
        table.setHorizontalHeaderLabels(list(columns))
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setMinimumHeight(180)
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setToolTip(str(value))
                table.setItem(row, column, item)

        def filter_rows(text):
            needle = text.strip().lower()
            for row, values in enumerate(rows):
                table.setRowHidden(row, bool(needle) and not any(needle in str(v).lower() for v in values))
        search.textChanged.connect(filter_rows)
        form.addRow(search)
        form.addRow(table)
        summary = QLabel(problem or (f"{len(rows)} keys" if kind == "ViewMetaData"
                                     else f"{len(rows)} differing keys"))
        summary.setObjectName("metadata-summary")
        summary.setWordWrap(True)
        form.addRow(summary)

    def begin_zdefocus_pick(self, key):
        node = self.dispatcher.document["nodes"].get(key)
        if node is None or node["type"] != "ZDefocus":
            return False
        self.viewer.zdefocus_picking = key
        self.viewer.setCursor(Qt.CursorShape.CrossCursor)
        self.statusBar().showMessage("ZDefocus: click a depth value in the viewer · Esc ends")
        return True

    def zdefocus_pick(self, key, x, y):
        node = self.dispatcher.document["nodes"].get(key)
        source = None if node is None else node["inputs"].get("image")
        if source is None:
            self._show_command_error(ValueError("Wire the ZDefocus image input first"))
            return False
        try:
            raster = self.evaluator.evaluate_raster(self.dispatcher.document, source,
                                                    frame=int(self.dispatcher.document["time"]["current"]), tier=1)
            layer = self.evaluator._layer_of("ZDefocus", raster, node["params"].get("depth_layer", "depth.Z"))
            ix, iy = x - layer.data.x, y - layer.data.y
            value = float(layer.pixels[iy, ix, 0])
            if not math.isfinite(value) or value <= 0:
                raise ValueError("The chosen viewer pixel has no positive depth")
        except (ValueError, IndexError) as error:
            self._show_command_error(error)
            return False
        self.command({"op": "set", "id": key, "param": "focal_plane", "value": value})
        self.statusBar().showMessage(f"ZDefocus focal plane: {value:.4g}")
        return True

    def _focus_pick_render(self, key):
        """The Render3D node whose picture a focus click on Camera3D `key` reads: the viewed one when it uses
        that camera, else the first Render3D that does."""
        document = self.dispatcher.document
        uses = [k for k, n in document["nodes"].items()
                if n["type"] == "Render3D" and n["inputs"].get("camera") == key]
        viewed = document.get("view")
        return viewed if viewed in uses else (uses[0] if uses else None)

    def begin_focus_pick(self, key):
        node = self.dispatcher.document["nodes"].get(key)
        if node is None or node["type"] != "Camera3D":
            self._show_command_error(ValueError("Focus picking needs a Camera3D node"))
            return False
        if self._focus_pick_render(key) is None:
            self._show_command_error(ValueError("Wire this camera into a Render3D node first"))
            return False
        self.viewer.focus_picking = key
        self.viewer.setCursor(Qt.CursorShape.CrossCursor)
        self.statusBar().showMessage("Camera3D: click a surface in the viewer to focus on it · Esc ends")
        return True

    def focus_pick(self, key, x, y):
        """Set Camera3D `key`'s focus distance to the view depth of the surface under full-resolution pixel (x, y) of
        the Render3D picture (`lens.pick_focus_distance`)."""
        from . import lens
        document = self.dispatcher.document
        render_key = self._focus_pick_render(key)
        if render_key is None:
            self._show_command_error(ValueError("Wire this camera into a Render3D node first"))
            return False
        render = document["nodes"][render_key]
        frame = int(document["time"]["current"])
        try:
            scene = self.evaluator.evaluate_raster(document, render["inputs"]["scene"], frame=frame, tier=1, typed=True)
            camera = self.evaluator.evaluate_raster(document, key, frame=frame, tier=1, typed=True)
            distance = lens.pick_focus_distance(camera, scene, render["params"]["width"], render["params"]["height"],
                                                x, y)
        except (ValueError, KeyError) as error:
            self._show_command_error(error)
            return False
        if distance is None or not math.isfinite(distance) or distance <= 0:
            self.statusBar().showMessage("Camera3D: nothing under the cursor to focus on")
            return False
        self.command({"op": "set", "id": key, "param": "focus_distance", "value": distance})
        self.statusBar().showMessage(f"Camera3D focus distance: {distance:.4g}")
        return True

    def begin_crypto_pick(self, key):
        node = self.dispatcher.document["nodes"].get(key)
        if node is None or node["type"] != "Cryptomatte":
            self._show_command_error(ValueError("Picking needs a Cryptomatte node"))
            return False
        self.viewer.crypto_picking = key
        self.viewer.setCursor(Qt.CursorShape.CrossCursor)
        self.statusBar().showMessage("Cryptomatte: click objects in the viewer to add them · Esc ends")
        return True

    def crypto_pick(self, key, x, y):
        """Add the object at full-resolution pixel (x, y) to `key`'s matte list, reading the id under
        the cursor from the node's input (the viewer only shows the finished matte)."""
        from . import cryptomatte
        document = self.dispatcher.document
        node = document["nodes"].get(key)
        source = None if node is None else node["inputs"].get("image")
        if source is None:
            self._show_command_error(ValueError("Wire the Cryptomatte node's image input first"))
            return False
        try:
            raster = self.evaluator.evaluate_raster(document, source, frame=int(document["time"]["current"]), tier=1)
            token = cryptomatte.pick_token(raster, node["params"].get("crypto_layer", ""), x, y)
        except ValueError as error:
            self._show_command_error(error)
            return False
        if token is None:
            self.statusBar().showMessage("Cryptomatte: nothing under the cursor")
            return False
        current = node["params"].get("matte_list", "")
        updated = cryptomatte.add_token(current, token)
        if updated != current:
            self.command({"op": "set", "id": key, "param": "matte_list", "value": updated})
        self.statusBar().showMessage(f"Cryptomatte: {token}" + ("" if updated != current else " (already listed)"))
        return True

    def begin_tracker_pick(self, key):
        if self._tracker_future is not None:
            self._show_command_error(ValueError("Cancel the running Tracker analysis before picking another point"))
            return False
        node = self.dispatcher.document["nodes"].get(key)
        if node is None or node["type"] not in ("Tracker", "Stabilize"):
            self._show_command_error(ValueError("Track-point picking requires a Tracker or Stabilize node"))
            return False
        if self.dispatcher.document.get("view") != key:
            self.command({"op": "view", "id": key})
        if self._tracker_context(key) is None:
            self._show_command_error(ValueError("View the Tracker's image input before picking a point"))
            return False
        self.viewer.tracker_picking = True
        self.viewer.setCursor(Qt.CursorShape.CrossCursor)
        self.statusBar().showMessage("Tracker: click a point in the reference frame · Esc cancels")
        return True

    def add_tracker_point(self, point):
        if self._tracker_future is not None:
            return False
        key = self.graph.selected_id()
        context = self._tracker_context(key)
        if context is None:
            return False
        payload = copy.deepcopy(context[2])
        if len(payload.get("tracks", [])) >= 16:
            self._show_command_error(ValueError("Tracker supports at most 16 track points"))
            return False
        names = {track.get("name") for track in payload.get("tracks", [])}
        index = 1
        while f"track{index}" in names:
            index += 1
        self._tracker_seed = {"name": f"track{index}", "enabled": 1,
                              "x": float(point[0]), "y": float(point[1])}
        self._tracker_key = key
        self._tracker_index = len(payload.get("tracks", []))
        self.statusBar().showMessage(f"Picked {self._tracker_seed['name']} at reference point; ready to analyze")
        return True

    def key_tracker_point(self, key):
        context = self._tracker_context(key)
        index = self._tracker_selected_index
        if context is None or index is None or index >= len(context[2].get("tracks", [])):
            self._show_command_error(ValueError("Select a track point in the viewer first"))
            return False
        frame = int(self.dispatcher.document["time"]["current"])
        tracks = copy.deepcopy(context[2]["tracks"])
        track = tracks[index]
        for field in ("x", "y"):
            value = shape_model.resolve_scalar(track[field], frame, field)
            track[field] = self.viewer._roto_scalar_at_frame(track[field], frame, value)
        self.command({"op": "set_tracks", "id": key, "tracks": tracks})
        self.statusBar().showMessage(f"Keyed {track['name']} at frame {frame}")
        return True

    def clear_tracker_range(self, key, first, last):
        context = self._tracker_context(key)
        index = self._tracker_selected_index
        if context is None or index is None or index >= len(context[2].get("tracks", [])):
            self._show_command_error(ValueError("Select a track point in the viewer first"))
            return False
        lo, hi = sorted((int(first), int(last)))
        tracks = copy.deepcopy(context[2]["tracks"])
        track = tracks[index]
        removed = 0
        for field in ("x", "y", "error"):
            scalar = track.get(field)
            if not isinstance(scalar, dict) or not scalar.get("curve"):
                continue
            curve = copy.deepcopy(scalar["curve"])
            before = curve["keys"]
            curve["keys"] = [item for item in before if not lo <= int(item["frame"]) <= hi]
            removed += len(before) - len(curve["keys"])
            track[field] = {"value": scalar["value"], "curve": curve if curve["keys"] else None}
        self.command({"op": "set_tracks", "id": key, "tracks": tracks})
        self.statusBar().showMessage(f"Cleared {removed} keys from {track['name']} in frames {lo}–{hi}")
        return True

    def _tracker_export_samples(self, node, tracks):
        time = self.dispatcher.document["time"]
        first, last = int(time["first"]), int(time["last"])
        if last-first > 5000:
            raise ValueError("Export is limited to 5,001 sampled frames")
        params = node["params"]
        stabilise = node["type"] == "Stabilize" or params.get("mode") == "stabilise"
        return [(frame, tracker_model.solve({"tracks": tracks}, frame, params,
                                            force_stabilise=stabilise))
                for frame in range(first, last+1)]

    def export_tracker_transform(self, key):
        node = self.dispatcher.document["nodes"].get(key)
        if node is None or node["type"] not in ("Tracker", "Stabilize"):
            return False
        source = node["inputs"].get("image")
        tracks = copy.deepcopy(self.dispatcher.document.get("node_data", {}).get(key, {}).get("tracks", []))
        if source is None or not tracks:
            self._show_command_error(ValueError("Connect the image input and analyze at least one track first"))
            return False
        try:
            samples = self._tracker_export_samples(node, tracks)
        except ValueError as error:
            self._show_command_error(error); return False
        ident = uuid.uuid4().hex[:12]
        params = {name: SPECS["Transform"]["params"][name] for name in
                  ("translate_x", "translate_y", "rotate", "scale", "center_x", "center_y", "filter", "mix")}
        curves = {field: {"interpolation": "linear", "keys": [
                    {"frame": frame, "value": float(solved[field])}
                    for frame, solved in samples]}
                  for field in tracker_model.SOLVED_FIELDS}
        for field, curve in curves.items():
            params[field] = curve["keys"][0]["value"]
        commands = [{"op": "create", "id": ident, "type": "Transform",
                     "name": f"{node['name']} Transform", "params": params,
                     "pos": [node["pos"][0], node["pos"][1]+100]},
                    {"op": "connect", "id": ident, "input": "image", "source": source}]
        commands.extend({"op": "set_curve", "id": ident, "param": field, "curve": curve}
                        for field, curve in curves.items())
        self.command({"op": "batch", "commands": commands})
        self.statusBar().showMessage(f"Exported {node['name']} as animated Transform")
        return True

    def export_tracker_cornerpin(self, key):
        node = self.dispatcher.document["nodes"].get(key)
        if node is None or node["type"] not in ("Tracker", "Stabilize"):
            return False
        source = node["inputs"].get("image")
        tracks = copy.deepcopy(self.dispatcher.document.get("node_data", {}).get(key, {}).get("tracks", []))
        usable = [track for track in tracks if shape_model.resolve_scalar(track["enabled"],
                     int(node["params"].get("reference_frame", 1)), "enabled") >= 0.5]
        if source is None or len(usable) < 4:
            self._show_command_error(ValueError("CornerPin export needs four enabled tracks and an image input"))
            return False
        time = self.dispatcher.document["time"]
        first, last = int(time["first"]), int(time["last"])
        if last-first > 5000:
            self._show_command_error(ValueError("Export is limited to 5,001 sampled frames")); return False
        ref = int(node["params"].get("reference_frame", 1))
        stabilise = node["type"] == "Stabilize" or node["params"].get("mode") == "stabilise"
        ident = uuid.uuid4().hex[:12]
        params = copy.deepcopy(SPECS["CornerPin"]["params"])
        keyed = {name: [] for i in range(1, 5) for axis in ("x", "y")
                 for name in (f"from{i}_{axis}", f"to{i}_{axis}")}
        for frame in range(first, last+1):
            current = shape_model.resolve_tracks({"tracks": usable}, frame)
            reference = shape_model.resolve_tracks({"tracks": usable}, ref)
            for index, (r, c) in enumerate(zip(reference[:4], current[:4]), start=1):
                a, b = (c, r) if stabilise else (r, c)
                for axis in ("x", "y"):
                    keyed[f"from{index}_{axis}"].append({"frame": frame, "value": float(a[axis])})
                    keyed[f"to{index}_{axis}"].append({"frame": frame, "value": float(b[axis])})
        for name, keys in keyed.items():
            params[name] = float(keys[0]["value"])
        commands = [{"op": "create", "id": ident, "type": "CornerPin",
                     "name": f"{node['name']} CornerPin", "params": params,
                     "pos": [node["pos"][0], node["pos"][1]+100]},
                    {"op": "connect", "id": ident, "input": "image", "source": source}]
        commands.extend({"op": "set_curve", "id": ident, "param": name,
                         "curve": {"interpolation": "linear", "keys": keys}}
                        for name, keys in keyed.items())
        self.command({"op": "batch", "commands": commands})
        self.statusBar().showMessage(f"Exported four tracks from {node['name']} as animated CornerPin")
        return True

    def cancel_tracker_analysis(self):
        if self._tracker_cancel is not None:
            self._tracker_cancel.set()
            self.statusBar().showMessage("Tracker analysis: cancelling…")

    def analyse_tracker(self, key, direction="forward", first_frame=None, last_frame=None):
        if self._tracker_future is not None:
            return False
        context = self._tracker_context(key)
        index = self._tracker_index
        if context is None or index is None or self._tracker_seed is None or self._tracker_key != key:
            self._show_command_error(ValueError("Pick a track point before analyzing"))
            return False
        source = context[1]["inputs"].get("image")
        if source is None:
            self._show_command_error(ValueError("Tracker analysis requires a connected image input"))
            return False
        snapshot = copy.deepcopy(self.dispatcher.document)
        frame = int(context[1]["params"].get("reference_frame", snapshot["time"]["current"]))
        last = int(snapshot["time"]["last"] if last_frame is None else last_frame)
        first = int(snapshot["time"]["first"] if first_frame is None else first_frame)
        if first > frame or last < frame:
            self._show_command_error(ValueError("Tracking range must include the reference frame"))
            return False
        point = (float(self._tracker_seed["x"]), float(self._tracker_seed["y"]))
        cancel = threading.Event()
        job = {"key": key, "index": index, "seed": copy.deepcopy(self._tracker_seed),
               "base_tracks": copy.deepcopy(context[2].get("tracks", [])),
               "reference_frame": frame, "progress": {"frame": frame, "completed": 0,
                                                          "total": abs((last if direction == "forward" else first) - frame)},
               "errors": {frame: 0.0}}
        self._tracker_job = job
        self._tracker_cancel = cancel
        self._tracker_future = self.executor.submit(
            lambda: tracker_model.analyse(
                lambda f: self.evaluator.evaluate_raster(snapshot, target=source, frame=f),
                frame, point, first_frame=first, last_frame=last, cancel=cancel,
                direction=direction,
                pattern_radius=int(context[1]["params"].get("pattern_radius", 8)),
                search_radius=int(context[1]["params"].get("search_radius", 24)),
                channels=context[1]["params"].get("tracking_channels", "luminance"),
                adaptive_update=bool(context[1]["params"].get("adaptive_update", 0)),
                progress=lambda f, n, total, score: (
                    job["progress"].update(frame=f, completed=n, total=total, score=score),
                    job["errors"].update({f: max(0.0, min(1.0, 1.0-score))}))))
        total = last - frame if direction == "forward" else frame - first
        self.statusBar().showMessage(f"Tracker {direction}: 0/{max(0, total)} frames")
        self.inspect(key)
        QTimer.singleShot(50, self._poll_tracker_analysis)
        return True

    def _poll_tracker_analysis(self):
        future = self._tracker_future
        if future is None:
            return
        if not future.done():
            job = self._tracker_job or {}
            progress = job.get("progress", {})
            self.statusBar().showMessage(
                f"Tracker analysis: {progress.get('completed', 0)}/{progress.get('total', 0)} "
                f"frames · frame {progress.get('frame', '?')}")
            QTimer.singleShot(50, self._poll_tracker_analysis)
            return
        self._tracker_future = None
        cancel = self._tracker_cancel
        self._tracker_cancel = None
        job = self._tracker_job
        self._tracker_job = None
        try:
            if job is None:
                raise tracker_model.AnalysisError("Tracker analysis lost its pending job state")
            results = future.result()
            if cancel is not None and cancel.is_set():
                raise CancelledError()
            key = job["key"]
            node = self.dispatcher.document["nodes"].get(key)
            if node is None or node["type"] not in ("Tracker", "Stabilize"):
                raise tracker_model.AnalysisError("Tracking node was deleted or replaced during analysis; results discarded")
            current_tracks = self.dispatcher.document.get("node_data", {}).get(key, {}).get("tracks", [])
            if current_tracks != job["base_tracks"]:
                raise tracker_model.AnalysisError("Tracker data changed during analysis; results discarded")
            payload = {"tracks": copy.deepcopy(job["base_tracks"])}
            index = job["index"]
            payload["tracks"].append(copy.deepcopy(job["seed"]))
            track = payload["tracks"][index]
            for field in ("x", "y"):
                keys = [{"frame": int(f), "value": float(position[0 if field == "x" else 1])}
                        for f, position in sorted(results.items())]
                track[field] = {"value": float(track[field]) if not isinstance(track[field], dict) else float(track[field]["value"]),
                                "curve": {"interpolation": "constant", "keys": keys}}
            track["error"] = {"value": float(track.get("error", 0.0)) if not isinstance(track.get("error", 0.0), dict)
                               else float(track["error"]["value"]),
                               "curve": {"interpolation": "linear", "keys": [
                                   {"frame": int(f), "value": float(error)}
                                   for f, error in sorted(job.get("errors", {f: 0.0 for f in results}).items())]}}
            self.command({"op": "set_tracks", "id": key, "tracks": payload["tracks"]})
            self._tracker_seed = None
            self._tracker_index = None
            self._tracker_key = None
            self.statusBar().showMessage(f"Tracker analysis complete: {len(results)} frames")
        except CancelledError:
            self.statusBar().showMessage("Tracker analysis cancelled; document unchanged")
        except (ValueError, KeyError, OSError) as error:
            self._show_command_error(error)
            self.statusBar().showMessage(f"Tracker analysis failed: {error}", 10000)
        finally:
            self.inspect(self.graph.selected_id())

    # -- Animation on numeric knobs -------------------------------------------------------
    #
    # The curve engine (nodebased/animation.py) and its three atomic undoable ops -- set_key,
    # delete_key, clear_curve -- already existed and were exercised only through the agent
    # bridge. Everything below is the missing half: reaching them from the properties panel.
    # No new document shape, no second code path; the button issues the same command an agent
    # would, through the same dispatcher, so it undoes and saves identically.

    def node_curve(self, key, param):
        """The curve for ``(node, param)``, or None when that parameter is not animated."""
        return (self.graph_document().get("animation") or {}).get("curves", {}).get(key, {}).get(param)

    def commit_param(self, key, param, value):
        """Route a knob edit to the base parameter, or to a key when the knob is animated.

        Editing an animated knob and having it silently change the base -- immediately overridden
        by the curve, so the viewer does not move -- is the single most confusing thing a
        keyframing UI can do. When a curve exists, typing a value means "make it that value here",
        which is a key at the current frame. Nuke behaves the same way.
        """
        if self.node_curve(key, param) is None:
            self.defer_command({"op": "set", "id": key, "param": param, "value": value})
            return
        frame = self.dispatcher.document["time"]["current"]
        self.defer_command({"op": "set_key", "id": key, "param": param,
                            "frame": int(frame), "value": float(value)})

    def animatable_row(self, key, param, control, expression=None):
        """Pack a numeric control next to its keyframe button."""
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(control, 1)
        button = QPushButton()
        button.setFixedWidth(26)
        button.setFlat(True)
        button.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        curve = self.node_curve(key, param)
        frame = self.dispatcher.document["time"]["current"]
        keyed_here = curve is not None and any(k["frame"] == frame for k in curve["keys"])
        if expression is not None:
            button.setText("ƒ")
            button.setEnabled(False)
            button.setStyleSheet("color: #c58cff; border: none; font-size: 14px")
            button.setToolTip("Expression-driven. Clear the expression before keying this knob.")
        elif keyed_here:
            button.setText("◆")
            button.setStyleSheet(f"color: {KEY_COLOR.name()}; border: none; font-size: 14px")
            button.setToolTip(f"Key set at frame {frame}. Click to remove it.\nRight-click for "
                              f"curve options.")
        elif curve is not None:
            button.setText("◇")
            button.setStyleSheet(f"color: {KEY_COLOR.name()}; border: none; font-size: 14px")
            button.setToolTip(f"Animated ({len(curve['keys'])} keys, {curve['interpolation']}). "
                              f"Click to key the current value at frame {frame}.\nRight-click for "
                              f"curve options.")
        else:
            button.setText("○")
            button.setStyleSheet("color: #6d6d78; border: none; font-size: 14px")
            button.setToolTip(f"Not animated. Click to set the first key at frame {frame}.")
        button.clicked.connect(
            lambda checked=False, k=key, p=param, w=control, on=keyed_here:
                self.toggle_key(k, p, w, on))
        button.customContextMenuRequested.connect(
            lambda point, k=key, p=param, w=control, b=button: self.curve_menu(k, p, w, b, point))
        layout.addWidget(button)
        return row

    def expression_row(self, key, param, expression=None):
        """Return the formula editor for one numeric knob.

        The editor intentionally commits only from an explicit Set button or Return. Typing into
        a field must not mutate the document on every keystroke: an incomplete formula is normal
        while editing, and the Dispatcher is the atomic validation/undo boundary. Clear uses the
        same boundary, so setting and removing a link undo exactly like an agent command.
        """
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        editor = QLineEdit(expression or "")
        editor.setObjectName("expression-editor")
        editor.setAccessibleName(f"{key}.{param} expression")
        editor.setPlaceholderText("e.g. frame * 2 + 1 or knob(\"node\", \"param\")")
        editor.setToolTip("Safe arithmetic formula. Use frame or knob(\"node id\", \"param\")")
        apply = QPushButton("Set")
        apply.setObjectName("set-expression")
        clear = QPushButton("Clear")
        clear.setObjectName("clear-expression")
        clear.setEnabled(bool(expression))
        layout.addWidget(editor, 1)
        layout.addWidget(apply)
        layout.addWidget(clear)

        def set_expression():
            text = editor.text().strip()
            if not text:
                self._show_command_error(ValueError("Enter an expression or use Clear"))
                return
            self.defer_command({"op": "set_expression", "id": key, "param": param,
                                "expression": text})

        apply.clicked.connect(set_expression)
        editor.returnPressed.connect(set_expression)
        clear.clicked.connect(lambda: self.defer_command(
            {"op": "clear_expression", "id": key, "param": param}))
        return row

    def install_expression_shortcut(self, field, key, param, control):
        """Make ``=`` open the expression editor without changing the numeric field."""
        field._expression_target = (key, param, control)
        field.installEventFilter(self)

    def open_expression_editor(self, key, param, control):
        """Insert the expression editor immediately below a numeric knob row."""
        widget = control
        form = None
        while widget is not None:
            layout = widget.layout()
            if isinstance(layout, QFormLayout):
                form = layout
                break
            widget = widget.parentWidget()
        if form is None:
            return None

        row_index = None
        for row in range(form.rowCount()):
            for role in (QFormLayout.ItemRole.LabelRole, QFormLayout.ItemRole.FieldRole,
                         QFormLayout.ItemRole.SpanningRole):
                item = form.itemAt(row, role)
                candidate = item.widget() if item is not None else None
                if candidate is not None and (candidate is control or candidate.isAncestorOf(control)):
                    row_index = row
                    break
            if row_index is not None:
                break
        if row_index is None:
            return None

        expressions = self.graph_document().get("expressions") or {}
        expression = expressions.get(key, {}).get(param)
        editor_row = self.expression_row(key, param, expression)
        form.insertRow(row_index + 1, "Expression", editor_row)
        editor = editor_row.findChild(QLineEdit, "expression-editor")
        editor.setFocus()
        editor.selectAll()
        return editor

    def toggle_key(self, key, param, control, keyed_here):
        frame = int(self.dispatcher.document["time"]["current"])
        if keyed_here:
            self.defer_command({"op": "delete_key", "id": key, "param": param, "frame": frame})
        else:
            self.defer_command({"op": "set_key", "id": key, "param": param, "frame": frame,
                                "value": float(control.value())})

    def curve_menu(self, key, param, control, button, point):
        """Nuke-style knob context menu, opened from the knob itself as well as its key button.

        In Nuke the animation menu belongs to the knob: right-clicking the number is how you key
        it, and the small diamond is a shortcut rather than the only door. The menu is parented to
        the *window*, not to `button` -- a menu owned by a panel widget dies with the panel if
        anything rebuilds the inspector while it is open.
        """
        menu = self.build_curve_menu(key, param, control)
        menu.exec(button.mapToGlobal(point))

    def build_curve_menu(self, key, param, control):
        """Construct the curve_menu contents without showing it (exec is a blocking modal call)."""
        frame = int(self.dispatcher.document["time"]["current"])
        curve = self.node_curve(key, param)
        expression = (self.graph_document().get("expressions") or {}).get(key, {}).get(param)
        menu = QMenu(self)
        menu.addAction("Edit expression…" if expression is not None else "Enter expression…",
                       lambda: self.open_expression_editor(key, param, control))
        if expression is not None:
            menu.addAction("Clear expression", lambda: self.defer_command(
                {"op": "clear_expression", "id": key, "param": param}))
        menu.addSeparator()
        menu.addAction(f"Set key at frame {frame}",
                       lambda: self.defer_command({"op": "set_key", "id": key, "param": param,
                                                   "frame": frame, "value": float(control.value())}))
        delete = menu.addAction(
            f"Delete key at frame {frame}",
            lambda: self.defer_command({"op": "delete_key", "id": key, "param": param,
                                        "frame": frame}))
        delete.setEnabled(curve is not None and any(k["frame"] == frame for k in curve["keys"]))
        clear = menu.addAction("Remove animation",
                               lambda: self.defer_command({"op": "clear_curve", "id": key,
                                                           "param": param}))
        clear.setEnabled(curve is not None)
        if curve is not None:
            menu.addSeparator()
            interpolation = menu.addMenu("Interpolation")
            for name in CURVE_INTERPOLATIONS:
                action = interpolation.addAction(
                    name,
                    # set_key re-keys the frame it is given and carries the interpolation for the
                    # whole curve, so re-setting any existing key is the atomic way to change it.
                    lambda n=name, c=curve: self.defer_command(
                        {"op": "set_key", "id": key, "param": param,
                         "frame": c["keys"][0]["frame"], "value": float(c["keys"][0]["value"]),
                         "interpolation": n}))
                action.setCheckable(True)
                action.setChecked(curve["interpolation"] == name)
        menu.addSeparator()
        node_type = self.graph_nodes()[key]["type"]
        default = SPECS[node_type]["params"][param]
        reset = menu.addAction(f"Set to default ({default})",
                              lambda: self.defer_command({"op": "set", "id": key, "param": param,
                                                          "value": default}))
        reset.setEnabled(param not in (self.graph_document().get("expressions") or {}).get(key, {}))
        return menu

    def node_search(self):
        graph_pos = self.graph.last_click_scene_pos
        global_pos = self.graph.viewport().mapToGlobal(self.graph.mapFromScene(graph_pos))
        kind = NodeSearch.choose(self, SPECS, global_pos)
        if kind:
            self.add_node(kind, position=graph_pos)

    def graph_center(self):
        """The scene point at the middle of the visible graph, where the NODES dock lands a node
        clicked (rather than dragged to a chosen spot) -- unless `add_node`'s own selected-node
        priority wires it into a branch instead, exactly as a Tab search click does."""
        return self.graph.mapToScene(self.graph.viewport().rect().center())

    def run_fluid_shelf_tool(self, tool):
        """Build the selected-geometry fluid setup through one ordinary undoable batch."""
        nodes = self.graph_nodes()
        selected = [item.key for item in self.graph.scene().selectedItems()
                    if isinstance(item, NodeItem) and item.key in nodes]
        center = self.graph_center()
        document = self.graph_document()
        ops = fluidshelf.build_ops(
            tool, nodes, selected, (document.get("animation") or {}),
            (center.x(), center.y()))
        if not ops:
            self.statusBar().showMessage("Select one or more 3D geometry nodes first", 3500)
            return False
        return self.command({"op": "batch", "commands": ops})

    def eventFilter(self, watched, event):
        if (watched.objectName() == "node-name-header"
                and event.type() == QEvent.Type.MouseButtonDblClick):
            content = getattr(watched, "_panel_content", None)
            collapse = getattr(watched, "_panel_collapse", None)
            if content is not None:
                expanded = not content.isVisible()
                content.setVisible(expanded)
                if collapse is not None:
                    collapse.setText("▾" if expanded else "▸")
            return True
        if event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Equal:
            target = getattr(watched, "_expression_target", None)
            if target is not None:
                self.open_expression_editor(*target)
                return True
        if event.type() in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease) and event.key() == Qt.Key.Key_Control:
            graph_point = self.graph.viewport().mapFromGlobal(QCursor.pos())
            if (self.graph.viewport().rect().contains(graph_point)
                    or watched in (self.graph, self.graph.viewport())
                    or self.graph.hasFocus()):
                self.graph.ctrl_handles_visible = event.type() == QEvent.Type.KeyPress
                self.graph.viewport().update()
        if (event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Tab
                and not QApplication.activePopupWidget()):
            graph_point = self.graph.viewport().mapFromGlobal(QCursor.pos())
            if self.graph.viewport().rect().contains(graph_point):
                self.graph.last_click_scene_pos = self.graph.mapToScene(graph_point)
                self.node_search()
                return True
        return super().eventFilter(watched, event)

    def node_position(self, desired, below=False, kind=None):
        """Find a nearby vacant location; never drop a new node on an existing one.

        `below` searches straight down the column first, which is where a node added to a
        selection belongs: the stream reads top to bottom, as in Nuke. The radial search is the
        fallback only once the column is exhausted.
        """
        desired = QPointF(round(desired.x()), round(desired.y()))
        candidates = [QPointF(0, 0)]
        if below:
            candidates.extend(QPointF(0, step) for step in range(40, 1201, 40))
        for radius in range(80, 801, 80):
            candidates.extend(QPointF(x, y) for x, y in
                              ((radius, 0), (-radius, 0), (0, radius), (0, -radius),
                               (radius, radius), (-radius, radius), (radius, -radius), (-radius, -radius)))
        occupied = [item.sceneBoundingRect().adjusted(-12, -12, 12, 12)
                    for item in self.graph.items_by_id.values() if not item.is_backdrop]
        for offset in candidates:
            pos = desired + offset
            rect = QRectF(pos.x(), pos.y(), *node_size({"type": kind or ""}, self.show_thumbnails))
            if not any(rect.intersects(other) for other in occupied):
                return pos
        return desired + QPointF(0, 880)

    def add_node(self, kind=None, params=None, position=None):
        if not kind:
            kind, ok = QInputDialog.getItem(self, "Create node", "Node (type to search)", list(SPECS), 0, True)
            if not ok:
                return
        # A selected node with a real input slot takes priority over click position: the new
        # node lands in its branch, wired from its output, rather than wherever Tab/hotkey
        # last recorded a click. Generators (Read/Constant/Checker) have no input slot, so
        # selecting one falls back to plain click placement instead of a no-op connect.
        # Ports are typed (image, geometry, light, camera, scene): wire the selection into the
        # first slot that accepts what it outputs, and leave it alone when nothing does, so
        # creating a Render3D under a Grade makes a free node instead of a rejected command.
        from .core import INPUT_TYPES, OUTPUT_TYPES
        source, slot = self.graph.selected_id(), None
        if source:
            produced = OUTPUT_TYPES.get(self.graph_nodes()[source]["type"], "image")
            slots = list(SPECS[kind]["inputs"])
            if OUTPUT_TYPES.get(kind, "image") != "image" or kind == "Render3D":
                slots += list(SPECS[kind].get("optional_inputs", []))
            slot = next((name for name in slots if produced in INPUT_TYPES.get(name, ("image",))), None)
        if slot is None:
            source = None
        if source:
            # Directly underneath the selection, centred on it -- a Dot is far narrower than a
            # node, so centre on its bounds rather than aligning left edges.
            selected = self.graph.items_by_id[source].sceneBoundingRect()
            anchor = QPointF(selected.center().x() - node_size({"type": kind}, self.show_thumbnails)[0] / 2,
                             selected.bottom() + 40)
        else:
            anchor = position if position is not None else self.graph.last_click_scene_pos
        pos = self.node_position(anchor, below=bool(source), kind=kind)
        if kind == "Backdrop":
            # Like Nuke: a backdrop made with nodes selected frames them, with room for its title.
            picked = [item.sceneBoundingRect() for item in self.graph.scene().selectedItems()
                      if isinstance(item, NodeItem) and not item.is_backdrop]
            if picked:
                frame = picked[0]
                for rect in picked[1:]:
                    frame = frame.united(rect)
                frame = frame.adjusted(-30, -30 - BACKDROP_TITLE_HEIGHT, 30, 30)
                pos = frame.topLeft()
                params = {**(params or {}), "width": int(frame.width()), "height": int(frame.height())}
                source = slot = None
        key = __import__("uuid").uuid4().hex[:12]
        commands = [{"op": "create", "id": key, "type": kind, "pos": [pos.x(), pos.y()], "params": params or {}}]
        if source:
            commands.append({"op": "connect", "id": key, "input": slot, "source": source})
            # Splice into the branch: anything currently reading from the selected node's
            # output is rewired to read from the new node instead, so it's inserted inline
            # rather than just forking a new dead-end off the selection.
            nodes = self.graph_nodes()
            # Only a node that outputs what the selection outputs can stand in for it downstream.
            downstream = [(dest, name) for dest, node in nodes.items()
                          for name, src in node["inputs"].items() if src == source
                          ] if OUTPUT_TYPES.get(kind, "image") == produced else []
            commands.extend({"op": "connect", "id": dest, "input": name, "source": key} for dest, name in downstream)
        if self.command({"op": "batch", "commands": commands}) is not None:
            self.graph.scene().clearSelection()
            self.graph.items_by_id[key].setSelected(True)
            self.node_toolbar.note_added(kind)
            if kind == "Read" and not params:
                self.browse_read(key)

    def browse_read(self, key):
        chosen = SequenceBrowser.choose(self, self.last_browse_directory)
        if chosen is None:
            return
        self.last_browse_directory = str(Path(chosen["path"]).parent)
        self.command({"op": "set", "id": key, "param": "path", "value": chosen["path"]})
        self.offer_sequence_range(chosen)

    def browse_lut(self, key):
        node = self.graph_nodes().get(key)
        if node is None:
            return
        current = str(node["params"].get("cube_path", ""))
        directory = str(Path(current).expanduser().parent) if current else self.last_browse_directory or ""
        path, _ = QFileDialog.getOpenFileName(self, "3D LUT file", directory,
                                              "3D LUT files (*.cube *.3dl);;All files (*)")
        if path:
            self.last_browse_directory = str(Path(path).parent)
            self.command({"op": "set", "id": key, "param": "cube_path", "value": path})

    def browse_geometry(self, key):
        path, _ = QFileDialog.getOpenFileName(self, "Choose geometry", self.last_browse_directory or "",
                                              "Wavefront OBJ (*.obj)")
        if path:
            self.last_browse_directory = str(Path(path).parent)
            self.command({"op": "set", "id": key, "param": "geo_path", "value": path})

    def offer_sequence_range(self, chosen):
        """Ask before re-ranging the comp to a freshly loaded sequence.

        Nuke sets the project range from the first clip you load; doing it silently on every load
        would quietly discard a range an artist set deliberately, so this asks and only when the
        range actually differs.
        """
        if not chosen.get("sequence") or chosen.get("first") is None:
            return
        time_range = self.dispatcher.document["time"]
        if (time_range["first"], time_range["last"]) == (chosen["first"], chosen["last"]):
            return
        gap = f" with {len(chosen['missing'])} frames missing" if chosen["missing"] else ""
        answer = QMessageBox.question(
            self, "Sequence range",
            f"{Path(chosen['path']).name} covers frames {chosen['first']}–{chosen['last']}{gap}.\n"
            f"Set the project range to match? "
            f"(currently {time_range['first']}–{time_range['last']})",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if answer == QMessageBox.StandardButton.Yes:
            self.set_time(first=chosen["first"], last=chosen["last"], current=chosen["first"])

    def browse_write(self, key):
        path, selected = QFileDialog.getSaveFileName(
            self, "Write output", self.dispatcher.document["nodes"][key]["params"]["path"] or "render.%04d.exr",
            "OpenEXR (*.exr);;PNG (*.png)")
        if path:
            self.command({"op": "set", "id": key, "param": "path", "value": path})

    def read_file(self):
        chosen = SequenceBrowser.choose(self, self.last_browse_directory)
        if chosen is None:
            return
        self.last_browse_directory = str(Path(chosen["path"]).parent)
        self.add_node("Read", {"path": chosen["path"]})
        self.command({"op": "view", "id": self.graph.selected_id()})
        self.offer_sequence_range(chosen)

    def confirm_discard(self):
        if self.dispatcher.document == self.saved_document:
            return True
        result = QMessageBox.question(self, "Unsaved project", "Save changes before continuing?", QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel)
        if result == QMessageBox.StandardButton.Save:
            return self.save_project()
        return result == QMessageBox.StandardButton.Discard

    def open_project(self):
        if not self.confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open project", "", "NodeBased (*.nbcomp)")
        if path and self.command({"op": "load", "path": path}) is not None:
            self.project_path = path
            self.saved_document = copy.deepcopy(self.dispatcher.document)
            self.update_title()
            self.graph.fit()

    def save_project(self, save_as=False):
        path = self.project_path
        if not path or save_as:
            path, _ = QFileDialog.getSaveFileName(self, "Save project", path or "Untitled.nbcomp", "NodeBased (*.nbcomp)")
        if not path:
            return False
        if not Path(path).suffix:
            path += ".nbcomp"
        if self.command({"op": "save", "path": path}, render=False) is None:
            return False
        self.project_path = path
        self.saved_document = copy.deepcopy(self.dispatcher.document)
        self.update_title()
        return True

    @staticmethod
    def _clip_to_roi(region, bounds, rect):
        """The region the tile executor is asked for once the ROI is on: the ROI box in the
        canvas's own (tier) pixels, cut down to whatever the viewport already asked for."""
        x0, y0, x1, y1 = viewframe.roi_pixels(rect, bounds.x, bounds.y, bounds.width, bounds.height)
        if region is not None and region != bounds:
            cx0, cy0 = max(x0, region.x), max(y0, region.y)
            cx1, cy1 = min(x1, region.right), min(y1, region.bottom)
            if cx1 > cx0 and cy1 > cy0:
                x0, y0, x1, y1 = cx0, cy0, cx1, cy1
        return TileRegion(x0, y0, x1 - x0, y1 - y0, full_x=bounds.x, full_y=bounds.y,
                          full_width=bounds.width, full_height=bounds.height)

    def request_preview(self, *_, playhead_only=False):
        """Queue a preview. ``playhead_only`` marks a transport advance rather than a content
        change: the queued work is replaced but an in-flight render is left to finish, because
        a frame that outlives its own tick is still the newest frame we have."""
        self.generation += 1
        self.rendered_identity = self.render_identity()
        snapshot = copy.deepcopy(self.dispatcher.document)
        frame = snapshot["time"]["current"]
        future = self.future_frames(frame) if self.playing else ()
        viewport = None
        # A first preview, or one whose source canvas just changed size, has no scene extent
        # worth clamping against: request the complete data window so the first frame establishes
        # honest bounds for `fit()` to compute from. Without this check, reconnecting the viewer
        # to a differently sized source inherited whatever fraction of the OLD canvas happened to
        # be visible in the viewport, silently clamping the very first request for the new source
        # to a small top-left crop that `fit()` then zoomed into — the image looked stuck in the
        # corner with no way to recenter, because every later request re-derived its viewport from
        # that same wrongly-zoomed view. The canvas-size check is a header-only read for a Read
        # source (sub-millisecond once the OS file cache is warm), so it costs nothing during
        # ordinary playback, where the canvas size does not change frame to frame.
        target = snapshot.get("view")
        current_extent = self.viewer.scene().itemsBoundingRect()
        same_canvas = False
        if target and not current_extent.isEmpty():
            try:
                if self.tile_executor.supports_tiled(snapshot, target):
                    tier = self.proxy.currentData()
                    bounds = self.tile_executor.canvas_region(snapshot, target, frame=frame, tier=tier)
                    claimed = self.viewer.sceneRect()
                    same_canvas = (bounds.width * tier, bounds.height * tier) == \
                                  (claimed.width(), claimed.height())
            except Exception:
                same_canvas = False
        if same_canvas:
            rect = self.viewer.mapToScene(self.viewer.viewport().rect()).boundingRect()
            viewport = (math.floor(rect.left()), math.floor(rect.top()),
                        math.ceil(rect.right()), math.ceil(rect.bottom()))
        cancel_active = not (playhead_only and self.playing)
        if cancel_active:
            # A real seek or content edit, not a plain playback tick (docs/PLAYBACK.md
            # criterion 3): drop pending decode-ahead work instead of caching frames a scrub
            # just made irrelevant. See decodepool.py -- a job already dispatched to OIIO still
            # runs to completion, but its result is discarded rather than stored.
            self.decode_pool.bump_epoch()
        self.preview_queue.replace(self.generation, frame, snapshot, future,
                                   tier=self.proxy.currentData(), viewport=viewport,
                                   playing=self.playing,
                                   cancel_active=cancel_active)
        # The decode-ahead pool only ever gets consulted by the tier != 1 Read path (tileexec.py's
        # `_generator_tile`); a tier-1 request takes the bounded-region-read fast path instead and
        # never looks at it. Prefetching at tier 1 would just spend the decode pool's own worker
        # threads (and the GIL) decoding full frames nobody will ever read back -- measured as a
        # net loss for full-resolution playback (docs/BENCHMARKS-v0.17-playback.md).
        if target and future and self.proxy.currentData() != 1:
            self.tile_executor.prefetch_reads(snapshot, target, future)
        self.timer.start(0 if self.playing else 35)

    def start_preview(self):
        if self.busy:
            return
        queued = self.preview_queue.take()
        if queued is None:
            return
        request, cancel = queued
        self.busy = True
        # The viewer always outranks thumbnails on the single preview worker.
        self.thumbnail_cancel.set()
        exposure, channel = self.exposure.value(), self.channels.currentText()
        view = self.effective_view(request.document)
        look = self.look_args(request.document)
        background = request.document["settings"]["viewer"]["background"]
        if request.display:
            self.statusBar().showMessage(f"Evaluating frame {request.frame}…")
        def report(stage, fraction, info):
            # Runs on the worker, between splat tiles: the cheapest place to notice a cancel.
            if cancel.is_set():
                raise Cancelled()
            self.signals.progress.emit((request, cancel), stage, float(fraction), dict(info))
        def work():
            with self.render_progress_router.handler(report if request.display else None):
                render()
        def render():
            start = time.perf_counter()
            try:
                target = request.document.get("view")
                render_region = None
                tiled = bool(target) and self.tile_executor.supports_tiled(request.document, target)
                if tiled:
                    bounds = self.tile_executor.canvas_region(request.document, target,
                                                              frame=request.frame, tier=request.tier)
                    if request.display and request.viewport is not None:
                        x0, y0, x1, y1 = request.viewport
                        wanted = TileRegion(x0, y0, max(0, x1 - x0), max(0, y1 - y0),
                                            full_x=bounds.x, full_y=bounds.y,
                                            full_width=bounds.width, full_height=bounds.height)
                        ox0, oy0 = max(wanted.x, bounds.x), max(wanted.y, bounds.y)
                        ox1, oy1 = min(wanted.right, bounds.right), min(wanted.bottom, bounds.bottom)
                        render_region = TileRegion(ox0, oy0, max(0, ox1 - ox0), max(0, oy1 - oy0),
                                                   full_x=bounds.x, full_y=bounds.y,
                                                   full_width=bounds.width, full_height=bounds.height)
                    else:
                        render_region = bounds
                roi = viewer_roi(request.document)
                if tiled and request.display and roi["on"]:
                    render_region = self._clip_to_roi(render_region, bounds, roi["rect"])
                region_key = (None if render_region is None else
                              (render_region.x, render_region.y,
                               render_region.width, render_region.height))
                display_key = DisplayCache.key(request.document, target, request.frame,
                                               request.tier)
                # The display cache is consulted before anything is composed. It holds the
                # scene-linear frame, so a hit only needs the display-time conversion: checking it
                # only after composing would still re-assemble every tile just to throw the result
                # away.
                cached = self.display_cache.get(display_key, region_key)
                crop = None
                if cached is None and tiled and render_region != bounds:
                    # Read-ahead caches whole frames; a zoomed-in view asks for its visible crop.
                    # A whole frame contains every crop of itself, so serve the crop from it.
                    cached = self.display_cache.get(
                        display_key, (bounds.x, bounds.y, bounds.width, bounds.height))
                    crop = QRect(render_region.x - bounds.x, render_region.y - bounds.y,
                                 render_region.width, render_region.height)
                if cached is not None:
                    if cancel.is_set():
                        raise Cancelled()
                    cached_or_cropped = cached
                    if crop is not None:
                        cached_or_cropped = cached[crop.top():crop.top() + crop.height(),
                                                   crop.left():crop.left() + crop.width()]
                    height, width = cached_or_cropped.shape[:2]
                    image = None
                    if request.display:
                        image = to_qimage(cached_or_cropped, exposure, channel,
                                          background=background, view=view, look=look)
                    image, compare_note = self._compare_outputs(
                        request, cancel, cached_or_cropped, image, render_region,
                        exposure, channel, background, view, look)
                    elapsed = (time.perf_counter() - start) * 1000
                    proxy = "" if request.tier == 1 else f"  ·  proxy 1/{request.tier}"
                    self.signals.finished.emit(
                        (request, cancel), cached_or_cropped, image,
                        f"{width * request.tier} × {height * request.tier}{proxy}  ·  "
                        f"{elapsed:.0f} ms  ·  display cache hit  ·  display {gpudisplay.status()}"
                        f"{compare_note}",
                        render_region)
                    return
                if tiled and request.display and request.tier == 1 and not request.playing:
                    # Nothing at full resolution yet, but playback with "Proxy while playing" on
                    # has usually already cached this frame at a proxy tier. Put that up at once so
                    # scrubbing over played frames is immediate, then refine to full resolution
                    # below. The stand-in is display-only: it never becomes `self.frame`.
                    self._offer_cached_proxy(request, cancel, target, view, exposure, channel,
                                             background, look)
                if tiled:
                    tile_result = self.tile_executor.compose_region(request.document, target, render_region,
                                                                     frame=request.frame,
                                                                     tier=request.tier, cancel=cancel)
                    frame = tile_result.pixels
                    tile_detail = (f"  ·  tiles {tile_result.tile_hits} hit/{tile_result.tile_misses} miss"
                                   if tile_result.tiled else "  ·  tile fallback")
                else:
                    frame = self.evaluator.evaluate(request.document, cancel=cancel,
                                                    frame=request.frame, tier=request.tier)
                    tile_detail = "  ·  full-frame fallback"
                if cancel.is_set():
                    raise Cancelled()
                self.display_cache.put(display_key, frame, region_key,
                                       is_data=_is_data_target(request.document, target))
                image = (to_qimage(frame, exposure, channel, background=background, view=view, look=look)
                         if request.display else None)
                image, compare_note = self._compare_outputs(
                    request, cancel, frame, image, render_region, exposure, channel, background, view, look)
                elapsed = (time.perf_counter() - start) * 1000
                proxy = "" if request.tier == 1 else f"  ·  proxy 1/{request.tier}"
                self.signals.finished.emit((request, cancel), frame, image, f"{frame.shape[1] * request.tier} × {frame.shape[0] * request.tier}{proxy}  ·  {elapsed:.0f} ms{tile_detail}  ·  cache {self.evaluator.bytes / 1048576:.1f} / {self.evaluator.budget / 1048576:.0f} MiB  ·  display {gpudisplay.status()}{compare_note}", render_region)
            except Cancelled:
                self.signals.finished.emit((request, cancel), None, None, "Cancelled", None)
            except (ValueError, OSError) as error:
                # Written for the user: a missing input, an unreadable file, a refused render.
                self.signals.finished.emit((request, cancel), None, None, str(error), None)
            except Exception as error:
                # A bug in NodeBased. str(KeyError('grade')) is just "'grade'", which put a bare
                # node name in the viewer with nothing to say it was a failure.
                self.signals.finished.emit(
                    (request, cancel), None, None,
                    f"Internal error while evaluating ({type(error).__name__}: {error}). "
                    "This is a NodeBased bug, not a problem with the graph.", None)
        self.executor.submit(work)

    def _render_b(self, request, cancel, b_target, a_frame, a_region):
        """Evaluate the B buffer at the request's own frame and tier, lined up pixel for pixel with
        the A frame. Runs on the worker; the display cache serves it like any viewed node."""
        document, tier, frame = request.document, request.tier, request.frame
        a_origin = (a_region.x, a_region.y) if a_region is not None else (0, 0)
        if self.tile_executor.supports_tiled(document, b_target):
            bounds = self.tile_executor.canvas_region(document, b_target, frame=frame, tier=tier)
            region = bounds
            if a_region is not None:
                x0, y0 = max(a_region.x, bounds.x), max(a_region.y, bounds.y)
                x1, y1 = min(a_region.right, bounds.right), min(a_region.bottom, bounds.bottom)
                region = TileRegion(x0, y0, max(0, x1 - x0), max(0, y1 - y0),
                                    full_x=bounds.x, full_y=bounds.y,
                                    full_width=bounds.width, full_height=bounds.height)
            if region.width <= 0 or region.height <= 0:
                return np.zeros(a_frame.shape[:2] + (4,), dtype=np.float32)
            key = DisplayCache.key(document, b_target, frame, tier)
            region_key = (region.x, region.y, region.width, region.height)
            pixels = self.display_cache.get(key, region_key)
            if pixels is None:
                pixels = self.tile_executor.compose_region(document, b_target, region, frame=frame,
                                                            tier=tier, cancel=cancel).pixels
                self.display_cache.put(key, pixels, region_key,
                                       is_data=_is_data_target(document, b_target))
            origin = (region.x, region.y)
        else:
            pixels = self.evaluator.evaluate(document, b_target, cancel=cancel, frame=frame, tier=tier)
            origin = (0, 0)
        return compare_model.align(a_frame.shape, a_origin, np.asarray(pixels, dtype=np.float32), origin)

    def _compare_outputs(self, request, cancel, frame, image, render_region, exposure, channel,
                         background, view, look=None):
        """Turn a finished A frame into what the viewer shows under the current compare mode.
        Returns (picture to put in the scene, status note). B's frame and picture travel to
        preview_ready through `compare_results`; nothing here touches the Qt widgets."""
        mode, b_target = compare_model.active_b(request.document)
        if b_target is None or frame is None:
            return image, ""
        try:
            frame_b = self._render_b(request, cancel, b_target, frame, render_region)
        except (ValueError, OSError) as error:
            # B failing must not take A down with it: A stays on screen and the status says why.
            return image, f"  ·  B unavailable: {error}"
        extras = {"frame_b": frame_b, "mode": mode, "image_b": None}
        if request.display:
            def show(pixels):
                return to_qimage(pixels, exposure, channel, background=background, view=view, look=look)
            if mode == "B only":
                image = show(frame_b)
            elif mode == "wipe":
                extras["image_b"] = show(frame_b)
            elif mode in compare_model.COMBINED_MODES:
                image = show(compare_model.combine(mode, np.asarray(frame, dtype=np.float32), frame_b))
            self.compare_results[(request.generation, request.frame)] = extras
        return image, f"  ·  compare {mode}"

    def _offer_cached_proxy(self, request, cancel, target, view, exposure, channel, background,
                            look=None):
        """Emit the best cached proxy picture of `request.frame`, if any. Runs on the worker."""
        for tier in PROXY_TIERS:
            if tier == 1 or cancel.is_set():
                continue
            try:
                bounds = self.tile_executor.canvas_region(request.document, target,
                                                          frame=request.frame, tier=tier)
            except Exception:
                return
            key = DisplayCache.key(request.document, target, request.frame, tier)
            cached = self.display_cache.get(key, (bounds.x, bounds.y, bounds.width, bounds.height))
            if cached is None:
                continue
            height, width = cached.shape[:2]
            image = to_qimage(cached, exposure, channel, background=background, view=view, look=look)
            self.signals.interim.emit(
                (request, cancel), image,
                f"{width * tier} × {height * tier}  ·  proxy 1/{tier} from cache  ·  "
                f"rendering full resolution…", tier, bounds)
            return

    def preview_progress(self, payload, stage, fraction, info):
        request, cancel = payload
        # Same rule as an interim picture: only the request the user is waiting on may speak.
        if (cancel.is_set() or request.generation != self.generation
                or self.frame_generation == request.generation):
            return
        self._show_render_progress(stage, fraction, info)

    def _show_render_progress(self, stage, fraction, info):
        text = progress_text(stage, fraction, info)
        if text is None:
            self.render_progress.hide()
            return
        # Indeterminate while splats are being prepared: there is no fraction yet.
        self.render_progress.setRange(0, 0 if stage == "prepare" else 1000)
        self.render_progress.setValue(int(1000 * fraction))
        self.render_progress.show()
        self.statusBar().showMessage(text)
        self.viewer_info.setText(text)

    def preview_interim(self, payload, image, status, scale, render_region):
        request, cancel = payload
        current = self.dispatcher.document["time"]["current"]
        # Only for the scrub it was made for, and never over the finished picture: the final
        # result for this generation always wins, whichever order the two arrive in.
        if (cancel.is_set() or self.playing or request.generation != self.generation
                or request.frame != current or self.frame_generation == request.generation):
            return
        self.statusBar().showMessage(status)
        self.viewer_info.setText(status)
        self._show_image(image, scale, render_region)

    def _show_image(self, image, scale, render_region, image_b=None):
        previous = self.viewer.sceneRect().size()
        self.viewer.set_compare_image(image_b, scale, render_region)
        # The pixmap is displayed in full-resolution scene coordinates, but the live frame stays
        # proxy-sized. Keep both coordinate systems beside the picture for cheap mouse lookups.
        self.viewer.last_scale = scale
        self.viewer.last_render_region = render_region
        self.viewer.last_frame_size = (image.width(), image.height())
        self.viewer.scene().clear()
        if scale != 1:
            # Show the proxy at the comp's real size so framing, pans and zooms do not change
            # when an artist drops the tier. Fast transform on purpose: this is a display upscale
            # of an approximation, and a smooth filter would only make it look more finished
            # than it is.
            image = image.scaled(image.width() * scale, image.height() * scale,
                                 Qt.AspectRatioMode.IgnoreAspectRatio,
                                 Qt.TransformationMode.FastTransformation)
        pixmap = self.viewer.scene().addPixmap(QPixmap.fromImage(image))
        pixmap.setTransformationMode(Qt.TransformationMode.FastTransformation)
        full_canvas = (render_region is None or (render_region.width == render_region.full_width
                                                 and render_region.height == render_region.full_height))
        if full_canvas:
            # The last complete picture, for the outside of a region of interest.
            self.viewer.backdrop = (pixmap.pixmap(), pixmap.pos())
        elif (viewer_roi(self.dispatcher.document)["on"] and self.viewer.backdrop is not None
              and self.viewer.backdrop[0].size() == QSize(round(render_region.full_width * scale),
                                                          round(render_region.full_height * scale))):
            backdrop = self.viewer.scene().addPixmap(self.viewer.backdrop[0])
            backdrop.setPos(render_region.full_x * scale, render_region.full_y * scale)
            backdrop.setOpacity(0.35)
            backdrop.setZValue(-1)
        if render_region is not None:
            # Tile result coordinates are data-window coordinates. Keep the pixmap at that
            # location instead of rebasing the crop to (0,0), otherwise a pan would make the
            # image visibly jump and EXR overscan would be lost at display.
            pixmap.setPos(render_region.x * scale, render_region.y * scale)
            scene_rect = QRectF(render_region.full_x * scale, render_region.full_y * scale,
                                render_region.full_width * scale, render_region.full_height * scale)
        else:
            scene_rect = QRectF(0, 0, image.width(), image.height())
        self.viewer.setSceneRect(scene_rect)
        self.viewer.draw_format_overlay(scene_rect)
        if self.viewer._initial_fit_pending:
            self.viewer._fit_initial_image()
        elif previous.width() != scene_rect.width() or previous.height() != scene_rect.height():
            self.viewer.fit()
        self.viewer._place_pixel_readout()

    def preview_ready(self, payload, frame, image, status, render_region=None):
        request, cancel = payload
        extras = self.compare_results.pop((request.generation, request.frame), None)
        self.preview_queue.finish(cancel)
        self.busy = False
        if request.display:
            self.render_progress.hide()
        if frame is None and not cancel.is_set():
            # Logged regardless of whether this result ends up on screen: a read-ahead request
            # can fail on a frame that never becomes "current" and so never reaches viewer_info,
            # but "frame 47 of this sequence is corrupt" is real information either way.
            self.render_errors.append({
                "frame": request.frame, "generation": request.generation,
                "target": request.document.get("view"), "message": status,
                "timestamp": time.time(),
            })
        if request.display and request.playing and not cancel.is_set():
            # The pacing signal playback_tick bounds the playhead against: a completed attempt
            # (success or evaluation error) is one frame's worth of render capacity spent, so the
            # transport may advance by one more. A cancelled request never finished its own
            # target and must not count, or the playhead could outrun render capacity that was
            # never actually delivered.
            self.playback_frames_rendered += 1
        # The preview timer is single-shot and a tick that arrives while busy drops its own
        # wakeup, so without this the next queued frame waits for the following tick even though
        # the worker is free. Kicking it here is what lets playback run at the renderer's
        # sustained rate instead of one frame per tick-that-happened-to-find-us-idle.
        if self.playing and len(self.preview_queue):
            self.timer.start(0)
        current = self.dispatcher.document["time"]["current"]
        # Outside playback the rule stays strict: a result is displayable only if it is still the
        # playhead and still the newest request (docs/PLAYBACK.md criterion 4). During playback
        # that rule is unsatisfiable for any frame costing more than one frame interval, because
        # the transport has already ticked past it by the time it finishes — which froze the
        # viewer entirely instead of dropping frames. While playing, accept any display result
        # newer than what is on screen and show it; the transport keeps following the wall clock,
        # so this drops timeline positions rather than stalling.
        fresh = request.generation == self.generation and request.frame == current
        catching_up = (self.playing and request.playing
                       and request.generation > self.frame_generation)
        if request.display and (fresh or catching_up):
            if self.playing:
                status += (f"  ·  ahead {len(self.preview_queue)}/{self.preview_queue.max_prefetch}"
                           f"  ·  dropped {self.playback_dropped_frames}")
            self.frame = frame
            self.frame_b = extras["frame_b"] if extras is not None and frame is not None else None
            self.frame_generation = request.generation
            self.statusBar().showMessage(status)
            self.viewer_info.setText(status if frame is not None else "Evaluation error")
            if frame is not None:
                self._show_image(image, request.tier, render_region,
                                 extras["image_b"] if extras is not None else None)
            else:
                self.viewer.scene().clear()
                self.viewer._hide_pixel_readout()
                text = self.viewer.scene().addText(status)
                text.setDefaultTextColor(QColor("#e3b18d"))
                self.viewer.fit()
                self.viewer.draw_format_overlay(None)
        # Every completed render -- display or read-ahead -- may have added a display-cache entry,
        # which is exactly when the orange band grows. Refreshing here is what makes the strip
        # fill in live during playback instead of only after the next document edit.
        self.refresh_timeline_marks()
        if len(self.preview_queue):
            self.start_preview()
        elif not self.playing and self.show_thumbnails:
            self.thumbnail_timer.start()

    def schedule_thumbnails(self):
        """Render postage stamps for nodes whose upstream picture changed, when the viewer is idle."""
        if (not self.show_thumbnails or self.thumbnails_closed or self.playing or self.busy
                or len(self.preview_queue)):
            return
        document = copy.deepcopy(self.graph_document())
        frame = document["time"]["current"]
        view = self.display_view.currentText()
        wanted = []
        for key, node in document["nodes"].items():
            if not wants_thumbnail(node):
                continue
            identity = thumbnail_key(document, key, frame, view)
            cached = self.thumbnails.get(key)
            if cached is None or cached[0] != identity:
                wanted.append((key, identity))
        for key in list(self.thumbnails):
            if key not in document["nodes"]:
                del self.thumbnails[key]
        if not wanted:
            return
        cancel = threading.Event()
        self.thumbnail_cancel = cancel
        tier = max(PROXY_TIERS)

        def work():
            for key, identity in wanted:
                if cancel.is_set():
                    return
                try:
                    pixels = self.evaluator.evaluate(document, key, cancel=cancel,
                                                     frame=frame, tier=tier)
                except Cancelled:
                    return
                except Exception:
                    # An unset Read path or a broken branch has no picture; the band stays empty.
                    continue
                height, width = pixels.shape[:2]
                if not width or not height:
                    continue
                # Decimate before the view transform so the costly part runs on ~13k pixels.
                step = max(1, int(math.ceil(max(width / THUMB_WIDTH, height / THUMB_HEIGHT))))
                image = to_qimage(pixels[::step, ::step], view=view)
                image = image.scaled(THUMB_WIDTH, THUMB_HEIGHT, Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation)
                self.signals.thumbnail.emit(key, identity, image)
        self.executor.submit(work)

    def thumbnail_ready(self, key, identity, image):
        if key not in self.graph_nodes():
            return
        self.thumbnails[key] = (identity, image)
        item = self.graph.items_by_id.get(key)
        if item is not None:
            item.set_thumbnail(image)

    def set_show_thumbnails(self, enabled):
        enabled = bool(enabled)
        if enabled == self.show_thumbnails:
            return
        self.show_thumbnails = enabled
        self.preferences.set_thumbnails(enabled)
        self.graph.rebuild()
        if enabled:
            self.thumbnail_timer.start()
        else:
            self.thumbnail_cancel.set()

    def export_geometry(self, key, single=True):
        from .geoexport import export_obj
        document = self.dispatcher.document
        time_range = document["time"]
        frames = ([time_range["current"]] if single
                  else range(time_range["first"], time_range["last"] + 1))
        try:
            written = export_obj(document, key, frames, document["nodes"][key]["params"]["geo_write_path"])
        except ValueError as error:
            self.statusBar().showMessage(str(error), 10000)
            QMessageBox.warning(self, "Geometry export", str(error))
            return
        self.statusBar().showMessage(f"Exported {len(written)} geometry file(s)", 10000)

    def export_lut(self, key):
        from .lutio import export_graph_cube
        node = self.dispatcher.document["nodes"][key]
        params = node["params"]
        try:
            path = export_graph_cube(self.dispatcher.document, key, params["lut_path"],
                                     params["lut_size"], params["colorspace_in"], params["colorspace_out"])
        except (OSError, ValueError, RuntimeError) as error:
            self.statusBar().showMessage(str(error), 10000)
            QMessageBox.warning(self, "LUT export", str(error))
            return
        self.statusBar().showMessage(f"Generated LUT: {path}", 10000)

    def export_splats(self, key, single=True):
        from .splatexport import export_splats
        document = self.dispatcher.document
        time_range = document["time"]
        frames = ([time_range["current"]] if single
                  else range(time_range["first"], time_range["last"] + 1))
        try:
            written = export_splats(document, key, frames)
        except ValueError as error:
            self.statusBar().showMessage(str(error), 10000)
            QMessageBox.warning(self, "Splat export", str(error))
            return
        self.statusBar().showMessage(f"Exported {len(written)} splat file(s)", 10000)

    def export_vdb(self, key, single=True):
        from .vdbexport import export_vdb
        document = self.dispatcher.document
        time_range = document["time"]
        frames = ([time_range["current"]] if single
                  else range(time_range["first"], time_range["last"] + 1))
        try:
            written = export_vdb(document, key, frames)
        except ValueError as error:
            self.statusBar().showMessage(str(error), 10000)
            QMessageBox.warning(self, "VDB export", str(error))
            return
        self.statusBar().showMessage(f"Exported {len(written)} VDB file(s)", 10000)

    def write_target(self, key):
        """Resolve a Write node's (path, format, bits), or raise with the reason it cannot render."""
        params = self.dispatcher.document["nodes"][key]["params"]
        path = params["path"].strip()
        if not path:
            raise ValueError("Set an output path on the Write node first")
        chosen = params["file_type"]
        suffix = Path(path).suffix.lower().lstrip(".")
        if chosen == "Auto":
            if suffix not in ("exr", "png"):
                raise ValueError(f"Cannot infer a format from {Path(path).name!r}; "
                                 f"choose exr or png, or use that extension")
            chosen = suffix
        elif suffix != chosen:
            # Honour the explicit choice and make the filename agree with it, rather than writing
            # PNG bytes into a file called .exr.
            path = str(Path(path).with_suffix("." + chosen))
        return path, chosen, params["bit_depth"]

    def render_write(self, key, single=True):
        """Render a Write node. Always full resolution through the reference evaluator."""
        try:
            path, file_type, bits = self.write_target(key)
        except ValueError as error:
            QMessageBox.warning(self, "Write", str(error))
            return
        time_range = self.dispatcher.document["time"]
        frames = ([time_range["current"]] if single
                  else list(range(time_range["first"], time_range["last"] + 1)))
        if len(frames) > 1 and not is_sequence(path):
            QMessageBox.warning(self, "Write",
                                f"{Path(path).name!r} is a single file, so a {len(frames)}-frame "
                                f"range would overwrite it every frame. Use a padded pattern "
                                f"such as render.%04d.exr.")
            return
        # A range render is long and must stay interruptible; the document is snapshotted once so
        # an edit landing mid-render cannot change what the remaining frames are rendered from.
        document = copy.deepcopy(self.dispatcher.document)
        bundle_on = bool(document["nodes"][key]["params"].get("bundle", 0))
        if bundle_on and file_type != "exr":
            QMessageBox.warning(self, "Write", "The conditioning bundle needs an EXR output path "
                                               "(it writes a multichannel EXR and a manifest beside it).")
            return
        Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        progress = QProgressDialog(f"Rendering {Path(path).name}…", "Cancel", 0, len(frames), self)
        progress.setWindowTitle("Write")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        written = 0
        try:
            for index, frame_number in enumerate(frames):
                if progress.wasCanceled():
                    break
                progress.setValue(index)
                progress.setLabelText(f"Rendering frame {frame_number} of "
                                      f"{frames[0]}–{frames[-1]}…")
                QApplication.processEvents()
                # Render the Write node's own upstream tree. Omitting the target would evaluate
                # the document's view instead, so a render would silently follow whatever the
                # Viewer was pointed at -- in Nuke a Write is independent of the Viewer, and an
                # export that changes with the current view is the worst kind of wrong output:
                # plausible-looking frames of the wrong tree.
                target = sequence_path(path, frame_number) if is_sequence(path) else path
                if bundle_on:
                    # Multichannel EXR plus the JSON manifest that says what each layer means.
                    write_bundle_frame(self.evaluator, document, key, frame_number, target, bits)
                    written += 1
                    continue
                raster = self.evaluator.evaluate_raster(document, key, frame=frame_number, tier=1)
                pixels = raster.to_display()
                if file_type == "exr":
                    # A multichannel input (Render3D's multichannel output, a multilayer Read)
                    # writes every named layer into the same part; PNG only ever writes the beauty.
                    write_exr(target, pixels, bits=bits, layers=raster_layer_arrays(raster), metadata=raster.meta)
                else:
                    write_png(target, pixels)
                written += 1
        except (ValueError, OSError) as error:
            progress.close()
            QMessageBox.warning(self, "Write failed",
                                f"{error}\n\n{written} of {len(frames)} frames written.")
            return
        finally:
            progress.close()
        detail = f"{file_type} · {bits + ' float' if file_type == 'exr' else '8-bit sRGB'}"
        cancelled = " · cancelled" if written < len(frames) else ""
        self.statusBar().showMessage(
            f"Wrote {written} frame{'' if written == 1 else 's'} to {path} · {detail}{cancelled}",
            12000)

    def export(self):
        if self.frame is None or self.frame_generation != self.generation:
            self.statusBar().showMessage("Wait for a valid current preview before exporting", 8000)
            return
        frame = self.frame  # Keep the chosen image stable across the modal dialog.
        path, selected_filter = QFileDialog.getSaveFileName(self, "Export image", "output.exr", "OpenEXR half RGBA ZIPS (*.exr);;OpenEXR 32-bit float RGBA ZIPS (*.exr);;PNG sRGB 8-bit (*.png)")
        if not path:
            return
        try:
            if not Path(path).suffix:
                path += ".png" if "PNG" in selected_filter else ".exr"
            # Preview may now be a bounded tile crop even at tier 1. Exports always render a
            # complete full-resolution reference frame; a viewport artifact can never escape.
            self.statusBar().showMessage("Rendering full resolution for export…")
            QApplication.processEvents()
            def report(stage, fraction, info):
                # The export renders on the GUI thread, so repaint by hand between tiles.
                # Input stays queued: a click landing mid-render must not start a second export.
                self._show_render_progress(stage, fraction, info)
                QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
            try:
                with self.render_progress_router.handler(report):
                    frame = self.evaluator.evaluate(copy.deepcopy(self.dispatcher.document),
                                                    frame=self.dispatcher.document["time"]["current"],
                                                    tier=1)
            finally:
                self.render_progress.hide()
            if Path(path).suffix.lower() == ".exr":
                # Half is the default; the second EXR filter is the explicit opt-in for data
                # passes that must keep all 32 bits.
                bits = "float" if "32-bit" in selected_filter else "half"
                write_exr(path, frame, bits=bits)
                detail = f"{bits} float, ZIPS"
            else:
                write_png(path, frame)
                detail = "8-bit sRGB"
            self.statusBar().showMessage(f"Exported {path} · {detail} · viewer exposure/channel controls are display-only", 10000)
        except (ValueError, OSError) as error:
            QMessageBox.warning(self, "Export failed", str(error))

    def update_status(self, state, detail, percent):
        labels = {"idle": "Check for updates", "checking": "Checking…",
                  "available": f"Download v{detail}", "downloading": f"Downloading {percent}%",
                  "ready": "Restart to update", "current": "Up to date",
                  "error": "Update failed · Retry", "unsupported": "Updates unavailable"}
        self.update_button.setText(labels.get(state, state))
        self.update_button.setEnabled(state not in ("checking", "downloading"))
        self.update_button.setToolTip(detail or f"NodeBased {__version__}")
        if state in ("error", "unsupported"):
            self.statusBar().showMessage(detail, 15000)

    def update_clicked(self):
        if self.updater.state == "available":
            self.updater.fetch()
        elif self.updater.state == "ready":
            if not self.confirm_discard():
                return
            try:
                self.updater.install()
            except (ValueError, OSError) as error:
                self.updater.changed.emit("error", str(error), 0)
                return
            self.update_exit = True
            self.close()
        else:
            self.updater.check()

    def closeEvent(self, event):
        if not self.update_exit and not self.confirm_discard():
            event.ignore()
            return
        self.updater.cancel.set()
        if self._tracker_cancel is not None:
            self._tracker_cancel.set()
        self.timer.stop()
        self.playback_timer.stop()
        self.thumbnails_closed = True
        self.thumbnail_timer.stop()
        self.thumbnail_cancel.set()
        self.preview_queue.cancel()
        # GL resources are thread-affine to the worker thread that built them; release them
        # there, before that thread stops, or the context can never be made current again.
        try:
            self.executor.submit(gpudisplay.shutdown).result(timeout=5)
        except Exception:
            pass
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.decode_pool.shutdown()
        if self.server:
            self.server.close()
        QApplication.instance().removeEventFilter(self)
        self.save_workspace()
        event.accept()


def _scene3d_selftest():
    """Whether the optional 3D packages load in this build; an adapter is not required."""
    result = {"usd": False, "wgpu": False, "gpu_adapter": False}
    try:
        from pxr import Usd, UsdGeom
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Mesh.Define(stage, "/probe")
        result["usd"] = bool(stage.GetPrimAtPath("/probe"))
    except Exception:
        pass
    try:
        import wgpu  # noqa: F401
        result["wgpu"] = True
        from . import gpu3d
        result["gpu_adapter"] = bool(gpu3d.available())
    except Exception:
        pass
    return result


def main():
    parser = argparse.ArgumentParser(description="NodeBased native 2D compositing workbench")
    parser.add_argument("project", nargs="?")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--network-probe", metavar="OUTPUT_JSON", help=argparse.SUPPRESS)
    parser.add_argument("--smoke-test", metavar="OUTPUT_JSON", help=argparse.SUPPRESS)
    parser.add_argument("--agent", metavar="LOCAL_NAME", help="opt-in user-local agent socket; no network listener")
    args = parser.parse_args()
    if args.network_probe:
        from .updater import open_url
        with open_url("https://api.github.com/repos/neodimo/NodeBased", timeout=20) as response:
            repository = json.loads(response.read())
        Path(args.network_probe).write_text(json.dumps({"ok": repository.get("full_name") == "neodimo/NodeBased"}), encoding="utf-8")
        return 0
    app = QApplication(sys.argv[:1])
    app.setApplicationName("NodeBased")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    try:
        window = Window(load_document(args.project) if args.project else None, args.agent)
    except (ValueError, OSError) as error:
        print(f"NodeBased: {error}", file=sys.stderr)
        return 1
    if args.project:
        window.project_path = str(Path(args.project).resolve())
        window.update_title()
    window.show()
    window.update_title()
    if args.smoke_test:
        deadline = time.monotonic() + 30
        def smoke():
            if window.frame is None and time.monotonic() < deadline:
                QTimer.singleShot(100, smoke)
                return
            from .media import selftest
            media = selftest()
            result = {"version": __version__, "ok": window.frame is not None and all(media.values()),
                      "update_button": window.update_button.text(), "media": media,
                      "scene3d": _scene3d_selftest(),
                      "shape": list(window.frame.shape) if window.frame is not None else None,
                      "display": gpudisplay.status()}
            Path(args.smoke_test).write_text(json.dumps(result), encoding="utf-8")
            window.saved_document = window.dispatcher.document
            window.close()
            app.exit(0 if result["ok"] else 1)
        QTimer.singleShot(100, smoke)
    return app.exec()
