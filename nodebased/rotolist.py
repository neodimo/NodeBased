"""The stroke/shape list of the Roto and RotoPaint panels, laid out as Nuke's: one tree table.

Columns, in Nuke's order: Name, Visible, Lock, Overlay color, Color, Invert, Blending, Motion blur, Lifetime, Source (View is
stereo-only and has no column here). Each cell edits a field the item already stores (or, for Overlay color, the one new field),
in one undo step through the same `set_shapes` / `set_paint_items` command the properties panel uses. Layers are folders in the
tree; the order of the list is the render order, and a layer's Visible and Lock apply to everything inside it.

The tree arithmetic (`build_tree`, `move_nodes`, `flatten` and friends) is plain Python so tests can drive it without a window.
"""
import copy

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout,
                               QHeaderView, QLineEdit, QMenu, QPushButton, QSpinBox, QTreeWidget, QTreeWidgetItem, QWidget)

from . import paintknobs, shapes as shape_model

COLUMNS = ("Name", "Visible", "Lock", "Overlay color", "Color", "Invert", "Blending", "Motion blur", "Lifetime", "Source")
COLUMN_TIPS = ("Name of the item or layer", "Show or hide", "Lock against editing", "Colour of the outline in the Viewer",
               "Rendered colour", "Invert the shape", "Blending mode", "Apply motion blur", "Frames the item exists on",
               "Source: colour, foreground or background")
PAINT_BLENDS = ("over", "add", "multiply", "screen")
DEFAULT_OVERLAY = "#58d7ff"      # the stock outline colour of the Viewer (Viewer.roto_handle_colour's default)
LIFETIME_NAMES = {"all": "All frames", "single": "Single", "start_to": "Start to", "to_end": "To end", "range": "Range",
                  "from_current": "From current"}


# --------------------------------------------------------------------------------------------------------- tree arithmetic
def build_tree(entries, layers):
    """The list as nested nodes `{"id": ("item", index) | ("layer", name), "children": [...]}` in render order.

    A layer sits where its first member sits; a layer with no members goes to the end of the level it belongs to."""
    by_name = {layer["name"]: layer for layer in layers}
    top, made = [], {}

    def ensure(name):
        if name in made:
            return made[name]
        node = made[name] = {"id": ("layer", name), "children": []}
        parent = by_name[name].get("layer")
        (ensure(parent)["children"] if parent in by_name else top).append(node)
        return node

    for index, entry in enumerate(entries):
        name = entry.get("layer")
        (ensure(name)["children"] if name in by_name else top).append({"id": ("item", index), "children": []})
    for layer in layers:
        ensure(layer["name"])
    return top


def flatten(top, entries, layers):
    """Write a tree back as (entries in render order, layers in outline order, old index of each new entry)."""
    by_name = {layer["name"]: layer for layer in layers}
    new_entries, new_layers, order = [], [], []

    def walk(children, parent):
        for node in children:
            kind, ref = node["id"]
            if kind == "item":
                entry = copy.deepcopy(entries[ref])
                if parent is None:
                    entry.pop("layer", None)
                else:
                    entry["layer"] = parent
                new_entries.append(entry); order.append(ref)
            else:
                layer = dict(by_name[ref])
                if parent is None:
                    layer.pop("layer", None)
                else:
                    layer["layer"] = parent
                new_layers.append(layer)
                walk(node["children"], ref)

    walk(top, None)
    return new_entries, new_layers, order


def _locate(children, node_id, parent=None):
    for position, node in enumerate(children):
        if node["id"] == node_id:
            return children, position, node
        found = _locate(node["children"], node_id)
        if found:
            return found
    return None


def _contains(node, node_id):
    return node["id"] == node_id or any(_contains(child, node_id) for child in node["children"])


def move_nodes(top, moving, target, where):
    """Move nodes by id to above/below/into a target node (`where` of "end", or no target, appends to the top level).

    Returns False and leaves the tree alone when a layer would move into itself."""
    found = [(_locate(top, node_id)) for node_id in moving]
    nodes = [f[2] for f in found if f]
    nodes = [n for n in nodes if not any(m is not n and _contains(m, n["id"]) for m in nodes)]
    if not nodes:
        return False
    if target is not None and any(_contains(n, target) for n in nodes):
        return False
    for node in nodes:
        siblings, position, _ = _locate(top, node["id"])
        del siblings[position]
    if target is None or where == "end":
        top.extend(nodes)
        return True
    siblings, position, anchor = _locate(top, target)
    if where == "into" and anchor["id"][0] == "layer":
        anchor["children"].extend(nodes)
    else:
        at = position + (1 if where in ("below", "into") else 0)
        siblings[at:at] = nodes
    return True


def remove_nodes(top, ids):
    for node_id in ids:
        found = _locate(top, node_id)
        if found:
            del found[0][found[1]]


def unique_name(base, taken):
    if base not in taken:
        return base
    number = 2
    while f"{base} {number}" in taken:
        number += 1
    return f"{base} {number}"


def convert_entry(entry, to_roto):
    """A copied curve as an entry of the other node kind (None for a stroke, which has no curve)."""
    entry = copy.deepcopy(entry)
    entry.pop("layer", None)
    if entry.get("kind") == "stroke":
        return None
    if to_roto:
        if entry.pop("kind", None) is not None:
            blend = entry.pop("blend", "over")
            entry["blend_mode"] = {"add": "plus"}.get(blend, blend)
        return entry
    if entry.get("kind") == "shape":
        return entry
    blend = {"plus": "add"}.get(entry.pop("blend_mode", "over"), "over")
    required = {"name", "mode", "opacity", "feather", "points"}
    item = {k: v for k, v in entry.items() if k in required or k in paintknobs.SHAPE_ITEM_FIELDS}
    item.update(kind="shape", blend=blend if blend in PAINT_BLENDS else "over", visible=bool(entry.get("visible", True)))
    return item


def lifetime_text(entry):
    life = entry.get("lifetime")
    if not life:
        return LIFETIME_NAMES["all"]
    kind = life.get("type", life.get("mode", "all"))
    first, last = life.get("start", life.get("first")), life.get("end", life.get("last"))
    name = LIFETIME_NAMES.get(kind, kind)
    if kind == "range":
        return f"{name} {first}-{last}"
    if kind in ("single", "to_end", "from_current"):
        return f"{name} {first}"
    if kind == "start_to":
        return f"{name} {last}"
    return name


# --------------------------------------------------------------------------------------------------------------- the widget
class ShapeListTree(QTreeWidget):
    """The Roto (`is_roto`) or RotoPaint list. `on_select(index)` is the panel's own item-selection routine; `extras(layout,
    index, name_edit)` lets the Roto panel add its key button to the Name cell."""

    def __init__(self, window, key, is_roto, on_select, extras=None):
        super().__init__()
        self.window, self.key, self.is_roto = window, key, is_roto
        self.slot, self.op = ("shapes", "set_shapes") if is_roto else ("items", "set_paint_items")
        self.prefix = "roto-shape" if is_roto else "rotopaint"
        self.on_select, self.extras = on_select, extras
        self.setObjectName("roto-shape-list" if is_roto else "rotopaint-layer-list")
        self.setColumnCount(len(COLUMNS)); self.setHeaderLabels(list(COLUMNS))
        header = self.header(); header.setStretchLastSection(False)
        for column, tip in enumerate(COLUMN_TIPS):
            self.headerItem().setToolTip(column, tip)
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch if column == 0
                                        else QHeaderView.ResizeMode.ResizeToContents)
        header.setMinimumSectionSize(24)
        self.setRootIsDecorated(True); self.setUniformRowHeights(False); self.setIndentation(14)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setDragEnabled(True); self.setAcceptDrops(True); self.setDropIndicatorShown(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_menu)
        self.setMinimumHeight(160)
        self._building = True
        self.nodes = {}
        self.populate()
        self._building = False
        self.currentItemChanged.connect(self._current_changed)

    # -- document access ---------------------------------------------------------------------------------------------------
    def payload(self):
        return self.window.dispatcher.document.get("node_data", {}).get(self.key, {})

    def entries(self):
        return copy.deepcopy(self.payload().get(self.slot, []))

    def layers(self):
        return copy.deepcopy(self.payload().get("layers", []))

    def frame(self):
        return int(self.window.dispatcher.document["time"]["current"])

    def selected_index(self):
        viewer = self.window.viewer
        if self.is_roto:
            return int(getattr(viewer, "roto_selected_shape_index", -1))
        index = getattr(viewer, "paint_selected_item_index", None)
        return -1 if index is None else int(index)

    def set_viewer_selection(self, index):
        viewer = self.window.viewer
        if self.is_roto:
            viewer.roto_selected_shape_index = index
        else:
            viewer.paint_selected_item_index = index if index >= 0 else None

    def commit(self, entries, layers=None, select=None):
        command = {"op": self.op, "id": self.key, self.slot: entries}
        if layers is not None:
            command["layers"] = layers
        if select is not None:
            self.set_viewer_selection(select)
        self.window.command(command)

    # -- building ----------------------------------------------------------------------------------------------------------
    @staticmethod
    def _id_text(node_id):
        return f"{node_id[0]}:{node_id[1]}"

    @staticmethod
    def parse_id(text):
        kind, ref = str(text).split(":", 1)
        return (kind, int(ref)) if kind == "item" else (kind, ref)

    def populate(self):
        entries, layers = self.entries(), self.layers()
        self.layer_names = {layer["name"] for layer in layers}
        by_name = {layer["name"]: layer for layer in layers}
        frame = self.frame()
        self.tree_items = {}

        def add(parent, node):
            kind, ref = node["id"]
            row = QTreeWidgetItem(parent if parent is not None else self.invisibleRootItem())
            row.setData(0, Qt.ItemDataRole.UserRole, self._id_text(node["id"]))
            self.tree_items[node["id"]] = row
            if kind == "item":
                self._item_cells(row, ref, entries[ref], frame)
            else:
                self._layer_cells(row, by_name[ref])
            for child in node["children"]:
                add(row, child)
            row.setExpanded(True)

        for node in build_tree(entries, layers):
            add(None, node)
        selected = self.selected_index()
        layer = getattr(self.window.viewer, "roto_list_layer", None)
        if 0 <= selected < len(entries) and ("item", selected) in self.tree_items and not layer:
            self.setCurrentItem(self.tree_items[("item", selected)])
        elif layer and ("layer", layer) in self.tree_items:
            self.setCurrentItem(self.tree_items[("layer", layer)])

    def _check(self, row, column, name, tip, value, setter, enabled=True):
        box = QCheckBox(); box.setObjectName(name); box.setToolTip(tip); box.setAccessibleName(tip)
        box.setChecked(bool(value)); box.setEnabled(enabled)
        if enabled:
            box.toggled.connect(lambda v: setter(bool(v)))
        wrap = QWidget(); layout = QHBoxLayout(wrap); layout.setContentsMargins(2, 0, 2, 0)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter); layout.addWidget(box)
        self.setItemWidget(row, column, wrap)
        return box

    def _swatch(self, row, column, name, tip, rgb, picker, enabled=True):
        button = QPushButton(); button.setObjectName(name); button.setToolTip(tip); button.setAccessibleName(tip)
        button.setFixedSize(34, 18); button.setEnabled(enabled)
        qcolor = QColor.fromRgbF(*[min(max(float(c), 0.0), 1.0) for c in rgb[:3]])
        button.setStyleSheet(f"background-color: {qcolor.name()}; border: 1px solid #666")
        button.setProperty("rgba", [float(c) for c in rgb])
        if enabled:
            button.clicked.connect(lambda _=False: picker())
        wrap = QWidget(); layout = QHBoxLayout(wrap); layout.setContentsMargins(2, 0, 2, 0)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter); layout.addWidget(button)
        self.setItemWidget(row, column, wrap)
        return button

    def _name_cell(self, row, name, tip, rename, object_name, index=None):
        wrap = QWidget(); layout = QHBoxLayout(wrap); layout.setContentsMargins(2, 0, 2, 0)
        edit = QLineEdit(name); edit.setObjectName(object_name); edit.setFrame(False); edit.setToolTip(tip)
        edit.setMinimumWidth(edit.fontMetrics().horizontalAdvance(name) + 10)
        edit.editingFinished.connect(lambda: edit.text() != name and edit.text() and rename(edit.text()))
        layout.addWidget(edit, 1)
        if self.extras is not None and index is not None:
            self.extras(layout, index, edit)
        self.setItemWidget(row, 0, wrap)
        return edit

    def _layer_cells(self, row, layer):
        name, p = layer["name"], f"{self.prefix}-layer"
        self._name_cell(row, name, f"Layer {name}", lambda text, old=name: self.rename_layer(old, text), f"{p}-name-{name}")
        self._check(row, 1, f"{p}-visible-{name}", "Show or hide everything in this layer", layer.get("visible", True),
                    lambda v, n=name: self.edit_layer(n, "visible", v))
        self._check(row, 2, f"{p}-locked-{name}", "Lock everything in this layer", layer.get("locked", False),
                    lambda v, n=name: self.edit_layer(n, "locked", v))
        overlay = layer.get("overlay_color")
        rgb = overlay if overlay else QColor(DEFAULT_OVERLAY).getRgbF()[:3]
        self._swatch(row, 3, f"{p}-overlay-{name}", "Outline colour of this layer in the Viewer", rgb,
                     lambda n=name, c=rgb: self.pick_color("Overlay color", c, lambda v: self.edit_layer(n, "overlay_color", v)))

    def _item_cells(self, row, index, entry, frame):
        p = f"{self.prefix}-row"
        stroke = entry.get("kind") == "stroke"
        edit = lambda field, value, i=index: self.edit_entry(i, field, value)
        name = entry["name"]
        shape_kind = ("Closed" if entry.get("closed", True) else "Open") + f" · {shape_model.resolve_scalar(entry.get('stroke_width', 1.0), frame, 'stroke_width'):g} px"
        tip = f"{'Stroke' if stroke else 'Shape'} name · {shape_kind}" if not stroke else "Stroke name"
        self._name_cell(row, name, tip, lambda text, i=index: edit("name", text), f"{p}-name-{index}", index)
        self._check(row, 1, f"{p}-visible-{index}", "Show or hide this item", entry.get("visible", True), lambda v: edit("visible", v))
        self._check(row, 2, f"{p}-locked-{index}", "Lock this item's points", entry.get("locked", False), lambda v: edit("locked", v))
        overlay = entry.get("overlay_color")
        rgb = overlay if overlay else QColor(DEFAULT_OVERLAY).getRgbF()[:3]
        self._swatch(row, 3, f"{p}-overlay-{index}", "Outline colour of this item in the Viewer", rgb,
                     lambda i=index, c=rgb: self.pick_color("Overlay color", c, lambda v: self.edit_entry(i, "overlay_color", v)))
        has_color = self.is_roto or stroke
        color = [shape_model.resolve_scalar(c, frame, "opacity") for c in entry.get("color", [1, 1, 1, 1])]
        self._swatch(row, 4, f"{p}-color-{index}", "Rendered colour", color,
                     lambda i=index, c=color: self.pick_color("Colour", c, lambda v: self.set_color(i, v), alpha=True), has_color)
        self._check(row, 5, f"{p}-invert-{index}", "Invert this shape's matte", entry.get("invert", False),
                    lambda v: edit("invert", v), not stroke)
        blend_field, values = ("blend_mode", shape_model.BLEND_MODES) if self.is_roto else ("blend", PAINT_BLENDS)
        combo = QComboBox(); combo.setObjectName(f"{p}-blend-{index}"); combo.setToolTip("Blending mode")
        for value in values:
            combo.addItem(value.title(), value)
        combo.setCurrentIndex(max(0, combo.findData(entry.get(blend_field, "over"))))
        combo.currentIndexChanged.connect(lambda _i, c=combo, f=blend_field: edit(f, c.currentData()))
        self.setItemWidget(row, 6, combo)
        self._check(row, 7, f"{p}-motion_blur-{index}", "Apply motion blur to this item", entry.get("motion_blur", False),
                    lambda v: edit("motion_blur", v))
        life = QPushButton(lifetime_text(entry)); life.setObjectName(f"{p}-lifetime-{index}"); life.setFlat(True)
        life.setToolTip("Frames this item exists on; click to edit its lifetime")
        life.clicked.connect(lambda _=False, i=index: self.open_lifetime_editor(i))
        self.setItemWidget(row, 8, life)
        source = QComboBox(); source.setObjectName(f"{p}-source-{index}"); source.setToolTip("Source: colour, foreground or background")
        for value in paintknobs.SOURCES:
            source.addItem(value.title(), value)
        source.setCurrentIndex(max(0, source.findData(entry.get("source", "color"))))
        source.currentIndexChanged.connect(lambda _i, c=source: edit("source", c.currentData()))
        self.setItemWidget(row, 9, source)

    # -- cell edits (each is one command, so one undo step) ----------------------------------------------------------------
    def edit_entry(self, index, field, value):
        entries = self.entries()
        if not 0 <= index < len(entries):
            return
        entries[index][field] = value
        self.commit(entries, select=index if self.selected_index() < 0 else None)

    def set_color(self, index, rgba):
        entries = self.entries()
        if not 0 <= index < len(entries):
            return
        old = entries[index].get("color", [1.0, 1.0, 1.0, 1.0])
        frame = self.frame()
        values = [min(max(float(v), 0.0), 1.0) for v in rgba]
        if entries[index].get("kind") == "stroke" or not self.is_roto:
            entries[index]["color"] = values
        else:
            entries[index]["color"] = [self._at_frame(channel, new, frame) for channel, new in zip(old, values)]
        self.commit(entries)

    @staticmethod
    def _at_frame(old, replacement, frame):
        if not isinstance(old, dict) or not old.get("curve"):
            return float(replacement)
        curve = copy.deepcopy(old["curve"])
        keys = [k for k in curve["keys"] if k["frame"] != frame] + [{"frame": frame, "value": float(replacement)}]
        keys.sort(key=lambda k: k["frame"])
        return {"value": shape_model.resolve_scalar(old, frame, "opacity"), "curve": {**curve, "keys": keys}}

    def edit_layer(self, name, field, value):
        layers = self.layers()
        for layer in layers:
            if layer["name"] == name:
                layer[field] = value
        self.commit(self.entries(), layers)

    def rename_layer(self, old, new):
        layers = self.layers()
        if new in {layer["name"] for layer in layers if layer["name"] != old}:
            self.window._show_command_error(ValueError(f"Another layer is already called {new}"))
            return
        entries = self.entries()
        for layer in layers:
            if layer["name"] == old:
                layer["name"] = new
            if layer.get("layer") == old:
                layer["layer"] = new
        for entry in entries:
            if entry.get("layer") == old:
                entry["layer"] = new
        self.commit(entries, layers)

    def pick_color(self, title, rgb, accept, alpha=False):
        start = [float(c) for c in rgb] + ([1.0] if len(rgb) == 3 and alpha else [])
        self.window.open_color_picker(start, title, lambda picked: accept(
            [min(max(float(v), 0.0), 1.0) for v in (picked if alpha else list(picked)[:3])]))

    def open_lifetime_editor(self, index):
        if self.selected_index() != index:
            self.set_viewer_selection(index)
            self.on_select(index)
        name = "roto-shape-lifetime-type" if self.is_roto else "rotopaint-layer-lifetime"
        def focus():
            panel = self.window.properties.widget()
            target = panel.findChild(QWidget, name) if panel is not None else None
            if target is not None:
                target.setFocus()
        QTimer.singleShot(0, focus)

    # -- selection -----------------------------------------------------------------------------------------------------------
    def _current_changed(self, current, _previous):
        if self._building or current is None:
            return
        kind, ref = self.parse_id(current.data(0, Qt.ItemDataRole.UserRole))
        if kind == "item":
            self.window.viewer.roto_list_layer = None
            self.set_viewer_selection(ref)
            self.on_select(ref)
        else:
            self.window.viewer.roto_list_layer = ref

    def count(self):
        """How many items (not layers) the list holds."""
        return sum(1 for kind, _ in self.tree_items if kind == "item")

    def currentRow(self):
        return self.selected_index()

    def setCurrentRow(self, row):
        if ("item", row) in self.tree_items:
            self.setCurrentItem(self.tree_items[("item", row)])

    def selected_ids(self):
        return [self.parse_id(item.data(0, Qt.ItemDataRole.UserRole)) for item in self.selectedItems()]

    def select_ids(self, ids):
        self.clearSelection()
        for node_id in ids:
            if node_id in self.tree_items:
                self.tree_items[node_id].setSelected(True)

    # -- structure commands ------------------------------------------------------------------------------------------------
    def _write(self, top, entries, layers, selected_old=None):
        new_entries, new_layers, order = flatten(top, entries, layers)
        select = None
        if selected_old is not None and selected_old in order:
            select = order.index(selected_old)
        elif selected_old is not None:
            select = min(max(self.selected_index(), 0), len(new_entries) - 1) if new_entries else -1
        self.commit(new_entries, new_layers, select)

    def move_ids(self, ids, target, where):
        """Drag-and-drop in plain form: move the nodes to `where` ("above", "below", "into", "end") of `target`."""
        entries, layers = self.entries(), self.layers()
        top = build_tree(entries, layers)
        if not move_nodes(top, ids, target, where):
            return False
        current = self.selected_index()
        self._write(top, entries, layers, current if 0 <= current < len(entries) else None)
        return True

    def dropEvent(self, event):
        target_item = self.itemAt(event.position().toPoint())
        ids = self.selected_ids()
        event.setDropAction(Qt.DropAction.IgnoreAction); event.accept()
        if not ids:
            return
        if target_item is None:
            self.move_ids(ids, None, "end"); return
        target = self.parse_id(target_item.data(0, Qt.ItemDataRole.UserRole))
        rect = self.visualItemRect(target_item)
        offset = event.position().toPoint().y() - rect.top()
        if target[0] == "layer" and rect.height() * 0.25 <= offset <= rect.height() * 0.75:
            where = "into"
        else:
            where = "above" if offset < rect.height() / 2 else "below"
        self.move_ids(ids, target, where)

    def cmd_move(self, delta):
        """Swap the selected item with its neighbour at the same level (inside its layer, or at the top)."""
        entries, layers = self.entries(), self.layers()
        index = self.selected_index()
        if not 0 <= index < len(entries):
            return False
        top = build_tree(entries, layers)
        siblings, position, _ = _locate(top, ("item", index))
        target = position + delta
        if not 0 <= target < len(siblings):
            return False
        siblings[position], siblings[target] = siblings[target], siblings[position]
        self._write(top, entries, layers, index)
        return True

    def cmd_add_layer(self):
        entries, layers = self.entries(), self.layers()
        taken = {layer["name"] for layer in layers}
        number = 1
        while f"Layer{number}" in taken:
            number += 1
        layer = {"name": f"Layer{number}"}
        anchor = (self.selected_ids() or [None])[-1]
        parent = None
        if anchor and anchor[0] == "item" and entries[anchor[1]].get("layer"):
            parent = entries[anchor[1]]["layer"]
        elif anchor and anchor[0] == "layer":
            parent = next((l.get("layer") for l in layers if l["name"] == anchor[1]), None)
        if parent:
            layer["layer"] = parent
        layers.append(layer)
        self.window.viewer.roto_list_layer = layer["name"]
        self.commit(entries, layers)
        return layer["name"]

    def cmd_duplicate(self):
        entries, layers = self.entries(), self.layers()
        ids = self.selected_ids()
        if not ids:
            return False
        names = {entry["name"] for entry in entries}
        layer_names = {layer["name"] for layer in layers}
        top_ids = []
        picked = []
        for node_id in ids:
            found = _locate(build_tree(entries, layers), node_id)
            picked.append(found[2] if found else None)
        picked = [n for n in picked if n]
        picked = [n for n in picked if not any(m is not n and _contains(m, n["id"]) for m in picked)]
        copied_layers, new_roots = [], []

        def copy_node(node, parent, mapping):
            kind, ref = node["id"]
            if kind == "item":
                entry = copy.deepcopy(entries[ref])
                entry["name"] = unique_name(entry["name"] + " copy", names); names.add(entry["name"])
                if parent:
                    entry["layer"] = parent
                else:
                    entry.pop("layer", None)
                entries.append(entry)
                return ("item", len(entries) - 1)
            definition = copy.deepcopy(next(l for l in layers if l["name"] == ref))
            definition["name"] = unique_name(ref + " copy", layer_names); layer_names.add(definition["name"])
            if parent:
                definition["layer"] = parent
            else:
                definition.pop("layer", None)
            layers.append(definition)
            for child in node["children"]:
                copy_node(child, definition["name"], mapping)
            return ("layer", definition["name"])

        for node in picked:
            kind, ref = node["id"]
            parent = entries[ref].get("layer") if kind == "item" else next((l.get("layer") for l in layers if l["name"] == ref), None)
            new_roots.append((node["id"], copy_node(node, parent, {})))
        top = build_tree(entries, layers)
        for original, new in reversed(new_roots):
            move_nodes(top, [new], original, "below")
        new_entries, new_layers, order = flatten(top, entries, layers)
        first_new = new_roots[-1][1]
        select = order.index(first_new[1]) if first_new[0] == "item" else None
        self.commit(new_entries, new_layers, select)
        return True

    def cmd_copy(self, cut=False):
        entries = self.entries()
        picked = [entries[ref] for kind, ref in self.selected_ids() if kind == "item" and entries[ref].get("kind") != "stroke"]
        picked = [e for e in picked]
        if not picked:
            return False
        self.window.roto_list_clipboard = {"from_roto": self.is_roto, "entries": copy.deepcopy(picked)}
        if cut:
            self.cmd_delete()
        return True

    def cmd_cut(self):
        return self.cmd_copy(cut=True)

    def cmd_paste(self):
        board = getattr(self.window, "roto_list_clipboard", None)
        if not board:
            return False
        entries, layers = self.entries(), self.layers()
        names = {entry["name"] for entry in entries}
        anchor = (self.selected_ids() or [None])[-1]
        parent = entries[anchor[1]].get("layer") if anchor and anchor[0] == "item" else (anchor[1] if anchor else None)
        added = []
        for original in board["entries"]:
            entry = convert_entry(original, self.is_roto)
            if entry is None:
                continue
            entry["name"] = unique_name(entry["name"], names); names.add(entry["name"])
            if parent:
                entry["layer"] = parent
            entries.append(entry); added.append(len(entries) - 1)
        if not added:
            return False
        top = build_tree(entries, layers)
        if anchor and anchor[0] == "item":
            move_nodes(top, [("item", i) for i in added], anchor, "below")
        new_entries, new_layers, order = flatten(top, entries, layers)
        self.commit(new_entries, new_layers, order.index(added[-1]))
        return True

    def cmd_delete(self):
        entries, layers = self.entries(), self.layers()
        ids = self.selected_ids()
        if not ids:
            return False
        top = build_tree(entries, layers)
        remove_nodes(top, ids)
        new_entries, new_layers, order = flatten(top, entries, layers)
        current = self.selected_index()
        select = min(max(current, 0), len(new_entries) - 1) if new_entries and self.is_roto else -1
        self.window.viewer.roto_list_layer = None
        self.commit(new_entries, new_layers, select)
        return True

    def cmd_set_lifetime(self, first, last):
        entries = self.entries()
        changed = False
        for kind, ref in self.selected_ids():
            if kind != "item":
                continue
            entry = entries[ref]
            entry["lifetime"] = ({"mode": "range", "first": int(first), "last": int(last)} if entry.get("kind") == "stroke"
                                 else {"type": "range", "start": int(first), "end": int(last)})
            changed = True
        if changed:
            self.commit(entries)
        return changed

    def cmd_planar_track(self):
        return self.window.track_roto_shape(self.key)

    def ask_lifetime(self):
        dialog = QDialog(self); dialog.setWindowTitle("Set lifetime / frame range")
        form = QFormLayout(dialog)
        first, last = QSpinBox(), QSpinBox()
        for spin in (first, last):
            spin.setRange(-100000, 100000)
        time = self.window.dispatcher.document["time"]
        first.setValue(int(time.get("start", time["current"]))); last.setValue(int(time.get("end", time["current"])))
        first.setObjectName("roto-list-lifetime-first"); last.setObjectName("roto-list-lifetime-last")
        form.addRow("First frame", first); form.addRow("Last frame", last)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.cmd_set_lifetime(min(first.value(), last.value()), max(first.value(), last.value()))

    def build_menu(self):
        menu = QMenu(self); menu.setObjectName("roto-list-menu")
        def add(target, text, slot, name, enabled=True):
            action = target.addAction(text); action.setObjectName(name); action.setEnabled(enabled)
            action.triggered.connect(lambda _=False, s=slot: s())
            return action
        has_items = bool(self.selected_ids())
        has_curves = any(kind == "item" and self.entries()[ref].get("kind") != "stroke" for kind, ref in self.selected_ids())
        add(menu, "Add new layer", self.cmd_add_layer, "roto-list-add-layer")
        add(menu, "Duplicate", self.cmd_duplicate, "roto-list-duplicate", has_items)
        menu.addSeparator()
        for title, slot, leaf, name, enabled in (("Copy", self.cmd_copy, "Curve", "copy", has_curves),
                                                 ("Cut", self.cmd_cut, "Curve", "cut", has_curves),
                                                 ("Paste", self.cmd_paste, "Spline", "paste",
                                                  bool(getattr(self.window, "roto_list_clipboard", None)))):
            sub = menu.addMenu(title); sub.setObjectName(f"roto-list-{name}-menu")
            add(sub, leaf, slot, f"roto-list-{name}", enabled)
        add(menu, "Delete", self.cmd_delete, "roto-list-delete", has_items)
        menu.addSeparator()
        add(menu, "Set lifetime / frame range", self.ask_lifetime, "roto-list-set-lifetime", has_curves)
        add(menu, "Planar-track", self.cmd_planar_track, "roto-list-planar-track", has_curves)
        return menu

    def _show_menu(self, position):
        item = self.itemAt(position)
        if item is not None and not item.isSelected():
            self.clearSelection(); item.setSelected(True)
        self.build_menu().exec(self.viewport().mapToGlobal(position))
