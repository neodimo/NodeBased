"""Roto shape-list and per-shape controls, isolated from the shared properties builder."""
import copy
from . import shapes as shape_model
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
                               QPushButton, QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit,
                               QColorDialog, QLabel, QToolButton)


def build_panel(window, key):
    root = QWidget(); root.setObjectName("roto-shapes-panel")
    layout = QVBoxLayout(root); layout.setContentsMargins(0, 0, 0, 0)
    shapes = copy.deepcopy(window.dispatcher.document.get("node_data", {}).get(key, {}).get("shapes", []))
    frame = int(window.dispatcher.document["time"]["current"])
    def key_state(value):
        if isinstance(value, dict):
            curve = value.get("curve")
            if curve:
                return True, any(item["frame"] == frame for item in curve["keys"])
            return (False, False) if not value else tuple(any(result[i] for result in (key_state(v) for v in value.values())) for i in range(2))
        if isinstance(value, list):
            results = [key_state(v) for v in value]
            return any(r[0] for r in results), any(r[1] for r in results)
        return False, False
    selected = getattr(window.viewer, "roto_selected_shape_index", -1)
    if selected < 0 and shapes:
        selected = window.viewer.roto_selected_shape_index = 0
    listing = QListWidget(); listing.setObjectName("roto-shape-list")
    for index, shape in enumerate(shapes):
        row = QListWidgetItem(); listing.addItem(row)
        widget = QWidget(); row_layout = QHBoxLayout(widget); row_layout.setContentsMargins(2, 0, 2, 0)
        label = QLineEdit(shape["name"]); label.setObjectName(f"roto-shape-row-name-{index}")
        label.setToolTip("Shape name")
        label.setFrame(False); label.editingFinished.connect(
            lambda idx=index, field=label: row_edit(idx, "name", field.text()))
        label.setMinimumWidth(label.fontMetrics().horizontalAdvance(shape["name"]) + 10)
        row_layout.addWidget(label, 1)
        width = shape_model.resolve_scalar(shape.get("stroke_width", 1.0), frame, "stroke_width")
        state = f"{'Closed' if shape.get('closed', True) else 'Open'} · {width:g} px"
        state_label = QLabel(state); state_label.setObjectName(f"roto-shape-row-state-{index}")
        row_layout.addWidget(state_label)
        animated, keyed_here = key_state(shape)
        shape_curve = {"interpolation": "smooth",
                       "keys": [{"frame": frame}] if keyed_here else ([{"frame": -1}] if animated else [])}
        key_button = window.make_key_button(
            key, f"shape-{index}", label, curve_override=shape_curve,
            on_toggle=lambda idx=index: key_shape(idx), object_name=f"roto-shape-key-{index}")
        key_button.setAccessibleName("Shape key")
        key_button.setToolTip("Key all shape points and controls at the current frame")
        row_layout.addWidget(key_button)
        for field, caption, tip in (("visible", "Visible", "Show or hide this shape"),
                                    ("locked", "Lock", "Lock this shape's points"),
                                    ("invert", "Invert", "Invert this shape's matte")):
            toggle = QCheckBox(caption); toggle.setToolTip(tip)
            toggle.setObjectName(f"roto-shape-row-{field}-{index}")
            toggle.setChecked(shape.get(field, field == "visible"))
            toggle.toggled.connect(lambda value, idx=index, f=field: row_edit(idx, f, bool(value)))
            row_layout.addWidget(toggle)
        listing.setItemWidget(row, widget)
    if 0 <= selected < len(shapes): listing.setCurrentRow(selected)
    layout.addWidget(QLabel("Curves")); layout.addWidget(listing)

    def current():
        return copy.deepcopy(window.dispatcher.document.get("node_data", {}).get(key, {}).get("shapes", []))
    def key_shape(index):
        entries = current()
        if not 0 <= index < len(entries): return
        frame = int(window.dispatcher.document["time"]["current"])
        from . import shapes as shape_model
        shape = entries[index]
        def keyed(value, field):
            current_value = shape_model.resolve_scalar(value, frame, field)
            if isinstance(value, dict) and value.get("curve"):
                curve = copy.deepcopy(value["curve"])
            else:
                curve = {"interpolation": "smooth", "keys": []}
            keys = [item for item in curve["keys"] if item["frame"] != frame]
            keys.append({"frame": frame, "value": current_value})
            keys.sort(key=lambda item: item["frame"])
            curve["keys"] = keys
            return {"value": current_value, "curve": curve}
        shape["opacity"] = keyed(shape["opacity"], "opacity")
        shape["feather"] = keyed(shape["feather"], "feather")
        shape["color"] = [keyed(c, "opacity") for c in shape.get("color", [1, 1, 1, 1])]
        resolved_transform = shape_model.resolve_shapes({"shapes": [shape]}, frame)[0]["transform"]
        transform = shape.setdefault("transform", {})
        for field, default in shape_model.TRANSFORM_DEFAULTS.items():
            transform[field] = keyed(transform.get(field, resolved_transform[field]), field)
        for point in shape["points"]:
            for field in (*shape_model.POINT_FIELDS, *shape_model.FEATHER_POINT_FIELDS):
                point.setdefault(field, 0.0)
                point[field] = keyed(point[field], field)
        save(entries, index)
        window.rebuild_properties_dock()
    def key_field(index, field):
        entries = current()
        if not 0 <= index < len(entries): return
        frame = int(window.dispatcher.document["time"]["current"])
        from . import shapes as shape_model
        shape = entries[index]
        names = ["color"] if field == "color" else [field]
        for name in names:
            target = shape
            leaf = name
            if name.startswith("transform."):
                target = shape.setdefault("transform", {})
                leaf = name.split(".", 1)[1]
            default = shape_model.TRANSFORM_DEFAULTS.get(leaf, None)
            if name.startswith("transform.") and leaf in ("center_x", "center_y"):
                default = shape_model.resolve_shapes({"shapes": [shape]}, frame)[0]["transform"][leaf]
            old = target.get(leaf, [1, 1, 1, 1] if name == "color" else default)
            if name == "color":
                shape[name] = [{"value": shape_model.resolve_scalar(c, frame, "opacity"),
                                "curve": {"interpolation": "smooth", "keys": [{"frame": frame, "value": shape_model.resolve_scalar(c, frame, "opacity")}]}}
                               for c in old]
            else:
                resolved = shape_model.resolve_scalar(old, frame, leaf)
                curve = copy.deepcopy(old.get("curve")) if isinstance(old, dict) and old.get("curve") else {"interpolation": "smooth", "keys": []}
                curve["keys"] = [k for k in curve["keys"] if k["frame"] != frame] + [{"frame": frame, "value": resolved}]
                curve["keys"].sort(key=lambda k: k["frame"])
                target[leaf] = {"value": resolved, "curve": curve}
        save(entries, index)
        window.rebuild_properties_dock()
    def row_edit(index, field, value):
        entries=current()
        if index < len(entries): entries[index][field]=value; save(entries,index)
    def edit_transform(index, field, value):
        entries = current()
        if index < len(entries):
            transform = entries[index].setdefault("transform", {})
            default = (shape_model.resolve_shapes({"shapes": [entries[index]]}, frame)[0]["transform"][field]
                       if field in ("center_x", "center_y") else shape_model.TRANSFORM_DEFAULTS[field])
            transform[field] = value_at_frame(transform.get(field, default), value, field)
            save(entries, index)
    def save(entries, select=None):
        window.command({"op": "set_shapes", "id": key, "shapes": entries})
        if select is not None: window.viewer.roto_selected_shape_index = select
    def value_at_frame(old, replacement, field):
        if not isinstance(old, dict) or not old.get("curve"):
            return float(replacement)
        curve = copy.deepcopy(old["curve"])
        keys = [entry for entry in curve["keys"] if entry["frame"] != frame]
        keys.append({"frame": frame, "value": float(replacement)})
        keys.sort(key=lambda entry: entry["frame"])
        return {"value": shape_model.resolve_scalar(old, frame, field), "curve": {**curve, "keys": keys}}
    def select(row):
        window.viewer.roto_selected_shape_index = row
        window.viewer.viewport().update()
        window.rebuild_properties_dock()
    listing.currentRowChanged.connect(select)
    buttons = QHBoxLayout()
    for label, delta in (("Move up", -1), ("Move down", 1)):
        b = QPushButton(label); b.setToolTip("Move the selected shape up" if delta < 0 else "Move the selected shape down"); b.setObjectName("roto-shape-up" if delta < 0 else "roto-shape-down")
        def move(_=False, d=delta):
            entries=current(); i=listing.currentRow(); j=i+d
            if 0 <= i < len(entries) and 0 <= j < len(entries): entries[i],entries[j]=entries[j],entries[i]; save(entries,j)
        b.clicked.connect(move); buttons.addWidget(b)
    for label, action in (("Duplicate", "duplicate"), ("Delete", "delete")):
        b=QPushButton(label); b.setObjectName("roto-shape-"+action)
        b.setToolTip(f"{label} the selected shape")
        def mutate(_=False, act=action):
            entries=current(); i=listing.currentRow()
            if not 0 <= i < len(entries): return
            if act == "delete": del entries[i]; save(entries, min(i,len(entries)-1))
            else:
                item=copy.deepcopy(entries[i]); item["name"] += " copy"; entries.insert(i+1,item); save(entries,i+1)
        b.clicked.connect(mutate); buttons.addWidget(b)
    layout.addLayout(buttons)
    i=listing.currentRow()
    if 0 <= i < len(shapes):
        shape=shapes[i]
        def edit(field,value):
            entries=current()
            if i < len(entries):
                old=entries[i].get(field)
                if field == "color":
                    old = old or [1.0,1.0,1.0,1.0]
                    entries[i][field]=[value_at_frame(channel, replacement, "opacity")
                                       for channel,replacement in zip(old,value)]
                elif field == "feather_model":
                    entries[i][field] = value
                else:
                    entries[i][field]=value_at_frame(old,value,field)
                save(entries,i)
        name=QLineEdit(shape["name"]); name.setObjectName("roto-shape-name")
        name.editingFinished.connect(lambda: edit("name",name.text())); layout.addWidget(name)
        follow=QComboBox(); follow.setObjectName(f"roto-shape-track-link-{i}")
        follow.addItem("No Tracker link", None)
        trackers=window.dispatcher.document["nodes"]
        for tracker_id, tracker_node in trackers.items():
            if tracker_node["type"] not in ("Tracker", "Stabilize"): continue
            for track in window.dispatcher.document.get("node_data", {}).get(tracker_id, {}).get("tracks", []):
                follow.addItem(f"{tracker_node['name']} · {track['name']}",
                               {"tracker_id": tracker_id, "track_name": track["name"]})
        link=shape.get("track_link")
        follow.setCurrentIndex(next((j for j in range(follow.count()) if follow.itemData(j)==link),0))
        def set_follow(index):
            entries=current()
            if i < len(entries):
                if follow.itemData(index) is None: entries[i].pop("track_link",None)
                else: entries[i]["track_link"]=follow.itemData(index)
                save(entries,i)
        follow.currentIndexChanged.connect(set_follow); layout.addWidget(QLabel("Transform follows")); layout.addWidget(follow)
        follow_mode=QComboBox(); follow_mode.setObjectName(f"roto-shape-track-follow-{i}")
        follow_mode.addItem("Translate only", "translate"); follow_mode.addItem("Full Tracker solve", "full")
        follow_mode.setCurrentIndex(follow_mode.findData(shape.get("track_follow", "translate")))
        follow_mode.currentIndexChanged.connect(lambda index, idx=i, box=follow_mode: row_edit(idx, "track_follow", box.itemData(index)))
        layout.addWidget(follow_mode)
        for field,label in (("locked","Lock"),("invert","Invert"),("motion_blur","Motion blur")):
            cb=QCheckBox(label); cb.setObjectName("roto-shape-"+field); cb.setChecked(shape.get(field,False)); cb.toggled.connect(lambda v,f=field:edit(f,bool(v))); layout.addWidget(cb)
        for field, label, lo, hi, step in (("stroke_width", "Stroke width", 0, 4096, 0.5),
                                          ("feather", "Feather", -500, 500, 0.5),
                                          ("opacity", "Opacity", 0, 1, 0.05)):
            spin = QDoubleSpinBox()
            spin.setObjectName("roto-shape-" + field)
            spin.setRange(lo, hi)
            spin.setSingleStep(step)
            spin.setValue(float(shape_model.resolve_scalar(
                shape.get(field, 1 if field == "stroke_width" else 0 if field == "feather" else 1), frame, field)))
            spin.setKeyboardTracking(False)
            spin.editingFinished.connect(lambda f=field, w=spin: edit(f, float(w.value())))
            spin.setToolTip(f"Set the shape {label.lower()}")
            layout.addWidget(QLabel(label))
            row = QHBoxLayout()
            row.addWidget(spin, 1)
            field_value = shape.get(field)
            field_curve = field_value.get("curve") if isinstance(field_value, dict) else None
            key_button = window.make_key_button(
                key, f"shape-{i}-{field}", spin, curve_override=field_curve,
                on_toggle=lambda f=field: key_field(i, f),
                object_name="roto-shape-key-" + field)
            key_button.setAccessibleName(f"{label} key")
            row.addWidget(key_button)
            layout.addLayout(row)
        transform = shape.get("transform", {})
        for field, label, lo, hi, step in (
                ("translate_x", "Translate X", -65536, 65536, 1),
                ("translate_y", "Translate Y", -65536, 65536, 1),
                ("rotate", "Rotate", -360000, 360000, 1),
                ("scale_x", "Scale X", -1000, 1000, .01),
                ("scale_y", "Scale Y", -1000, 1000, .01),
                ("center_x", "Center X", -65536, 65536, 1),
                ("center_y", "Center Y", -65536, 65536, 1)):
            spin = QDoubleSpinBox(); spin.setObjectName(f"roto-shape-transform-{field}-{i}")
            spin.setRange(lo, hi); spin.setSingleStep(step); spin.setKeyboardTracking(False)
            default = shape_model.resolve_shapes({"shapes": [shape]}, frame)[0]["transform"][field]
            spin.setValue(shape_model.resolve_scalar(transform.get(field, default), frame, field))
            spin.editingFinished.connect(lambda f=field, w=spin: edit_transform(i, f, float(w.value())))
            row = QHBoxLayout(); row.addWidget(spin, 1)
            raw = transform.get(field)
            curve = raw.get("curve") if isinstance(raw, dict) else None
            button = window.make_key_button(key, f"shape-{i}-transform-{field}", spin,
                curve_override=curve, on_toggle=lambda f=field: key_field(i, "transform." + f),
                object_name=f"roto-shape-key-transform-{field}-{i}")
            button.setAccessibleName(f"{label} key"); row.addWidget(button)
            layout.addWidget(QLabel(label)); layout.addLayout(row)
        for field,values in (("feather_falloff",("linear","smooth","gaussian")),("blend_mode",("over","plus","minus","multiply"))):
            combo=QComboBox(); combo.setObjectName("roto-shape-"+field)
            for val in values: combo.addItem(val.title(),val)
            combo.setCurrentIndex(combo.findData(shape.get(field,"linear" if field=="feather_falloff" else "over")))
            combo.currentIndexChanged.connect(lambda idx,f=field,c=combo:edit(f,c.itemData(idx))); layout.addWidget(QLabel(field.replace("_"," ").title())); layout.addWidget(combo)
        if shape.get("feather_model", "box") == "box":
            switch = QPushButton("Switch to edge feather")
            switch.setObjectName("roto-shape-edge-feather")
            switch.setToolTip("Use the outward Nuke-style feather for this shape")
            switch.clicked.connect(lambda: edit("feather_model", "edge"))
            layout.addWidget(switch)
        reset_feather = QPushButton("Reset feather on selected points")
        reset_feather.setObjectName("roto-reset-feather-points")
        reset_feather.setEnabled(not shape.get("locked") and any(si == i for si, _ in window.viewer.roto_selected_points))
        def reset_selected_feather():
            entries = current()
            if i >= len(entries) or entries[i].get("locked"):
                return
            for si, pi in window.viewer.roto_selected_points:
                if si != i or pi >= len(entries[i]["points"]):
                    continue
                point = entries[i]["points"][pi]
                for field in shape_model.FEATHER_POINT_FIELDS:
                    point[field] = value_at_frame(point.get(field, 0.0), 0.0, field)
            save(entries, i)
        reset_feather.clicked.connect(reset_selected_feather)
        layout.addWidget(reset_feather)
        color_row = QHBoxLayout()
        color=QPushButton("Choose colour…"); color.setObjectName("roto-shape-color")
        color.setToolTip("Choose the shape colour")
        swatch = QLabel(); swatch.setObjectName("roto-shape-color-swatch")
        swatch.setFixedSize(18, 18)
        swatch.setToolTip("Current shape colour")
        def resolved_color(value):
            return [shape_model.resolve_scalar(channel, frame, "opacity") for channel in value]
        def paint_swatch(value):
            rgba = resolved_color(value or [1, 1, 1, 1])
            qcolor = QColor.fromRgbF(*rgba)
            swatch.setStyleSheet(f"background-color: {qcolor.name(QColor.NameFormat.HexArgb)}; border: 1px solid #666")
            swatch.setProperty("rgba", rgba)
        paint_swatch(shape.get("color", [1, 1, 1, 1]))
        def choose():
            rgba=resolved_color(shape.get("color",[1,1,1,1])); picked=QColorDialog.getColor(QColor.fromRgbF(*rgba),window,"Roto colour",QColorDialog.ColorDialogOption.ShowAlphaChannel)
            if picked.isValid(): edit("color",[picked.redF(),picked.greenF(),picked.blueF(),picked.alphaF()])
        color.clicked.connect(choose); color_row.addWidget(color); color_row.addWidget(swatch)
        color_curve = shape.get("color", [])
        animated_color, keyed_color = key_state(color_curve)
        color_proxy = {"interpolation": "smooth",
                       "keys": [{"frame": frame}] if keyed_color else ([{"frame": -1}] if animated_color else [])}
        color_key=window.make_key_button(key, "shape-color", color, curve_override=color_proxy,
                                         on_toggle=lambda: key_field(i,"color"),
                                         object_name="roto-shape-key-color")
        color_key.setAccessibleName("Colour key")
        color_key.setToolTip("Set colour keys at the current frame")
        color_row.addWidget(color_key); layout.addLayout(color_row)
    draw=QPushButton("Draw shape…"); draw.setObjectName("roto-draw-shape"); draw.clicked.connect(lambda:window.begin_roto_draw(key)); layout.addWidget(draw)
    return root
