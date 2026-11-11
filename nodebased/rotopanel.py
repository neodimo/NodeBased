"""Roto shape-list and per-shape controls, isolated from the shared properties builder."""
import copy
from . import shapes as shape_model
from . import rotolist
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
                               QPushButton, QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit,
                               QColorDialog, QLabel, QToolButton, QSpinBox, QGroupBox)


def build_paint_timing_controls(window, key, index):
    """Selected RotoPaint spline keys and key-scope controls for shape items."""
    items = window.dispatcher.document.get("node_data", {}).get(key, {}).get("items", [])
    if not 0 <= index < len(items) or items[index].get("kind") != "shape":
        return None
    root = QWidget(); root.setObjectName("rotopaint-timing-keys")
    layout = QVBoxLayout(root); layout.setContentsMargins(0, 0, 0, 0)
    frame = int(window.dispatcher.document["time"]["current"])
    def frames_for(item):
        found = set()
        def walk(value):
            if isinstance(value, dict):
                if value.get("curve"):
                    found.update(int(k["frame"]) for k in value["curve"].get("keys", []))
                else:
                    for child in value.values(): walk(child)
            elif isinstance(value, list):
                for child in value: walk(child)
        walk(item)
        return sorted(found)
    frames = frames_for(items[index])
    bar = QHBoxLayout()
    count = QLabel(f"Key {frames.index(frame)+1 if frame in frames else 0} / {len(frames)}")
    count.setObjectName("rotopaint-spline-key-count"); bar.addWidget(count)
    for title, delta in (("Previous key", -1), ("Next key", 1)):
        target = max((f for f in frames if f < frame), default=None) if delta < 0 else min((f for f in frames if f > frame), default=None)
        button = QPushButton(title); button.setObjectName("rotopaint-spline-previous-key" if delta < 0 else "rotopaint-spline-next-key")
        button.setEnabled(target is not None)
        button.clicked.connect(lambda _=False, at=target: window.set_time(current=at) if at is not None else None)
        bar.addWidget(button)
    layout.addLayout(bar)
    for title, delete, name in (("Set spline key", False, "rotopaint-spline-set-key"),
                                ("Delete spline key", True, "rotopaint-spline-delete-key")):
        button = QPushButton(title); button.setObjectName(name)
        def edit_spline(_=False, remove=delete):
            at = int(window.dispatcher.document["time"]["current"])
            entries = copy.deepcopy(window.dispatcher.document.get("node_data", {}).get(key, {}).get("items", []))
            if index >= len(entries): return
            item = entries[index]
            for point in item.get("points", []):
                for field in (*shape_model.POINT_FIELDS, *shape_model.FEATHER_POINT_FIELDS):
                    old = point.get(field, 0.0)
                    point[field] = delete_scalar_key(old, at) if remove else set_scalar_key(old, field, at)
            window.command({"op": "set_paint_items", "id": key, "items": entries})
        button.clicked.connect(edit_spline); layout.addWidget(button)
    for title, delete, name in (("Set selected point keys", False, "rotopaint-point-set-keys"),
                                ("Delete selected point keys", True, "rotopaint-point-delete-keys")):
        button = QPushButton(title); button.setObjectName(name)
        def edit_points(_=False, remove=delete):
            at = int(window.dispatcher.document["time"]["current"])
            selected = {pi for si, pi in getattr(window.viewer, "roto_selected_points", set()) if si == 0}
            entries = copy.deepcopy(window.dispatcher.document.get("node_data", {}).get(key, {}).get("items", []))
            if index >= len(entries): return
            item = entries[index]
            for pi in selected:
                if 0 <= pi < len(item.get("points", [])):
                    point = item["points"][pi]
                    for field in (*shape_model.POINT_FIELDS, *shape_model.FEATHER_POINT_FIELDS):
                        old = point.get(field, 0.0)
                        point[field] = delete_scalar_key(old, at) if remove else set_scalar_key(old, field, at)
            if selected: window.command({"op": "set_paint_items", "id": key, "items": entries})
        button.clicked.connect(edit_points); layout.addWidget(button)
    scope = QComboBox(); scope.setObjectName("rotopaint-key-scope")
    for label, value in (("All", "all"), ("Spline", "spline"), ("Transform", "transform"), ("Attributes", "attributes")):
        scope.addItem(label, value)
    layout.addWidget(scope)
    for title, delete, name in (("Set item key", False, "rotopaint-item-set-key"),
                                ("Delete item key", True, "rotopaint-item-delete-key")):
        button = QPushButton(title); button.setObjectName(name)
        def edit_scope(_=False, remove=delete):
            at = int(window.dispatcher.document["time"]["current"])
            entries = copy.deepcopy(window.dispatcher.document.get("node_data", {}).get(key, {}).get("items", []))
            if index >= len(entries): return
            item = entries[index]; mode = scope.currentData()
            paths = []
            if mode in ("all", "spline"):
                paths.extend(("points", pi, field) for pi, point in enumerate(item.get("points", []))
                             for field in (*shape_model.POINT_FIELDS, *shape_model.FEATHER_POINT_FIELDS))
            if mode in ("all", "transform"):
                item.setdefault("transform", {})
                paths.extend(("transform", field) for field in shape_model.TRANSFORM_DEFAULTS)
            if mode in ("all", "attributes"):
                paths.extend((field,) for field in ("opacity", "feather"))
                paths.extend(("color", ci) for ci in range(len(item.get("color", []))))
            for path in paths:
                target = item
                for part in path[:-1]: target = target[part]
                field = path[-1]
                old = target.get(field, shape_model.TRANSFORM_DEFAULTS.get(field, 0.0)) if isinstance(target, dict) else target[field]
                target[field] = delete_scalar_key(old, at) if remove else set_scalar_key(old, field, at)
            window.command({"op": "set_paint_items", "id": key, "items": entries})
        button.clicked.connect(edit_scope); layout.addWidget(button)
    return root


def set_scalar_key(value, field, at):
    number = shape_model.resolve_scalar(value, at, field)
    curve = copy.deepcopy(value.get("curve")) if isinstance(value, dict) and value.get("curve") else {"interpolation": "smooth", "keys": []}
    keys = [item for item in curve["keys"] if item["frame"] != at]
    keys.append({"frame": at, "value": number}); keys.sort(key=lambda item: item["frame"])
    return {"value": number, "curve": {**curve, "keys": keys}}


def delete_scalar_key(value, at):
    if not isinstance(value, dict) or not value.get("curve"):
        return value
    updated = copy.deepcopy(value)
    updated["curve"]["keys"] = [item for item in updated["curve"]["keys"] if item["frame"] != at]
    return updated["value"] if not updated["curve"]["keys"] else updated


def add_point_tracking(layout, window, key):
    """Track points beside Track shape: direction and frame range, then the command (lane 9 step I2, past Nuke)."""
    from PySide6.QtWidgets import QComboBox, QSpinBox
    time = window.dispatcher.document.get("time", {})
    row = QHBoxLayout()
    direction = QComboBox(); direction.setObjectName("roto-track-points-direction")
    for label, value in (("Forward and backward", "both"), ("Forward", "forward"), ("Backward", "backward")):
        direction.addItem(label, value)
    first, last = QSpinBox(), QSpinBox()
    for spin, name, value in ((first, "roto-track-points-first", time.get("first", 1)),
                              (last, "roto-track-points-last", time.get("last", 1))):
        spin.setRange(-100000, 100000); spin.setValue(int(value)); spin.setObjectName(name)
    row.addWidget(direction); row.addWidget(QLabel("From")); row.addWidget(first)
    row.addWidget(QLabel("to")); row.addWidget(last)
    layout.addLayout(row)
    button = QPushButton("Track points")
    button.setObjectName("roto-track-points")
    button.setToolTip("Track every control point of the selected shape from the current frame; points keep their own patch, "
                      "so the shape follows a deforming subject. Keys every tracked frame; low-confidence points are flagged.")
    button.clicked.connect(lambda: window.track_roto_points(key, direction.currentData(), first.value(), last.value()))
    layout.addWidget(button)
    stick = QPushButton("Stick to plate")
    stick.setObjectName("roto-stick-to-plate")
    stick.setToolTip("Carry the selected paint stroke with the picture from the current frame, over the same frames: its points are "
                     "keyed to follow the plate, and its brush size and clone offset follow the motion. Past Nuke.")
    stick.clicked.connect(lambda: window.stick_roto_strokes(key, direction.currentData(), first.value(), last.value()))
    layout.addWidget(stick)


def add_tracking_actions(layout, window, key):
    track = QPushButton("Track selected shape or stroke")
    track.setObjectName("roto-track-shape")
    track.setToolTip("Track the selected item's bounding quad from the reference frame")
    track.clicked.connect(lambda: window.track_roto_shape(key))
    layout.addWidget(track)
    add_point_tracking(layout, window, key)
    layout.addWidget(QLabel("Export"))
    export = QPushButton("Create")
    export.setObjectName("roto-create-tracking-export")
    export.clicked.connect(lambda: window.export_roto_tracking(key))
    layout.addWidget(export)


def build_panel(window, key):
    root = QWidget(); root.setObjectName("roto-shapes-panel")
    layout = QVBoxLayout(root); layout.setContentsMargins(0, 0, 0, 0)
    shapes = copy.deepcopy(window.dispatcher.document.get("node_data", {}).get(key, {}).get("shapes", []))
    frame = int(window.dispatcher.document["time"]["current"])
    def float_slider(value, lo, hi, step, name):
        from .app import FloatSliderControl
        control = FloatSliderControl((lo, hi), (lo, hi), float(value))
        control.motion = window.motion; control.slider.motion = window.motion
        control.spin.setObjectName(name); control.spin.setSingleStep(step)
        return control
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
    layout.addWidget(QLabel("Curves")); list_slot = layout.count()

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
    def keyframes(shape):
        frames = set()
        def walk(value):
            if isinstance(value, dict):
                curve = value.get("curve")
                if isinstance(curve, dict):
                    frames.update(int(item["frame"]) for item in curve.get("keys", []))
                else:
                    for child in value.values(): walk(child)
            elif isinstance(value, list):
                for child in value: walk(child)
        walk(shape)
        return sorted(frames)
    def set_scalar_key(value, field, at):
        from . import shapes as shape_model
        number = shape_model.resolve_scalar(value, at, field)
        curve = copy.deepcopy(value.get("curve")) if isinstance(value, dict) and value.get("curve") else {"interpolation": "smooth", "keys": []}
        keys = [item for item in curve["keys"] if item["frame"] != at]
        keys.append({"frame": at, "value": number}); keys.sort(key=lambda item: item["frame"])
        return {"value": number, "curve": {**curve, "keys": keys}}
    def delete_scalar_key(value, at):
        if not isinstance(value, dict) or not value.get("curve"):
            return value
        updated = copy.deepcopy(value)
        updated["curve"]["keys"] = [item for item in updated["curve"]["keys"] if item["frame"] != at]
        return updated["value"] if not updated["curve"]["keys"] else updated
    def edit_spline_keys(index, delete=False):
        entries = current()
        if not 0 <= index < len(entries): return
        at = int(window.dispatcher.document["time"]["current"])
        shape = entries[index]
        for point in shape["points"]:
            for field in ("x", "y", "in_x", "in_y", "out_x", "out_y", "feather_x", "feather_y"):
                point[field] = delete_scalar_key(point.get(field, 0.0), at) if delete else set_scalar_key(point.get(field, 0.0), field, at)
        save(entries, index)
        window.rebuild_properties_dock()
    def edit_selected_point_keys(index, delete=False):
        selected_points = {(si, pi) for si, pi in getattr(window.viewer, "roto_selected_points", set()) if si == index}
        if not selected_points: return
        entries = current()
        if not 0 <= index < len(entries): return
        at = int(window.dispatcher.document["time"]["current"])
        shape = entries[index]
        for _, point_index in selected_points:
            if not 0 <= point_index < len(shape["points"]): continue
            point = shape["points"][point_index]
            for field in ("x", "y", "in_x", "in_y", "out_x", "out_y", "feather_x", "feather_y"):
                point[field] = delete_scalar_key(point.get(field, 0.0), at) if delete else set_scalar_key(point.get(field, 0.0), field, at)
        save(entries, index)
        window.rebuild_properties_dock()
    def edit_key_scope(index, scope, delete=False):
        entries = current()
        if not 0 <= index < len(entries): return
        shape = entries[index]
        paths = []
        if scope in ("all", "spline"):
            paths.extend(("points", pi, field) for pi, point in enumerate(shape.get("points", []))
                         for field in (*shape_model.POINT_FIELDS, *shape_model.FEATHER_POINT_FIELDS))
        if scope in ("all", "transform"):
            transform = shape.setdefault("transform", {})
            paths.extend(("transform", field) for field in shape_model.TRANSFORM_DEFAULTS)
        if scope in ("all", "attributes"):
            paths.extend((field,) for field in ("opacity", "feather"))
            paths.extend(("color", ci) for ci in range(len(shape.get("color", []))))
        for path in paths:
            target = shape
            for part in path[:-1]: target = target[part]
            field = path[-1]
            old = target.get(field, shape_model.TRANSFORM_DEFAULTS.get(field, 0.0)) if isinstance(target, dict) else target[field]
            target[field] = delete_scalar_key(old, frame) if delete else set_scalar_key(old, "opacity" if path[0] == "color" else field, frame)
        save(entries, index)
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
        window.refresh_panel_soon(window.rebuild_properties_dock)
    def row_extras(row_layout, index, label):
        """The key button of the Name cell: it keys every point and control of the shape at the current frame."""
        animated, keyed_here = key_state(shapes[index])
        shape_curve = {"interpolation": "smooth",
                       "keys": [{"frame": frame}] if keyed_here else ([{"frame": -1}] if animated else [])}
        key_button = window.make_key_button(
            key, f"shape-{index}", label, curve_override=shape_curve,
            on_toggle=lambda idx=index: key_shape(idx), object_name=f"roto-shape-key-{index}")
        key_button.setAccessibleName("Shape key")
        key_button.setToolTip("Key all shape points and controls at the current frame")
        row_layout.addWidget(key_button)
    listing = rotolist.ShapeListTree(window, key, True, select, row_extras)
    layout.insertWidget(list_slot, listing)
    buttons = QHBoxLayout()
    for label, delta in (("Move up", -1), ("Move down", 1)):
        b = QPushButton(label); b.setToolTip("Move the selected shape up" if delta < 0 else "Move the selected shape down"); b.setObjectName("roto-shape-up" if delta < 0 else "roto-shape-down")
        b.clicked.connect(lambda _=False, d=delta: listing.cmd_move(d)); buttons.addWidget(b)
    for label, action in (("Duplicate", listing.cmd_duplicate), ("Delete", listing.cmd_delete)):
        b=QPushButton(label); b.setObjectName("roto-shape-"+label.lower())
        b.setToolTip(f"{label} the selected shape")
        def mutate(_=False, act=action):
            index = listing.selected_index()
            if 0 <= index < len(shapes):
                listing.select_ids([("item", index)])
                act()
        b.clicked.connect(mutate); buttons.addWidget(b)
    layout.addLayout(buttons)
    layer_name = getattr(window.viewer, "roto_list_layer", None)
    layer_entry = next((l for l in window.dispatcher.document.get("node_data", {}).get(key, {}).get("layers", [])
                        if l["name"] == layer_name), None) if layer_name else None
    if layer_entry is not None:
        _layer_transform_controls(window, key, layout, layer_entry, frame, float_slider)
    i=listing.selected_index()
    if 0 <= i < len(shapes):
        shape=shapes[i]
        frames = keyframes(shape)
        current_frame = int(window.dispatcher.document["time"]["current"])
        count = QLabel(f"Key {frames.index(current_frame)+1 if current_frame in frames else 0} / {len(frames)}")
        count.setObjectName("roto-spline-key-count")
        key_controls = QHBoxLayout(); key_controls.addWidget(count)
        for title, delta in (("Previous key", -1), ("Next key", 1)):
            button = QPushButton(title); button.setObjectName("roto-spline-previous-key" if delta < 0 else "roto-spline-next-key")
            target = (max((f for f in frames if f < current_frame), default=None) if delta < 0
                      else min((f for f in frames if f > current_frame), default=None))
            button.setEnabled(target is not None)
            button.clicked.connect(lambda _=False, at=target: window.set_time(current=at) if at is not None else None)
            key_controls.addWidget(button)
        set_key = QPushButton("Set spline key"); set_key.setObjectName("roto-spline-set-key")
        set_key.clicked.connect(lambda _=False, idx=i: edit_spline_keys(idx))
        delete_key = QPushButton("Delete spline key"); delete_key.setObjectName("roto-spline-delete-key")
        delete_key.clicked.connect(lambda _=False, idx=i: edit_spline_keys(idx, True))
        key_controls.addWidget(set_key); key_controls.addWidget(delete_key); layout.addLayout(key_controls)
        point_controls = QHBoxLayout()
        for title, delete in (("Set selected point keys", False), ("Delete selected point keys", True)):
            button = QPushButton(title); button.setObjectName("roto-point-delete-keys" if delete else "roto-point-set-keys")
            button.clicked.connect(lambda _=False, idx=i, d=delete: edit_selected_point_keys(idx, d))
            point_controls.addWidget(button)
        layout.addLayout(point_controls)
        scope_controls = QHBoxLayout()
        scope = QComboBox(); scope.setObjectName("roto-key-scope")
        for label, value in (("All", "all"), ("Spline", "spline"), ("Transform", "transform"), ("Attributes", "attributes")):
            scope.addItem(label, value)
        scope_controls.addWidget(scope)
        for title, delete, name in (("Set item key", False, "roto-item-set-key"),
                                    ("Delete item key", True, "roto-item-delete-key")):
            button = QPushButton(title); button.setObjectName(name)
            button.clicked.connect(lambda _=False, idx=i, d=delete, box=scope: edit_key_scope(idx, box.currentData(), d))
            scope_controls.addWidget(button)
        layout.addLayout(scope_controls)
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
                elif field == "motion_blur_samples":
                    entries[i][field] = value_at_frame(old, int(value), field) if isinstance(old, dict) and old.get("curve") else int(value)
                else:
                    entries[i][field]=value_at_frame(old,value,field)
                save(entries,i)
        name=QLineEdit(shape["name"]); name.setObjectName("roto-shape-name")
        name.editingFinished.connect(lambda: edit("name",name.text())); layout.addWidget(name)
        follow=QComboBox(); follow.setObjectName(f"roto-shape-track-link-{i}")
        follow.addItem("No Tracker link", None)
        trackers=window.dispatcher.document["nodes"]
        for tracker_id, tracker_node in trackers.items():
            if tracker_node["type"] == "PlanarTracker":
                follow.addItem(f"{tracker_node['name']} · plane", {"tracker_id": tracker_id, "track_name": "corner1"})
                continue
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
        follow_mode.addItem("Planar", "planar")
        follow_mode.setCurrentIndex(follow_mode.findData(shape.get("track_follow", "translate")))
        follow_mode.currentIndexChanged.connect(lambda index, idx=i, box=follow_mode: row_edit(idx, "track_follow", box.itemData(index)))
        layout.addWidget(follow_mode)
        for field,label in (("locked","Lock"),("invert","Invert"),("motion_blur","Motion blur")):
            cb=QCheckBox(label); cb.setObjectName("roto-shape-"+field); cb.setChecked(shape.get(field,False)); cb.toggled.connect(lambda v,f=field:edit(f,bool(v))); layout.addWidget(cb)
        source = QComboBox(); source.setObjectName("roto-shape-source")
        for label, value in (("Colour", "color"), ("Foreground", "foreground"), ("Background", "background")):
            source.addItem(label, value)
        source.setCurrentIndex(max(0, source.findData(shape.get("source", "color"))))
        source.currentIndexChanged.connect(lambda index, box=source: edit("source", box.itemData(index)))
        layout.addWidget(QLabel("Shape source")); layout.addWidget(source)
        # Nuke's Clone tab controls belong to this shape and only affect picture sources.
        clone_box = QWidget(); clone_layout = QVBoxLayout(clone_box)
        clone_layout.setContentsMargins(8, 2, 2, 2)
        clone_box.setObjectName(f"roto-shape-clone-{i}")
        clone_layout.addWidget(QLabel("Clone"))
        def clone_edit(field, value):
            entries = current()
            if i < len(entries):
                entries[i][field] = value
                save(entries, i)
        def clone_vec(field, part, value):
            entries = current()
            if i < len(entries):
                vector = list(entries[i].get(field, [0.0, 0.0])); vector[part] = value
                entries[i][field] = vector
                save(entries, i)
        for part, label in enumerate(("Translate X", "Translate Y")):
            spin = QDoubleSpinBox(); spin.setObjectName(f"roto-shape-source-transform-translate-{part}")
            spin.setRange(-65536, 65536); spin.setValue(float(shape.get("source_transform_translate", [0, 0])[part]))
            spin.setKeyboardTracking(False); spin.editingFinished.connect(
                lambda p=part, w=spin: clone_vec("source_transform_translate", p, float(w.value())))
            clone_layout.addWidget(QLabel(label)); clone_layout.addWidget(spin)
        for field, label, default in (("source_transform_rotate", "Rotate", 0.0),
                                      ("source_transform_scale", "Scale", 1.0),
                                      ("source_transform_skewX", "Skew X", 0.0),
                                      ("source_transform_skewY", "Skew Y", 0.0)):
            spin = QDoubleSpinBox(); spin.setObjectName("roto-shape-" + field)
            spin.setRange(-1000, 1000); spin.setValue(float(shape.get(field, default))); spin.setKeyboardTracking(False)
            spin.editingFinished.connect(lambda f=field, w=spin: clone_edit(f, float(w.value())))
            clone_layout.addWidget(QLabel(label)); clone_layout.addWidget(spin)
        for field, label, values, default in (("source_transform_skew_order", "Skew order", ("XY", "YX"), "XY"),
                                               ("source_filter", "Filter", ("impulse", "cubic", "keys", "simon", "rifman", "mitchell", "parzen", "notch"), "cubic"),
                                               ("source_time_offset_type", "Time offset type", ("relative", "absolute"), "relative")):
            combo = QComboBox(); combo.setObjectName("roto-shape-" + field)
            for value in values: combo.addItem(value.capitalize(), value)
            combo.setCurrentIndex(max(0, combo.findData(shape.get(field, default))))
            combo.currentIndexChanged.connect(lambda idx, f=field, w=combo: clone_edit(f, w.itemData(idx)))
            clone_layout.addWidget(QLabel(label)); clone_layout.addWidget(combo)
        center = shape.get("source_transform_center", [0, 0])
        for part, label in enumerate(("Center X", "Center Y")):
            spin = QDoubleSpinBox(); spin.setObjectName(f"roto-shape-source-transform-center-{part}")
            spin.setRange(-65536, 65536); spin.setValue(float(center[part])); spin.setKeyboardTracking(False)
            spin.editingFinished.connect(lambda p=part, w=spin: clone_vec("source_transform_center", p, float(w.value())))
            clone_layout.addWidget(QLabel(label)); clone_layout.addWidget(spin)
        for field, label, default in (("source_translate_round", "Round translation", False),
                                      ("source_black_outside", "Black outside source", False)):
            check = QCheckBox(label); check.setObjectName("roto-shape-" + field)
            check.setChecked(bool(shape.get(field, default))); check.toggled.connect(lambda v, f=field: clone_edit(f, bool(v)))
            clone_layout.addWidget(check)
        for field, label in (("source_time_offset", "Source time offset"),):
            spin = QDoubleSpinBox(); spin.setObjectName("roto-shape-" + field)
            spin.setRange(-100000, 100000); spin.setValue(float(shape.get(field, 0))); spin.setKeyboardTracking(False)
            spin.editingFinished.connect(lambda f=field, w=spin: clone_edit(f, float(w.value())))
            clone_layout.addWidget(QLabel(label)); clone_layout.addWidget(spin)
        layout.addWidget(clone_box)
        clone_box.setEnabled(shape.get("source", "color") != "color")
        source.currentIndexChanged.connect(lambda index, box=clone_box, control=source:
                                           box.setEnabled(control.itemData(index) != "color"))
        for field, label, lo, hi, step in (("stroke_width", "Stroke width", 0, 4096, 0.5),
                                          ("feather", "Feather", -500, 500, 0.5),
                                          ("opacity", "Opacity", 0, 1, 0.05)):
            shown = shape_model.resolve_scalar(
                shape.get(field, 1 if field == "stroke_width" else 0 if field == "feather" else 1), frame, field)
            control = float_slider(shown, lo, hi, step, "roto-shape-" + field)
            spin = control.spin; spin.setKeyboardTracking(False)
            spin.editingFinished.connect(lambda f=field, w=spin: edit(f, float(w.value())))
            spin.setToolTip(f"Set the shape {label.lower()}")
            layout.addWidget(QLabel(label))
            row = QHBoxLayout()
            row.addWidget(control, 1)
            field_value = shape.get(field)
            field_curve = field_value.get("curve") if isinstance(field_value, dict) else None
            key_button = window.make_key_button(
                key, f"shape-{i}-{field}", control, curve_override=field_curve,
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
                ("center_y", "Center Y", -65536, 65536, 1),
                ("skew_x", "Skew X", -89, 89, 1),
                ("skew_y", "Skew Y", -89, 89, 1)):
            default = shape_model.resolve_shapes({"shapes": [shape]}, frame)[0]["transform"][field]
            control = float_slider(shape_model.resolve_scalar(transform.get(field, default), frame, field), lo, hi,
                                   step, f"roto-shape-transform-{field}-{i}")
            spin = control.spin; spin.setKeyboardTracking(False)
            spin.editingFinished.connect(lambda f=field, w=spin: edit_transform(i, f, float(w.value())))
            row = QHBoxLayout(); row.addWidget(control, 1)
            raw = transform.get(field)
            curve = raw.get("curve") if isinstance(raw, dict) else None
            button = window.make_key_button(key, f"shape-{i}-transform-{field}", control,
                curve_override=curve, on_toggle=lambda f=field: key_field(i, "transform." + f),
                object_name=f"roto-shape-key-transform-{field}-{i}")
            button.setAccessibleName(f"{label} key"); row.addWidget(button)
            layout.addWidget(QLabel(label)); layout.addLayout(row)
        def plain(field, value):
            """Set a setting that is a plain value on the shape (a flag, a choice, a whole number), not an animatable scalar."""
            entries = current()
            if i < len(entries):
                entries[i][field] = value
                save(entries, i)
        def plain_transform(field, value):
            entries = current()
            if i < len(entries):
                entries[i].setdefault("transform", {})[field] = value
                save(entries, i)
        def plain_check(field, label, default):
            box = QCheckBox(label); box.setObjectName("roto-shape-" + field); box.setChecked(bool(shape.get(field, default)))
            box.toggled.connect(lambda v, f=field: plain(f, bool(v))); layout.addWidget(box)
        def plain_combo(field, label, values, default, setter=None):
            combo = QComboBox(); combo.setObjectName("roto-shape-" + field)
            for val in values: combo.addItem(val.replace("_", " ").capitalize(), val)
            source = shape.get("transform", {}) if setter is plain_transform else shape
            combo.setCurrentIndex(max(0, combo.findData(source.get(field, default))))
            combo.currentIndexChanged.connect(lambda idx, f=field, c=combo: (setter or plain)(f, c.itemData(idx)))
            layout.addWidget(QLabel(label)); layout.addWidget(combo)
        def plain_spin(field, label, lo, hi, step, default, whole=False):
            raw = shape.get(field, default)
            shown = shape_model.resolve_scalar(raw, frame, field) if isinstance(raw, dict) else raw
            if whole:
                spin = QSpinBox(); spin.setRange(int(lo), int(hi)); spin.setValue(int(round(shown)))
            else:
                from .app import FloatSliderControl
                spin = FloatSliderControl((lo, hi), (lo, hi), float(shown))
                spin.motion = window.motion; spin.slider.motion = window.motion
                spin.spin.setSingleStep(step); spin.spin.setKeyboardTracking(False)
            editor = spin if whole else spin.spin
            editor.setObjectName("roto-shape-" + field); editor.setKeyboardTracking(False)
            editor.editingFinished.connect(lambda f=field, w=editor: edit(f, int(w.value()) if whole else float(w.value())))
            layout.addWidget(QLabel(label))
            row = QHBoxLayout(); row.addWidget(spin, 1)
            if field in ("motion_blur_samples", "motion_blur_shutter", "motion_blur_offset"):
                curve = raw.get("curve") if isinstance(raw, dict) else None
                def toggle_motion_key(f=field):
                    entries = current()
                    if i >= len(entries): return
                    old = entries[i].get(f, default)
                    if isinstance(old, dict) and old.get("curve") and any(k["frame"] == frame for k in old["curve"]["keys"]):
                        entries[i][f] = delete_scalar_key(old, frame)
                    else:
                        entries[i][f] = set_scalar_key(old, f, frame)
                    save(entries, i); window.rebuild_properties_dock()
                button = window.make_key_button(key, f"shape-{i}-{field}", spin, curve_override=curve,
                    on_toggle=toggle_motion_key, object_name="roto-shape-key-" + field)
                button.setAccessibleName(f"{label} key")
                row.addWidget(button)
            layout.addLayout(row)
        plain_combo("skew_order", "Skew order", ("XY", "YX"), "XY", plain_transform)
        plain_check("feather_on", "Feather on", True)
        plain_spin("falloff_power", "Feather falloff", 0.0, 16.0, 0.1, 1.0)
        plain_spin("motion_blur_samples", "Motion blur samples", 1, 64, 1, 1, whole=True)
        plain_spin("motion_blur_shutter", "Motion blur shutter", 0.0, 8.0, 0.05, 0.5)
        plain_combo("motion_blur_offset_type", "Motion blur shutter offset", shape_model.OFFSET_TYPES, "start")
        plain_spin("motion_blur_offset", "Motion blur custom offset", -8.0, 8.0, 0.05, 0.0)
        life = shape.get("lifetime") or {"type": "all"}
        def set_life(part, value):
            live_entries = current()
            if i >= len(live_entries): return
            now = dict(live_entries[i].get("lifetime") or {"type": "all"})
            now[part] = value
            if now["type"] == "range" and now.get("end", now.get("start", 1)) < now.get("start", 1):
                now["end"] = now["start"]
            if now["type"] == "all":
                now = {"type": "all"}
            plain("lifetime", now)
        life_type = QComboBox(); life_type.setObjectName("roto-shape-lifetime-type")
        for val in shape_model.LIFETIME_TYPES: life_type.addItem(val.replace("_", " ").capitalize(), val)
        life_type.setCurrentIndex(max(0, life_type.findData(life.get("type", "all"))))
        life_from_label = QLabel("From"); life_from_label.setObjectName("roto-shape-lifetime-from-label")
        life_to_label = QLabel("To"); life_to_label.setObjectName("roto-shape-lifetime-to-label")
        def update_lifetime_type(idx):
            kind = life_type.itemData(idx)
            set_life("type", kind)
            from_visible = kind in ("single", "to_end", "range")
            to_visible = kind in ("start_to", "range")
            life_from_label.setVisible(from_visible); spin_from.setVisible(from_visible)
            life_to_label.setVisible(to_visible); spin_to.setVisible(to_visible)
        life_type.currentIndexChanged.connect(update_lifetime_type)
        layout.addWidget(QLabel("Lifetime")); layout.addWidget(life_type)
        spin_from = QSpinBox(); spin_from.setRange(-100000, 100000); spin_from.setObjectName("roto-shape-lifetime-start")
        spin_from.setValue(int(life.get("start", 1))); spin_from.setKeyboardTracking(False)
        spin_from.editingFinished.connect(lambda w=spin_from: set_life("start", int(w.value())))
        spin_to = QSpinBox(); spin_to.setRange(-100000, 100000); spin_to.setObjectName("roto-shape-lifetime-end")
        spin_to.setValue(int(life.get("end", 1))); spin_to.setKeyboardTracking(False)
        spin_to.editingFinished.connect(lambda w=spin_to: set_life("end", int(w.value())))
        for label, widget in ((life_from_label, spin_from), (life_to_label, spin_to)):
            layout.addWidget(label); layout.addWidget(widget)
        kind = life.get("type", "all")
        from_visible = kind in ("single", "to_end", "range")
        to_visible = kind in ("start_to", "range")
        life_from_label.setVisible(from_visible); spin_from.setVisible(from_visible)
        life_to_label.setVisible(to_visible); spin_to.setVisible(to_visible)
        if shape.get("closed", True) is False:
            plain_combo("cap_start", "Start cap", shape_model.END_CAPS, "rounded")
            plain_combo("cap_end", "End cap", shape_model.END_CAPS, "rounded")
        matrix_edit = QLineEdit(", ".join(f"{v:g}" for v in shape["extra_matrix"]) if shape.get("extra_matrix") else "")
        matrix_edit.setObjectName("roto-shape-extra_matrix"); matrix_edit.setPlaceholderText("16 numbers, row by row; empty for none")
        def set_matrix():
            text = matrix_edit.text().strip()
            try:
                values = [float(part) for part in text.replace(",", " ").split()] if text else None
            except ValueError:
                return
            if values is None or len(values) == 16:
                entries = current()
                if i < len(entries):
                    if values is None: entries[i].pop("extra_matrix", None)
                    else: entries[i]["extra_matrix"] = values
                    save(entries, i)
        matrix_edit.editingFinished.connect(set_matrix)
        layout.addWidget(QLabel("Extra matrix")); layout.addWidget(matrix_edit)
        for field,values in (("feather_falloff",("linear","smooth","gaussian")),("blend_mode",shape_model.BLEND_MODES)):
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
            # The linear float picker (DiMo 10/18): the shape's floats go in and come back unclamped.
            rgba=resolved_color(shape.get("color",[1,1,1,1]))
            window.open_color_picker([float(v) for v in rgba], "Roto colour", lambda picked: edit("color", [min(max(float(v), 0.0), 1.0) for v in picked]))   # a shape colour is 0-1
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
    add_tracking_actions(layout, window, key)
    return root


def _layer_transform_controls(window, key, layout, layer, frame, float_slider):
    """The Transform controls of the selected layer: translate, rotate, scale, skew and centre, applied to every shape inside it
    (identity by default). A value already animated is keyed at the current frame; otherwise the edit sets the value."""
    from . import shapes as shape_model
    group = QGroupBox(f"Layer transform: {layer['name']}"); group.setObjectName("roto-layer-transform")
    box = QVBoxLayout(group)
    payload = window.dispatcher.document.get("node_data", {}).get(key, {})
    shown = shape_model.resolve_layer_transform(payload, layer, frame,
                                                shape_model.resolve_shapes({**payload, "layers": [
                                                    {k: v for k, v in l.items() if k != "transform"} for l in payload.get("layers", [])]}, frame))
    name = layer["name"]

    def edit(field, value):
        data = copy.deepcopy(window.dispatcher.document.get("node_data", {}).get(key, {}))
        target = next((l for l in data.get("layers", []) if l["name"] == name), None)
        if target is None:
            return
        transform = target.setdefault("transform", {})
        for centre in ("center_x", "center_y"):           # pin the centre so an edit does not move it afterwards
            transform.setdefault(centre, shown[centre])
        old = transform.get(field, shape_model.TRANSFORM_DEFAULTS[field])
        if isinstance(old, dict) and old.get("curve"):
            curve = copy.deepcopy(old["curve"])
            keys = [entry for entry in curve["keys"] if entry["frame"] != frame]
            keys.append({"frame": frame, "value": float(value)}); keys.sort(key=lambda entry: entry["frame"])
            transform[field] = {"value": float(value), "curve": {**curve, "keys": keys}}
        else:
            transform[field] = float(value)
        window.command({"op": "set_shapes", "id": key, "shapes": data.get("shapes", []), "layers": data["layers"]})

    for field, label, lo, hi, step in (("translate_x", "Translate X", -65536, 65536, 1), ("translate_y", "Translate Y", -65536, 65536, 1),
                                       ("rotate", "Rotate", -360000, 360000, 1), ("scale_x", "Scale X", -1000, 1000, .01),
                                       ("scale_y", "Scale Y", -1000, 1000, .01), ("center_x", "Center X", -65536, 65536, 1),
                                       ("center_y", "Center Y", -65536, 65536, 1), ("skew_x", "Skew X", -89, 89, 1),
                                       ("skew_y", "Skew Y", -89, 89, 1)):
        control = float_slider(shown[field], lo, hi, step, f"roto-layer-transform-{field}")
        spin = control.spin; spin.setKeyboardTracking(False)
        spin.editingFinished.connect(lambda f=field, w=spin: edit(f, float(w.value())))
        box.addWidget(QLabel(label)); box.addWidget(control)
    layout.addWidget(group)
