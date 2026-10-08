"""Artist-facing point list for Tracker and Stabilize nodes."""
import copy

from PySide6.QtWidgets import (QCheckBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QPushButton, QVBoxLayout, QWidget)


def build_panel(window, key):
    root = QWidget()
    root.setObjectName("tracker-points-panel")
    layout = QVBoxLayout(root)
    layout.setContentsMargins(0, 0, 0, 0)
    tracks = copy.deepcopy(window.dispatcher.document.get("node_data", {}).get(key, {}).get("tracks", []))
    listing = QListWidget()
    listing.setObjectName("tracker-track-list")

    def current_tracks():
        return copy.deepcopy(window.dispatcher.document.get("node_data", {}).get(key, {}).get("tracks", []))

    def replace_tracks(updated):
        commands = [{"op": "set_tracks", "id": key, "tracks": updated}]
        old_names = {item["name"] for item in current_tracks()} - {item["name"] for item in updated}
        if old_names:
            document = window.dispatcher.document
            for roto_id, node in document["nodes"].items():
                if node["type"] != "Roto":
                    continue
                payload = copy.deepcopy(document.get("node_data", {}).get(roto_id, {}))
                changed = False
                for shape in payload.get("shapes", []):
                    link = shape.get("track_link")
                    if link and link.get("tracker_id") == key and link.get("track_name") in old_names:
                        shape["track_link"] = None
                        changed = True
                if changed:
                    commands.append({"op": "set_shapes", "id": roto_id, "shapes": payload["shapes"]})
        window.command({"op": "batch", "commands": commands})

    def rename(index, name):
        updated = current_tracks()
        if not 0 <= index < len(updated):
            return
        name = name.strip()
        if not name:
            return
        old = updated[index]["name"]
        if name == old:
            return
        updated[index]["name"] = name
        commands = [{"op": "set_tracks", "id": key, "tracks": updated}]
        document = window.dispatcher.document
        for roto_id, node in document["nodes"].items():
            if node["type"] != "Roto":
                continue
            payload = copy.deepcopy(document.get("node_data", {}).get(roto_id, {}))
            changed = False
            for shape in payload.get("shapes", []):
                link = shape.get("track_link")
                if link and link.get("tracker_id") == key and link.get("track_name") == old:
                    link["track_name"] = name
                    changed = True
            if changed:
                commands.append({"op": "set_shapes", "id": roto_id, "shapes": payload["shapes"]})
        window.command({"op": "batch", "commands": commands})

    timeline = window.dispatcher.document["time"]
    first, last = int(timeline["first"]), int(timeline["last"])
    selected = getattr(window, "_tracker_selected_index", None)
    for index, track in enumerate(tracks):
        row = QListWidgetItem()
        listing.addItem(row)
        widget = QWidget()
        row_layout = QHBoxLayout(widget)
        row_layout.setContentsMargins(2, 0, 2, 0)
        name = QLineEdit(track["name"])
        name.setObjectName(f"tracker-track-name-{index}")
        name.setMinimumWidth(name.fontMetrics().horizontalAdvance(track["name"]) + 12)
        name.setToolTip("Track point name")
        name.editingFinished.connect(lambda i=index, field=name: rename(i, field.text()))
        row_layout.addWidget(name, 1)
        enabled = QCheckBox("Enabled")
        enabled.setObjectName(f"tracker-track-enabled-{index}")
        enabled.setChecked(bool(track.get("enabled", 1)))
        def set_enabled(value, i=index):
            updated = current_tracks()
            if i < len(updated):
                updated[i]["enabled"] = 1 if value else 0
                replace_tracks(updated)
        enabled.toggled.connect(set_enabled)
        row_layout.addWidget(enabled)
        analyzed = any(
            isinstance(track.get(field), dict) and track[field].get("curve") and
            any(first <= int(item["frame"]) <= last for item in track[field]["curve"].get("keys", []))
            for field in ("x", "y"))
        row_layout.addWidget(QLabel("Analyzed" if analyzed else "No keys"))
        listing.setItemWidget(row, widget)
    if selected is not None and 0 <= selected < len(tracks):
        listing.setCurrentRow(selected)
    layout.addWidget(QLabel("Track points"))
    layout.addWidget(listing)
    controls = QHBoxLayout()
    add = QPushButton("Add point…")
    add.setObjectName("tracker-track-add")
    add.clicked.connect(lambda: window.begin_tracker_pick(key))
    controls.addWidget(add)
    remove = QPushButton("Delete point")
    remove.setObjectName("tracker-track-delete")
    remove.setEnabled(bool(tracks))
    def delete_selected():
        index = listing.currentRow()
        updated = current_tracks()
        if 0 <= index < len(updated):
            del updated[index]
            replace_tracks(updated)
    remove.clicked.connect(delete_selected)
    controls.addWidget(remove)
    layout.addLayout(controls)

    def select(row):
        window._tracker_selected_index = row if row >= 0 else None
        window.viewer.viewport().update()
    listing.currentRowChanged.connect(select)
    return root
