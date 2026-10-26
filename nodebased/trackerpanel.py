"""Artist-facing point list for Tracker and Stabilize nodes."""
import copy

from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QPushButton, QSpinBox, QVBoxLayout, QWidget)

from . import trackerknobs


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
        # Nuke's T, R and S: which of the tracks give translation, rotation and scale (none ticked is all three).
        flags = trackerknobs.track_flags(track)
        for letter, field, on, tip in (("T", "use_t", flags[0], "This track gives the transform's translation"),
                                       ("R", "use_r", flags[1], "This track gives the transform's rotation"),
                                       ("S", "use_s", flags[2], "This track gives the transform's scale")):
            box = QCheckBox(letter)
            box.setObjectName(f"tracker-track-{letter.lower()}-{index}")
            box.setToolTip(tip)
            box.setChecked(on)

            def set_flag(value, i=index, name=field):
                updated = current_tracks()
                if i >= len(updated):
                    return
                effective = trackerknobs.track_flags(updated[i])
                for position, other in enumerate(("use_t", "use_r", "use_s")):
                    updated[i][other] = 1 if effective[position] else 0
                updated[i][name] = 1 if value else 0
                replace_tracks(updated)
            box.toggled.connect(set_flag)
            row_layout.addWidget(box)
        # The track's own pattern and search areas; empty (the node's) until set here or dragged on the viewer.
        pattern_box = QSpinBox()
        pattern_box.setObjectName(f"tracker-track-pattern-{index}")
        pattern_box.setRange(1, 256)
        pattern_box.setToolTip("This track's pattern area (half-width in pixels)")
        search_box = QSpinBox()
        search_box.setObjectName(f"tracker-track-search-{index}")
        search_box.setRange(1, 256)
        search_box.setToolTip("This track's search area (half-width in pixels)")
        node_params = window.dispatcher.document["nodes"][key]["params"]
        own_pattern, own_search = trackerknobs.track_radii(node_params, track)
        pattern_box.setValue(own_pattern)
        search_box.setValue(own_search)

        def set_areas(_value=0, i=index, a=pattern_box, b=search_box):
            updated = current_tracks()
            if i >= len(updated):
                return
            pattern, search = a.value(), max(b.value(), a.value())
            if (updated[i].get("pattern_radius"), updated[i].get("search_radius")) == (pattern, search):
                return
            updated[i].update(pattern_radius=pattern, search_radius=search)
            replace_tracks(updated)
        pattern_box.editingFinished.connect(set_areas)
        search_box.editingFinished.connect(set_areas)
        row_layout.addWidget(pattern_box)
        row_layout.addWidget(search_box)
        analyzed = any(
            isinstance(track.get(field), dict) and track[field].get("curve") and
            any(first <= int(item["frame"]) <= last for item in track[field]["curve"].get("keys", []))
            for field in ("x", "y"))
        row_layout.addWidget(QLabel("Analyzed" if analyzed else "No keys"))
        listing.setItemWidget(row, widget)
    if selected is not None and 0 <= selected < len(tracks):
        listing.setCurrentRow(selected)
    layout.addWidget(QLabel("Track points"))
    listing.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
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
        indices = sorted({listing.row(item) for item in listing.selectedItems()} | ({listing.currentRow()} - {-1}))
        updated = current_tracks()
        kept = trackerknobs.remove_tracks(updated, indices)
        if len(kept) != len(updated):
            window._tracker_selected_index = None
            replace_tracks(kept)
    remove.clicked.connect(delete_selected)
    controls.addWidget(remove)
    layout.addLayout(controls)
    # Nuke's four buttons over the list: add track (a new track at the middle of the picture), delete track (every selected one),
    # select all, and average tracks (a new track at the mean of the selected ones).
    row = QHBoxLayout()
    for name, text, tip in (("tracker-add-track", "Add track", "Add a new track at the middle of the picture"),
                            ("tracker-del-tracks", "Delete tracks", "Delete every selected track"),
                            ("tracker-select-all", "Select all", "Select every track"),
                            ("tracker-average-tracks", "Average tracks", "Add a track that follows the mean of the selected tracks")):
        button = QPushButton(text)
        button.setObjectName(name)
        button.setToolTip(tip)
        row.addWidget(button)
        if name == "tracker-add-track":
            button.clicked.connect(lambda _c=False: window.add_track_at_centre(key))
        elif name == "tracker-del-tracks":
            button.clicked.connect(lambda _c=False: delete_selected())
            button.setEnabled(bool(tracks))
        elif name == "tracker-select-all":
            button.clicked.connect(lambda _c=False: listing.selectAll())
            button.setEnabled(bool(tracks))
        else:
            button.clicked.connect(lambda _c=False: window.average_selected_tracks(
                key, sorted({listing.row(item) for item in listing.selectedItems()})))
            button.setEnabled(len(tracks) > 1)
    layout.addLayout(row)

    def select(row):
        window._tracker_selected_index = row if row >= 0 else None
        window.viewer.viewport().update()
    listing.currentRowChanged.connect(select)
    return root
