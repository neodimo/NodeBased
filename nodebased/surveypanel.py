"""Artist-facing Survey list for the CameraSolve3D node (Rendering 13 step L2).

The Survey fixes the scale of a monocular camera solve, and with three positioned points its orientation and
position too. A pick is a 2D point clicked in the viewer on one frame; it may be given a position in the scene
(X Y Z), and a distance between two picks may be typed. The list edits the node's `survey` knob (JSON, see
`camera_solve.parse_survey`); the sentence at the top is the same one the solve reports."""
import json

from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QPushButton, QVBoxLayout, QWidget)

from . import camera_solve


def _format_position(position):
    return " ".join(f"{value:g}" for value in position) if position else ""


def _parse_position(text):
    """`X Y Z` (spaces or commas) as three floats, or None for an empty field; ValueError for anything else."""
    parts = text.replace(",", " ").split()
    if not parts:
        return None
    if len(parts) != 3:
        raise ValueError("a position is three numbers: X Y Z")
    return [float(part) for part in parts]


def build_panel(window, key):
    root = QWidget()
    root.setObjectName("survey-panel")
    layout = QVBoxLayout(root)
    layout.setContentsMargins(0, 0, 0, 0)

    def current():
        return camera_solve.parse_survey(window.dispatcher.document["nodes"][key]["params"].get("survey", ""))

    try:
        survey = current()
        problem = None
    except ValueError as error:
        survey, problem = {"picks": [], "distances": []}, str(error)

    def commit(updated):
        window.command({"op": "set", "id": key, "param": "survey", "value": json.dumps(updated, separators=(",", ":"))})

    status = QLabel(problem and f"Survey unreadable: {problem}" or camera_solve.survey_plan(survey)["message"])
    status.setObjectName("survey-status")
    status.setWordWrap(True)
    layout.addWidget(QLabel("Survey"))
    layout.addWidget(status)

    picks = QListWidget()
    picks.setObjectName("survey-pick-list")
    for index, pick in enumerate(survey["picks"]):
        row = QListWidgetItem()
        picks.addItem(row)
        widget = QWidget()
        row_layout = QHBoxLayout(widget)
        row_layout.setContentsMargins(2, 0, 2, 0)
        label = QLabel(f"{pick['name']}  frame {pick['frame']}  ({pick['x']:.1f}, {pick['y']:.1f})")
        label.setObjectName(f"survey-pick-label-{index}")
        row_layout.addWidget(label, 1)
        position = QLineEdit(_format_position(pick["position"]))
        position.setObjectName(f"survey-pick-position-{index}")
        position.setPlaceholderText("X Y Z (optional)")
        position.setToolTip("Where this point is in the scene, in scene units. Three or more positioned picks fix the "
                            "scale, orientation and position of the solve.")

        def set_position(i=index, field=position):
            updated = current()
            try:
                updated["picks"][i]["position"] = _parse_position(field.text())
            except ValueError as error:
                window._show_command_error(error)
                return
            if updated != current():
                commit(updated)
        position.editingFinished.connect(set_position)
        row_layout.addWidget(position)
        picks.setItemWidget(row, widget)
        row.setSizeHint(widget.sizeHint())
    layout.addWidget(picks)

    controls = QHBoxLayout()
    add = QPushButton("Pick point in viewer…")
    add.setObjectName("survey-pick-add")
    add.setToolTip("View the plate on the frame you want, then click a surveyed point; Esc ends")
    add.clicked.connect(lambda: window.begin_survey_pick(key))
    controls.addWidget(add)
    remove = QPushButton("Delete pick")
    remove.setObjectName("survey-pick-delete")
    remove.setEnabled(bool(survey["picks"]))

    def delete_pick():
        index = picks.currentRow()
        updated = current()
        if 0 <= index < len(updated["picks"]):
            name = updated["picks"][index]["name"]
            del updated["picks"][index]
            updated["distances"] = [d for d in updated["distances"] if name not in (d["a"], d["b"])]
            commit(updated)
    remove.clicked.connect(delete_pick)
    controls.addWidget(remove)
    layout.addLayout(controls)

    layout.addWidget(QLabel("Distances"))
    distances = QListWidget()
    distances.setObjectName("survey-distance-list")
    for entry in survey["distances"]:
        distances.addItem(f"{entry['a']} – {entry['b']}: {entry['distance']:g}")
    layout.addWidget(distances)
    names = [pick["name"] for pick in survey["picks"]]
    names_row, value_row = QHBoxLayout(), QHBoxLayout()
    first, second = QComboBox(), QComboBox()
    first.setObjectName("survey-distance-a")
    second.setObjectName("survey-distance-b")
    for combo in (first, second):
        combo.addItems(names)
    if len(names) > 1:
        second.setCurrentIndex(1)
    value = QDoubleSpinBox()
    value.setObjectName("survey-distance-value")
    value.setDecimals(4)
    value.setRange(0.0, 1000000.0)
    value.setValue(1.0)
    value.setToolTip("The real distance between the two picks, in scene units")
    add_distance = QPushButton("Add")
    add_distance.setObjectName("survey-distance-add")
    add_distance.setEnabled(len(names) > 1)

    def new_distance():
        a, b = first.currentText(), second.currentText()
        if not a or a == b or value.value() <= 0:
            window._show_command_error(ValueError("Pick two different points and a distance above zero"))
            return
        updated = current()
        updated["distances"] = [d for d in updated["distances"] if {d["a"], d["b"]} != {a, b}]
        updated["distances"].append({"a": a, "b": b, "distance": value.value()})
        commit(updated)
    add_distance.clicked.connect(new_distance)
    for widget in (first, second):
        names_row.addWidget(widget)
    value_row.addWidget(value, 1)
    value_row.addWidget(add_distance)
    layout.addLayout(names_row)
    layout.addLayout(value_row)
    drop = QPushButton("Delete distance")
    drop.setObjectName("survey-distance-delete")
    drop.setEnabled(bool(survey["distances"]))

    def delete_distance():
        index = distances.currentRow()
        updated = current()
        if 0 <= index < len(updated["distances"]):
            del updated["distances"][index]
            commit(updated)
    drop.clicked.connect(delete_distance)
    layout.addWidget(drop)
    return root
