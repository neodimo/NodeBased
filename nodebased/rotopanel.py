"""Roto shape-list and per-shape controls, isolated from the shared properties builder."""
import copy
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
                               QPushButton, QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit,
                               QColorDialog, QLabel)


def build_panel(window, key):
    root = QWidget(); root.setObjectName("roto-shapes-panel")
    layout = QVBoxLayout(root); layout.setContentsMargins(0, 0, 0, 0)
    shapes = copy.deepcopy(window.dispatcher.document.get("node_data", {}).get(key, {}).get("shapes", []))
    selected = getattr(window.viewer, "roto_selected_shape_index", -1)
    if selected < 0 and shapes:
        selected = window.viewer.roto_selected_shape_index = 0
    listing = QListWidget(); listing.setObjectName("roto-shape-list")
    for index, shape in enumerate(shapes):
        row = QListWidgetItem(); listing.addItem(row)
        widget = QWidget(); row_layout = QHBoxLayout(widget); row_layout.setContentsMargins(2, 0, 2, 0)
        label = QLineEdit(shape["name"]); label.setObjectName(f"roto-shape-row-name-{index}")
        label.setFrame(False); label.editingFinished.connect(
            lambda idx=index, field=label: row_edit(idx, "name", field.text()))
        row_layout.addWidget(label, 1)
        for field, caption in (("visible", "V"), ("locked", "L"), ("invert", "I")):
            toggle = QCheckBox(caption); toggle.setToolTip({"visible":"Shape visibility", "locked":"Lock shape points", "invert":"Invert shape matte"}[field])
            toggle.setObjectName(f"roto-shape-row-{field}-{index}")
            toggle.setChecked(shape.get(field, field == "visible"))
            toggle.toggled.connect(lambda value, idx=index, f=field: row_edit(idx, f, bool(value)))
            row_layout.addWidget(toggle)
        listing.setItemWidget(row, widget)
    if 0 <= selected < len(shapes): listing.setCurrentRow(selected)
    layout.addWidget(QLabel("Curves")); layout.addWidget(listing)

    def current():
        return copy.deepcopy(window.dispatcher.document.get("node_data", {}).get(key, {}).get("shapes", []))
    def row_edit(index, field, value):
        entries=current()
        if index < len(entries): entries[index][field]=value; save(entries,index)
    def save(entries, select=None):
        window.command({"op": "set_shapes", "id": key, "shapes": entries})
        if select is not None: window.viewer.roto_selected_shape_index = select
    def select(row):
        window.viewer.roto_selected_shape_index = row
        window.viewer.viewport().update()
        window.rebuild_properties_dock()
    listing.currentRowChanged.connect(select)
    buttons = QHBoxLayout()
    for label, delta in (("↑", -1), ("↓", 1)):
        b = QPushButton(label); b.setToolTip("Move shape"); b.setObjectName("roto-shape-up" if delta < 0 else "roto-shape-down")
        def move(_=False, d=delta):
            entries=current(); i=listing.currentRow(); j=i+d
            if 0 <= i < len(entries) and 0 <= j < len(entries): entries[i],entries[j]=entries[j],entries[i]; save(entries,j)
        b.clicked.connect(move); buttons.addWidget(b)
    for label, action in (("Duplicate", "duplicate"), ("Delete", "delete")):
        b=QPushButton(label); b.setObjectName("roto-shape-"+action)
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
            if i < len(entries): entries[i][field]=value; save(entries,i)
        name=QLineEdit(shape["name"]); name.setObjectName("roto-shape-name")
        name.editingFinished.connect(lambda: edit("name",name.text())); layout.addWidget(name)
        for field,label in (("locked","Lock"),("invert","Invert"),("motion_blur","Motion blur")):
            cb=QCheckBox(label); cb.setObjectName("roto-shape-"+field); cb.setChecked(shape.get(field,False)); cb.toggled.connect(lambda v,f=field:edit(f,bool(v))); layout.addWidget(cb)
        for field,label,lo,hi,step in (("feather","Feather",0,500,0.5),("opacity","Opacity",0,1,0.05)):
            spin=QDoubleSpinBox(); spin.setObjectName("roto-shape-"+field); spin.setRange(lo,hi); spin.setSingleStep(step); spin.setValue(float(shape.get(field,0 if field=="feather" else 1))); spin.setKeyboardTracking(False); spin.editingFinished.connect(lambda f=field,w=spin:edit(f,float(w.value()))); layout.addWidget(QLabel(label)); layout.addWidget(spin)
        for field,values in (("feather_falloff",("linear","smooth","gaussian")),("blend_mode",("over","plus","minus","multiply"))):
            combo=QComboBox(); combo.setObjectName("roto-shape-"+field)
            for val in values: combo.addItem(val.title(),val)
            combo.setCurrentIndex(combo.findData(shape.get(field,"linear" if field=="feather_falloff" else "over")))
            combo.currentIndexChanged.connect(lambda idx,f=field,c=combo:edit(f,c.itemData(idx))); layout.addWidget(QLabel(field.replace("_"," ").title())); layout.addWidget(combo)
        color=QPushButton("Choose colour…"); color.setObjectName("roto-shape-color")
        def choose():
            rgba=shape.get("color",[1,1,1,1]); picked=QColorDialog.getColor(QColor.fromRgbF(*rgba),window,"Roto colour",QColorDialog.ColorDialogOption.ShowAlphaChannel)
            if picked.isValid(): edit("color",[picked.redF(),picked.greenF(),picked.blueF(),picked.alphaF()])
        color.clicked.connect(choose); layout.addWidget(color)
    draw=QPushButton("Draw shape…"); draw.setObjectName("roto-draw-shape"); draw.clicked.connect(lambda:window.begin_roto_draw(key)); layout.addWidget(draw)
    return root
