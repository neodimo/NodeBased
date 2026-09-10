"""Neutral charcoal surfaces; limited, legible accents identify node families."""
COLORS = {"Read": "#d9b879", "Checker": "#d9b879", "Constant": "#d9b879",
          "Grade": "#83cbb7", "ColorCorrect": "#6fc9b0", "Blur": "#79c7d9",
          "Transform": "#89aff0", "Crop": "#6f9be0", "Shuffle": "#a889d9",
          "Merge": "#bd9ee3", "Dot": "#b8a6db", "Switch": "#c6a5db",
          "Viewer": "#a2a2ac"}
STYLE = """
QMainWindow, QWidget { background: #242426; color: #e4e4e7; font: 12px 'Inter', 'Segoe UI', sans-serif; }
QMenuBar, QMenu, QToolBar { background: #29292c; border: 0; }
QMenu::item:selected { background: #3d3d42; }
QDockWidget { font-weight: 600; }
QDockWidget::title { background: #2e2e31; padding: 9px; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background: #1b1b1d; border: 1px solid #414146; border-radius: 5px; padding: 5px; selection-background-color: #477f75; }
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border: 1px solid #83cbb7; }
QPushButton { background: #343438; border: 1px solid #49494f; border-radius: 5px; padding: 6px 12px; }
QPushButton:hover { background: #414146; border-color: #777780; }
QPushButton:pressed { background: #28282b; }
QPushButton:disabled { color: #85858f; }
QPushButton#update { color: #9edccb; border-color: #476b62; background: #2b3532; }
QPushButton#update:hover { background: #344940; }
QToolButton { padding: 7px; }
QToolButton:hover { background: #3c3c40; }
QSplitter::handle { background: #3b3b3f; height: 4px; width: 4px; }
QStatusBar { background: #1c1c1e; color: #aaaab4; }
QLabel#muted { color: #a1a1aa; }
QLabel#brand { color: #91d5c2; font-size: 16px; font-weight: 700; padding: 6px; }
QScrollBar:vertical { background: #202022; width: 10px; }
QScrollBar::handle:vertical { background: #4b4b51; min-height: 25px; }
"""
