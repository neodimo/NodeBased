"""Write the shipped "Hot pour" preset (nodebased/data/presets/fluids/hot_pour.json) from `nodebased.hotpour`.

    python tools/make_hot_pour_preset.py

The preset file is generated, never hand-edited: tests/test_hot_pour.py fails when it differs from what this writes.
"""
import json
from pathlib import Path

from nodebased import hotpour

TARGET = Path(__file__).resolve().parents[1] / "nodebased" / "data" / "presets" / "fluids" / "hot_pour.json"
DESCRIPTION = ("Hot liquid poured from a moving spout into a glass on a table, steam rising off its surface. FLIP liquid "
               "at up to 256 cells with surface tension, an open top and whitewater; steam at up to 192 cells that takes "
               "the liquid as its heat source and collider. Set the Time row to 1 - 120.")


def manifest():
    return {"name": "Hot pour", "category": "Fluids", "description": DESCRIPTION, "thumbnail": "hot_pour.png",
            "ops": hotpour.hot_pour_ops()}


if __name__ == "__main__":
    TARGET.write_text(json.dumps(manifest(), indent=2) + "\n")
    print("wrote", TARGET)
