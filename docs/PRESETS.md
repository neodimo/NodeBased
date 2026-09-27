# Presets

A preset is one JSON file describing a wired sub-graph. It appears in the NODES dock's "Presets"
browser (search, category filter, thumbnails) and on the radial menu; placing it drops the
sub-graph into the current document and, where its `ops` reference `$selected`, wires it onto
whatever is selected.

Backed by `nodebased/presets.py`. Shipped presets live under `nodebased/data/presets/<area>/*.json`
(one sub-folder per contributing area, e.g. `particles/`, and later `fluids/` for lane 6's presets
so they show up in the same browser without any code change). User presets live in
`presets.user_presets_directory()`, the same per-machine app-data location `radialcommands` uses
for user radial commands.

## File format

```json
{
  "name": "Sparks",
  "category": "Particles",
  "description": "Short-lived, fast particles that bounce off the ground as instanced streaks.",
  "thumbnail": "sparks.png",
  "ops": [
    {"op": "create", "id": "$new:emit", "type": "ParticleEmitter3D", "params": {"emit_rate": 200.0}},
    {"op": "create", "id": "$new:gravity", "type": "ParticleGravity3D", "params": {}},
    {"op": "connect", "id": "$new:gravity", "input": "particles", "source": "$new:emit"}
  ]
}
```

- `name` (required, non-empty string): shown in the browser and the radial menu.
- `category` (required, non-empty string): groups presets in the browser and the category filter
  combo. Free text; every distinct value seen becomes a filterable category.
- `description` (optional string, default `""`): shown as a tooltip and matched by the browser's
  search box, alongside `name`.
- `thumbnail` (optional string): a filename resolved next to the manifest -- a small PNG. Absent,
  the browser falls back to a generic icon for the category.
- `ops` (required, non-empty list): the same op vocabulary `core.Dispatcher` accepts (`create`,
  `connect`, `set`, ...), with the same placeholders a user radial command's `ops` body accepts
  (`nodebased/radialcommands.py`):
  - `$selected` -- the whole selection, as a list of node ids.
  - `$selected[N]` -- the Nth selected node id (negative indices count from the end).
  - `$downstream_of(ID)` -- every node id downstream of `ID` (`ID` may itself be a placeholder).
  - `$new:NAME` -- a fresh node id, the same value every time `$new:NAME` appears in one preset's
    `ops`, so a later op can wire onto a node an earlier op created.

  A `create` op with no `pos` of its own lands near wherever the preset was dropped, offset so
  several creates in one preset do not stack on top of each other -- exactly a radial command's
  placement rule.

Placing a preset with nothing selected simply skips any op whose placeholder has nothing to
resolve to (an empty `$selected` list, an out-of-range `$selected[N]`); write presets so their
non-selection-dependent ops (the sub-graph itself) still make sense on their own, and reserve
`$selected`/`$selected[N]` for the parts that only matter when something is selected (e.g. wiring
a new collider's `geometry` input onto the selected mesh).

## Saving a selection as a preset

`presets.capture_selection_ops(nodes, selected_ids)` turns a live selection into an `ops` body:
every selected node becomes a `create` with a fresh `$new:` id, every wire between two selected
nodes is preserved, and -- when the selection forms one unbranched chain -- an input the chain's
root reads from outside the selection becomes `$selected[0]`, so the saved preset re-wires its
root onto whatever is selected the next time it is placed. `presets.save_selection_as_preset`
writes the resulting manifest into the user presets directory (or any directory a caller names)
and validates it before returning, the same way `radialcommands.save_ops_command` does for user
commands.

## Adding presets for another area

A new sub-folder under `nodebased/data/presets/` (or a document naming a different `category`)
needs no code change: `presets.load_all` walks every shipped and user directory and
`presets.categories`/`presets.search` work over whatever `category` values the files declare. Lane
6 (fluids) should add its own presets this way once fluid nodes exist, rather than special-casing
fluids anywhere in `presets.py`.
