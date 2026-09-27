"""Shelf tools for particles (DiMo 9/27, "particle artist tools"): pure ops-builders that act on
the selected 3D nodes. Each is exposed as a built-in radial/dock command
(`nodebased/radialcommands.py` `DEFAULT_COMMANDS`), so it shows up on the ring and in Preferences
-> Radial commands, editable there like any other command, the same way the two worked examples
already in `DEFAULT_COMMANDS` are.

Every function here takes the graph's `nodes` (document-shaped, as `window.graph_nodes()` gives a
radial command script), the selected node ids, and a `new_id` callable (a script's `ctx.new_id`,
or `uuid.uuid4().hex[:12]` in a test) and returns a plain ops list ready for one `batch` command --
never touching a document or a window directly, so it is tested the same way `presets.build_ops`
is.
"""
from __future__ import annotations

from typing import Callable, Dict, FrozenSet, List

from .radialcommands import consumers_of

FORCE_KINDS = ("ParticleWind3D", "ParticleTurbulence3D", "ParticleDrag3D")


def emit_particles_from_selected(nodes: Dict[str, dict], selected_ids: List[str],
                                 new_id: Callable[[], str]) -> list:
    """One `ParticleEmitter3D` per selected node, wired to emit from that node's surface."""
    ops = []
    for node_id in selected_ids:
        emitter_id = new_id()
        ops.append({"op": "create", "id": emitter_id, "type": "ParticleEmitter3D",
                   "params": {"emit_from": "surface"}})
        ops.append({"op": "connect", "id": emitter_id, "input": "geo", "source": node_id})
    return ops


def make_selected_a_particle_collider(nodes: Dict[str, dict], selected_ids: List[str],
                                      new_id: Callable[[], str],
                                      animated_ids: FrozenSet[str] = frozenset()) -> list:
    """One `ParticleBounce3D` per selected node, wired to collide against it. `animated` is set on
    a collider whose node id is in `animated_ids` -- the caller's own read of whether that node
    carries motion (a transform with keyframes, say); this module has no access to a document's
    animation curves, only the plain node dict `window.graph_nodes()` gives a radial command, so
    the caller (the NODES dock button, which does have the document) supplies it."""
    ops = []
    for node_id in selected_ids:
        collider_id = new_id()
        params = {"animated": 1} if node_id in animated_ids else {}
        ops.append({"op": "create", "id": collider_id, "type": "ParticleBounce3D", "params": params})
        ops.append({"op": "connect", "id": collider_id, "input": "geometry", "source": node_id})
    return ops


def scatter_instances_on_selected(nodes: Dict[str, dict], selected_ids: List[str],
                                  new_id: Callable[[], str]) -> list:
    """One `Instance3D` per selected node, scattering a small default `Sphere3D` over its
    vertices: a static emission (the points are the node's own geometry, never a particle
    stream)."""
    ops = []
    for node_id in selected_ids:
        instance_id, mesh_id = new_id(), new_id()
        ops.append({"op": "create", "id": mesh_id, "type": "Sphere3D",
                   "params": {"sphere_radius": 0.05, "rows": 4, "columns": 8}})
        ops.append({"op": "create", "id": instance_id, "type": "Instance3D", "params": {}})
        ops.append({"op": "connect", "id": instance_id, "input": "points", "source": node_id})
        ops.append({"op": "connect", "id": instance_id, "input": "instance", "source": mesh_id})
    return ops


def add_force_to_selected_particles(nodes: Dict[str, dict], selected_ids: List[str], kind: str,
                                    new_id: Callable[[], str]) -> list:
    """A `kind` force (one of `FORCE_KINDS`) spliced in right after each selected particle-stream
    node: every consumer that reads "particles" from the selected node is rewired to read from the
    new force instead, and the new force reads from the selected node -- inserted mid-chain, not
    appended as a dead end nothing downstream sees."""
    if kind not in FORCE_KINDS:
        raise ValueError(f"not a force kind: {kind!r}")
    ops = []
    for node_id in selected_ids:
        force_id = new_id()
        ops.append({"op": "create", "id": force_id, "type": kind, "params": {}})
        ops.append({"op": "connect", "id": force_id, "input": "particles", "source": node_id})
        for consumer_id in consumers_of(nodes, node_id):
            consumer = nodes[consumer_id]
            for slot, source in (consumer.get("inputs") or {}).items():
                if source == node_id and slot == "particles":
                    ops.append({"op": "connect", "id": consumer_id, "input": slot, "source": force_id})
    return ops
