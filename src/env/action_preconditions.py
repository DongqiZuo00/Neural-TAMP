from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx

from src.core.graph_schema import Relation, normalize_state


@dataclass(frozen=True)
class PreconditionResult:
    ok: bool
    reason: str


def check_action_preconditions(action: dict[str, Any], graph: nx.DiGraph) -> PreconditionResult:
    action_name = action.get("action")
    if not isinstance(action_name, str):
        return PreconditionResult(False, "missing_action_name")

    if action_name == "NavigateTo":
        target = action.get("target")
        if target and target not in graph.nodes:
            return PreconditionResult(False, f"target_not_in_graph:{target}")
        return PreconditionResult(True, "ok")

    if action_name in {"Open", "Close"}:
        target = action.get("target")
        if not target or target not in graph.nodes:
            return PreconditionResult(False, f"target_not_in_graph:{target}")
        if not _is_openable(graph, target):
            return PreconditionResult(False, f"target_not_openable:{target}")
        return PreconditionResult(True, "ok")

    if action_name == "PickUp":
        target = action.get("target")
        if not target or target not in graph.nodes:
            return PreconditionResult(False, f"target_not_in_graph:{target}")
        if not _is_pickupable(graph, target):
            return PreconditionResult(False, f"target_not_pickupable:{target}")
        if _is_held(graph, target):
            return PreconditionResult(False, f"target_already_held:{target}")
        return PreconditionResult(True, "ok")

    if action_name == "PutObject":
        receptacle_id = action.get("receptacle_id")
        if not receptacle_id or receptacle_id not in graph.nodes:
            return PreconditionResult(False, f"receptacle_not_in_graph:{receptacle_id}")
        if not _is_receptacle(graph, receptacle_id):
            return PreconditionResult(False, f"receptacle_not_receptacle:{receptacle_id}")
        if _is_openable(graph, receptacle_id) and _is_closed(graph, receptacle_id):
            return PreconditionResult(False, f"receptacle_closed:{receptacle_id}")

        target = action.get("target")
        if target:
            if target not in graph.nodes:
                return PreconditionResult(False, f"target_not_in_graph:{target}")
            if not _is_pickupable(graph, target):
                return PreconditionResult(False, f"target_not_pickupable:{target}")
        elif not _find_held_object(graph):
            return PreconditionResult(False, "no_held_object_for_put")
        return PreconditionResult(True, "ok")

    return PreconditionResult(True, "ok")


def _node_geometry(graph: nx.DiGraph, node_id: str) -> dict[str, Any]:
    data = graph.nodes[node_id]
    raw_node = data.get("raw_node")
    if raw_node and isinstance(getattr(raw_node, "geometry", None), dict):
        return raw_node.geometry
    geometry = data.get("geometry")
    if isinstance(geometry, dict):
        return geometry
    return {}


def _node_state(graph: nx.DiGraph, node_id: str) -> dict[str, Any]:
    data = graph.nodes[node_id]
    return normalize_state(data.get("state"))


def _is_openable(graph: nx.DiGraph, node_id: str) -> bool:
    return bool(_node_geometry(graph, node_id).get("openable"))


def _is_pickupable(graph: nx.DiGraph, node_id: str) -> bool:
    return bool(_node_geometry(graph, node_id).get("pickupable"))


def _is_receptacle(graph: nx.DiGraph, node_id: str) -> bool:
    return bool(_node_geometry(graph, node_id).get("receptacle"))


def _is_closed(graph: nx.DiGraph, node_id: str) -> bool:
    state = _node_state(graph, node_id)
    return state.get("open_state") == "closed"


def _is_held(graph: nx.DiGraph, node_id: str) -> bool:
    state = _node_state(graph, node_id)
    return bool(state.get("held"))


def _find_held_object(graph: nx.DiGraph) -> str | None:
    robot_id = _find_robot_id(graph)
    if not robot_id:
        return None
    for _, target, data in graph.out_edges(robot_id, data=True):
        if data.get("relation") == Relation.HOLDING:
            return target
    return None


def _find_robot_id(graph: nx.DiGraph) -> str | None:
    for node_id, data in graph.nodes(data=True):
        if data.get("type") == "agent":
            return node_id
    if "robot_agent" in graph.nodes:
        return "robot_agent"
    return None
