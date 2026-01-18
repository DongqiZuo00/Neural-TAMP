from __future__ import annotations

from typing import Any

import networkx as nx

from src.core.graph_schema import Relation, normalize_state
from src.env.action_adapter import ProcTHORActionAdapter


def execute_symbolic_action(
    env,
    adapter: ProcTHORActionAdapter,
    action: dict[str, Any],
    scene_graph: nx.DiGraph,
) -> tuple[bool, str]:
    ok, reason = adapter.validate_action_dict(action)
    if not ok:
        return False, f"INVALID_ACTION_SCHEMA: {reason}"

    action_name = action["action"]

    if action_name == "NavigateTo":
        return _execute_thor_action(env, adapter, action, scene_graph, scene_cache=_scene_cache(env))

    if action_name == "PutObject":
        target = action.get("target")
        if not isinstance(target, str):
            return False, "symbolic_precondition_failed"
        if not _is_holding(scene_graph, target):
            nav_ok, nav_err = execute_symbolic_action(
                env,
                adapter,
                {"action": "NavigateTo", "target": target},
                scene_graph,
            )
            if not nav_ok:
                return False, nav_err
            pickup_ok, pickup_err = execute_symbolic_action(
                env,
                adapter,
                {"action": "PickUp", "target": target},
                scene_graph,
            )
            if not pickup_ok:
                return False, pickup_err

    if not _symbolic_preconditions_met(action, scene_graph):
        return False, "symbolic_precondition_failed"

    return _execute_thor_action(env, adapter, action, scene_graph)


def _symbolic_preconditions_met(action: dict[str, Any], scene_graph: nx.DiGraph) -> bool:
    action_name = action["action"]
    target = action.get("target")
    robot_id = _find_robot_id(scene_graph)
    if not robot_id:
        return False

    if action_name == "PickUp":
        return (
            _has_affordance(scene_graph, target, "pickupable")
            and not _is_holding_any(scene_graph)
            and _is_near(scene_graph, robot_id, target)
        )

    if action_name == "Open":
        return (
            _has_affordance(scene_graph, target, "openable")
            and _open_state(scene_graph, target) == "closed"
            and _is_near(scene_graph, robot_id, target)
        )

    if action_name == "Close":
        return _open_state(scene_graph, target) == "open" and _is_near(scene_graph, robot_id, target)

    if action_name == "PutObject":
        receptacle_id = action.get("receptacle_id")
        return (
            _is_holding(scene_graph, target)
            and _has_affordance(scene_graph, receptacle_id, "receptacle")
            and _is_near(scene_graph, robot_id, receptacle_id)
        )

    return False


def _execute_thor_action(
    env,
    adapter: ProcTHORActionAdapter,
    action: dict[str, Any],
    scene_graph: nx.DiGraph,
    scene_cache: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    if action.get("action") == "PutObject":
        receptacle_id = action.get("receptacle_id")
        if not receptacle_id:
            return False, "missing receptacle_id for PutObject"
        xy = _receptacle_xy_from_metadata(env, receptacle_id)
        if xy is not None:
            thor_kwargs = {
                "action": "PutObject",
                "x": float(xy[0]),
                "y": float(xy[1]),
                "forceAction": True,
            }
            return _step_controller(env, thor_kwargs)

    thor_kwargs, ok, reason = adapter.to_thor_step(action, scene_graph, scene_cache=scene_cache)
    if not ok:
        return False, f"INVALID_ACTION_SCHEMA: {reason}"

    if thor_kwargs.get("action") in {"OpenObject", "CloseObject", "PickupObject", "PutObject"}:
        thor_kwargs["forceAction"] = True

    return _step_controller(env, thor_kwargs)


def _step_controller(env, thor_kwargs: dict[str, Any]) -> tuple[bool, str]:
    try:
        event = env.controller.step(**thor_kwargs)
    except Exception as exc:
        raise RuntimeError(f"API_SCHEMA_BUG: {exc}") from exc
    success = bool(event.metadata.get("lastActionSuccess"))
    error_msg = event.metadata.get("errorMessage") or ""
    return success, error_msg


def _scene_cache(env) -> dict[str, Any]:
    positions = []
    try:
        event = env.controller.step(action="GetReachablePositions")
    except Exception as exc:
        raise RuntimeError(f"API_SCHEMA_BUG: {exc}") from exc
    positions = event.metadata.get("actionReturn") or []
    if not isinstance(positions, list):
        positions = []
    return {"reachable_positions": positions}


def _receptacle_xy_from_metadata(env, receptacle_id: str) -> tuple[float, float] | None:
    try:
        event = env.controller.step(action="GetObjectScreenPosition", objectId=receptacle_id)
    except Exception as exc:
        if "Invalid action" in str(exc):
            return None
        raise RuntimeError(f"API_SCHEMA_BUG: {exc}") from exc
    if not event.metadata.get("lastActionSuccess"):
        return None
    screen_pos = event.metadata.get("actionReturn") or {}
    if "x" in screen_pos and "y" in screen_pos:
        return float(screen_pos["x"]), float(screen_pos["y"])
    return None


def _find_robot_id(scene_graph: nx.DiGraph) -> str | None:
    for node_id, data in scene_graph.nodes(data=True):
        if data.get("type") == "agent":
            return node_id
    if "robot_agent" in scene_graph.nodes:
        return "robot_agent"
    return None


def _is_holding_any(scene_graph: nx.DiGraph) -> bool:
    robot_id = _find_robot_id(scene_graph)
    if not robot_id:
        return False
    for _, target, data in scene_graph.out_edges(robot_id, data=True):
        if data.get("relation") == Relation.HOLDING:
            return True
        if _is_held_state(scene_graph, target):
            return True
    return _any_held_state(scene_graph)


def _is_holding(scene_graph: nx.DiGraph, obj_id: str | None) -> bool:
    if not obj_id:
        return False
    robot_id = _find_robot_id(scene_graph)
    if not robot_id:
        return False
    if scene_graph.has_edge(robot_id, obj_id) and scene_graph.edges[robot_id, obj_id].get("relation") == Relation.HOLDING:
        return True
    return _is_held_state(scene_graph, obj_id)


def _open_state(scene_graph: nx.DiGraph, obj_id: str | None) -> str | None:
    if not obj_id or obj_id not in scene_graph:
        return None
    state = normalize_state(scene_graph.nodes[obj_id].get("state"))
    return state.get("open_state")


def _is_held_state(scene_graph: nx.DiGraph, obj_id: str | None) -> bool:
    if not obj_id or obj_id not in scene_graph:
        return False
    state = normalize_state(scene_graph.nodes[obj_id].get("state"))
    return bool(state.get("held"))


def _any_held_state(scene_graph: nx.DiGraph) -> bool:
    for node_id, data in scene_graph.nodes(data=True):
        if data.get("type") != "object":
            continue
        state = normalize_state(data.get("state"))
        if state.get("held"):
            return True
    return False


def _has_affordance(scene_graph: nx.DiGraph, obj_id: str | None, affordance: str) -> bool:
    if not obj_id or obj_id not in scene_graph:
        return False
    raw_node = scene_graph.nodes[obj_id].get("raw_node")
    geometry = raw_node.geometry if raw_node and isinstance(raw_node.geometry, dict) else {}
    return bool(geometry.get(affordance))


def _is_near(scene_graph: nx.DiGraph, robot_id: str, obj_id: str | None) -> bool:
    if not obj_id or obj_id not in scene_graph:
        return False
    if scene_graph.has_edge(robot_id, obj_id) and scene_graph.edges[robot_id, obj_id].get("relation") == Relation.NEAR:
        return True
    if scene_graph.has_edge(obj_id, robot_id) and scene_graph.edges[obj_id, robot_id].get("relation") == Relation.NEAR:
        return True
    return False
