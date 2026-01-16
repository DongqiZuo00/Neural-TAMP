from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import networkx as nx

from src.core.graph_schema import Relation


@dataclass(frozen=True)
class ActionSchema:
    thor_action: str
    required_keys: frozenset[str]
    optional_keys: frozenset[str] = frozenset()

    def allowed_keys(self) -> frozenset[str]:
        return self.required_keys | self.optional_keys


class ProcTHORActionAdapter:
    """Adapter for Phase 1 action dictionaries to ProcTHOR controller.step kwargs."""

    SUPPORTED_ACTIONS = {
        "NavigateTo",
        "Open",
        "Close",
        "PickUp",
        "PutObject",
    }

    _ACTION_SCHEMAS = {
        "TeleportFull": ActionSchema(
            thor_action="TeleportFull",
            required_keys=frozenset({"action", "x", "y", "z", "rotation", "horizon", "standing"}),
        ),
        "OpenObject": ActionSchema(
            thor_action="OpenObject",
            required_keys=frozenset({"action", "objectId"}),
        ),
        "CloseObject": ActionSchema(
            thor_action="CloseObject",
            required_keys=frozenset({"action", "objectId"}),
        ),
        "PickupObject": ActionSchema(
            thor_action="PickupObject",
            required_keys=frozenset({"action", "objectId"}),
        ),
        "PutObject": ActionSchema(
            thor_action="PutObject",
            required_keys=frozenset({"action", "objectId"}),
        ),
    }

    _ACTION_MAPPING = {
        "NavigateTo": "TeleportFull",
        "Open": "OpenObject",
        "Close": "CloseObject",
        "PickUp": "PickupObject",
        "PutObject": "PutObject",
    }

    _TARGET_REQUIRED = {
        "NavigateTo",
        "Open",
        "Close",
        "PickUp",
    }

    def validate_action_dict(self, action_dict: dict[str, Any]) -> tuple[bool, str]:
        if not isinstance(action_dict, dict):
            return False, "action_dict must be a dict"

        action = action_dict.get("action")
        if not isinstance(action, str) or not action:
            return False, "action must be a non-empty string"
        if action not in self.SUPPORTED_ACTIONS:
            return False, f"unsupported action: {action}"

        target = action_dict.get("target")
        if action in self._TARGET_REQUIRED:
            if target is None:
                return False, f"{action} requires target"
            if not isinstance(target, str):
                return False, "target must be a string"
        elif target is not None and not isinstance(target, str):
            return False, "target must be a string"

        receptacle_id = action_dict.get("receptacle_id")
        if receptacle_id is not None and not isinstance(receptacle_id, str):
            return False, "receptacle_id must be a string"

        allowed_fields = {"action", "target", "receptacle_id"}
        extra_fields = sorted(set(action_dict.keys()) - allowed_fields)
        if extra_fields:
            return True, f"ignored extra fields: {', '.join(extra_fields)}"

        return True, "ok"

    def to_thor_step(
        self,
        action_dict: dict[str, Any],
        graph_t: nx.DiGraph,
        scene_cache: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], bool, str]:
        valid, reason = self.validate_action_dict(action_dict)
        if not valid:
            return {}, False, reason

        action = action_dict["action"]
        thor_action = self._ACTION_MAPPING[action]

        if thor_action == "TeleportFull":
            target = action_dict.get("target")
            if target not in graph_t.nodes:
                return {}, False, f"target not found in graph: {target}"
            node = graph_t.nodes[target]
            pos = node.get("pos")
            if not pos or len(pos) < 3:
                return {}, False, f"target {target} missing pos"
            reachable = []
            if scene_cache is not None:
                reachable = scene_cache.get("reachable_positions", []) or []
            if not reachable:
                return {}, False, "reachable_positions unavailable for NavigateTo"
            x, y, z = self._nearest_reachable(pos, reachable)
            thor_kwargs = {
                "action": thor_action,
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "rotation": {"x": 0, "y": 0, "z": 0},
                "horizon": 0,
                "standing": True,
            }
            return thor_kwargs, True, "ok"

        if thor_action in {"OpenObject", "CloseObject", "PickupObject"}:
            target = action_dict.get("target")
            thor_kwargs = {"action": thor_action, "objectId": target}
            return thor_kwargs, True, "ok"

        if thor_action == "PutObject":
            target = action_dict.get("target")
            object_id = target or self._find_held_object(graph_t)
            if not object_id:
                return {}, False, "no target or held object for PutObject"
            thor_kwargs = {"action": thor_action, "objectId": object_id}
            if action_dict.get("receptacle_id"):
                return thor_kwargs, True, "ignored receptacle_id for Phase 1"
            return thor_kwargs, True, "ok"

        return {}, False, f"unhandled action mapping: {thor_action}"

    def _find_held_object(self, graph_t: nx.DiGraph) -> str | None:
        robot_id = self._find_robot_id(graph_t)
        if not robot_id:
            return None
        for _, target, data in graph_t.out_edges(robot_id, data=True):
            if data.get("relation") == Relation.HOLDING:
                return target
        return None

    def _find_robot_id(self, graph_t: nx.DiGraph) -> str | None:
        for node_id, data in graph_t.nodes(data=True):
            if data.get("type") == "agent":
                return node_id
        if "robot_agent" in graph_t.nodes:
            return "robot_agent"
        return None

    def allowed_kwargs(self, thor_action: str) -> Iterable[str]:
        schema = self._ACTION_SCHEMAS.get(thor_action)
        if not schema:
            return []
        return schema.allowed_keys()

    def _nearest_reachable(self, target_pos: Any, reachable: list[dict[str, Any]]) -> tuple[float, float, float]:
        tx, ty, tz = target_pos
        best = reachable[0]
        best_dist = float("inf")
        for pos in reachable:
            dx = pos.get("x", 0.0) - tx
            dy = pos.get("y", 0.0) - ty
            dz = pos.get("z", 0.0) - tz
            dist = dx * dx + dy * dy + dz * dz
            if dist < best_dist:
                best = pos
                best_dist = dist
        return float(best.get("x", 0.0)), float(best.get("y", 0.0)), float(best.get("z", 0.0))
