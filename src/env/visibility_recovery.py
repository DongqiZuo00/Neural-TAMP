from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx

from src.env.action_adapter import ProcTHORActionAdapter
from src.env.action_visibility import ensure_object_visible
from scripts.data_collection import execute_action


@dataclass(frozen=True)
class VisibilityRecoveryResult:
    ok: bool
    reason: str


def ensure_visible_or_navigate(
    env,
    adapter: ProcTHORActionAdapter,
    target_id: str,
    graph: nx.DiGraph,
    scene_cache: dict[str, Any] | None = None,
) -> VisibilityRecoveryResult:
    if not target_id:
        return VisibilityRecoveryResult(False, "missing_target")
    if target_id not in graph.nodes:
        return VisibilityRecoveryResult(False, f"target_not_in_graph:{target_id}")

    if ensure_object_visible(env.controller, target_id):
        return VisibilityRecoveryResult(True, "visible")

    nav_action = {"action": "NavigateTo", "target": target_id}
    success, error_msg = execute_action(env, adapter, nav_action, graph, scene_cache=scene_cache)
    if not success:
        return VisibilityRecoveryResult(False, error_msg or "navigate_failed")

    if ensure_object_visible(env.controller, target_id):
        return VisibilityRecoveryResult(True, "visible_after_navigate")

    return VisibilityRecoveryResult(False, "object_not_visible_after_navigation_scan")
