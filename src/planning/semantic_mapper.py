from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.core.graph_schema import SceneGraph


@dataclass(frozen=True)
class MappingResult:
    actions: list[dict]
    corrections: list[dict]


_GENERIC_SYNONYMS = {
    "shelf": {"tvstand", "bookshelf", "shelf", "cabinet"},
    "stand": {"tvstand", "nightstand", "stand"},
    "table": {"table", "diningtable", "coffeetable", "sidetable"},
    "counter": {"countertop", "counter"},
    "container": {"cabinet", "drawer", "fridge", "microwave", "box"},
}


def map_actions_to_scene(
    actions: Iterable[dict],
    scene_graph: SceneGraph,
) -> MappingResult:
    """Map action targets to valid scene graph object IDs.

    Only rewrites IDs that are not present in the scene graph.
    """
    valid_ids = set(scene_graph.nodes.keys())
    label_index = _build_label_index(scene_graph)
    affordance_index = _build_affordance_index(scene_graph)
    mapped_actions: list[dict] = []
    corrections: list[dict] = []

    for action in actions:
        mapped = dict(action)
        action_name = mapped.get("action", "")
        for key in ("target", "receptacle_id"):
            raw_id = mapped.get(key)
            if not isinstance(raw_id, str) or not raw_id:
                continue
            if raw_id in valid_ids:
                continue
            candidate = _resolve_id(
                raw_id=raw_id,
                action_name=action_name,
                label_index=label_index,
                affordance_index=affordance_index,
                valid_ids=valid_ids,
            )
            if candidate and candidate != raw_id:
                mapped[key] = candidate
                corrections.append(
                    {
                        "field": key,
                        "action": action_name,
                        "from": raw_id,
                        "to": candidate,
                    }
                )
        mapped_actions.append(mapped)

    return MappingResult(actions=mapped_actions, corrections=corrections)


def _build_label_index(scene_graph: SceneGraph) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for node_id, node in scene_graph.nodes.items():
        label = (node.label or "").strip().lower()
        if not label:
            continue
        index.setdefault(label, []).append(node_id)
    return index


def _build_affordance_index(scene_graph: SceneGraph) -> dict[str, set[str]]:
    affordances = {"openable": set(), "pickupable": set(), "receptacle": set()}
    for node_id, node in scene_graph.nodes.items():
        geometry = node.geometry or {}
        if geometry.get("openable"):
            affordances["openable"].add(node_id)
        if geometry.get("pickupable"):
            affordances["pickupable"].add(node_id)
        if geometry.get("receptacle"):
            affordances["receptacle"].add(node_id)
    return affordances


def _resolve_id(
    raw_id: str,
    action_name: str,
    label_index: dict[str, list[str]],
    affordance_index: dict[str, set[str]],
    valid_ids: set[str],
) -> str | None:
    label_hint = _extract_label_hint(raw_id)
    if label_hint:
        direct = label_index.get(label_hint)
        if direct:
            return _pick_by_affordance(action_name, direct, valid_ids, affordance_index)

        generic = _expand_generic_labels(label_hint)
        for label in generic:
            candidates = label_index.get(label)
            if candidates:
                return _pick_by_affordance(action_name, candidates, valid_ids, affordance_index)

    # Fallback: if raw_id looks like a label, try substring match in IDs
    raw_lower = raw_id.lower()
    candidates = [node_id for node_id in valid_ids if raw_lower in node_id.lower()]
    if candidates:
        return _pick_by_affordance(action_name, candidates, valid_ids, affordance_index)
    return None


def _extract_label_hint(raw_id: str) -> str:
    if "|" in raw_id:
        label = raw_id.split("|", 1)[0]
    else:
        label = raw_id
    return label.strip().lower()


def _expand_generic_labels(label: str) -> list[str]:
    expanded = set()
    for key, synonyms in _GENERIC_SYNONYMS.items():
        if label == key or label in synonyms:
            expanded.update(synonyms)
    if not expanded:
        expanded.add(label)
    return sorted(expanded)


def _pick_by_affordance(
    action_name: str,
    candidates: list[str],
    valid_ids: set[str],
    affordance_index: dict[str, set[str]],
) -> str:
    # Prefer stable ordering for repeatability.
    candidates_sorted = sorted(set(candidates), key=str)
    required = _required_affordance(action_name)
    if required:
        filtered = [c for c in candidates_sorted if c in affordance_index.get(required, set())]
        if filtered:
            candidates_sorted = filtered
    for candidate in candidates_sorted:
        if candidate in valid_ids:
            return candidate
    return candidates_sorted[0]


def _required_affordance(action_name: str) -> str | None:
    if action_name in {"Open", "Close"}:
        return "openable"
    if action_name in {"PickUp"}:
        return "pickupable"
    if action_name in {"PutObject"}:
        return "receptacle"
    return None
