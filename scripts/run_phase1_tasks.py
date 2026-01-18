import argparse
import copy
import json
import os
import random
from collections import Counter
from datetime import datetime
from typing import Any

import numpy as np
import networkx as nx

from scripts.data_collection import (
    execute_action,
    export_graph_canonical,
    get_reachable_positions,
    graph_diff,
    sanitize_graph,
)
from src.core.graph_schema import Relation, normalize_state
from src.env.action_adapter import ProcTHORActionAdapter
from src.memory.graph_manager import GraphManager
from src.perception.oracle_interface import OracleInterface


TASK_TYPES = [
    "A1_PICK_NAVIGATE_ROOM",
    "A2_PICK_PUT",
    "A3_PICK_OPEN",
    "A5_OPEN_PICK",
    "A6_OPEN_PUT",
    "A7_OPEN_CLOSE",
    "A8_OPEN_NAVIGATE_ROOM",
    "A9_NAVIGATE_OBJ_PICK",
    "A10_NAVIGATE_OBJ_OPEN",
    "A11_NAVIGATE_ROOM_NAVIGATE_ROOM",
    "A12_NAVIGATE_OBJ_PUT",
    "A13_PICK_CLOSE",
    "A14_CLOSE_OPEN",
    "A15_PICK_NAVIGATE_OBJ",
]

TASK_MAX_STEPS = {
    "A1_PICK_NAVIGATE_ROOM": 4,
    "A2_PICK_PUT": 6,
    "A3_PICK_OPEN": 6,
    "A5_OPEN_PICK": 6,
    "A6_OPEN_PUT": 5,
    "A7_OPEN_CLOSE": 4,
    "A8_OPEN_NAVIGATE_ROOM": 5,
    "A9_NAVIGATE_OBJ_PICK": 4,
    "A10_NAVIGATE_OBJ_OPEN": 4,
    "A11_NAVIGATE_ROOM_NAVIGATE_ROOM": 3,
    "A12_NAVIGATE_OBJ_PUT": 5,
    "A13_PICK_CLOSE": 5,
    "A14_CLOSE_OPEN": 4,
    "A15_PICK_NAVIGATE_OBJ": 5,
}


def _state_dict(node_data: dict[str, Any]) -> dict[str, Any]:
    state = node_data.get("state")
    if isinstance(state, dict):
        return state
    return normalize_state(state)


def _geometry_dict(node_data: dict[str, Any]) -> dict[str, Any]:
    raw_node = node_data.get("raw_node")
    if raw_node is not None and hasattr(raw_node, "geometry"):
        geometry = raw_node.geometry
        if isinstance(geometry, dict):
            return geometry
    geometry = node_data.get("geometry")
    if isinstance(geometry, dict):
        return geometry
    return {}


def build_inventory(scene_graph, reachable_positions: list[dict[str, Any]]) -> dict[str, Any]:
    rooms = []
    objects = []
    for node in scene_graph.nodes.values():
        if node.id.startswith("Room|"):
            rooms.append({"id": node.id, "pos": node.pos})
        else:
            geometry = node.geometry or {}
            state = normalize_state(node.state)
            objects.append(
                {
                    "id": node.id,
                    "room_id": node.room_id,
                    "pos": node.pos,
                    "pickupable": bool(geometry.get("pickupable", False)),
                    "openable": bool(geometry.get("openable", False)),
                    "receptacle": bool(geometry.get("receptacle", False)),
                    "open_state": state.get("open_state", "none"),
                    "held": state.get("held", False),
                }
            )

    return {
        "rooms": rooms,
        "objects": objects,
        "robot_pose": scene_graph.robot_pose,
        "reachable_positions": reachable_positions,
    }


def _choose_from(rng: random.Random, primary: list[Any], fallback: list[Any]) -> Any:
    if primary:
        return rng.choice(primary)
    if fallback:
        return rng.choice(fallback)
    return None


def _choose_pick_object(
    rng: random.Random,
    pickupable: list[dict[str, Any]],
    non_pickupable: list[dict[str, Any]],
    noise_p_pick_bad: float,
) -> dict[str, Any] | None:
    if rng.random() < noise_p_pick_bad:
        return _choose_from(rng, non_pickupable, pickupable)
    return _choose_from(rng, pickupable, non_pickupable)


def _choose_open_container(
    rng: random.Random,
    openable: list[dict[str, Any]],
    non_openable: list[dict[str, Any]],
    noise_p_open_bad: float,
) -> dict[str, Any] | None:
    if rng.random() < noise_p_open_bad:
        return _choose_from(rng, non_openable, openable)
    return _choose_from(rng, openable, non_openable)


def _choose_receptacle(
    rng: random.Random,
    receptacles: list[dict[str, Any]],
    non_receptacles: list[dict[str, Any]],
    noise_p_put_bad_dest: float,
) -> dict[str, Any] | None:
    if rng.random() < noise_p_put_bad_dest:
        return _choose_from(rng, non_receptacles, receptacles)
    return _choose_from(rng, receptacles, non_receptacles)


def _choose_room(rng: random.Random, rooms: list[dict[str, Any]], exclude_id: str | None = None) -> dict[str, Any] | None:
    if not rooms:
        return None
    if exclude_id is None or len(rooms) <= 1:
        return rng.choice(rooms)
    candidates = [room for room in rooms if room["id"] != exclude_id]
    return rng.choice(candidates) if candidates else rng.choice(rooms)


def generate_task(
    inventory: dict[str, Any],
    rng: random.Random,
    noise_p_pick_bad: float,
    noise_p_open_bad: float,
    noise_p_put_bad_dest: float,
) -> dict[str, Any]:
    rooms = inventory["rooms"]
    objects = inventory["objects"]

    pickupable = [obj for obj in objects if obj["pickupable"]]
    non_pickupable = [obj for obj in objects if not obj["pickupable"]]
    openable = [obj for obj in objects if obj["openable"]]
    non_openable = [obj for obj in objects if not obj["openable"]]
    receptacles = [obj for obj in objects if obj["receptacle"]]
    non_receptacles = [obj for obj in objects if not obj["receptacle"]]
    openable_open = [obj for obj in openable if obj["open_state"] == "open"]
    openable_closed = [obj for obj in openable if obj["open_state"] == "closed"]

    task_type = rng.choice(TASK_TYPES)

    if task_type == "A1_PICK_NAVIGATE_ROOM":
        obj = _choose_pick_object(rng, pickupable, non_pickupable, noise_p_pick_bad)
        room2 = _choose_room(rng, rooms, exclude_id=obj["room_id"] if obj else None)
        return {"task_type": task_type, "obj": obj["id"] if obj else None, "room": room2["id"] if room2 else None}
    if task_type == "A2_PICK_PUT":
        obj = _choose_pick_object(rng, pickupable, non_pickupable, noise_p_pick_bad)
        dest = _choose_receptacle(rng, receptacles, non_receptacles, noise_p_put_bad_dest)
        return {"task_type": task_type, "obj": obj["id"] if obj else None, "dest": dest["id"] if dest else None}
    if task_type == "A3_PICK_OPEN":
        obj = _choose_pick_object(rng, pickupable, non_pickupable, noise_p_pick_bad)
        container = _choose_open_container(rng, openable_closed or openable, non_openable, noise_p_open_bad)
        return {
            "task_type": task_type,
            "obj": obj["id"] if obj else None,
            "container": container["id"] if container else None,
        }
    if task_type == "A5_OPEN_PICK":
        container = _choose_open_container(rng, openable_closed or openable, non_openable, noise_p_open_bad)
        obj = _choose_pick_object(rng, pickupable, non_pickupable, noise_p_pick_bad)
        return {
            "task_type": task_type,
            "container": container["id"] if container else None,
            "obj": obj["id"] if obj else None,
        }
    if task_type == "A6_OPEN_PUT":
        container = _choose_open_container(rng, openable_closed or openable, non_openable, noise_p_open_bad)
        obj = _choose_from(rng, pickupable, objects)
        return {
            "task_type": task_type,
            "container": container["id"] if container else None,
            "obj": obj["id"] if obj else None,
        }
    if task_type == "A7_OPEN_CLOSE":
        container = _choose_open_container(rng, openable_closed or openable, non_openable, noise_p_open_bad)
        return {"task_type": task_type, "container": container["id"] if container else None}
    if task_type == "A8_OPEN_NAVIGATE_ROOM":
        container = _choose_open_container(rng, openable_closed or openable, non_openable, noise_p_open_bad)
        room2 = _choose_room(rng, rooms, exclude_id=container["room_id"] if container else None)
        return {
            "task_type": task_type,
            "container": container["id"] if container else None,
            "room": room2["id"] if room2 else None,
        }
    if task_type == "A9_NAVIGATE_OBJ_PICK":
        obj = _choose_pick_object(rng, pickupable, non_pickupable, noise_p_pick_bad)
        return {"task_type": task_type, "obj": obj["id"] if obj else None}
    if task_type == "A10_NAVIGATE_OBJ_OPEN":
        container = _choose_open_container(rng, openable_closed or openable, non_openable, noise_p_open_bad)
        return {"task_type": task_type, "container": container["id"] if container else None}
    if task_type == "A11_NAVIGATE_ROOM_NAVIGATE_ROOM":
        room1 = _choose_room(rng, rooms)
        room2 = _choose_room(rng, rooms, exclude_id=room1["id"] if room1 else None)
        return {
            "task_type": task_type,
            "room_start": room1["id"] if room1 else None,
            "room": room2["id"] if room2 else None,
        }
    if task_type == "A12_NAVIGATE_OBJ_PUT":
        dest = _choose_receptacle(rng, receptacles, non_receptacles, noise_p_put_bad_dest)
        obj = _choose_from(rng, pickupable, objects)
        return {"task_type": task_type, "dest": dest["id"] if dest else None, "obj": obj["id"] if obj else None}
    if task_type == "A13_PICK_CLOSE":
        obj = _choose_pick_object(rng, pickupable, non_pickupable, noise_p_pick_bad)
        container = _choose_from(rng, openable_open or openable, objects)
        return {
            "task_type": task_type,
            "obj": obj["id"] if obj else None,
            "container": container["id"] if container else None,
        }
    if task_type == "A14_CLOSE_OPEN":
        container = _choose_from(rng, openable_open or openable, objects)
        return {"task_type": task_type, "container": container["id"] if container else None}
    if task_type == "A15_PICK_NAVIGATE_OBJ":
        obj = _choose_pick_object(rng, pickupable, non_pickupable, noise_p_pick_bad)
        other_objs = [candidate for candidate in objects if obj is None or candidate["id"] != obj["id"]]
        obj2 = _choose_from(rng, other_objs, objects)
        return {
            "task_type": task_type,
            "obj": obj["id"] if obj else None,
            "target_obj": obj2["id"] if obj2 else None,
        }

    return {"task_type": task_type}


def compile_task_to_actions(
    task: dict[str, Any],
    inventory: dict[str, Any] | None = None,
    reachable_positions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    task_type = task["task_type"]
    actions: list[dict[str, Any]] = []

    if task_type == "A1_PICK_NAVIGATE_ROOM":
        actions = [
            {"action": "NavigateTo", "target": task.get("obj")},
            {"action": "PickUp", "target": task.get("obj")},
            {"action": "NavigateTo", "target": task.get("room")},
        ]
    elif task_type == "A2_PICK_PUT":
        actions = [
            {"action": "NavigateTo", "target": task.get("obj")},
            {"action": "PickUp", "target": task.get("obj")},
            {"action": "NavigateTo", "target": task.get("dest")},
            {
                "action": "PutObject",
                "target": task.get("obj"),
                "receptacle_id": task.get("dest"),
            },
        ]
    elif task_type == "A3_PICK_OPEN":
        actions = [
            {"action": "NavigateTo", "target": task.get("obj")},
            {"action": "PickUp", "target": task.get("obj")},
            {"action": "NavigateTo", "target": task.get("container")},
            {"action": "Open", "target": task.get("container")},
        ]
    elif task_type == "A5_OPEN_PICK":
        actions = [
            {"action": "NavigateTo", "target": task.get("container")},
            {"action": "Open", "target": task.get("container")},
            {"action": "NavigateTo", "target": task.get("obj")},
            {"action": "PickUp", "target": task.get("obj")},
        ]
    elif task_type == "A6_OPEN_PUT":
        actions = [
            {"action": "NavigateTo", "target": task.get("container")},
            {"action": "Open", "target": task.get("container")},
            {"action": "PutObject", "target": task.get("obj")},
        ]
    elif task_type == "A7_OPEN_CLOSE":
        actions = [
            {"action": "NavigateTo", "target": task.get("container")},
            {"action": "Open", "target": task.get("container")},
            {"action": "Close", "target": task.get("container")},
        ]
    elif task_type == "A8_OPEN_NAVIGATE_ROOM":
        actions = [
            {"action": "NavigateTo", "target": task.get("container")},
            {"action": "Open", "target": task.get("container")},
            {"action": "NavigateTo", "target": task.get("room")},
        ]
    elif task_type == "A9_NAVIGATE_OBJ_PICK":
        actions = [
            {"action": "NavigateTo", "target": task.get("obj")},
            {"action": "PickUp", "target": task.get("obj")},
        ]
    elif task_type == "A10_NAVIGATE_OBJ_OPEN":
        actions = [
            {"action": "NavigateTo", "target": task.get("container")},
            {"action": "Open", "target": task.get("container")},
        ]
    elif task_type == "A11_NAVIGATE_ROOM_NAVIGATE_ROOM":
        actions = [
            {"action": "NavigateTo", "target": task.get("room_start")},
            {"action": "NavigateTo", "target": task.get("room")},
        ]
    elif task_type == "A12_NAVIGATE_OBJ_PUT":
        actions = [
            {"action": "NavigateTo", "target": task.get("dest")},
            {"action": "PutObject", "target": task.get("obj"), "receptacle_id": task.get("dest")},
        ]
    elif task_type == "A13_PICK_CLOSE":
        actions = [
            {"action": "NavigateTo", "target": task.get("obj")},
            {"action": "PickUp", "target": task.get("obj")},
            {"action": "NavigateTo", "target": task.get("container")},
            {"action": "Close", "target": task.get("container")},
        ]
    elif task_type == "A14_CLOSE_OPEN":
        actions = [
            {"action": "NavigateTo", "target": task.get("container")},
            {"action": "Close", "target": task.get("container")},
            {"action": "Open", "target": task.get("container")},
        ]
    elif task_type == "A15_PICK_NAVIGATE_OBJ":
        actions = [
            {"action": "NavigateTo", "target": task.get("obj")},
            {"action": "PickUp", "target": task.get("obj")},
            {"action": "NavigateTo", "target": task.get("target_obj")},
        ]

    return [action for action in actions if action.get("target") is not None or action["action"] == "PutObject"]


def _get_robot_id(G: nx.DiGraph) -> str:
    if "robot_agent" in G.nodes:
        return "robot_agent"
    for node_id, data in G.nodes(data=True):
        if data.get("type") == "agent":
            return node_id
    return "robot_agent"


def _get_robot_pos(G: nx.DiGraph) -> list[float] | None:
    robot_id = _get_robot_id(G)
    node = G.nodes.get(robot_id)
    if not node:
        return None
    return list(node.get("pos", [])) if node.get("pos") is not None else None


def _is_held(G: nx.DiGraph, obj_id: str | None) -> bool:
    if not obj_id or obj_id not in G.nodes:
        return False
    state = _state_dict(G.nodes[obj_id])
    if state.get("held"):
        return True
    robot_id = _get_robot_id(G)
    for _, target, data in G.out_edges(robot_id, data=True):
        if target == obj_id and data.get("relation") == Relation.HOLDING:
            return True
    return False


def _is_open(G: nx.DiGraph, obj_id: str | None) -> bool:
    if not obj_id or obj_id not in G.nodes:
        return False
    state = _state_dict(G.nodes[obj_id])
    return state.get("open_state") == "open"


def _is_closed(G: nx.DiGraph, obj_id: str | None) -> bool:
    if not obj_id or obj_id not in G.nodes:
        return False
    state = _state_dict(G.nodes[obj_id])
    return state.get("open_state") == "closed"


def _near_position(pos_a: list[float] | None, pos_b: list[float] | None, threshold: float = 1.0) -> bool:
    if not pos_a or not pos_b:
        return False
    return float(np.linalg.norm(np.array(pos_a) - np.array(pos_b))) <= threshold


def _navigate_success(G: nx.DiGraph, target_id: str | None, threshold: float = 1.0) -> bool:
    if not target_id or target_id not in G.nodes:
        return False
    robot_pos = _get_robot_pos(G)
    target_pos = G.nodes[target_id].get("pos")
    if target_pos is None:
        return False
    return _near_position(robot_pos, list(target_pos), threshold=threshold)


def check_task_success(task: dict[str, Any], G_start: nx.DiGraph, G_end: nx.DiGraph) -> tuple[bool, dict[str, Any]]:
    task_type = task["task_type"]
    meta: dict[str, Any] = {"intents": {}}

    if task_type == "A1_PICK_NAVIGATE_ROOM":
        meta["intents"]["pick"] = _is_held(G_end, task.get("obj"))
        meta["intents"]["navigate_room"] = _navigate_success(G_end, task.get("room"))
    elif task_type == "A2_PICK_PUT":
        meta["intents"]["put"] = not _is_held(G_end, task.get("obj"))
    elif task_type == "A3_PICK_OPEN":
        meta["intents"]["pick"] = _is_held(G_end, task.get("obj"))
        meta["intents"]["open"] = _is_open(G_end, task.get("container"))
    elif task_type == "A5_OPEN_PICK":
        meta["intents"]["open"] = _is_open(G_end, task.get("container"))
        meta["intents"]["pick"] = _is_held(G_end, task.get("obj"))
    elif task_type == "A6_OPEN_PUT":
        meta["intents"]["open"] = _is_open(G_end, task.get("container"))
        meta["intents"]["put"] = not _is_held(G_end, task.get("obj"))
    elif task_type == "A7_OPEN_CLOSE":
        meta["intents"]["close"] = _is_closed(G_end, task.get("container"))
    elif task_type == "A8_OPEN_NAVIGATE_ROOM":
        meta["intents"]["open"] = _is_open(G_end, task.get("container"))
        meta["intents"]["navigate_room"] = _navigate_success(G_end, task.get("room"))
    elif task_type == "A9_NAVIGATE_OBJ_PICK":
        meta["intents"]["navigate_obj"] = _navigate_success(G_end, task.get("obj"))
        meta["intents"]["pick"] = _is_held(G_end, task.get("obj"))
    elif task_type == "A10_NAVIGATE_OBJ_OPEN":
        meta["intents"]["navigate_obj"] = _navigate_success(G_end, task.get("container"))
        meta["intents"]["open"] = _is_open(G_end, task.get("container"))
    elif task_type == "A11_NAVIGATE_ROOM_NAVIGATE_ROOM":
        meta["intents"]["navigate_room2"] = _navigate_success(G_end, task.get("room"))
    elif task_type == "A12_NAVIGATE_OBJ_PUT":
        meta["intents"]["put"] = not _is_held(G_end, task.get("obj"))
    elif task_type == "A13_PICK_CLOSE":
        meta["intents"]["pick"] = _is_held(G_end, task.get("obj"))
        meta["intents"]["close"] = _is_closed(G_end, task.get("container"))
    elif task_type == "A14_CLOSE_OPEN":
        meta["intents"]["open"] = _is_open(G_end, task.get("container"))
    elif task_type == "A15_PICK_NAVIGATE_OBJ":
        meta["intents"]["pick"] = _is_held(G_end, task.get("obj"))
        meta["intents"]["navigate_obj"] = _navigate_success(G_end, task.get("target_obj"))

    goal_satisfied = all(meta["intents"].values()) if meta["intents"] else False

    return goal_satisfied, meta


def _maybe_swap_actions(rng: random.Random, actions: list[dict[str, Any]], p_swap: float = 0.05) -> list[dict[str, Any]]:
    if len(actions) < 2 or rng.random() >= p_swap:
        return actions
    idx = rng.randrange(len(actions) - 1)
    swapped = actions[:]
    swapped[idx], swapped[idx + 1] = swapped[idx + 1], swapped[idx]
    return swapped


def _print_episode_stats(
    episode_idx: int,
    stats: Counter,
    action_stats: Counter,
    task_stats: Counter,
    task_success_stats: Counter,
) -> None:
    total_steps = stats["total_steps"]
    env_fail = stats["env_fail_steps"]
    env_fail_rate = env_fail / total_steps if total_steps else 0.0
    schema_invalid = stats["schema_invalid_steps"]
    schema_invalid_rate = schema_invalid / total_steps if total_steps else 0.0
    print(
        f"[Episode {episode_idx}] env_fail_rate={env_fail_rate:.2%} "
        f"schema_invalid_rate={schema_invalid_rate:.2%} total_steps={total_steps}"
    )

    action_totals = {k.replace("_total", ""): v for k, v in action_stats.items() if k.endswith("_total")}
    for action, total in sorted(action_totals.items()):
        success = action_stats.get(action, 0)
        rate = success / total if total else 0.0
        print(f"  {action}: {success}/{total} ({rate:.2%})")

    for task_type in sorted(task_stats.keys()):
        total = task_stats[task_type]
        success = task_success_stats.get(task_type, 0)
        rate = success / total if total else 0.0
        print(f"  {task_type}: {success}/{total} ({rate:.2%})")


def run_phase1_tasks(
    num_scenes: int,
    episodes_per_scene: int,
    max_steps_per_episode: int,
    output_dir: str,
    seed: int,
    noise_p_pick_bad: float,
    noise_p_open_bad: float,
    noise_p_put_bad_dest: float,
    log_every: int,
) -> dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    steps_path = os.path.join(output_dir, f"phase1_tasks_steps_{timestamp}.jsonl")
    episodes_path = os.path.join(output_dir, f"phase1_tasks_episodes_{timestamp}.jsonl")

    rng = random.Random(seed)
    stats = Counter()
    action_stats = Counter()
    task_stats = Counter()
    task_success_stats = Counter()
    error_log: list[dict[str, Any]] = []

    from src.env.procthor_wrapper import ProcTHOREnv

    env = ProcTHOREnv()
    adapter = ProcTHORActionAdapter()

    episode_idx = 0
    try:
        with open(steps_path, "w", encoding="utf-8") as step_handle, open(
            episodes_path, "w", encoding="utf-8"
        ) as episode_handle:
            for scene_index in range(num_scenes):
                env.change_scene(scene_index)
                scene_id = f"ProcTHOR-Train-{scene_index}"

                for ep in range(episodes_per_scene):
                    if ep > 0:
                        env.reset()
                    episode_id = f"{scene_id}_ep_{ep}"
                    scene_cache = {"reachable_positions": get_reachable_positions(env.controller)}
                    oracle = OracleInterface(env)
                    manager = GraphManager(debug=False)

                    sg = oracle.get_hierarchical_graph()
                    manager.override_global_graph(sg)
                    ok, errors = sanitize_graph(manager.G)
                    if not ok:
                        error_log.append({"scene_id": scene_id, "episode_id": episode_id, "errors": errors})
                        continue

                    inventory = build_inventory(sg, scene_cache["reachable_positions"])
                    task = generate_task(
                        inventory,
                        rng,
                        noise_p_pick_bad,
                        noise_p_open_bad,
                        noise_p_put_bad_dest,
                    )
                    task_params = {k: v for k, v in task.items() if k != "task_type"}
                    actions = compile_task_to_actions(
                        task,
                        inventory=inventory,
                        reachable_positions=scene_cache.get("reachable_positions", []),
                    )
                    actions = _maybe_swap_actions(rng, actions)
                    task_max_steps = TASK_MAX_STEPS.get(task["task_type"], max_steps_per_episode)
                    max_steps = min(max_steps_per_episode, task_max_steps)
                    actions = actions[:max_steps]

                    step_successes = []
                    executed_actions: list[dict[str, Any]] = []
                    graph_start = copy.deepcopy(manager.G)
                    retry_used: dict[tuple[str, str | None], bool] = {}
                    t = 0
                    while t < len(actions) and t < max_steps:
                        action = actions[t]
                        graph_t = copy.deepcopy(manager.G)
                        try:
                            success, error_msg = execute_action(
                                env, adapter, action, graph_t, scene_cache=scene_cache
                            )
                        except RuntimeError as exc:
                            raise RuntimeError(f"API_SCHEMA_BUG: {exc}") from exc

                        sg_next = oracle.get_hierarchical_graph()
                        manager.override_global_graph(sg_next)
                        graph_t1 = manager.G

                        ok, errors = sanitize_graph(graph_t1)
                        if not ok:
                            error_log.append(
                                {
                                    "scene_id": scene_id,
                                    "episode_id": episode_id,
                                    "t": t,
                                    "errors": errors,
                                }
                            )
                            stats["schema_invalid_steps"] += 1

                        if error_msg.startswith("INVALID_ACTION_SCHEMA"):
                            stats["api_schema_rejected_steps"] += 1

                        stats["total_steps"] += 1
                        if success:
                            action_stats[action["action"]] += 1
                        else:
                            stats["env_fail_steps"] += 1
                        action_stats[f"{action['action']}_total"] += 1

                        step_entry = {
                            "scene_id": scene_id,
                            "episode_id": episode_id,
                            "t": t,
                            "task_type": task["task_type"],
                            "task_params": task_params,
                            "action": action,
                            "success": success,
                            "error_msg": error_msg,
                            "G_t": export_graph_canonical(graph_t),
                            "G_t1": export_graph_canonical(graph_t1),
                            "delta": graph_diff(graph_t, graph_t1),
                        }
                        step_handle.write(json.dumps(step_entry) + "\n")
                        step_successes.append(success)
                        executed_actions.append(action)
                        if (
                            not success
                            and action["action"] in {"PickUp", "Open"}
                            and t + 2 <= max_steps
                        ):
                            retry_key = (action["action"], action.get("target"))
                            if not retry_used.get(retry_key, False):
                                retry_used[retry_key] = True
                                retry_actions = [
                                    {"action": "NavigateTo", "target": action.get("target")},
                                    {"action": action["action"], "target": action.get("target")},
                                ]
                                actions[t + 1 : t + 1] = retry_actions
                        t += 1

                    goal_satisfied, _ = check_task_success(task, graph_start, manager.G)
                    task_stats[task["task_type"]] += 1
                    if goal_satisfied:
                        task_success_stats[task["task_type"]] += 1

                    episode_entry = {
                        "scene_id": scene_id,
                        "episode_id": episode_id,
                        "task_type": task["task_type"],
                        "task_params": task_params,
                        "actions": executed_actions,
                        "step_successes": step_successes,
                        "goal_satisfied": goal_satisfied,
                        "num_steps": len(executed_actions),
                    }
                    episode_handle.write(json.dumps(episode_entry) + "\n")

                    episode_idx += 1
                    if log_every and episode_idx % log_every == 0:
                        _print_episode_stats(
                            episode_idx, stats, action_stats, task_stats, task_success_stats
                        )
    finally:
        env.stop()

    return {
        "steps_path": steps_path,
        "episodes_path": episodes_path,
        "stats": stats,
        "action_stats": action_stats,
        "task_stats": task_stats,
        "task_success_stats": task_success_stats,
        "errors": error_log,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 task-driven data collection")
    parser.add_argument("--num_scenes", type=int, default=1)
    parser.add_argument("--episodes_per_scene", type=int, default=200)
    parser.add_argument("--max_steps_per_episode", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="datasets/phase1_tasks/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise_p_pick_bad", type=float, default=0.20)
    parser.add_argument("--noise_p_open_bad", type=float, default=0.15)
    parser.add_argument("--noise_p_put_bad_dest", type=float, default=0.10)
    parser.add_argument("--log_every", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_phase1_tasks(
        num_scenes=args.num_scenes,
        episodes_per_scene=args.episodes_per_scene,
        max_steps_per_episode=args.max_steps_per_episode,
        output_dir=args.output_dir,
        seed=args.seed,
        noise_p_pick_bad=args.noise_p_pick_bad,
        noise_p_open_bad=args.noise_p_open_bad,
        noise_p_put_bad_dest=args.noise_p_put_bad_dest,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
