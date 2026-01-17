import argparse
import copy
import json
import os
import random
import time
from collections import Counter
from datetime import datetime
from typing import Any

import networkx as nx

from src.core.graph_schema import CANONICAL_RELATIONS, Relation, normalize_state
from src.env.action_adapter import ProcTHORActionAdapter
from src.env.action_preconditions import check_action_preconditions
from src.memory.graph_manager import GraphManager, sync_bidirectional_edges, validate_graph_schema


def export_graph_canonical(graph: nx.DiGraph) -> dict[str, Any]:
    nodes = []
    for node_id, data in graph.nodes(data=True):
        state = normalize_state(data.get("state"))
        nodes.append(
            {
                "id": node_id,
                "type": data.get("type"),
                "pos": list(data.get("pos", [])),
                "room_id": data.get("room_id"),
                "state": state,
            }
        )

    edges = []
    for u, v, data in graph.edges(data=True):
        relation = data.get("relation")
        if relation in CANONICAL_RELATIONS:
            edges.append({"u": u, "v": v, "relation": relation})

    return {"nodes": nodes, "edges": edges}


def graph_diff(graph_t: nx.DiGraph, graph_t1: nx.DiGraph) -> dict[str, Any]:
    def canonical_edge_set(graph: nx.DiGraph) -> set[tuple[str, str, str]]:
        return {
            (u, v, data.get("relation"))
            for u, v, data in graph.edges(data=True)
            if data.get("relation") in CANONICAL_RELATIONS
        }

    edges_t = canonical_edge_set(graph_t)
    edges_t1 = canonical_edge_set(graph_t1)

    added_edges = [list(edge) for edge in sorted(edges_t1 - edges_t)]
    removed_edges = [list(edge) for edge in sorted(edges_t - edges_t1)]

    state_changes = []
    for node_id, data in graph_t.nodes(data=True):
        if node_id not in graph_t1:
            continue
        next_data = graph_t1.nodes[node_id]
        state_t = normalize_state(data.get("state"))
        state_t1 = normalize_state(next_data.get("state"))
        pos_t = list(data.get("pos", []))
        pos_t1 = list(next_data.get("pos", []))

        changed = {}
        if state_t.get("open_state") != state_t1.get("open_state"):
            changed["open_state"] = {"from": state_t.get("open_state"), "to": state_t1.get("open_state")}
        if state_t.get("held") != state_t1.get("held"):
            changed["held"] = {"from": state_t.get("held"), "to": state_t1.get("held")}
        if pos_t != pos_t1:
            changed["pos"] = {"from": pos_t, "to": pos_t1}

        if changed:
            state_changes.append({"id": node_id, "changes": changed})

    return {
        "added_edges": added_edges,
        "removed_edges": removed_edges,
        "state_changes": state_changes,
    }


def sanitize_graph(graph: nx.DiGraph) -> tuple[bool, list[dict[str, Any]]]:
    sync_bidirectional_edges(graph)
    return validate_graph_schema(graph)


def _random_action(graph: nx.DiGraph, rng: random.Random) -> dict[str, Any]:
    object_nodes = [n for n, d in graph.nodes(data=True) if d.get("type") == "object"]
    target = rng.choice(object_nodes) if object_nodes else None
    action = rng.choice(["NavigateTo", "Open", "Close", "PickUp", "PutObject"])
    action_dict = {"action": action, "target": target}
    if action == "PutObject":
        action_dict["receptacle_id"] = rng.choice(object_nodes) if object_nodes else None
    return action_dict


def get_reachable_positions(controller) -> list[dict[str, Any]]:
    try:
        event = controller.step(action="GetReachablePositions")
    except Exception as exc:
        print(f"[Reachable] Failed to query positions: {exc}")
        return []
    positions = event.metadata.get("actionReturn") or []
    if not isinstance(positions, list):
        print("[Reachable] Unexpected response for reachable positions")
        return []
    return positions


def execute_action(
    env,
    adapter: ProcTHORActionAdapter,
    action: dict[str, Any],
    graph: nx.DiGraph,
    scene_cache: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    controller = env.controller

    preconditions = check_action_preconditions(action, graph)
    if not preconditions.ok:
        return False, f"AFFORDANCE_MISMATCH: {preconditions.reason}"

    ok, reason = adapter.validate_action_dict(action)
    if not ok:
        return False, f"INVALID_ACTION_SCHEMA: {reason}"

    thor_kwargs, ok, reason = adapter.to_thor_step(action, graph, scene_cache=scene_cache)
    if not ok:
        return False, f"INVALID_ACTION_SCHEMA: {reason}"

    try:
        event = controller.step(**thor_kwargs)
    except Exception as exc:
        raise RuntimeError(f"API_SCHEMA_BUG: {exc}") from exc

    success = bool(event.metadata.get("lastActionSuccess"))
    error_msg = event.metadata.get("errorMessage") or ""
    return success, error_msg


def _collect_scene(
    env,
    scene_id: str,
    steps: int,
    output_handle,
    rng: random.Random,
    stats: Counter,
    error_log: list[dict[str, Any]],
    scene_cache: dict[str, Any],
) -> tuple[int, int]:
    from src.perception.oracle_interface import OracleInterface

    oracle = OracleInterface(env)
    manager = GraphManager(debug=False)
    adapter = ProcTHORActionAdapter()

    sg = oracle.get_hierarchical_graph()
    manager.override_global_graph(sg)

    ok, errors = sanitize_graph(manager.G)
    if not ok:
        error_log.append({"scene_id": scene_id, "t": 0, "errors": errors})
        return 0, 0

    valid_samples = 0
    total_steps = 0

    for t in range(steps):
        graph_t = copy.deepcopy(manager.G)
        action = _random_action(graph_t, rng)
        success, error_msg = execute_action(env, adapter, action, graph_t, scene_cache=scene_cache)

        sg_next = oracle.get_hierarchical_graph()
        manager.override_global_graph(sg_next)
        graph_t1 = manager.G

        ok, errors = sanitize_graph(graph_t1)
        total_steps += 1
        stats[action["action"]] += int(success)
        stats[f"{action['action']}_total"] += 1

        if not ok:
            error_log.append({"scene_id": scene_id, "t": t, "errors": errors})
            continue

        sample = {
            "scene_id": scene_id,
            "t": t,
            "action": action,
            "success": success,
            "error_msg": error_msg,
            "G_t": export_graph_canonical(graph_t),
            "G_t1": export_graph_canonical(graph_t1),
            "delta": graph_diff(graph_t, graph_t1),
        }
        output_handle.write(json.dumps(sample) + "\n")
        valid_samples += 1

    return total_steps, valid_samples


def run_rollouts(
    num_scenes: int,
    steps_per_scene: int,
    output_dir: str,
    seed: int,
    policy: str,
    save_mode: str,
) -> dict[str, Any]:
    if policy != "random":
        raise ValueError(f"Unsupported policy: {policy}")
    if save_mode != "jsonl":
        raise ValueError(f"Unsupported save_mode: {save_mode}")

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"phase1_rollouts_{timestamp}.jsonl")

    rng = random.Random(seed)
    stats = Counter()
    error_log: list[dict[str, Any]] = []

    from src.env.procthor_wrapper import ProcTHOREnv

    env = ProcTHOREnv()
    total_steps = 0
    total_valid = 0

    try:
        with open(output_path, "w", encoding="utf-8") as handle:
            for scene_index in range(num_scenes):
                env.change_scene(scene_index)
                scene_id = f"ProcTHOR-Train-{scene_index}"
                scene_cache = {"reachable_positions": get_reachable_positions(env.controller)}
                steps, valid = _collect_scene(
                    env=env,
                    scene_id=scene_id,
                    steps=steps_per_scene,
                    output_handle=handle,
                    rng=rng,
                    stats=stats,
                    error_log=error_log,
                    scene_cache=scene_cache,
                )
                total_steps += steps
                total_valid += valid
    finally:
        env.stop()

    return {
        "output_path": output_path,
        "total_steps": total_steps,
        "valid_samples": total_valid,
        "stats": stats,
        "errors": error_log,
    }


def _dry_run(output_dir: str, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "phase1_dry_run.jsonl")
    stats = Counter()
    error_log: list[dict[str, Any]] = []

    graph_t = nx.DiGraph()
    graph_t.add_node("robot_agent", type="agent", pos=[0.0, 0.0, 0.0], state=normalize_state({}))
    graph_t.add_node("room_1", type="room", pos=[0.0, 0.0, 0.0], state=normalize_state({}))
    graph_t.add_node("fridge_1", type="object", pos=[1.0, 0.0, 0.0], state=normalize_state({"open_state": "closed"}))
    graph_t.add_node("apple_1", type="object", pos=[1.0, 0.0, 0.0], state=normalize_state({}))
    graph_t.add_edge("room_1", "fridge_1", relation=Relation.CONTAINS)
    graph_t.add_edge("fridge_1", "apple_1", relation=Relation.INSIDE)

    graph_t1 = copy.deepcopy(graph_t)
    graph_t1.nodes["fridge_1"]["state"] = normalize_state({"open_state": "open"})
    graph_t1.add_edge("robot_agent", "apple_1", relation=Relation.HOLDING)
    graph_t1.remove_edge("fridge_1", "apple_1")

    ok, errors = sanitize_graph(graph_t1)
    if not ok:
        error_log.append({"scene_id": "dry_run", "t": 0, "errors": errors})

    with open(output_path, "w", encoding="utf-8") as handle:
        samples = [
            {
                "scene_id": "dry_run",
                "t": 0,
                "action": {"action": "PickUp", "target": "apple_1"},
                "success": True,
                "error_msg": "",
                "G_t": export_graph_canonical(graph_t),
                "G_t1": export_graph_canonical(graph_t1),
                "delta": graph_diff(graph_t, graph_t1),
            },
            {
                "scene_id": "dry_run",
                "t": 1,
                "action": {"action": "Open", "target": "fridge_1"},
                "success": True,
                "error_msg": "",
                "G_t": export_graph_canonical(graph_t1),
                "G_t1": export_graph_canonical(graph_t1),
                "delta": graph_diff(graph_t1, graph_t1),
            },
        ]
        for sample in samples:
            action = sample["action"]
            stats[action["action"]] += int(sample["success"])
            stats[f"{action['action']}_total"] += 1
            handle.write(json.dumps(sample) + "\n")

    return {
        "output_path": output_path,
        "total_steps": 2,
        "valid_samples": 2,
        "stats": stats,
        "errors": error_log,
    }


def _print_summary(result: dict[str, Any]) -> None:
    print(f"Output: {result['output_path']}")
    print(f"Total steps: {result['total_steps']}")
    print(f"Valid samples: {result['valid_samples']}")
    action_totals = {k.replace("_total", ""): v for k, v in result["stats"].items() if k.endswith("_total")}
    for action, total in sorted(action_totals.items()):
        success = result["stats"].get(action, 0)
        rate = success / total if total else 0.0
        print(f"{action}: {success}/{total} ({rate:.2%})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 data collection rollouts.")
    parser.add_argument("--num_scenes", type=int, default=1)
    parser.add_argument("--steps_per_scene", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="datasets/phase1/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", type=str, default="random")
    parser.add_argument("--save_mode", type=str, default="jsonl")
    parser.add_argument("--dry_run", action="store_true", help="Run a synthetic pipeline check without ProcTHOR.")
    args = parser.parse_args()

    start = time.time()
    if args.dry_run:
        result = _dry_run(args.output_dir, args.seed)
    else:
        result = run_rollouts(
            num_scenes=args.num_scenes,
            steps_per_scene=args.steps_per_scene,
            output_dir=args.output_dir,
            seed=args.seed,
            policy=args.policy,
            save_mode=args.save_mode,
        )
    elapsed = time.time() - start
    _print_summary(result)
    print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
