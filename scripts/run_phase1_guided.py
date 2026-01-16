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

from scripts.data_collection import execute_action, export_graph_canonical, graph_diff, sanitize_graph
from src.core.graph_schema import Relation, normalize_state
from src.env.action_adapter import ProcTHORActionAdapter
from src.memory.graph_manager import GraphManager
from src.perception.oracle_interface import OracleInterface


FAILURE_EXPLORATION_PROB = 0.15


def _get_robot_id(G: nx.DiGraph) -> str:
    if "robot_agent" in G.nodes:
        return "robot_agent"
    for node_id, data in G.nodes(data=True):
        if data.get("type") == "agent":
            return node_id
    return "robot_agent"


def _get_held_object(G: nx.DiGraph) -> str | None:
    robot_id = _get_robot_id(G)
    for _, target, data in G.out_edges(robot_id, data=True):
        if data.get("relation") == Relation.HOLDING:
            return target
    return None


def _get_objects(G: nx.DiGraph) -> list[str]:
    return [n for n, d in G.nodes(data=True) if d.get("type") == "object"]


def _get_pos_nodes(G: nx.DiGraph) -> list[str]:
    return [n for n, d in G.nodes(data=True) if d.get("pos") is not None]


def _pick_target(rng: random.Random, candidates: list[str], fallback: list[str]) -> str:
    if candidates:
        return rng.choice(candidates)
    if fallback:
        return rng.choice(fallback)
    return "robot_agent"


def _state_dict(node_data: dict[str, Any]) -> dict[str, Any]:
    state = node_data.get("state")
    if isinstance(state, dict):
        return state
    return normalize_state(state)


def _sample_action_state_aware(G: nx.DiGraph, rng: random.Random) -> tuple[dict[str, Any], str]:
    objects = _get_objects(G)
    pos_nodes = _get_pos_nodes(G)
    held_object = _get_held_object(G)

    openable_nodes = []
    closed_nodes = []
    open_nodes = []
    non_openable_nodes = []

    for obj_id in objects:
        state = _state_dict(G.nodes[obj_id])
        open_state = state.get("open_state", "none")
        if open_state == "closed":
            closed_nodes.append(obj_id)
            openable_nodes.append(obj_id)
        elif open_state == "open":
            open_nodes.append(obj_id)
            openable_nodes.append(obj_id)
        else:
            non_openable_nodes.append(obj_id)

    if rng.random() < FAILURE_EXPLORATION_PROB:
        failure_actions = ["PickUp", "Open", "PutObject"]
        action = rng.choice(failure_actions)
        reason = "failure_exploration"
    elif held_object:
        action = rng.choices(
            population=["PutObject", "NavigateTo", "Open", "Close"],
            weights=[0.5, 0.25, 0.15, 0.1],
            k=1,
        )[0]
        reason = "holding_bias"
    elif closed_nodes:
        action = rng.choices(
            population=["Open", "PickUp", "NavigateTo", "Close"],
            weights=[0.45, 0.25, 0.2, 0.1],
            k=1,
        )[0]
        reason = "closed_openable_bias"
    else:
        action = rng.choices(
            population=["PickUp", "NavigateTo", "Open", "Close"],
            weights=[0.4, 0.3, 0.2, 0.1],
            k=1,
        )[0]
        reason = "explore_bias"

    target = None
    receptacle_id = None

    if action == "NavigateTo":
        target = _pick_target(rng, objects, pos_nodes)
    elif action == "Open":
        target = _pick_target(rng, closed_nodes or openable_nodes, objects)
    elif action == "Close":
        target = _pick_target(rng, open_nodes or openable_nodes, objects)
    elif action == "PickUp":
        target = _pick_target(rng, objects, pos_nodes)
    elif action == "PutObject":
        if held_object:
            target = held_object
        else:
            target = _pick_target(rng, objects, pos_nodes)
        if objects and rng.random() < 0.2:
            receptacle_id = rng.choice(objects)

    action_dict = {"action": action}
    if target is not None:
        action_dict["target"] = target
    if receptacle_id is not None:
        action_dict["receptacle_id"] = receptacle_id

    return action_dict, reason


def sample_action_state_aware(G: nx.DiGraph, rng: random.Random) -> dict[str, Any]:
    action_dict, _ = _sample_action_state_aware(G, rng)
    return action_dict


def _print_step_stats(step_idx: int, stats: dict[str, Any]) -> None:
    print(
        f"[Step {step_idx}] valid_written={stats['valid_written_steps']} "
        f"schema_invalid={stats['schema_invalid_steps']} "
        f"env_fail={stats['env_fail_steps']}"
    )


def _print_scene_summary(scene_id: str, stats: dict[str, Any], action_stats: Counter) -> None:
    print(f"Scene {scene_id} summary:")
    print(f"  total_steps: {stats['total_steps']}")
    print(f"  valid_written_steps: {stats['valid_written_steps']}")
    print(f"  schema_invalid_steps: {stats['schema_invalid_steps']}")
    print(f"  env_fail_steps: {stats['env_fail_steps']}")
    print(f"  api_schema_rejected_steps: {stats['api_schema_rejected_steps']}")

    action_totals = {k.replace("_total", ""): v for k, v in action_stats.items() if k.endswith("_total")}
    for action, total in sorted(action_totals.items()):
        success = action_stats.get(action, 0)
        rate = success / total if total else 0.0
        print(f"  {action}: {success}/{total} ({rate:.2%})")


def run_guided_rollouts(
    num_scenes: int,
    steps_per_scene: int,
    output_dir: str,
    seed: int,
    save_mode: str,
    max_consecutive_failures: int,
    policy_debug: bool,
) -> dict[str, Any]:
    if save_mode != "jsonl":
        raise ValueError(f"Unsupported save_mode: {save_mode}")

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"phase1_guided_{timestamp}.jsonl")

    rng = random.Random(seed)
    overall_stats = Counter()
    error_log: list[dict[str, Any]] = []
    overall_action_stats = Counter()

    from src.env.procthor_wrapper import ProcTHOREnv

    env = ProcTHOREnv()
    adapter = ProcTHORActionAdapter()

    try:
        with open(output_path, "w", encoding="utf-8") as handle:
            for scene_index in range(num_scenes):
                env.change_scene(scene_index)
                scene_id = f"ProcTHOR-Train-{scene_index}"

                oracle = OracleInterface(env)
                manager = GraphManager(debug=False)
                manager.override_global_graph(oracle.get_hierarchical_graph())

                scene_stats = Counter()
                scene_action_stats = Counter()

                ok, errors = sanitize_graph(manager.G)
                if not ok:
                    error_log.append({"scene_id": scene_id, "t": 0, "errors": errors})
                    continue

                consecutive_failures = 0
                for t in range(steps_per_scene):
                    graph_t = copy.deepcopy(manager.G)
                    action, reason = _sample_action_state_aware(graph_t, rng)

                    if policy_debug:
                        print(f"[Policy] t={t} action={action} reason={reason}")

                    try:
                        success, error_msg = execute_action(env, adapter, action, graph_t)
                    except RuntimeError as exc:
                        raise RuntimeError(f"API_SCHEMA_BUG: {exc}") from exc

                    if error_msg.startswith("INVALID_ACTION_SCHEMA"):
                        scene_stats["schema_invalid_steps"] += 1
                        scene_stats["api_schema_rejected_steps"] += 1
                    elif not success:
                        scene_stats["env_fail_steps"] += 1

                    if success:
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1

                    if consecutive_failures >= max_consecutive_failures:
                        env.change_scene(scene_index)
                        manager.override_global_graph(oracle.get_hierarchical_graph())
                        consecutive_failures = 0

                    sg_next = oracle.get_hierarchical_graph()
                    manager.override_global_graph(sg_next)
                    graph_t1 = manager.G

                    ok, errors = sanitize_graph(graph_t1)
                    scene_stats["total_steps"] += 1
                    scene_action_stats[action["action"]] += int(success)
                    scene_action_stats[f"{action['action']}_total"] += 1

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
                    handle.write(json.dumps(sample) + "\n")
                    scene_stats["valid_written_steps"] += 1

                    if (t + 1) % 50 == 0:
                        _print_step_stats(t + 1, scene_stats)

                _print_scene_summary(scene_id, scene_stats, scene_action_stats)
                overall_stats.update(scene_stats)
                overall_action_stats.update(scene_action_stats)
    finally:
        env.stop()

    return {
        "output_path": output_path,
        "stats": overall_stats,
        "action_stats": overall_action_stats,
        "errors": error_log,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 guided data collection rollouts.")
    parser.add_argument("--num_scenes", type=int, default=1)
    parser.add_argument("--steps_per_scene", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="datasets/phase1_guided/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_mode", type=str, default="jsonl")
    parser.add_argument("--max_consecutive_failures", type=int, default=50)
    parser.add_argument("--policy_debug", action="store_true", help="Print per-step action decisions")
    args = parser.parse_args()

    start = time.time()
    result = run_guided_rollouts(
        num_scenes=args.num_scenes,
        steps_per_scene=args.steps_per_scene,
        output_dir=args.output_dir,
        seed=args.seed,
        save_mode=args.save_mode,
        max_consecutive_failures=args.max_consecutive_failures,
        policy_debug=args.policy_debug,
    )
    elapsed = time.time() - start

    print(f"Output: {result['output_path']}")
    print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
