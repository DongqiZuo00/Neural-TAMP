import sys
import os
import random
import shutil
import json
import copy

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from src.env.procthor_wrapper import ProcTHOREnv
from src.env.action_adapter import ProcTHORActionAdapter
from src.env.action_visibility import ensure_object_visible
from src.memory.graph_manager import GraphManager
from src.perception.oracle_interface import OracleInterface
from src.task.task_generator import TaskGenerator
from src.planning.decomposer import TaskDecomposer
from scripts.data_collection import export_graph_canonical, graph_diff, sanitize_graph, execute_action, get_reachable_positions


def _plan_subgoals(planner: TaskDecomposer, instruction, scene_graph):
    decomp_prompt = planner.prompter.build_decomposition_prompt(instruction, scene_graph)
    result = planner.llm.predict("You are a helper.", decomp_prompt)
    return result.get("subgoals", [])


def _plan_actions_for_subgoal(planner: TaskDecomposer, subgoal, scene_graph):
    action_prompt = planner.prompter.build_action_prompt([subgoal], scene_graph)
    result = planner.llm.predict("You are a robot executor.", action_prompt)
    return result.get("actions", [])


def _is_executable_error(error_msg: str) -> bool:
    return not error_msg.startswith("INVALID_ACTION_SCHEMA") and not error_msg.startswith("API_SCHEMA_BUG")


def main():
    print("=" * 60)
    print("🚀 Neural-TAMP: LLM Planner Data Collection")
    print("=" * 60)

    output_dir = "Neural-TAMP/vis_output/main_test"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    try:
        env = ProcTHOREnv()
        oracle = OracleInterface(env)
        memory = GraphManager(save_dir="Neural-TAMP/memory_data")
        task_gen = TaskGenerator()
        planner = TaskDecomposer(model_name="gpt-4o")
        adapter = ProcTHORActionAdapter()
        print("✅ Modules Ready.")
    except Exception as e:
        print(f"❌ Init Failed: {e}")
        return

    dataset_log = []
    total_subgoals = 0
    executable_subgoals = 0
    success_subgoals = 0

    rng = random.Random(0)
    candidate_indices = rng.sample(range(10000), 50)
    task_count = 0

    for idx in candidate_indices:
        if task_count >= 10:
            break
        print(f"\n🎬 Scene {idx} ({task_count + 1}/10)")

        try:
            env.change_scene(idx)
        except Exception:
            continue

        scene_cache = {"reachable_positions": get_reachable_positions(env.controller)}

        memory.override_global_graph(oracle.get_hierarchical_graph())
        ok, errors = sanitize_graph(memory.G)
        if not ok:
            print(f"   ❌ Invalid initial graph: {errors}")
            continue

        current_sg = memory.to_scene_graph()
        instruction, meta = task_gen.generate(current_sg)
        if not instruction:
            continue
        print(f"   Task: {instruction}")

        subgoals = _plan_subgoals(planner, instruction, current_sg)
        if not subgoals:
            print("   ❌ No subgoals returned.")
            continue

        task_record = {
            "scene": idx,
            "task": instruction,
            "task_meta": meta,
            "subgoal_records": [],
        }

        task_failed = False
        for subgoal in subgoals:
            total_subgoals += 1
            graph_t = copy.deepcopy(memory.G)
            current_sg = memory.to_scene_graph()

            actions = _plan_actions_for_subgoal(planner, subgoal, current_sg)
            action_results = []
            subgoal_success = True
            subgoal_executable = True
            reject_reason = ""

            if not actions:
                subgoal_success = False
                subgoal_executable = False
                reject_reason = "no_actions_generated"
            else:
                for action in actions:
                    target_id = action.get("target")
                    if target_id:
                        visible = ensure_object_visible(env.controller, target_id)
                        if not visible:
                            success = False
                            error_msg = "object_not_visible_after_scan"
                            action_results.append(
                                {
                                    "action": action,
                                    "success": success,
                                    "error_msg": error_msg,
                                }
                            )
                            subgoal_success = False
                            subgoal_executable = False
                            reject_reason = error_msg
                            break

                    success, error_msg = execute_action(env, adapter, action, memory.G, scene_cache=scene_cache)
                    action_results.append(
                        {
                            "action": action,
                            "success": success,
                            "error_msg": error_msg,
                        }
                    )
                    if not success:
                        subgoal_success = False
                    if not _is_executable_error(error_msg):
                        subgoal_executable = False
                    if not success:
                        reject_reason = error_msg or "action_failed"
                        break

            memory.override_global_graph(oracle.get_hierarchical_graph())
            graph_t1 = memory.G

            task_record["subgoal_records"].append(
                {
                    "subgoal": subgoal,
                    "actions": actions,
                    "action_results": action_results,
                    "success": subgoal_success,
                    "executable": subgoal_executable,
                    "reject_reason": reject_reason,
                    "G_t": export_graph_canonical(graph_t),
                    "G_t1": export_graph_canonical(graph_t1),
                    "delta": graph_diff(graph_t, graph_t1),
                }
            )

            if subgoal_executable:
                executable_subgoals += 1
            if subgoal_success:
                success_subgoals += 1

            if not subgoal_success:
                reason_note = f" ({reject_reason})" if reject_reason else ""
                print(f"   ❌ Subgoal rejected: {subgoal}{reason_note}")
                task_failed = True
                break

        dataset_log.append(task_record)

        with open(f"{output_dir}/log.json", "w") as f:
            json.dump(dataset_log, f, indent=2)

        task_count += 1
        if task_failed:
            continue

    env.stop()

    success_rate = success_subgoals / total_subgoals if total_subgoals else 0.0
    executable_rate = executable_subgoals / total_subgoals if total_subgoals else 0.0
    summary = {
        "tasks_completed": task_count,
        "total_subgoals": total_subgoals,
        "success_subgoals": success_subgoals,
        "executable_subgoals": executable_subgoals,
        "success_rate": success_rate,
        "executable_rate": executable_rate,
    }

    with open(f"{output_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n✅ Pipeline Finished.")
    print(f"Subgoal success rate: {success_rate:.2%}")
    print(f"Subgoal executable rate: {executable_rate:.2%}")


if __name__ == "__main__":
    main()
