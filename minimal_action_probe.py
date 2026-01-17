import sys
import os
import json
import math

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from src.env.procthor_wrapper import ProcTHOREnv
from src.env.action_adapter import ProcTHORActionAdapter
from src.memory.graph_manager import GraphManager
from src.perception.oracle_interface import OracleInterface
from src.core.graph_schema import normalize_state
from scripts.data_collection import execute_action, get_reachable_positions
from src.env.visibility_recovery import ensure_visible_or_navigate


def _find_object_by_affordance(graph, *, pickupable=False, receptacle=False, openable=None):
    for node_id, data in graph.nodes(data=True):
        if data.get("type") != "object":
            continue
        raw_node = data.get("raw_node")
        geometry = raw_node.geometry if raw_node and isinstance(raw_node.geometry, dict) else {}
        if pickupable and not geometry.get("pickupable"):
            continue
        if receptacle and not geometry.get("receptacle"):
            continue
        if openable is not None and bool(geometry.get("openable")) != openable:
            continue
        return node_id, geometry
    return None, {}


def _is_closed(graph, node_id):
    state = normalize_state(graph.nodes[node_id].get("state"))
    return state.get("open_state") == "closed"


def _get_robot_id(graph):
    if "robot_agent" in graph.nodes:
        return "robot_agent"
    for node_id, data in graph.nodes(data=True):
        if data.get("type") == "agent":
            return node_id
    return "robot_agent"


def _get_robot_pos(graph):
    robot_id = _get_robot_id(graph)
    node = graph.nodes.get(robot_id)
    if not node:
        return None
    pos = node.get("pos")
    return list(pos) if pos is not None else None


def _near_position(pos_a, pos_b, threshold=1.0):
    if not pos_a or not pos_b:
        return False
    dx = pos_a[0] - pos_b[0]
    dy = pos_a[1] - pos_b[1]
    dz = pos_a[2] - pos_b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz) <= threshold


def _focus_target(action):
    if action["action"] == "PutObject":
        return action.get("receptacle_id")
    return action.get("target")


def main():
    print("=" * 60)
    print("🧪 Minimal Action Probe (no task generation)")
    print("=" * 60)

    try:
        env = ProcTHOREnv()
        oracle = OracleInterface(env)
        memory = GraphManager(save_dir="Neural-TAMP/memory_data")
        adapter = ProcTHORActionAdapter()
        print("✅ Modules Ready.")
    except Exception as exc:
        print(f"❌ Init Failed: {exc}")
        return

    try:
        env.change_scene(0)
    except Exception as exc:
        print(f"❌ Failed to load scene: {exc}")
        env.stop()
        return

    memory.override_global_graph(oracle.get_hierarchical_graph())
    scene_cache = {"reachable_positions": get_reachable_positions(env.controller)}

    pickup_id, _ = _find_object_by_affordance(memory.G, pickupable=True)
    receptacle_id, receptacle_geo = _find_object_by_affordance(memory.G, receptacle=True)

    if not pickup_id or not receptacle_id:
        print("❌ Could not find pickupable or receptacle object in scene.")
        env.stop()
        return

    if pickup_id == receptacle_id:
        print("❌ Pickup target equals receptacle; rerun with a different scene.")
        env.stop()
        return

    actions = [
        {"action": "NavigateTo", "target": pickup_id},
        {"action": "PickUp", "target": pickup_id},
        {"action": "NavigateTo", "target": receptacle_id},
    ]

    if receptacle_geo.get("openable") and _is_closed(memory.G, receptacle_id):
        actions.append({"action": "Open", "target": receptacle_id})

    actions.append({"action": "PutObject", "target": pickup_id, "receptacle_id": receptacle_id})

    if receptacle_geo.get("openable"):
        actions.append({"action": "Close", "target": receptacle_id})

    print("Selected IDs:")
    print(json.dumps({"pickup_id": pickup_id, "receptacle_id": receptacle_id}, indent=2))
    print("Planned actions:")
    print(json.dumps(actions, indent=2))

    for step, action in enumerate(actions):
        if action["action"] != "NavigateTo":
            focus_target = _focus_target(action)
            if focus_target:
                robot_pos = _get_robot_pos(memory.G)
                target_pos = memory.G.nodes.get(focus_target, {}).get("pos")
                if not _near_position(robot_pos, target_pos, threshold=1.0):
                    nav_action = {"action": "NavigateTo", "target": focus_target}
                    nav_ok, nav_err = execute_action(
                        env,
                        adapter,
                        nav_action,
                        memory.G,
                        scene_cache=scene_cache,
                    )
                    nav_status = "✅" if nav_ok else "❌"
                    print(f"Step {step}: auto NavigateTo {nav_status} {nav_err}")
                    memory.override_global_graph(oracle.get_hierarchical_graph())
                    if not nav_ok:
                        break
            visibility = ensure_visible_or_navigate(
                env,
                adapter,
                focus_target,
                memory.G,
                scene_cache=scene_cache,
            )
            if not visibility.ok:
                print(f"Step {step}: visibility recovery ❌ {visibility.reason}")
                break
            memory.override_global_graph(oracle.get_hierarchical_graph())
        success, error_msg = execute_action(env, adapter, action, memory.G, scene_cache=scene_cache)
        status = "✅" if success else "❌"
        print(f"Step {step}: {action['action']} {status} {error_msg}")
        memory.override_global_graph(oracle.get_hierarchical_graph())
        if not success:
            break

    env.stop()


if __name__ == "__main__":
    main()
