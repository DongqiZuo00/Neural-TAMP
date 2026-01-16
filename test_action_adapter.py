import argparse
from typing import Any

import networkx as nx

from src.core.graph_schema import Relation, normalize_state
from src.env.action_adapter import ProcTHORActionAdapter


def build_min_graph(with_holding: bool = True) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node("robot_agent", type="agent", pos=[0.0, 0.0, 0.0], state=normalize_state({}))
    graph.add_node("fridge_1", type="object", pos=[1.0, 0.0, 0.0], state=normalize_state({}))
    graph.add_node("apple_1", type="object", pos=[1.5, 0.0, 0.5], state=normalize_state({}))
    graph.add_node("table_1", type="object", pos=[2.0, 0.0, 1.0], state=normalize_state({}))
    if with_holding:
        graph.add_edge("robot_agent", "apple_1", relation=Relation.HOLDING)
    return graph


def _assert_keys(action: str, thor_kwargs: dict[str, Any], allowed: set[str]) -> None:
    extra = set(thor_kwargs.keys()) - allowed
    assert not extra, f"{action} has unexpected keys: {extra}"


def run_unit_tests() -> None:
    adapter = ProcTHORActionAdapter()
    graph = build_min_graph()

    action_sets = {
        "NavigateTo": [
            {"action": "NavigateTo", "target": "fridge_1"},
            {"action": "NavigateTo", "target": "apple_1", "foo": "bar"},
            {"action": "NavigateTo", "target": "table_1"},
        ],
        "Open": [
            {"action": "Open", "target": "fridge_1"},
            {"action": "Open", "target": "apple_1", "extra": "ignored"},
            {"action": "Open", "target": "table_1"},
        ],
        "Close": [
            {"action": "Close", "target": "fridge_1"},
            {"action": "Close", "target": "apple_1"},
            {"action": "Close", "target": "table_1"},
        ],
        "PickUp": [
            {"action": "PickUp", "target": "apple_1"},
            {"action": "PickUp", "target": "fridge_1"},
            {"action": "PickUp", "target": "table_1"},
        ],
        "PutObject": [
            {"action": "PutObject", "target": "apple_1"},
            {"action": "PutObject", "receptacle_id": "table_1"},
            {"action": "PutObject", "target": "fridge_1", "receptacle_id": "table_1"},
        ],
    }

    allowed_keys = {
        "TeleportFull": {"action", "x", "y", "z", "rotation", "horizon", "standing"},
        "OpenObject": {"action", "objectId"},
        "CloseObject": {"action", "objectId"},
        "PickupObject": {"action", "objectId"},
        "PutObject": {"action", "objectId"},
    }

    for action, candidates in action_sets.items():
        for action_dict in candidates:
            ok, reason = adapter.validate_action_dict(action_dict)
            assert ok, f"validate_action_dict failed for {action_dict}: {reason}"

            thor_kwargs, ok, reason = adapter.to_thor_step(action_dict, graph)
            assert ok, f"to_thor_step failed for {action_dict}: {reason}"
            assert "action" in thor_kwargs
            schema_keys = allowed_keys[thor_kwargs["action"]]
            _assert_keys(action, thor_kwargs, schema_keys)
            if thor_kwargs["action"] == "TeleportFull":
                assert all(k in thor_kwargs for k in ("x", "y", "z")), "NavigateTo missing coords"
            if thor_kwargs["action"] in {"OpenObject", "CloseObject", "PickupObject"}:
                assert "objectId" in thor_kwargs
            if thor_kwargs["action"] == "PutObject":
                assert "receptacleObjectId" not in thor_kwargs

    print("[Unit] All adapter schema checks passed.")


def run_smoke_test() -> None:
    from src.env.procthor_wrapper import ProcTHOREnv
    from src.memory.graph_manager import GraphManager
    from src.perception.oracle_interface import OracleInterface

    adapter = ProcTHORActionAdapter()
    env = ProcTHOREnv()
    manager = GraphManager(debug=False)

    try:
        env.change_scene(0)
        oracle = OracleInterface(env)
        manager.override_global_graph(oracle.get_hierarchical_graph())
        graph_t = manager.G
        targets = [n for n, d in graph_t.nodes(data=True) if d.get("type") == "object"]
        if not targets:
            raise AssertionError("No object targets found for smoke test")
        target = targets[0]

        action_sequence = [
            {"action": "NavigateTo", "target": target},
            {"action": "Open", "target": target},
            {"action": "PickUp", "target": target},
            {"action": "PutObject", "target": target},
        ]

        for idx in range(10):
            action_dict = action_sequence[idx % len(action_sequence)]
            thor_kwargs, ok, reason = adapter.to_thor_step(action_dict, graph_t)
            assert ok, f"to_thor_step failed during smoke test: {reason}"
            try:
                env.controller.step(**thor_kwargs)
            except Exception as exc:
                raise AssertionError(f"Controller step raised exception: {exc}") from exc

        print("[Smoke] Controller step schema accepted for 10 steps.")
    finally:
        env.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Action adapter tests")
    parser.add_argument("--smoke", action="store_true", help="Run lightweight env smoke test")
    args = parser.parse_args()

    run_unit_tests()
    if args.smoke:
        run_smoke_test()


if __name__ == "__main__":
    main()
