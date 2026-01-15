import networkx as nx
from src.core.graph_schema import SceneGraph, Node, Edge, Relation
from src.memory.graph_manager import GraphManager, sync_bidirectional_edges, validate_graph_schema
from src.world_model.rule_dynamics import RuleBasedDynamics

def assert_edge(graph: nx.DiGraph, source, target, relation):
    assert graph.has_edge(source, target), f"Missing edge {source}->{target}"
    assert graph.edges[source, target].get("relation") == relation, (
        f"Expected {relation} on {source}->{target}"
    )

def main():
    sg = SceneGraph()
    sg.robot_pose = {"position": {"x": 0, "y": 0, "z": 0}, "rotation": {"y": 0}}

    room = Node("room_1", "Room", (0.0, 0.0, 0.0), state={"open_state": "none", "held": False})
    table = Node("table_1", "Table", (1.0, 0.0, 0.0), state={"open_state": "none", "held": False})
    fridge = Node("fridge_1", "Fridge", (2.0, 0.0, 0.0), state={"open_state": "closed", "held": False})
    apple = Node("apple_1", "Apple", (2.0, 0.0, 0.0), state={"open_state": "none", "held": False})

    sg.add_node(room)
    sg.add_node(table)
    sg.add_node(fridge)
    sg.add_node(apple)

    sg.add_edge(Edge("room_1", "table_1", Relation.CONTAINS))
    sg.add_edge(Edge("room_1", "fridge_1", Relation.CONTAINS))
    sg.add_edge(Edge("fridge_1", "apple_1", Relation.INSIDE))

    manager = GraphManager(debug=True)
    manager.override_global_graph(sg)
    graph = manager.G

    assert_edge(graph, "fridge_1", "apple_1", Relation.INSIDE)
    assert_edge(graph, "apple_1", "fridge_1", Relation.IN_CONTAINER)
    ok, errors = validate_graph_schema(graph)
    assert ok, f"Validation failed after override: {errors}"

    dynamics = RuleBasedDynamics()
    graph, ok, msg = dynamics.predict(graph, {"action": "Open", "target": "fridge_1"})
    assert ok, msg
    graph, ok, msg = dynamics.predict(graph, {"action": "PickUp", "target": "apple_1"})
    assert ok, msg
    sync_bidirectional_edges(graph)
    ok, errors = validate_graph_schema(graph)
    assert ok, f"Validation failed after pickup: {errors}"
    assert_edge(graph, "robot_agent", "apple_1", Relation.HOLDING)
    assert_edge(graph, "apple_1", "robot_agent", Relation.HELD_BY)
    assert graph.nodes["apple_1"]["state"]["held"] is True

    graph, ok, msg = dynamics.predict(
        graph,
        {"action": "PutObject", "target": "apple_1", "receptacle_id": "table_1"},
    )
    assert ok, msg
    sync_bidirectional_edges(graph)
    ok, errors = validate_graph_schema(graph)
    assert ok, f"Validation failed after place: {errors}"
    assert_edge(graph, "table_1", "apple_1", Relation.ON)
    assert_edge(graph, "apple_1", "table_1", Relation.ON_TOP_OF)

    in_container = [
        e for e in graph.out_edges("apple_1", data=True)
        if e[2].get("relation") == Relation.IN_CONTAINER
    ]
    on_top = [
        e for e in graph.out_edges("apple_1", data=True)
        if e[2].get("relation") == Relation.ON_TOP_OF
    ]
    assert not (in_container and on_top), "apple_1 cannot be in_container and on_top_of"

    print("Phase 0 minimal test passed.")

if __name__ == "__main__":
    main()
