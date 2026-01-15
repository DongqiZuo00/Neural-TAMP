import networkx as nx
from src.memory.graph_manager import sync_bidirectional_edges, validate_graph_schema

def sanitize_graph(graph: nx.DiGraph) -> tuple[bool, list[dict]]:
    """
    Sync bidirectional physical edges and validate schema.
    Returns (ok, errors).
    """
    sync_bidirectional_edges(graph)
    return validate_graph_schema(graph)

def should_keep_sample(graph: nx.DiGraph) -> bool:
    ok, _ = sanitize_graph(graph)
    return ok
