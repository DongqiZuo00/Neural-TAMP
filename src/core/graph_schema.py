import json
import numpy as np

class Relation:
    # Canonical physical relations
    CONTAINS = "contains"
    INSIDE = "inside"
    ON = "on"
    HOLDING = "holding"

    # Derived physical relations (mirrors)
    IN_ROOM = "in_room"
    IN_CONTAINER = "in_container"
    ON_TOP_OF = "on_top_of"
    HELD_BY = "held_by"

    # Action relations
    INITIATES = "initiates"
    TARGETS = "targets"

    # Geometric auxiliary relations
    NEAR = "near"

MIRROR_RELATION = {
    Relation.CONTAINS: Relation.IN_ROOM,
    Relation.INSIDE: Relation.IN_CONTAINER,
    Relation.ON: Relation.ON_TOP_OF,
    Relation.HOLDING: Relation.HELD_BY,
    Relation.IN_ROOM: Relation.CONTAINS,
    Relation.IN_CONTAINER: Relation.INSIDE,
    Relation.ON_TOP_OF: Relation.ON,
    Relation.HELD_BY: Relation.HOLDING,
}

CANONICAL_RELATIONS = {
    Relation.CONTAINS,
    Relation.INSIDE,
    Relation.ON,
    Relation.HOLDING,
}

DERIVED_RELATIONS = {
    Relation.IN_ROOM,
    Relation.IN_CONTAINER,
    Relation.ON_TOP_OF,
    Relation.HELD_BY,
}

ACTION_RELATIONS = {
    Relation.INITIATES,
    Relation.TARGETS,
}

GEOMETRIC_RELATIONS = {
    Relation.NEAR,
}

PHYSICAL_RELATIONS = CANONICAL_RELATIONS | DERIVED_RELATIONS

VALID_OPEN_STATES = {"open", "closed", "none"}

def normalize_state(state):
    if isinstance(state, dict):
        open_state = state.get("open_state", "none")
        held = state.get("held", False)
        return {"open_state": open_state, "held": bool(held)}
    if not isinstance(state, str):
        return {"open_state": "none", "held": False}
    state_lower = state.lower()
    if "open" in state_lower:
        open_state = "open"
    elif "closed" in state_lower:
        open_state = "closed"
    else:
        open_state = "none"
    held = "held" in state_lower
    return {"open_state": open_state, "held": held}

class Node:
    def __init__(self, id, label, pos, bbox=None, state=None, geometry=None, room_id=None):
        """
        :param room_id: [新增] 该物体所属的房间ID (e.g., "Room|0")
        """
        self.id = id
        self.label = label
        self.pos = pos
        self.bbox = bbox
        self.state = normalize_state(state)
        self.geometry = geometry if geometry else {}
        self.room_id = room_id # 核心新增：记录归属
    
    def to_dict(self):
        return {
            "id": self.id, 
            "label": self.label, 
            "pos": self.pos, 
            "bbox": self.bbox, 
            "state": self.state,
            "room_id": self.room_id, # 序列化
            "geometry": str(self.geometry)
        }

class Edge:
    def __init__(self, source_id, target_id, relation, attributes=None):
        self.source_id = source_id
        self.target_id = target_id
        self.relation = relation
        self.attributes = attributes if attributes else {}
    
    def to_dict(self):
        return {
            "source": self.source_id, 
            "target": self.target_id, 
            "relation": self.relation,
            "attributes": self.attributes
        }

class SceneGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.robot_pose = None

    def add_node(self, node: Node):
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge):
        self.edges.append(edge)

    def to_json_str(self):
        data = {
            "robot_pose": self.robot_pose, 
            "nodes": [n.to_dict() for n in self.nodes.values()], 
            "edges": [e.to_dict() for e in self.edges]
        }
        return json.dumps(data, indent=2)

    def to_prompt_text(self):
        lines = [f"Robot Pose: {self.robot_pose}"]
        lines.append("Nodes:")
        for n in self.nodes.values():
            # 优化 Prompt 显示
            loc_str = f" in {n.room_id}" if n.room_id else ""
            state = n.state if isinstance(n.state, dict) else normalize_state(n.state)
            open_state = state.get("open_state", "none")
            held = state.get("held", False)
            state_str = f" state(open={open_state}, held={held})"
            lines.append(f"- {n.id} ({n.label}){loc_str}: {n.pos}{state_str}")
        return "\n".join(lines)
