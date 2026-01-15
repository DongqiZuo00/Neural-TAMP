import json
import numpy as np

class Node:
    def __init__(self, id, label, pos, bbox=None, state="default", geometry=None, room_id=None):
        """
        :param room_id: [新增] 该物体所属的房间ID (e.g., "Room|0")
        """
        self.id = id
        self.label = label
        self.pos = pos
        self.bbox = bbox
        self.state = state
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
            lines.append(f"- {n.id} ({n.label}){loc_str}: {n.pos}")
        return "\n".join(lines)