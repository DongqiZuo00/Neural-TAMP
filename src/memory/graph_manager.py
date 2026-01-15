import numpy as np
import os
import networkx as nx
from src.core.graph_schema import (
    SceneGraph,
    Node,
    Edge,
    Relation,
    MIRROR_RELATION,
    CANONICAL_RELATIONS,
    DERIVED_RELATIONS,
    PHYSICAL_RELATIONS,
    ACTION_RELATIONS,
    GEOMETRIC_RELATIONS,
    VALID_OPEN_STATES,
    normalize_state,
)

def sync_bidirectional_edges(G: nx.DiGraph) -> None:
    edges = list(G.edges(data=True))
    to_remove = []
    to_add = []

    for u, v, data in edges:
        relation = data.get("relation")
        if relation in ACTION_RELATIONS:
            continue
        if relation in DERIVED_RELATIONS:
            mirror = MIRROR_RELATION.get(relation)
            if mirror is None or not G.has_edge(v, u) or G.edges[v, u].get("relation") != mirror:
                to_remove.append((u, v))
        elif relation in CANONICAL_RELATIONS:
            mirror = MIRROR_RELATION.get(relation)
            if mirror is None:
                continue
            if not G.has_edge(v, u) or G.edges[v, u].get("relation") != mirror:
                new_attrs = dict(data)
                new_attrs["relation"] = mirror
                to_add.append((v, u, new_attrs))

    if to_remove:
        G.remove_edges_from(to_remove)
    for u, v, attrs in to_add:
        G.add_edge(u, v, **attrs)

def validate_graph_schema(G: nx.DiGraph):
    errors = []

    for u, v, data in G.edges(data=True):
        relation = data.get("relation")
        if relation in ACTION_RELATIONS:
            continue
        if relation in CANONICAL_RELATIONS:
            mirror = MIRROR_RELATION.get(relation)
            if mirror and (not G.has_edge(v, u) or G.edges[v, u].get("relation") != mirror):
                errors.append({
                    "type": "MIRROR_MISSING",
                    "edge": (u, v, relation),
                    "expected_mirror": (v, u, mirror),
                })
        elif relation in DERIVED_RELATIONS:
            mirror = MIRROR_RELATION.get(relation)
            if mirror and (not G.has_edge(v, u) or G.edges[v, u].get("relation") != mirror):
                errors.append({
                    "type": "ORPHAN_DERIVED",
                    "edge": (u, v, relation),
                    "expected_canonical": (v, u, mirror),
                })

    for node_id, data in G.nodes(data=True):
        if data.get("type") != "object":
            continue
        derived_out = {
            Relation.IN_ROOM: [],
            Relation.IN_CONTAINER: [],
            Relation.ON_TOP_OF: [],
            Relation.HELD_BY: [],
        }
        for _, v, edata in G.out_edges(node_id, data=True):
            rel = edata.get("relation")
            if rel in derived_out:
                derived_out[rel].append((node_id, v))

        for rel, edges in derived_out.items():
            if len(edges) > 1:
                errors.append({
                    "type": "CARDINALITY_VIOLATION",
                    "node": node_id,
                    "relation": rel,
                    "edges": edges,
                })

        if derived_out[Relation.IN_CONTAINER] and derived_out[Relation.ON_TOP_OF]:
            errors.append({
                "type": "MUTEX_VIOLATION",
                "node": node_id,
                "relations": [Relation.IN_CONTAINER, Relation.ON_TOP_OF],
            })
        if derived_out[Relation.HELD_BY] and (derived_out[Relation.IN_CONTAINER] or derived_out[Relation.ON_TOP_OF]):
            errors.append({
                "type": "MUTEX_VIOLATION",
                "node": node_id,
                "relations": [Relation.HELD_BY, Relation.IN_CONTAINER, Relation.ON_TOP_OF],
            })

    for node_id, data in G.nodes(data=True):
        if data.get("type") != "object":
            continue
        state = data.get("state")
        if not isinstance(state, dict):
            errors.append({
                "type": "STATE_INVALID",
                "node": node_id,
                "state": state,
                "reason": "state must be a dict with open_state and held",
            })
            continue
        open_state = state.get("open_state")
        held = state.get("held")
        if open_state not in VALID_OPEN_STATES:
            errors.append({
                "type": "STATE_INVALID",
                "node": node_id,
                "state": state,
                "reason": "invalid open_state",
            })
        if not isinstance(held, bool):
            errors.append({
                "type": "STATE_INVALID",
                "node": node_id,
                "state": state,
                "reason": "held must be bool",
            })

    return len(errors) == 0, errors

def sync_bidirectional_edges(G: nx.DiGraph) -> None:
    edges = list(G.edges(data=True))
    to_remove = []
    to_add = []

    for u, v, data in edges:
        relation = data.get("relation")
        if relation in ACTION_RELATIONS:
            continue
        if relation in DERIVED_RELATIONS:
            mirror = MIRROR_RELATION.get(relation)
            if mirror is None or not G.has_edge(v, u) or G.edges[v, u].get("relation") != mirror:
                to_remove.append((u, v))
        elif relation in CANONICAL_RELATIONS:
            mirror = MIRROR_RELATION.get(relation)
            if mirror is None:
                continue
            if not G.has_edge(v, u):
                new_attrs = dict(data)
                new_attrs["relation"] = mirror
                to_add.append((v, u, new_attrs))

    if to_remove:
        G.remove_edges_from(to_remove)
    for u, v, attrs in to_add:
        G.add_edge(u, v, **attrs)

def validate_graph_schema(G: nx.DiGraph):
    errors = []

    for u, v, data in G.edges(data=True):
        relation = data.get("relation")
        if relation in ACTION_RELATIONS:
            continue
        if relation in CANONICAL_RELATIONS:
            mirror = MIRROR_RELATION.get(relation)
            if mirror and (not G.has_edge(v, u) or G.edges[v, u].get("relation") != mirror):
                errors.append({
                    "type": "MIRROR_MISSING",
                    "edge": (u, v, relation),
                    "expected_mirror": (v, u, mirror),
                })
        elif relation in DERIVED_RELATIONS:
            mirror = MIRROR_RELATION.get(relation)
            if mirror and (not G.has_edge(v, u) or G.edges[v, u].get("relation") != mirror):
                errors.append({
                    "type": "ORPHAN_DERIVED",
                    "edge": (u, v, relation),
                    "expected_canonical": (v, u, mirror),
                })

    for node_id, data in G.nodes(data=True):
        if data.get("type") != "object":
            continue
        derived_out = {
            Relation.IN_ROOM: [],
            Relation.IN_CONTAINER: [],
            Relation.ON_TOP_OF: [],
            Relation.HELD_BY: [],
        }
        for _, v, edata in G.out_edges(node_id, data=True):
            rel = edata.get("relation")
            if rel in derived_out:
                derived_out[rel].append((node_id, v))

        for rel, edges in derived_out.items():
            if len(edges) > 1:
                errors.append({
                    "type": "CARDINALITY_VIOLATION",
                    "node": node_id,
                    "relation": rel,
                    "edges": edges,
                })

        if derived_out[Relation.IN_CONTAINER] and derived_out[Relation.ON_TOP_OF]:
            errors.append({
                "type": "MUTEX_VIOLATION",
                "node": node_id,
                "relations": [Relation.IN_CONTAINER, Relation.ON_TOP_OF],
            })
        if derived_out[Relation.HELD_BY] and (derived_out[Relation.IN_CONTAINER] or derived_out[Relation.ON_TOP_OF]):
            errors.append({
                "type": "MUTEX_VIOLATION",
                "node": node_id,
                "relations": [Relation.HELD_BY, Relation.IN_CONTAINER, Relation.ON_TOP_OF],
            })

    for node_id, data in G.nodes(data=True):
        if data.get("type") != "object":
            continue
        state = data.get("state")
        if not isinstance(state, dict):
            errors.append({
                "type": "STATE_INVALID",
                "node": node_id,
                "state": state,
                "reason": "state must be a dict with open_state and held",
            })
            continue
        open_state = state.get("open_state")
        held = state.get("held")
        if open_state not in VALID_OPEN_STATES:
            errors.append({
                "type": "STATE_INVALID",
                "node": node_id,
                "state": state,
                "reason": "invalid open_state",
            })
        if not isinstance(held, bool):
            errors.append({
                "type": "STATE_INVALID",
                "node": node_id,
                "state": state,
                "reason": "held must be bool",
            })

    return len(errors) == 0, errors

class GraphManager:
    def __init__(self, save_dir="Neural-TAMP/memory_data", debug=False):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        # [核心架构] 底层存储升级为 NetworkX
        self.G = nx.DiGraph()
        self.robot_id = "robot_agent"
        self.debug = debug

    def override_global_graph(self, oracle_graph: SceneGraph):
        """从 Oracle 初始化全知图 (Init State)"""
        self.G.clear()
        
        # 1. 添加机器人
        r_pos = oracle_graph.robot_pose['position'] if oracle_graph.robot_pose else {'x':0,'y':0,'z':0}
        self.G.add_node(self.robot_id, type='agent', pos=[r_pos['x'], r_pos['y'], r_pos['z']], label='Robot')
        
        # 2. 添加实体节点
        for nid, node in oracle_graph.nodes.items():
            node_type = "room" if nid.startswith("Room|") or node.label == "Room" else "object"
            # 将原始 Node 对象存入 raw_node 以便兼容导出
            self.G.add_node(nid, type=node_type, pos=node.pos, label=node.label, 
                            state=normalize_state(node.state), room_id=node.room_id, raw_node=node)
            
        # 3. 添加物理关系 (canonical)
        for edge in oracle_graph.edges:
            if edge.relation in CANONICAL_RELATIONS:
                self.G.add_edge(edge.source_id, edge.target_id, relation=edge.relation)
                
        # 4. 计算初始几何关系
        self._compute_geometry_edges()
        sync_bidirectional_edges(self.G)
        ok, errors = validate_graph_schema(self.G)
        if not ok:
            if self.debug:
                raise AssertionError(f"Graph schema validation failed: {errors}")
            print(f"[GraphManager] Schema validation failed: {errors}")

    def inject_action(self, action_dict, step_idx):
        """
        [Action as Node] 将动作直接写入图中，形成因果链
        """
        action_id = f"Action_{step_idx}_{action_dict['action']}"
        
        # 1. 插入 Action Node
        self.G.add_node(action_id, type='action', step=step_idx, **action_dict)
        
        # 2. 建立 Robot -> Action (Initiates)
        self.G.add_edge(self.robot_id, action_id, relation=Relation.INITIATES)
        
        # 3. 建立 Action -> Target (Targets)
        target = action_dict.get('target')
        if target and self.G.has_node(target):
            self.G.add_edge(action_id, target, relation=Relation.TARGETS)
            
        return action_id

    def update_state(self, new_graph: nx.DiGraph):
        """更新记忆状态 (State Transition)"""
        self.G = new_graph

    def to_scene_graph(self, export_mode="canonical_only") -> SceneGraph:
        """[兼容接口] NetworkX -> SceneGraph 对象"""
        sg = SceneGraph()
        # 恢复节点 (跳过 Action 节点，因为规划器只需要环境状态)
        for n, d in self.G.nodes(data=True):
            if d.get('type') == 'action': continue
            
            if 'raw_node' in d:
                raw_node = d['raw_node']
                raw_node.state = normalize_state(raw_node.state)
                sg.add_node(raw_node)
            else:
                # 容错：从 NetworkX 属性重建 Node
                sg.add_node(Node(n, d.get('label','?'), d.get('pos',[0,0,0]), state=d.get("state")))
        
        # 恢复边
        for u, v, d in self.G.edges(data=True):
            relation = d.get('relation')
            if relation in ACTION_RELATIONS:
                continue
            if export_mode == "canonical_only":
                if relation in CANONICAL_RELATIONS:
                    sg.add_edge(Edge(u, v, relation))
            else:
                if relation in PHYSICAL_RELATIONS:
                    sg.add_edge(Edge(u, v, relation))
        return sg

    def _compute_geometry_edges(self):
        """基于 NetworkX 的几何关系计算"""
        # 清除旧的几何辅助边
        remove_list = [
            (u, v)
            for u, v, d in self.G.edges(data=True)
            if d.get("relation") in GEOMETRIC_RELATIONS
        ]
        self.G.remove_edges_from(remove_list)

        obj_nodes = [n for n, d in self.G.nodes(data=True) if d.get('type')=='object']
        
        for i in range(len(obj_nodes)):
            for j in range(len(obj_nodes)):
                if i == j: continue
                u, v = obj_nodes[i], obj_nodes[j]
                
                # 隔墙过滤
                if self.G.nodes[u].get('room_id') != self.G.nodes[v].get('room_id'): continue
                
                # 几何计算
                pos_u = np.array(self.G.nodes[u]['pos'])
                pos_v = np.array(self.G.nodes[v]['pos'])
                dist = np.linalg.norm(pos_u - pos_v)
                
                if dist < 1.0: # 简化阈值
                    self.G.add_edge(u, v, relation=Relation.NEAR, distance=float(dist))
