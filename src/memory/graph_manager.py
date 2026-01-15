import numpy as np
import os
import networkx as nx
from src.core.graph_schema import SceneGraph, Node, Edge
from src.utils.semantic_matcher import SemanticMatcher

class GraphManager:
    def __init__(self, save_dir="Neural-TAMP/memory_data"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        # [核心架构] 底层存储升级为 NetworkX
        self.G = nx.DiGraph()
        self.matcher = SemanticMatcher()
        self.robot_id = "robot_agent"

    def override_global_graph(self, oracle_graph: SceneGraph):
        """从 Oracle 初始化全知图 (Init State)"""
        self.G.clear()
        
        # 1. 添加机器人
        r_pos = oracle_graph.robot_pose['position'] if oracle_graph.robot_pose else {'x':0,'y':0,'z':0}
        self.G.add_node(self.robot_id, type='agent', pos=[r_pos['x'], r_pos['y'], r_pos['z']], label='Robot')
        
        # 2. 添加实体节点
        for nid, node in oracle_graph.nodes.items():
            # 将原始 Node 对象存入 raw_node 以便兼容导出
            self.G.add_node(nid, type='object', pos=node.pos, label=node.label, 
                            state=node.state, room_id=node.room_id, raw_node=node)
            
        # 3. 添加 Room 包含关系
        for edge in oracle_graph.edges:
            if edge.relation == 'contains':
                self.G.add_edge(edge.source_id, edge.target_id, relation='contains')
                
        # 4. 计算初始几何关系
        self._compute_geometry_edges()

    def inject_action(self, action_dict, step_idx):
        """
        [Action as Node] 将动作直接写入图中，形成因果链
        """
        action_id = f"Action_{step_idx}_{action_dict['action']}"
        
        # 1. 插入 Action Node
        self.G.add_node(action_id, type='action', step=step_idx, **action_dict)
        
        # 2. 建立 Robot -> Action (Initiates)
        self.G.add_edge(self.robot_id, action_id, relation='initiates')
        
        # 3. 建立 Action -> Target (Targets)
        target = action_dict.get('target')
        if target and self.G.has_node(target):
            self.G.add_edge(action_id, target, relation='targets')
            
        return action_id

    def update_state(self, new_graph: nx.DiGraph):
        """更新记忆状态 (State Transition)"""
        self.G = new_graph

    def to_scene_graph(self) -> SceneGraph:
        """[兼容接口] NetworkX -> SceneGraph 对象"""
        sg = SceneGraph()
        # 恢复节点 (跳过 Action 节点，因为规划器只需要环境状态)
        for n, d in self.G.nodes(data=True):
            if d.get('type') == 'action': continue
            
            if 'raw_node' in d:
                sg.add_node(d['raw_node'])
            else:
                # 容错：从 NetworkX 属性重建 Node
                sg.add_node(Node(n, d.get('label','?'), d.get('pos',[0,0,0])))
        
        # 恢复边
        for u, v, d in self.G.edges(data=True):
            if d.get('relation') in ['on', 'inside', 'contains']:
                sg.add_edge(Edge(u, v, d['relation']))
        return sg

    def _compute_geometry_edges(self):
        """基于 NetworkX 的几何关系计算"""
        # 清除旧的几何边
        remove_list = [(u,v) for u,v,d in self.G.edges(data=True) if d.get('relation') in ['on', 'inside']]
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
                    # 这里可以复用 semantic matcher 的逻辑，为简洁略去，核心是 add_edge
                    pass