import numpy as np
import os
from src.core.graph_schema import SceneGraph, Node, Edge
from src.utils.semantic_matcher import SemanticMatcher

class GraphManager:
    def __init__(self, save_dir="Neural-TAMP/memory_data"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.global_graph = SceneGraph()
        self.matcher = SemanticMatcher()

    def override_global_graph(self, perfect_graph: SceneGraph):
        """
        覆盖模式：只计算严格的同房间几何关系
        """
        self.global_graph = perfect_graph
        self._recompute_edges_strict()

    def _recompute_edges_strict(self):
        """
        [严格模式] 计算 Edge
        1. 保留 Oracle 给出的 Room->contains->Object 关系
        2. 计算 Object<->Object 关系，但必须满足:
           - 处于同一个 Room (room_id 相同)
           - 满足几何条件 (on / inside)
        3. 彻底移除 close_to
        """
        # 1. 保留现有的语义 Edge (即 Room contains Object)
        semantic_edges = [e for e in self.global_graph.edges if e.relation == "contains"]
        self.global_graph.edges = semantic_edges 

        # 获取所有物体节点 (排除房间节点)
        obj_nodes = [n for n in self.global_graph.nodes.values() if "Room|" not in n.id]

        for i in range(len(obj_nodes)):
            for j in range(len(obj_nodes)):
                if i == j: continue
                a, b = obj_nodes[i], obj_nodes[j]
                
                # --- 核心过滤: 隔墙无 Edge ---
                # 如果两个物体不在同一个房间，直接跳过，不做任何计算
                if a.room_id is None or b.room_id is None:
                    continue # 甚至不属于任何房间的物体也不计算
                
                if a.room_id != b.room_id:
                    continue 

                # --- 计算几何关系 ---
                vec = np.array(a.pos) - np.array(b.pos)
                
                h_dist = np.linalg.norm(vec[[0, 2]]) # 水平距离
                v_dist = vec[1] # 垂直距离 (y轴, a - b)
                
                # 判定 On / Inside
                # 这里的阈值可以根据需要微调
                if h_dist < 0.5: 
                    # 如果水平很近，且 a 在 b 上方 (0.05 ~ 0.8米)
                    if 0.05 < v_dist < 0.8:
                        if self.matcher.is_anchor(b.label): 
                            self.global_graph.add_edge(Edge(a.id, b.id, "on"))
                        elif self.matcher.is_container(b.label): 
                            self.global_graph.add_edge(Edge(a.id, b.id, "inside"))

    # (保留 get_rag_context 和 save_snapshot，不需要修改)
    def get_rag_context(self, query):
        q_emb = self.matcher.model.encode(query, convert_to_tensor=True)
        from sentence_transformers import util
        hits = set()
        for node in self.global_graph.nodes.values():
            n_emb = self.matcher.model.encode(node.label, convert_to_tensor=True)
            if util.cos_sim(q_emb, n_emb) > 0.4: 
                hits.add(node.id)
                # 关联房间
                if node.room_id: hits.add(node.room_id)
        
        # 如果没搜到，搜房间名
        if not hits:
            for node in self.global_graph.nodes.values():
                if "Room" in node.id:
                     n_emb = self.matcher.model.encode(node.label, convert_to_tensor=True)
                     if util.cos_sim(q_emb, n_emb) > 0.4: hits.add(node.id)

        return self.global_graph.to_prompt_text()

    def save_snapshot(self):
        with open(f"{self.save_dir}/memory.json", 'w') as f: 
            f.write(self.global_graph.to_json_str())