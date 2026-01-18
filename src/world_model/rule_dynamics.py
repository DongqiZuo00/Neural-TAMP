import networkx as nx
import copy
from src.core.graph_schema import Relation, normalize_state

class RuleBasedDynamics:
    """
    [World Model] 规则动力学引擎
    职责：输入当前图和动作，预测下一时刻的图结构。
    特点：不修改原图，返回新的图状态 (Deep Copy)。
    """
    def __init__(self, robot_id="robot_agent"):
        self.robot_id = robot_id

    def predict(self, current_graph: nx.DiGraph, action_dict: dict) -> tuple[nx.DiGraph, bool, str]:
        # 1. 在平行宇宙中推演 (Deep Copy)
        next_graph = copy.deepcopy(current_graph)
        
        # 2. 解析动作参数
        action_type = action_dict.get('action')
        target_id = action_dict.get('target')
        
        # 3. 路由物理规则
        try:
            if action_type == "PickUp":
                return self._rule_pick(next_graph, target_id)
            elif action_type == "PutObject":
                dest_id = action_dict.get('receptacle_id')
                return self._rule_place(next_graph, target_id, dest_id)
            elif action_type == "NavigateTo":
                if target_id in next_graph and "pos" in next_graph.nodes[target_id]:
                    next_graph.nodes[self.robot_id]["pos"] = next_graph.nodes[target_id]["pos"]
                return next_graph, True, f"Navigated to {target_id}"
            elif action_type in ["Open", "Close"]:
                return self._rule_state_change(next_graph, target_id, action_type)
            else:
                return next_graph, False, f"Unknown Action: {action_type}"
        except Exception as e:
            return next_graph, False, f"Dynamics Error: {str(e)}"

    def _rule_pick(self, G, obj_id):
        if not G.has_node(obj_id): return G, False, "Target not found"
        
        # [Effect] 1. 断开物理连接 (on, inside)
        edges_to_remove = [
            (u, v)
            for u, v, d in G.in_edges(obj_id, data=True)
            if d.get("relation") in [Relation.ON, Relation.INSIDE]
        ]
        G.remove_edges_from(edges_to_remove)
        
        # [Effect] 2. 建立 Holding 连接
        G.add_edge(self.robot_id, obj_id, relation=Relation.HOLDING)
        state = normalize_state(G.nodes[obj_id].get("state"))
        state["held"] = True
        G.nodes[obj_id]["state"] = state
        
        # [Effect] 3. 位置同步
        if self.robot_id in G:
            G.nodes[obj_id]['pos'] = G.nodes[self.robot_id]['pos']
            
        return G, True, "Picked Up"

    def _rule_place(self, G, obj_id, dest_id):
        # 检查是否持有
        if not G.has_edge(self.robot_id, obj_id) or G.edges[self.robot_id, obj_id].get("relation") != Relation.HOLDING:
            return G, False, "Not holding object"
        
        # [Effect] 移除 Holding，建立 On
        G.remove_edge(self.robot_id, obj_id)
        G.add_edge(dest_id, obj_id, relation=Relation.ON)
        state = normalize_state(G.nodes[obj_id].get("state"))
        state["held"] = False
        G.nodes[obj_id]["state"] = state
        
        # 位置更新 (简单堆叠)
        if dest_id in G:
            dest_pos = G.nodes[dest_id]['pos']
            new_pos = [dest_pos[0], dest_pos[1] + 0.3, dest_pos[2]]
            G.nodes[obj_id]['pos'] = new_pos
            
        return G, True, "Placed"

    def _rule_state_change(self, G, obj_id, action):
        if not G.has_node(obj_id): return G, False, "Target not found"
        state = G.nodes[obj_id].get("state")
        if not isinstance(state, dict) or "open_state" not in state:
            return G, False, "Target not openable"
        state = normalize_state(state)
        new_state = "open" if action == "Open" else "closed"
        state["open_state"] = new_state
        G.nodes[obj_id]["state"] = state
        return G, True, new_state
