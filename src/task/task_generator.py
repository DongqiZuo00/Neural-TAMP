import random
import numpy as np
from typing import Tuple, Dict, List
from src.core.graph_schema import SceneGraph, Node

class TaskGenerator:
    """
    RL-Driven Adversarial Task Generator (with Hard Constraints)
    
    机制：
    1. Filter: 严格筛选符合物理逻辑的 (Target, Dest) 对。
    2. Attack: 使用 Value Function 寻找最刁钻的组合。
    3. Generate: 生成 >60 字符的详细指令。
    """
    def __init__(self):
        # --- 1. 物体分类 (用于尺寸/类型检查) ---
        self.categories = {
            "small_pickupable": {
                "Apple", "Bread", "Tomato", "Lettuce", "Potato", "Egg", 
                "Mug", "Cup", "Bowl", "Plate", "Fork", "Knife", "Spoon",
                "Remote", "CellPhone", "CD", "SoapBottle", "Candle", "AlarmClock", "Statue", "Vase"
            },
            "medium_pickupable": {
                "Laptop", "Book", "SprayBottle", "TeddyBear", "Boots", "Pillow"
            },
            "small_container": {
                 "Bowl", "Cup", "Mug", "Pot", "Pan" # 只能装极小的东西
            },
            "enclosed_receptacle": {
                "Fridge", "Microwave", "Cabinet", "Drawer", "Safe", "Box", "Bin", "GarbageCan"
            },
            "open_surface": {
                "Table", "CoffeeTable", "DiningTable", "SideTable", "CounterTop", 
                "Sofa", "ArmChair", "Bed", "Chair", "Shelf", "Sink", "Bathtub", "Toilet"
            }
        }
        
        # 聚合 Set 方便查询
        self.all_pickupable = self.categories["small_pickupable"] | self.categories["medium_pickupable"]
        self.all_receptacles = self.categories["small_container"] | self.categories["enclosed_receptacle"] | self.categories["open_surface"]

        # --- 2. 对抗性 Reward 权重 ---
        self.w_distance = 2.0  # 距离越远越好
        self.w_wall = 3.0      # 离墙越近越好 (边缘测试)
        self.w_clutter = 1.5   # 目的地杂物越多越好

    def generate(self, scene_graph: SceneGraph) -> Tuple[str, Dict]:
        """
        生成任务的主入口
        """
        # 1. 扫描所有物体
        targets, receptacles = self._scan_objects(scene_graph)
        
        if not targets or not receptacles:
            return None, {"error": "Scene too empty to generate tasks."}

        # 2. 暴力搜索 + 硬约束检查
        valid_pairs = []
        for t in targets:
            for r in receptacles:
                if self._check_hard_constraints(t, r, scene_graph):
                    valid_pairs.append((t, r))

        if not valid_pairs:
            return None, {"error": "No logically valid tasks found (constraints too strict?)."}

        # 3. 计算攻击性 Reward (Simulated RL Critic)
        ranked_tasks = []
        for target, dest in valid_pairs:
            reward, debug_info = self._calculate_adversarial_reward(target, dest, scene_graph)
            ranked_tasks.append({
                "target": target,
                "dest": dest,
                "reward": reward,
                "debug": debug_info
            })

        # 4. 贪婪采样 (Top-K)
        ranked_tasks.sort(key=lambda x: x["reward"], reverse=True)
        # 选前3难的，保证极高难度
        top_k = min(len(ranked_tasks), 3)
        selected_task = random.choice(ranked_tasks[:top_k])
        
        t_node = selected_task["target"]
        d_node = selected_task["dest"]

        # 5. 生成长文本指令
        instruction = self._generate_long_instruction(t_node, d_node, scene_graph)
        
        task_meta = {
            "type": "PickPlace",
            "target_id": t_node.id,
            "target_class": t_node.label,
            "dest_id": d_node.id,
            "dest_class": d_node.label,
            "adversarial_score": selected_task["reward"],
            "difficulty_factors": selected_task["debug"]
        }

        return instruction, task_meta

    def _scan_objects(self, graph):
        """扫描当前 Graph 中实际存在的物体"""
        targets = []
        receptacles = []
        for node in graph.nodes.values():
            # 必须在白名单里，防止生成 ProcTHOR 不支持的交互
            if node.label in self.all_pickupable:
                targets.append(node)
            elif node.label in self.all_receptacles:
                receptacles.append(node)
        return targets, receptacles

    def _check_hard_constraints(self, target: Node, dest: Node, graph: SceneGraph) -> bool:
        """
        硬约束检查器：确保任务绝对可执行
        """
        # A. 自身排斥
        if target.id == dest.id: return False
        
        # B. 尺寸/类型兼容性 (Size Physics)
        t_type = target.label
        d_type = dest.label
        
        # 规则 1: 中型物体不能放进小型容器 (e.g. Laptop -> Cup 是不行的)
        if t_type in self.categories["medium_pickupable"] and d_type in self.categories["small_container"]:
            return False
            
        # 规则 2: 如果是把物体放入自身类型的容器 (e.g. Cup into Cup)，通常避免，除非为了堆叠
        if t_type == d_type:
            return False

        # C. 初始状态检查 (不能已经在里面了)
        # 检查图中的 Edge
        for edge in graph.edges:
            if edge.source_id == target.id and edge.target_id == dest.id and edge.relation in ["inside", "on"]:
                return False # 任务已完成，无意义

        # D. 可达性预判 (Reachability Proxy)
        # 虽然不能做完整的 Path Planning，但至少保证它们不在“不可达”的封闭区域
        # 这里我们假设只要在 Graph 里且有 room_id，就是可达的
        if target.room_id is None or dest.room_id is None:
            return False

        return True

    def _calculate_adversarial_reward(self, target: Node, dest: Node, graph: SceneGraph):
        """
        计算攻击性 Reward (RL Critic)
        """
        # 1. 距离 (Distance)
        dist = np.linalg.norm(np.array(target.pos) - np.array(dest.pos))
        r_dist = min(dist, 10.0) / 10.0 
        
        # 2. 靠墙检测 (Target Near Wall)
        # 利用 Room Bounds 近似计算
        r_wall = 0.0
        if target.room_id and target.room_id in graph.nodes:
            room = graph.nodes[target.room_id]
            if "bounds" in room.geometry:
                min_x, min_z, max_x, max_z = room.geometry["bounds"]
                tx, tz = target.pos[0], target.pos[2]
                # 计算到最近墙壁的距离
                d_edge = min(abs(tx - min_x), abs(tx - max_x), abs(tz - min_z), abs(tz - max_z))
                # 越近分越高 (0.2m 以内满分)
                if d_edge < 0.2: r_wall = 1.0
                elif d_edge < 1.0: r_wall = 0.5
                else: r_wall = 0.0

        # 3. 遮挡/拥挤检测 (Clutter at Dest)
        # 统计目的地当前有多少 edge 连接 (inside/on)
        clutter_count = 0
        for edge in graph.edges:
            if edge.target_id == dest.id and edge.relation in ["inside", "on"]:
                clutter_count += 1
        r_clutter = min(clutter_count, 5) / 5.0

        total_reward = (self.w_distance * r_dist) + (self.w_wall * r_wall) + (self.w_clutter * r_clutter)
        
        return total_reward, {
            "dist_m": round(dist, 1),
            "is_near_wall": r_wall > 0.5,
            "clutter_items": clutter_count
        }

    def _generate_long_instruction(self, target: Node, dest: Node, graph: SceneGraph) -> str:
        """
        生成 > 60 字符的指令，并保证语义连贯。
        """
        # 1. 获取空间描述
        def get_loc_desc(node):
            if not node.room_id: return "nearby"
            room_label = graph.nodes[node.room_id].label
            return f"in the {room_label}"
        
        t_loc = get_loc_desc(target)
        d_loc = get_loc_desc(dest)

        # 2. 获取状态描述
        d_state = ""
        if "closed" in dest.state.lower(): d_state = " (which is currently closed)"

        # 3. 动词变换
        verbs = ["locate", "find", "identify", "spot"]
        actions = ["grab", "pick up", "retrieve", "take"]
        moves = ["carry it to", "transport it to", "bring it over to", "move it towards"]
        
        v = random.choice(verbs)
        a = random.choice(actions)
        m = random.choice(moves)

        # 4. 组装长句
        # 模板 A
        text = f"Please {v} the {target.label} located {t_loc}, carefully {a} it, and then {m} the {dest.label} {d_loc}{d_state} to place it there safely."
        
        # 模板 B (更加强调困难)
        if random.random() > 0.5:
            text = f"Your mission is to retrieve the {target.label} found {t_loc}. Once you have it, navigate through the room and deposit it into the {dest.label} {d_loc}."

        # 5. 长度硬约束补全
        if len(text) < 60:
            text += " Make sure to avoid any obstacles during the navigation."
            
        return text