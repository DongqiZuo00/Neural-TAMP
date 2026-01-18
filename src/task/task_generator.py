import random
import copy
from typing import Tuple, Dict, List
from collections import defaultdict
from src.core.graph_schema import SceneGraph, Node, Edge

class VirtualSandbox:
    """
    [v9.1 Sandbox] 增强版沙盒
    负责: 状态追踪、容量估算、房间记录、物理规则校验
    """
    def __init__(self, graph: SceneGraph):
        self.container_states = {} 
        self.container_contents = defaultdict(set)
        self.container_capacities = {}
        self.object_rooms = {} 
        
        for node in graph.nodes.values():
            # 1. 记录房间信息
            self.object_rooms[node.id] = node.room_id

            # 2. 初始化容器状态
            if self._is_openable(node.label):
                self.container_states[node.id] = 'Closed' # 默认关着，增加难度
            else:
                self.container_states[node.id] = 'Open'   # 永远敞开

            # 3. 初始化容量
            cap = self._estimate_capacity(node.label)
            if cap > 0: self.container_capacities[node.id] = cap
            self.container_contents[node.id] = set()

    def get_room(self, obj_id):
        return self.object_rooms.get(obj_id, "Unknown")

    def update_room(self, obj_id, new_room_id):
        self.object_rooms[obj_id] = new_room_id

    def can_open(self, target_id):
        """物理检查: 必须是 Openable 且当前状态为 Closed"""
        return self.container_states.get(target_id) == 'Closed'

    def can_close(self, target_id):
        """物理检查: 必须是 Openable 且当前状态为 Open"""
        return self.container_states.get(target_id) == 'Open'

    def can_place(self, target_id):
        """
        物理检查: 
        1. 是容器
        2. 是 Open 状态
        3. 容量未满
        """
        if target_id not in self.container_capacities: return False
        if self.container_states.get(target_id) == 'Closed': return False
        current_load = len(self.container_contents[target_id])
        return current_load < self.container_capacities[target_id]

    def has_capacity_if_opened(self, target_id):
        """
        预判: 假设把门打开，里面还有位置吗？
        用于避免 '打开了门却发现满了' 的尴尬情况
        """
        if target_id not in self.container_capacities: return False
        current_load = len(self.container_contents[target_id])
        return current_load < self.container_capacities[target_id]

    def apply_open(self, target_id):
        self.container_states[target_id] = 'Open'

    def apply_close(self, target_id):
        self.container_states[target_id] = 'Closed'

    def apply_place(self, obj_id, dest_id):
        # 移除旧位置
        for c_id, contents in self.container_contents.items():
            if obj_id in contents:
                contents.remove(obj_id)
                break
        # 加入新位置
        self.container_contents[dest_id].add(obj_id)
        # 更新房间
        new_room = self.get_room(dest_id)
        self.update_room(obj_id, new_room)

    def _is_openable(self, label):
        """
        [严格白名单] 只有这些能生成 Open/Close 指令
        ❌ Sink, Bathtub 绝对不能加进来
        """
        return label in {
            "Fridge", "Refrigerator", "Cabinet", "Microwave", "Safe", 
            "Drawer", "Box", "Laptop", "Toilet", "WashingMachine", "Dryer", "Blinds"
        }

    def _estimate_capacity(self, label):
        if label in ["Fridge", "Refrigerator"]: return 6
        if label in ["Cabinet", "Shelf", "CounterTop", "Table", "DiningTable", "Bed", "Sofa"]: return 10
        if label in ["Microwave", "Box", "Drawer", "Safe", "Sink", "Bathtub", "Toilet"]: return 3
        return 0

class TaskGenerator:
    """
    [v9.1 - The Nomad] Cross-Room Priority Generator
    核心升级: 
    1. 移除孤儿 Open 动作
    2. 严格校验 Openable 属性
    """
    def __init__(self):
        self.target_steps = 10 
        
        # 幻觉攻击映射表
        self.hallucination_map = {
            "electronic": {"Fridge", "Microwave", "Sink", "Toilet", "Bathtub"},
            "paper": {"Fridge", "Microwave", "Sink"},
            "food": {"Safe", "Toilet", "Drawer", "Sofa", "Bed"}
        }
        self.categories = {
            "electronic": {"Laptop", "CellPhone", "RemoteControl", "AlarmClock"},
            "paper": {"Book", "Newspaper", "CreditCard"},
            "food": {"Apple", "Bread", "Egg", "Tomato", "Potato"},
        }
        self.large_objects = {"Laptop", "Pillow", "Box", "Pot", "Pan"}
        self.small_containers = {"Cup", "Mug", "Bowl"}

    def generate(self, scene_graph: SceneGraph) -> Tuple[str, Dict]:
        # 1. 初始化
        sandbox = VirtualSandbox(scene_graph)
        movables = self._scan_movables(scene_graph)
        containers = self._scan_containers(scene_graph)
        
        if not movables or not containers: return None, {"error": "Scene empty"}

        # 2. 生成序列
        return self._gen_cross_room_sequence(sandbox, movables, containers)

    def _gen_cross_room_sequence(self, sandbox, movables, containers):
        chain = []
        current_step = 1
        
        todo = movables[:]
        random.shuffle(todo)
        
        while current_step <= self.target_steps:
            
            # --- Phase A: 选择 Source Object ---
            if not todo: 
                 todo = movables[:]
                 random.shuffle(todo)
            obj_node = todo.pop(0)
            
            source_room = sandbox.get_room(obj_node.id)

            # --- Phase B: 容器分类 (Far vs Near) ---
            far_containers = []  
            near_containers = [] 

            for c in containers:
                c_room = sandbox.get_room(c.id)
                # 基础物理过滤 (大不入小)
                if not self._is_physically_feasible(obj_node, c): continue
                
                if c_room != source_room:
                    far_containers.append(c)
                else:
                    near_containers.append(c)

            # --- Phase C: 目标选择策略 (Nomad Strategy) ---
            target_candidates = []
            strategy_type = "Normal"
            score_multiplier = 1.0

            # 1. 跨房间 + 幻觉
            hallucination_targets = self._get_hallucination_targets(obj_node, far_containers)
            if hallucination_targets:
                target_candidates = hallucination_targets
                strategy_type = "CrossRoom_Hallucination"
                score_multiplier = 5.0
            
            # 2. 跨房间 + 普通
            if not target_candidates and far_containers:
                target_candidates = far_containers
                strategy_type = "CrossRoom_Normal"
                score_multiplier = 3.0 

            # 3. 同房间 + 幻觉
            if not target_candidates:
                hallucination_targets = self._get_hallucination_targets(obj_node, near_containers)
                if hallucination_targets:
                    target_candidates = hallucination_targets
                    strategy_type = "SameRoom_Hallucination"
                    score_multiplier = 1.5

            # 4. 同房间 + 普通 (保底)
            if not target_candidates:
                target_candidates = near_containers
                strategy_type = "SameRoom_Fallback"
                score_multiplier = 0.5

            random.shuffle(target_candidates)

            # --- Phase D: 动作生成 (Bonding Logic) ---
            action_generated = False
            for dest in target_candidates:
                
                # 检查: 是否需要开门?
                need_open = sandbox.can_open(dest.id)
                
                # 检查: 容量够吗? (如果需要开门，检查开门后的容量)
                if need_open:
                    has_space = sandbox.has_capacity_if_opened(dest.id)
                else:
                    has_space = sandbox.can_place(dest.id)
                
                if not has_space: continue # 容量不足，跳过此容器

                # === 确定生成序列 ===
                
                # Step 1: 插入 Open (如果必要)
                if need_open:
                    chain.append({
                        "step": current_step,
                        "action": "OpenObject",
                        "target": dest.label, "target_id": dest.id,
                        "desc": f"open the {dest.label} (ID: {dest.id})",
                        "score": 1.0,
                        "room_info": f"Room {sandbox.get_room(dest.id)}"
                    })
                    sandbox.apply_open(dest.id) # 更新沙盒状态
                    current_step += 1
                    if current_step > self.target_steps: break
                
                # Step 2: 插入 Place (必然执行)
                chain.append({
                    "step": current_step,
                    "action": "Place",
                    "target": obj_node.label, "target_id": obj_node.id,
                    "dest": dest.label, "dest_id": dest.id,
                    "desc": f"move the {obj_node.label} (ID: {obj_node.id}) to the {dest.label} (ID: {dest.id})",
                    "score": 1.0 * score_multiplier,
                    "type": strategy_type,
                    "room_transfer": f"{source_room} -> {sandbox.get_room(dest.id)}"
                })
                sandbox.apply_place(obj_node.id, dest.id)
                current_step += 1
                action_generated = True
                
                # Step 3: 随机 Close (如果它是 Openable 的，且当前是 Open 的)
                # 再次做 can_close 检查，确保不会去关 Sink
                if sandbox.can_close(dest.id) and random.random() < 0.3 and current_step <= self.target_steps:
                    chain.append({
                        "step": current_step,
                        "action": "CloseObject",
                        "target": dest.label, "target_id": dest.id,
                        "desc": f"close the {dest.label} (ID: {dest.id})",
                        "score": 0.5,
                        "room_info": f"Room {sandbox.get_room(dest.id)}"
                    })
                    sandbox.apply_close(dest.id)
                    current_step += 1
                
                break # Place 成功，跳出，处理下一个物体

            if not action_generated: continue

        return self._finalize_text(chain)

    def _get_hallucination_targets(self, obj, containers):
        cat = self._get_category(obj.label)
        if cat in self.hallucination_map:
            bad_labels = self.hallucination_map[cat]
            return [c for c in containers if c.label in bad_labels]
        return []

    def _finalize_text(self, chain):
        if not chain: return None, {}
        parts = []
        total_score = 0
        for i, step in enumerate(chain):
            prefix = "First," if i==0 else ("Then," if i < len(chain)-1 else "Finally,")
            parts.append(f"{prefix} {step['desc']}.")
            total_score += step.get('score', 0)
            
        return " ".join(parts), {
            "type": "Nomad_Cross_Room_10",
            "length": len(chain),
            "total_score": total_score,
            "chain_details": chain
        }

    # --- Helpers ---
    def _scan_movables(self, graph):
        whitelist = {
            "Apple", "Bread", "Egg", "Lettuce", "Potato", "Tomato", "Sandwich",
            "Bowl", "Plate", "Cup", "Mug", "Fork", "Knife", "Spoon", "Pot", "Pan",
            "Laptop", "CellPhone", "Book", "CreditCard", "KeyChain", "RemoteControl",
            "Box", "Newspaper", "Pillow", "SoapBar", "SprayBottle", "Statue", "Vase", 
            "Watch", "AlarmClock", "Pencil", "Pen", "TeddyBear", "Candle", "Plunger"
        }
        return [n for n in graph.nodes.values() if n.label in whitelist]

    def _scan_containers(self, graph):
        whitelist = {
            "Table", "DiningTable", "CoffeeTable", "SideTable", "Desk", "CounterTop", 
            "Shelf", "Cabinet", "Drawer", "Fridge", "Safe", "Microwave",
            "Sofa", "ArmChair", "Bed", "Chair", "Stool", "Sink", "Bathtub", "Toilet",
            "Box", "Bin", "GarbageCan"
        }
        return [n for n in graph.nodes.values() if n.label in whitelist]

    def _is_physically_feasible(self, obj, dest):
        if obj.id == dest.id: return False
        if obj.label in self.large_objects and dest.label in self.small_containers: return False
        return True

    def _get_category(self, label):
        for cat, items in self.categories.items():
            if label in items: return cat
        return "misc"