import random
import copy
import numpy as np
from typing import Tuple, Dict, List
from collections import defaultdict
from src.core.graph_schema import SceneGraph, Node, Edge

class TaskGenerator:
    """
    Scene-Aware & Adversarial Task Generator (Compatible with NetworkX GraphManager)
    
    核心策略：
    1. Scene Profiling: 宏观调度 (去最挤的房间，或者找关着的容器)。
    2. Adversarial Scoring: 微观选品 (挑干扰项多、嵌套深、反常识的物体)。
    """
    def __init__(self):
        self.min_steps = 2
        self.max_steps = 4
        
        # --- 攻击权重 ---
        self.weights = {
            "distractor_count": 2.5,
            "nesting_depth": 3.0,
            "state_constraint": 2.0,
            "semantic_oddity": 1.5,
            "step_coherence": 1.0
        }

        # --- 语义常识库 ---
        self.receptacles = {
            "Table", "CounterTop", "Shelf", "Cabinet", "Fridge", "Microwave", 
            "Box", "Bin", "Safe", "Drawer", "Sofa", "Bed", "ArmChair", "Desk",
            "Cup", "Mug", "Bowl", "Pot", "Pan", "CoffeeTable", "DiningTable", "SideTable"
        }
        self.large_objects = {"Laptop", "Pillow", "Box", "Boots", "TeddyBear", "Pot", "Pan"}
        self.small_containers = {"Cup", "Mug", "Bowl"}
        self.common_pairs = {"Apple": "Fridge", "Book": "Shelf", "Pen": "Desk", "Bread": "Plate"}

    def generate(self, scene_graph: SceneGraph) -> Tuple[str, Dict]:
        """
        标准接口：接收 SceneGraph 对象 (由 GraphManager.to_scene_graph() 提供)
        """
        # 0. 防御性检查：确保输入是 SceneGraph
        if not isinstance(scene_graph, SceneGraph):
            return None, {"error": f"Invalid input type: {type(scene_graph)}. Expected SceneGraph."}

        # 1. 场景画像 (Profiling)
        profile = self._profile_scene(scene_graph)
        
        if not profile['movable_objects'] or not profile['receptacles']:
            return None, {"error": "Scene too empty (no movables or receptacles)"}

        # 2. 策略尝试
        # 优先生成复杂任务，尝试 5 次
        for _ in range(5):
            # 策略 A: 跨房间搬运 (Logistic Attack) - 70% 概率
            if len(profile['rooms']) > 1 and random.random() < 0.7:
                instruction, meta = self._gen_cross_room_chain(scene_graph, profile)
            # 策略 B: 房间内隐藏 (Hide & Seek Attack) - 30% 概率
            else:
                instruction, meta = self._gen_intra_room_chain(scene_graph, profile)
                
            if instruction:
                return instruction, meta
        
        return None, {"error": "Failed to generate valid adversarial task after 5 attempts"}

    # =========================================================================
    #  Phase 0: 场景画像 (Scene Profiling)
    # =========================================================================
    
    def _profile_scene(self, graph: SceneGraph) -> Dict:
        """建立场景索引：哪里挤？哪里有锁？"""
        profile = {
            "rooms": defaultdict(list),         
            "room_labels": {},                  
            "movable_objects": [],              
            "receptacles": [],                  
            "closed_containers": [],            
            "crowded_rooms": [],                
            "empty_rooms": []                   
        }
        
        room_obj_counts = defaultdict(int)
        
        for nid, node in graph.nodes.items():
            if "Room" in nid:
                profile['room_labels'][nid] = node.label
            elif node.room_id:
                profile['rooms'][node.room_id].append(node)
                room_obj_counts[node.room_id] += 1
                
                if self._is_movable(node.label):
                    profile['movable_objects'].append(node)
                if self._is_receptacle(node.label):
                    profile['receptacles'].append(node)
                    # 检测 closed 状态 (适配 NetworkX 转换后的 Node)
                    if isinstance(node.state, dict) and node.state.get("open_state") == "closed":
                        profile['closed_containers'].append(node)

        # 分析房间密度
        sorted_rooms = sorted(room_obj_counts.items(), key=lambda x: x[1], reverse=True)
        if sorted_rooms:
            profile['crowded_rooms'] = [r[0] for r in sorted_rooms[:max(1, len(sorted_rooms)//2)]]
            # 剩下的作为空旷房间
            profile['empty_rooms'] = [r[0] for r in sorted_rooms[len(sorted_rooms)//2:]]
        
        if not profile['empty_rooms'] and profile['crowded_rooms']:
             profile['empty_rooms'] = profile['crowded_rooms']

        return profile

    # =========================================================================
    #  Phase 1: 生成策略实现
    # =========================================================================

    def _gen_cross_room_chain(self, graph: SceneGraph, profile: Dict):
        # 1. 确定 Source (拥挤) 和 Dest (空旷)
        source_rid = random.choice(profile['crowded_rooms'])
        dest_rids = [r for r in list(profile['rooms'].keys()) if r != source_rid]
        if not dest_rids: return None, {}
        dest_rid = random.choice(dest_rids)

        virtual_graph = self._clone_graph(graph)
        task_chain = []
        
        # Step 1: 在 Source Room 挑选 Adversarial Score 最高的物体
        source_objs = [o for o in profile['rooms'][source_rid] if o in profile['movable_objects']]
        dest_receps = [r for r in profile['rooms'][dest_rid] if r in profile['receptacles']]
        
        if not source_objs or not dest_receps: return None, {}
        
        step1 = self._find_best_adversarial_step(virtual_graph, source_objs, dest_receps)
        if not step1: return None, {}
        
        self._add_step(task_chain, virtual_graph, step1, 1)
        
        # Step 2: 尝试多步任务
        curr_obj_id = step1['target'].id
        
        # 60% 概率继续搬运同一个物体 (连贯性测试)
        if random.random() < 0.6 and curr_obj_id in virtual_graph.nodes:
            candidates = [virtual_graph.nodes[curr_obj_id]]
            # 随机去任意房间
            next_rid = random.choice(list(profile['rooms'].keys()))
            next_receps = [r for r in profile['rooms'][next_rid] if r in profile['receptacles']]
            
            step2 = self._find_best_adversarial_step(virtual_graph, candidates, next_receps, last_obj_id=curr_obj_id)
            if step2:
                self._add_step(task_chain, virtual_graph, step2, 2)

        return self._finalize_output(task_chain, "Adversarial_Logistic")

    def _gen_intra_room_chain(self, graph: SceneGraph, profile: Dict):
        # 优先选有 Closed 容器的房间
        candidates = [r for r in profile['rooms'] if any(o in profile['movable_objects'] for o in profile['rooms'][r])]
        if not candidates: return None, {}
        
        rid = random.choice(candidates)
        
        virtual_graph = self._clone_graph(graph)
        task_chain = []
        
        local_objs = [o for o in profile['rooms'][rid] if o in profile['movable_objects']]
        local_receps = [r for r in profile['rooms'][rid] if r in profile['receptacles']]
        
        # 寻找最难的一步
        step1 = self._find_best_adversarial_step(virtual_graph, local_objs, local_receps)
        if not step1: return None, {}
        
        self._add_step(task_chain, virtual_graph, step1, 1)
        return self._finalize_output(task_chain, "Adversarial_Tidy")

    # =========================================================================
    #  Phase 2: 对抗性评分核心 (Adversarial Scoring)
    # =========================================================================

    def _find_best_adversarial_step(self, graph, target_candidates, dest_candidates, last_obj_id=None):
        scored_pairs = []
        max_trials = 80
        count = 0
        
        # Shuffle 保证随机性
        random.shuffle(target_candidates)
        random.shuffle(dest_candidates)
        
        for t in target_candidates:
            for d in dest_candidates:
                if count > max_trials: break
                
                if not self._is_physically_feasible(t, d, graph): continue
                
                score, debug = self._calculate_attack_score(t, d, graph, last_obj_id)
                scored_pairs.append({"target": t, "dest": d, "score": score, "debug": debug})
                count += 1
                
        if not scored_pairs: return None
        # 贪婪选择分数最高的
        scored_pairs.sort(key=lambda x: x["score"], reverse=True)
        return scored_pairs[0]

    def _calculate_attack_score(self, target: Node, dest: Node, graph: SceneGraph, last_obj_id: str):
        # 1. 干扰项攻击: 同名物体越多越好
        same_label_count = len([n for n in graph.nodes.values() if n.label == target.label])
        score_dist = min(same_label_count - 1, 5) / 5.0
        
        # 2. 状态攻击: 优先放入 Closed 容器
        score_state = 0.0
        # 如果 Dest 是关着的，加分 (诱导 Agent 去开门)
        if isinstance(dest.state, dict) and dest.state.get("open_state") == "closed":
            score_state += 1.0
        
        # 3. 语义反常识
        score_sem = 1.0 if self._is_semantic_oddity(target, dest) else 0.0
        
        # 4. 连贯性奖励
        score_cohere = 1.0 if last_obj_id and target.id == last_obj_id else 0.0

        total = (
            self.weights["distractor_count"] * score_dist + 
            self.weights["state_constraint"] * score_state +
            self.weights["semantic_oddity"] * score_sem +
            self.weights["step_coherence"] * score_cohere
        )
        return total, {"distractors": same_label_count, "closed": bool(score_state > 0)}

    # =========================================================================
    #  Utils
    # =========================================================================

    def _is_physically_feasible(self, t, d, graph):
        if t.id == d.id: return False
        if t.label == d.label: return False
        # 大不入小
        if t.label in self.large_objects and d.label in self.small_containers: return False
        return True

    def _add_step(self, chain, graph, step_data, step_num):
        t, d = step_data['target'], step_data['dest']
        chain.append({
            "step": step_num,
            "target": t.label, "target_id": t.id,
            "dest": d.label, "dest_id": d.id,
            "adversarial_score": step_data['score'],
            "reason": step_data['debug']
        })
        # 简单的虚拟状态更新 (为了计算多步任务)
        self._virtual_move(graph, t.id, d.id)

    def _finalize_output(self, chain, mode):
        if len(chain) < 1: return None, {}
        instr_parts = []
        for i, step in enumerate(chain):
            prefix = "First," if i==0 else ("Then," if i < len(chain)-1 else "Finally,")
            instr_parts.append(f"{prefix} move the {step['target']} (ID: {step['target_id']}) to the {step['dest']} (ID: {step['dest_id']}).")
        
        return " ".join(instr_parts), {"type": mode, "length": len(chain), "chain_details": chain}

    def _virtual_move(self, graph, obj_id, dest_id):
        # 移除旧边
        graph.edges = [e for e in graph.edges if e.target_id != obj_id]
        # 添加新边
        graph.add_edge(Edge(dest_id, obj_id, "on")) # 简化假设为 on
        # 更新 room_id
        if dest_id in graph.nodes:
            graph.nodes[obj_id].room_id = graph.nodes[dest_id].room_id

    def _clone_graph(self, graph):
        new_g = SceneGraph()
        for nid, node in graph.nodes.items():
            new_g.add_node(Node(nid, node.label, node.pos, bbox=node.bbox, state=node.state, room_id=node.room_id))
        for e in graph.edges:
            new_g.add_edge(Edge(e.source_id, e.target_id, e.relation))
        return new_g

    def _is_movable(self, label):
        immovable = {"Wall", "Floor", "Room", "Bed", "Sofa", "Table", "CounterTop", "Fridge", "Cabinet", "Shelf", "Sink", "Bathtub", "Toilet", "Stool", "Chair", "ArmChair", "Window"}
        return label not in immovable

    def _is_receptacle(self, label):
        return label in self.receptacles

    def _is_semantic_oddity(self, t, d):
        return t.label in self.common_pairs and self.common_pairs[t.label] not in d.label
