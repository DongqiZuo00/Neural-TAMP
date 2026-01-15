from typing import List, Dict
from src.core.graph_schema import SceneGraph
from src.planning.llm_interface import LLMInterface
from src.planning.prompt_builder import PromptBuilder

class TaskDecomposer:
    def __init__(self, model_name="gpt-4o"):
        self.llm = LLMInterface(model=model_name)
        self.prompter = PromptBuilder()

    def plan(self, task_instruction: str, scene_graph: SceneGraph) -> List[Dict]:
        """
        MLDT 风格的双层规划：
        1. Decompose: Task -> Sub-goals
        2. Translate: Sub-goals -> Atomic Actions
        """
        print(f"\n🧠 [Planner] Starting Hierarchical Planning for: \"{task_instruction}\"")

        # --- Step 1: High-Level Decomposition ---
        print("   1️⃣  Phase 1: Decomposing into Sub-goals...")
        decomp_prompt = self.prompter.build_decomposition_prompt(task_instruction, scene_graph)
        
        # 为了分解任务，我们只需要 System Prompt 里的设定，不需要复杂的 User Prompt 区分
        # 这里为了简化调用，直接把完整的 prompt 作为 system/user 组合
        decomp_result = self.llm.predict(
            system_prompt="You are a helper.", # 简单占位，主要逻辑在 decomp_prompt
            user_prompt=decomp_prompt
        )
        
        subgoals = decomp_result.get("subgoals", [])
        thought = decomp_result.get("thought", "")
        
        if not subgoals:
            print("   ❌ Decomposition failed (No subgoals found).")
            return []
            
        print(f"      Thought: {thought}")
        print(f"      Sub-goals: {subgoals}")

        # --- Step 2: Low-Level Action Generation ---
        print("   2️⃣  Phase 2: Translating to Atomic Actions...")
        action_prompt = self.prompter.build_action_prompt(subgoals, scene_graph)
        
        action_result = self.llm.predict(
            system_prompt="You are a robot executor.",
            user_prompt=action_prompt
        )
        
        actions = action_result.get("actions", [])
        
        if actions:
            print(f"      Generated {len(actions)} atomic actions.")
            return actions
        else:
            print("   ❌ Action translation failed.")
            return []