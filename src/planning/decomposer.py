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
        Input: 任务指令 + 场景图快照
        Output: 原子动作列表 (不负责执行，只负责生成)
        """
        print(f"   🧠 Planning: {task_instruction}")

        # 1. 任务分解
        decomp_prompt = self.prompter.build_decomposition_prompt(task_instruction, scene_graph)
        decomp_result = self.llm.predict("You are a helper.", decomp_prompt)
        subgoals = decomp_result.get("subgoals", [])
        
        if not subgoals:
            return []

        # 2. 动作生成
        action_prompt = self.prompter.build_action_prompt(subgoals, scene_graph)
        action_result = self.llm.predict("You are a robot executor.", action_prompt)
        actions = action_result.get("actions", [])
        
        return actions