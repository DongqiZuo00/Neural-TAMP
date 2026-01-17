from typing import List, Dict
from src.core.graph_schema import SceneGraph, Relation

class PromptBuilder:
    def __init__(self):
        pass

    # --- Phase 1: High-Level Decomposition ---
    def build_decomposition_prompt(self, task_instruction: str, scene_graph: SceneGraph) -> str:
        """
        生成分解任务的 Prompt。
        目标：将指令分解为逻辑子步骤 (Sub-goals)。
        """
        scene_desc = self._graph_to_text(scene_graph)
        return f"""You are a High-Level Task Planner.
Your goal is to break down a complex instruction into a list of short, logical sub-goals.

Current Scene:
{scene_desc}

Task: "{task_instruction}"

RULES:
1. Break the task into sequential steps.
2. Use ONLY objects that exist in the scene, and always include the exact object ID in each sub-goal.
3. Explicitly handle container states (e.g., if target is in a closed fridge, add "Open Fridge|1" as a step).
4. Respect affordances: only open objects marked openable, only pick up objects marked pickupable, and only place into objects marked receptacle.
5. Do NOT generate API code yet, just natural language steps.

EXAMPLE OUTPUT FORMAT (JSON):
{{
  "thought": "To put the apple in the fridge, I need to find it, grab it, go to the fridge, open it, and place it.",
  "subgoals": [
    "Navigate to the Apple|1",
    "Pick up the Apple|1",
    "Navigate to the Fridge|1",
    "Open the Fridge|1",
    "Put the Apple|1 inside the Fridge|1",
    "Close the Fridge|1"
  ]
}}

Please output the JSON for the given Task.
"""

    # --- Phase 2: Low-Level Action Generation ---
    def build_action_prompt(self, subgoals: List[str], scene_graph: SceneGraph) -> str:
        """
        生成原子动作的 Prompt。
        目标：将子步骤列表转化为具体的 API 调用。
        """
        scene_desc = self._graph_to_text(scene_graph)
        subgoals_text = "\n".join([f"{i+1}. {sg}" for i, sg in enumerate(subgoals)])
        
        return f"""You are a Low-Level Robotic Agent.
Your goal is to translate a list of High-Level Sub-goals into executable API Actions.

Current Scene:
{scene_desc}

Planned Sub-goals:
{subgoals_text}

AVAILABLE API ACTIONS:
1. NavigateTo(object_id)
2. PickUp(object_id)
3. PutObject(target_id, receptacle_id)
4. Open(object_id)
5. Close(object_id)

RULES:
- Map each sub-goal to the correct API call.
- Use exact Object IDs from the Scene description (e.g., "Fridge|1").
- If a sub-goal implies multiple actions (e.g., "Get the apple" -> Navigate + Pick), generate both.
- Ensure preconditions (must be at location to Pick/Open).
- Only Open/Close objects marked openable.
- Only PickUp objects marked pickupable.
- Only PutObject into receptacles marked receptacle.
- HARD RULE: For every non-NavigateTo action, always insert a NavigateTo(target_id) immediately before it.
- For PutObject, always include both "target" (the object to place, or held object) and "receptacle_id" (the destination).

EXAMPLE OUTPUT FORMAT (JSON):
{{
  "actions": [
    {{"action": "NavigateTo", "target": "Fridge|1"}},
    {{"action": "Open", "target": "Fridge|1"}},
    {{"action": "NavigateTo", "target": "Apple|1"}},
    {{"action": "PickUp", "target": "Apple|1"}},
    {{"action": "NavigateTo", "target": "Fridge|1"}},
    {{"action": "PutObject", "target": "Apple|1", "receptacle_id": "Fridge|1"}}
  ]
}}

Please output the JSON execution plan.
"""

    def _graph_to_text(self, scene_graph: SceneGraph) -> str:
        """(保持不变，负责将图转为文本描述)"""
        lines = []
        nodes = scene_graph.nodes
        rooms = [n for n in nodes.values() if "Room" in n.id]
        rooms.sort(key=lambda x: x.id)
        
        for room in rooms:
            lines.append(f"LOCATION: {room.label} (ID: {room.id})")
            room_objs = [n for n in nodes.values() if n.room_id == room.id]
            
            if not room_objs:
                lines.append("  (Empty)")
                continue

            for obj in room_objs:
                state_info = ""
                if isinstance(obj.state, dict):
                    if obj.state.get("open_state") == "closed":
                        state_info = "[State: CLOSED]"
                    elif obj.state.get("open_state") == "open":
                        state_info = "[State: OPEN]"

                affordances = []
                if isinstance(obj.geometry, dict):
                    if obj.geometry.get("pickupable"):
                        affordances.append("pickupable")
                    if obj.geometry.get("openable"):
                        affordances.append("openable")
                    if obj.geometry.get("receptacle"):
                        affordances.append("receptacle")
                affordance_text = f"[Affordances: {', '.join(affordances)}]" if affordances else ""

                relation_desc = ""
                for edge in scene_graph.edges:
                    if edge.target_id == obj.id and edge.relation in [Relation.INSIDE, Relation.ON]:
                        container_id = edge.source_id
                        relation_desc = f"(is {edge.relation} {container_id})"

                lines.append(
                    f"  - {obj.label} (ID: {obj.id}) {state_info} {affordance_text} {relation_desc}"
                )
            lines.append("")
        return "\n".join(lines)
