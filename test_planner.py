import json
import os
import sys

# 路径修正
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.graph_schema import SceneGraph, Node, Edge
from src.planning.llm_planner import LLMPlanner

def load_graph_from_log(dataset_path, index=0):
    """从生成的 dataset_tasks.json 中加载场景和任务"""
    if not os.path.exists(dataset_path):
        print("❌ Dataset not found! Please run main.py first.")
        return None, None

    with open(dataset_path, 'r') as f:
        data = json.load(f)
        
    if index >= len(data):
        print("❌ Index out of bounds.")
        return None, None
        
    entry = data[index]
    instruction = entry['instruction']
    
    # 重建 Graph 对象 (这里我们只读 JSON，实际上 main 流程里是内存传递)
    # 为了测试方便，我们简单 Mock 一个 graph 或者需要改 main 保存完整 graph json
    # *临时方案*: 我们手动创建一个简单的 graph 用于测试 LLM 逻辑，
    # 或者修改 main.py 保存 graph json (上一步代码注释里提到了)
    
    # 既然上一步没保存 graph json，我们这里先手动构造一个 Mock Graph 
    # 来验证 LLMPlanner 模块是否工作正常。
    print(f"⚠️ Note: Using Mock Graph for specific logic testing.")
    
    graph = SceneGraph()
    # 模拟一个典型的厨房场景
    graph.add_node(Node(id="Room|0", label="Kitchen", pos=(0,0,0), room_id=None))
    graph.add_node(Node(id="Fridge|1", label="Fridge", pos=(1,0,1), state="closed", room_id="Room|0"))
    graph.add_node(Node(id="Table|2", label="DiningTable", pos=(3,0,3), state="default", room_id="Room|0"))
    # 苹果在冰箱里
    apple = Node(id="Apple|3", label="Apple", pos=(1,0.5,1), state="default", room_id="Room|0")
    graph.add_node(apple)
    
    # 增加 Edge
    graph.add_edge(Edge(source_id="Apple|3", target_id="Fridge|1", relation="inside"))
    
    return instruction, graph

def main():
    print("="*60)
    print("🧠 Testing LLM Planner")
    print("="*60)
    
    # 1. 设置 Key (请确保环境变量里有，或者直接填在这里测试)
    # os.environ["OPENAI_API_KEY"] = "sk-......"
    
    # 2. 初始化 Planner
    planner = LLMPlanner(model="gpt-4o") # 或者 gpt-3.5-turbo
    
    # 3. 准备数据
    # case 1: 苹果在冰箱里 (需要 Open)
    task1 = "Put the Apple on the DiningTable."
    
    graph1 = SceneGraph()
    graph1.add_node(Node("Room|0", "Kitchen", (0,0,0)))
    graph1.add_node(Node("Fridge|1", "Fridge", (1,0,1), state="closed", room_id="Room|0"))
    graph1.add_node(Node("Table|2", "DiningTable", (3,0,3), state="default", room_id="Room|0"))
    graph1.add_node(Node("Apple|3", "Apple", (1,0,1), state="default", room_id="Room|0"))
    graph1.add_edge(Edge("Apple|3", "Fridge|1", "inside")) # Apple inside Fridge
    
    print(f"\n🧪 Test Case 1: {task1}")
    actions = planner.plan(task1, graph1)
    
    print("\n[Generated Plan]:")
    print(json.dumps(actions, indent=2))
    
    # 验证逻辑
    if not actions:
        print("❌ Failed to generate plan.")
        return

    # 简单检查: 是否有 Open 冰箱的动作?
    has_open = any(a['action'] == 'Open' and 'Fridge' in a['target'] for a in actions)
    if has_open:
        print("✅ Logic Check Passed: Robot decided to OPEN the fridge.")
    else:
        print("❌ Logic Check Failed: Robot forgot to OPEN the fridge!")

if __name__ == "__main__":
    main()