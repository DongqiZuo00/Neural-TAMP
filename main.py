import sys
import os
import random
import shutil
import json
import networkx as nx

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path: sys.path.append(current_dir)

from src.env.procthor_wrapper import ProcTHOREnv
from src.memory.graph_manager import GraphManager
from src.utils.visualizer import BEVVisualizer
from src.perception.oracle_interface import OracleInterface
from src.task.task_generator import TaskGenerator
from src.planning.decomposer import TaskDecomposer 
from src.world_model.rule_dynamics import RuleBasedDynamics # [新增]

def main():
    print("="*60)
    print("🚀 Neural-TAMP: Graph Dynamics Pipeline")
    print("="*60)

    # --- 1. 系统初始化 ---
    output_dir = "Neural-TAMP/vis_output/dynamics_dataset"
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    try:
        env = ProcTHOREnv()
        oracle = OracleInterface(env)
        memory = GraphManager(save_dir="Neural-TAMP/memory_data") # NetworkX Backend
        dynamics = RuleBasedDynamics()                            # World Model
        viz = BEVVisualizer(save_dir=output_dir)
        task_gen = TaskGenerator()
        planner = TaskDecomposer(model_name="gpt-4o")
        print("✅ Modules Ready.")
    except Exception as e:
        print(f"❌ Init Failed: {e}")
        return

    # --- 2. 批量处理循环 ---
    candidate_indices = random.sample(range(10000), 1) # 跑50个场景
    dataset_log = []

    for i, idx in enumerate(candidate_indices):
        print(f"\n🎬 Scene {idx} ({i+1}/50)")
        
        # A. 加载场景
        try: env.change_scene(idx)
        except: continue

        # B. 感知 -> 记忆 (G_0)
        # Oracle 抓取真值 -> GraphManager 转化为 NetworkX 图
        memory.override_global_graph(oracle.get_hierarchical_graph())

        # C. 任务生成
        # 导出 SceneGraph 供 Generator 使用
        current_sg = memory.to_scene_graph()
        instruction, meta = task_gen.generate(current_sg)
        if not instruction: continue
        print(f"   Task: {instruction}")

        # D. 规划 (Policy)
        # Planner 返回 Action List
        actions = planner.plan(instruction, current_sg)
        if not actions: continue

        # E. 图动力学推演 (Simulation Loop)
        # 遍历动作列表，一步步修改图
        trace = []
        for step, action in enumerate(actions):
            # 1. [Action Injection] 动作入图
            act_id = memory.inject_action(action, step)
            
            # 2. [Dynamics] 预测未来 (G_t -> G_t+1)
            # 直接调用 dynamic.predict，传入当前图和动作
            next_G, success, msg = dynamics.predict(memory.G, action)
            
            # 3. [Update] 更新记忆
            if success:
                memory.update_state(next_G)
                print(f"      Step {step}: {action['action']} -> ✅ {msg}")
            else:
                print(f"      Step {step}: {action['action']} -> ❌ {msg}")
                break # 模拟失败则停止该序列
            
            trace.append({"step": step, "action": action, "msg": msg})

        # F. 可视化与保存
        # 保存最终状态的图 (包含了所有动作节点和最终物体位置)
        filename = f"scene_{idx}_final.png"
        viz.render(memory.to_scene_graph(), filename=filename)
        
        dataset_log.append({
            "scene": idx, 
            "task": instruction, 
            "trace": trace,
            "final_image": filename
        })
        
        # 实时写入
        with open(f"{output_dir}/log.json", "w") as f:
            json.dump(dataset_log, f, indent=2)

    env.stop()
    print("\n✅ Pipeline Finished.")

if __name__ == "__main__":
    main()