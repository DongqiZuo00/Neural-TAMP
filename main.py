import sys
import os
import random
import shutil
import json
import time

# --- 路径修正 (防止 ModuleNotFoundError) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 导入核心模块
from src.env.procthor_wrapper import ProcTHOREnv
from src.memory.graph_manager import GraphManager
from src.utils.visualizer import BEVVisualizer
from src.perception.oracle_interface import OracleInterface
from src.task.task_generator import TaskGenerator

# [关键修改] 使用新的分层规划器 (MLDT Logic)
from src.planning.decomposer import TaskDecomposer 

def main():
    print("="*60)
    print("🚀 Neural-TAMP: Batch Generation Pipeline (MLDT Planner Integration)")
    print("="*60)

    # 0. 检查 API Key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  WARNING: OPENAI_API_KEY is not set. The Planner will likely fail.")
        # 你可以在这里选择 return，或者让它报错
        # return 

    # 1. 准备输出目录
    output_dir = "Neural-TAMP/vis_output/batch_dataset"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir) # 清空旧数据
    os.makedirs(output_dir)
    print(f"📂 Output Directory: {output_dir}")

    # 2. 初始化所有 AI 模块
    try:
        print("[System] Initializing Modules...")
        env = ProcTHOREnv()
        oracle = OracleInterface(env)
        memory = GraphManager(save_dir="Neural-TAMP/memory_data")
        viz = BEVVisualizer(save_dir=output_dir)
        task_gen = TaskGenerator()
        
        # 使用新的分层规划器
        planner = TaskDecomposer(model_name="gpt-4o") 
        
        print("✅ All Systems Ready.")
    except Exception as e:
        print(f"❌ Initialization Failed: {e}")
        return

    # 3. 配置批量生成参数
    TOTAL_SAMPLES = 50  # 目标样本数
    count = 0
    # 随机抽取 200 个备选场景 index
    candidate_indices = random.sample(range(10000), 200)
    
    # 数据集日志列表
    dataset_log = []

    print("\n🎬 Starting Data Generation Loop...")
    
    start_time = time.time()

    for idx in candidate_indices:
        if count >= TOTAL_SAMPLES:
            break

        print(f"\n" + "-"*40)
        print(f"🔄 Processing Candidate Index: {idx}")

        # --- A. 场景加载 (Environment) ---
        try:
            obs = env.change_scene(idx)
        except Exception as e:
            print(f"   ⚠️ Scene Load Failed: {e}")
            continue

        # --- B. 场景筛选 (Filtering) ---
        # 只保留多房间的大户型，保证导航任务的复杂度
        num_rooms = len(env.current_scene.get("rooms", []))
        if num_rooms < 2:
            print(f"   ⚠️ Skipped: Single Room Layout ({num_rooms} room)")
            continue 

        # --- C. 感知与记忆构建 (Perception & Memory) ---
        # 1. Oracle 获取带几何信息和 Room ID 的 Graph
        hierarchical_graph = oracle.get_hierarchical_graph()
        
        # 2. 存入记忆，并计算严格的 "Same-Room" Edge
        memory.override_global_graph(hierarchical_graph)
        
        # --- D. 任务生成 (Task Generation) ---
        # 基于当前的 Graph 生成一个可行的 Pick & Place 任务
        instruction, task_meta = task_gen.generate(memory.global_graph)
        
        if instruction is None:
            print(f"   ⚠️ Task Gen Failed: {task_meta.get('error')}")
            continue

        print(f"   ✅ Task Generated: \"{instruction}\"")

        # --- E. [核心] 分层规划 (Hierarchical Planning) ---
        # 使用 TaskDecomposer: Task -> Subgoals -> Actions
        try:
            plan_actions = planner.plan(instruction, memory.global_graph)
        except Exception as e:
            print(f"   ❌ Planning Exception: {e}")
            plan_actions = []

        if not plan_actions:
            print("   ⚠️ Planner returned empty plan. Skipping sample.")
            continue

        # --- F. 数据保存 (Data Saving) ---
        # 1. 保存图片 (GT vs AI Perception)
        gt_filename = f"scene_{idx:05d}_GT.png"
        ai_filename = f"scene_{idx:05d}_AI.png"
        
        env.save_ground_truth_bev(os.path.join(output_dir, gt_filename))
        viz.render(memory.global_graph, filename=ai_filename)

        # 2. 构造数据条目
        log_entry = {
            "scene_index": idx,
            "num_rooms": num_rooms,
            "instruction": instruction,
            "task_metadata": task_meta,     # 包含 target_id, dest_id，方便 GNN 训练
            "plan": plan_actions,           # 这是你的 Ground Truth Actions
            "visualization": {
                "ground_truth_bev": gt_filename,
                "ai_semantic_map": ai_filename
            }
            # 如果需要，这里也可以保存 memory.global_graph.to_dict()
        }
        dataset_log.append(log_entry)
        count += 1
        
        print(f"   💾 Sample Saved. Total: {count}/{TOTAL_SAMPLES}")

        # 实时写入 JSON，防止中途崩溃数据丢失
        with open(os.path.join(output_dir, "dataset_full.json"), "w") as f:
            json.dump(dataset_log, f, indent=2)

    # 4. 结束与统计
    env.stop()
    duration = time.time() - start_time
    print("\n" + "="*60)
    print(f"🎉 Pipeline Finished in {duration:.1f}s.")
    print(f"✅ Successfully collected {count} samples.")
    print(f"📂 Dataset location: {os.path.join(output_dir, 'dataset_full.json')}")
    print("="*60)

if __name__ == "__main__":
    main()