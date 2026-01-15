import sys
import os
import random
import shutil
import json
import time

# --- 路径修正 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from src.env.procthor_wrapper import ProcTHOREnv
from src.memory.graph_manager import GraphManager
from src.utils.visualizer import BEVVisualizer
from src.perception.oracle_interface import OracleInterface
# 导入我们刚才重写的对抗性生成器
from src.task.task_generator import TaskGenerator

def main():
    print("="*60)
    print("🧪 Neural-TAMP: Adversarial Task Generation Test")
    print("   Target: Valid, Hard Tasks with Instructions > 60 chars")
    print("="*60)

    # 1. 准备输出目录
    output_dir = "Neural-TAMP/vis_output/task_test"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # 2. 初始化模块 (不包含 Planner)
    try:
        env = ProcTHOREnv()
        oracle = OracleInterface(env)
        memory = GraphManager(save_dir="Neural-TAMP/memory_data")
        viz = BEVVisualizer(save_dir=output_dir)
        task_gen = TaskGenerator()
        print("✅ Modules Initialized.")
    except Exception as e:
        print(f"❌ Init Failed: {e}")
        return

    # 3. 配置参数
    TOTAL_SAMPLES = 50
    count = 0
    candidate_indices = random.sample(range(10000), 200)
    dataset_log = []

    print("\n🎬 Starting Generation Loop...")
    start_time = time.time()

    for idx in candidate_indices:
        if count >= TOTAL_SAMPLES:
            break

        # --- A. 场景加载 ---
        try:
            # 切换场景
            obs = env.change_scene(idx)
        except Exception:
            continue

        # 过滤掉单间，只测多房间的大户型 (增加难度)
        if len(env.current_scene.get("rooms", [])) < 2:
            continue

        # --- B. 构建语义图 ---
        hierarchical_graph = oracle.get_hierarchical_graph()
        memory.override_global_graph(hierarchical_graph)

        # --- C. 生成对抗性任务 ---
        # 这里的 generate 内部已经包含了 Hard Constraint Check 和 RL Reward Calculation
        instruction, task_meta = task_gen.generate(memory.global_graph)

        if instruction is None:
            print(f"   ⚠️ Scene {idx}: Generation Failed ({task_meta.get('error')})")
            continue

        # --- D. 验证与日志 ---
        
        # 1. 长度检查
        char_len = len(instruction)
        len_check = "✅" if char_len >= 60 else "❌ TOO SHORT"
        
        # 2. 攻击性检查 (打印 Reward 详情)
        factors = task_meta.get("difficulty_factors", {})
        dist_str = f"Dist: {factors.get('dist_m', 0)}m"
        wall_str = "WALL" if factors.get('is_near_wall') else "open"
        clutter_str = f"Clutter: {factors.get('clutter_items', 0)}"
        score = task_meta.get('adversarial_score', 0)

        print(f"\n[{count+1}/{TOTAL_SAMPLES}] Scene {idx}")
        print(f"   🎯 Task: {instruction}")
        print(f"   📏 Length: {char_len} chars {len_check}")
        print(f"   😈 Difficulty: {score:.2f} ({dist_str}, {wall_str}, {clutter_str})")
        print(f"   📍 Logic: {task_meta['target_class']} -> {task_meta['dest_class']}")

        # --- E. 可视化保存 ---
        gt_filename = f"task_{idx:05d}_GT.png"
        ai_filename = f"task_{idx:05d}_AI.png"
        
        env.save_ground_truth_bev(os.path.join(output_dir, gt_filename))
        viz.render(memory.global_graph, filename=ai_filename)

        # 记录数据
        dataset_log.append({
            "scene_index": idx,
            "instruction": instruction,
            "length": char_len,
            "metadata": task_meta,
            "images": [gt_filename, ai_filename]
        })
        
        count += 1

    # 4. 结束
    env.stop()
    
    # 保存 JSON
    with open(os.path.join(output_dir, "task_dataset.json"), "w") as f:
        json.dump(dataset_log, f, indent=2)

    print("\n" + "="*60)
    print(f"🎉 Test Complete. Generated {count} tasks.")
    print(f"📂 Results saved to: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()