import sys
import os
import random
import shutil

# --- 路径修正 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from src.env.procthor_wrapper import ProcTHOREnv
from src.memory.graph_manager import GraphManager
from src.utils.visualizer import BEVVisualizer
from src.perception.oracle_interface import OracleInterface

def main():
    print("="*60)
    print("🚀 Neural-TAMP: 50-Scene Multi-Room Batch Test")
    print("="*60)

    # 0. 准备输出目录 (清空旧数据，保持整洁)
    output_dir = "Neural-TAMP/vis_output/batch_test"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    print(f"📂 Output Directory: {output_dir}")

    # 1. 初始化系统
    try:
        env = ProcTHOREnv()
        oracle = OracleInterface(env)
        memory = GraphManager(save_dir="Neural-TAMP/memory_data")
        viz = BEVVisualizer(save_dir=output_dir)
        print("✅ System Initialized.")
    except Exception as e:
        print(f"❌ Init Failed: {e}")
        return

    # 2. 配置批量参数
    TOTAL_SAMPLES = 50
    count = 0
    # 从 ProcTHOR-10k 的训练集中随机选 200 个备选，然后从中筛选出 50 个多房间的
    # (因为很多 index 其实是单间，我们需要过滤掉它们)
    candidate_indices = random.sample(range(10000), 200)

    print("\n🎬 Starting Batch Generation Loop...")
    
    for idx in candidate_indices:
        if count >= TOTAL_SAMPLES:
            break

        print(f"\n[Attempting Scene Index {idx}]...")
        
        # --- A. 切换场景 ---
        try:
            obs = env.change_scene(idx)
        except Exception as e:
            print(f"   ⚠️ Load Failed: {e}, skipping...")
            continue

        # --- B. 检查是否为多房间 (Multi-Room Check) ---
        # 我们通过检查户型数据里的 rooms 列表长度
        house = env.current_scene
        num_rooms = len(house.get("rooms", []))
        
        if num_rooms < 2:
            print(f"   ⚠️ Skipped: Single Room (Count: {num_rooms})")
            continue # 跳过单间，寻找大户型
            
        print(f"   ✅ Accepted: Found {num_rooms}-Room Layout.")

        # --- C. 拍摄真值鸟瞰图 (Ground Truth) ---
        gt_filename = f"scene_{idx:05d}_rooms_{num_rooms}_GT.png"
        gt_path = os.path.join(output_dir, gt_filename)
        env.save_ground_truth_bev(gt_path)

        # --- D. 构建 AI 语义地图 (AI Perception) ---
        # 1. Oracle 解析
        hierarchical_graph = oracle.get_hierarchical_graph()
        
        # 2. 存入记忆并计算 Edge
        memory.override_global_graph(hierarchical_graph)
        
        # 3. 渲染 AI 地图
        ai_filename = f"scene_{idx:05d}_rooms_{num_rooms}_AI.png"
        viz.render(memory.global_graph, filename=ai_filename)
        
        print(f"   -> Saved Pair: {gt_filename} | {ai_filename}")
        
        # --- E. 简单统计 ---
        node_count = len(memory.global_graph.nodes)
        edge_count = len(memory.global_graph.edges)
        print(f"   -> Stats: {node_count} Nodes, {edge_count} Edges")
        
        count += 1

    # 3. 结束
    env.stop()
    print("\n" + "="*60)
    print(f"🎉 Batch Test Complete. Generated {count} Multi-Room Scenes.")
    print(f"📂 Please check: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()