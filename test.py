import sys
import os
import random
import time
import json

# --- 路径设置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# --- 导入核心模块 ---
from src.env.procthor_wrapper import ProcTHOREnv
from src.perception.oracle_interface import OracleInterface
from src.memory.graph_manager import GraphManager
from src.task.task_generator import TaskGenerator 

def test_task_generation():
    print("="*60)
    print("🛠️  Testing: Scene-Aware Adversarial Task Generator (Headless Mode)")
    print("="*60)

    # 1. 初始化最小系统 (严格参照 main.py)
    try:
        print("[Init] Loading ProcTHOR Environment...")
        
        # [修正] 严格保持无参数初始化，适应服务器无头环境
        env = ProcTHOREnv() 
        
        oracle = OracleInterface(env)
        memory = GraphManager(save_dir="Neural-TAMP/debug_data")
        task_gen = TaskGenerator()
        print("✅ Modules Initialized.\n")
    except Exception as e:
        print(f"❌ Init Failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. 随机抽取几个场景进行测试
    # 为了快速验证，只测 3 个
    test_indices = random.sample(range(10000), 3) 
    
    for i, idx in enumerate(test_indices):
        print(f"▶️  Test Case {i+1}/3 | Scene Index: {idx}")
        
        # --- Step A: 加载场景 ---
        try:
            env.change_scene(idx)
        except Exception as e:
            print(f"   ⚠️ Load Failed: {e}")
            continue

        # --- Step B: 过滤简单场景 ---
        # 必须确保有 rooms 字段
        current_rooms = env.current_scene.get("rooms", [])
        num_rooms = len(current_rooms)
        
        # if num_rooms < 2:
        #     print(f"   ⚠️ Skipped (Single Room: {num_rooms})")
        #     continue

        # --- Step C: 感知与建图 ---
        print(f"   👀 Perception: Scanning {num_rooms} rooms...")
        # 这一步会调用 Oracle 获取全知图
        graph = oracle.get_hierarchical_graph()
        memory.override_global_graph(graph)
        
        # --- Step D: 生成任务 ---
        start_t = time.time()
        
        # 调用生成器
        instruction, meta = task_gen.generate(memory.to_scene_graph())
        
        duration = time.time() - start_t
        
        # --- Step E: 结果验证与打印 ---
        if instruction:
            print(f"   ✅ Task Generated ({duration:.3f}s):")
            print(f"      📜 Instruction: \"{instruction}\"")
            print(f"      📊 Type: {meta.get('type')}")
            
            # 打印任务链详情 (如果存在)
            if 'chain_details' in meta:
                print(f"      ⛓️  Task Chain ({meta.get('length')} Steps):")
                for step in meta['chain_details']:
                    # 获取详细信息
                    t_label = step.get('target', 'Unknown')
                    d_label = step.get('dest', 'Unknown')
                    adv_score = step.get('adversarial_score', 0)
                    reason = step.get('reason', {})                    
        else:
            # 如果生成失败，打印原因 (可能是场景太空，或者没找到符合攻击条件的物体)
            print(f"   ❌ Generation Failed: {meta.get('error')}")

        print("-" * 40)

    # 结束清理
    try:
        env.stop()
    except:
        pass
    print("\n✅ Test Finished.")

if __name__ == "__main__":
    test_task_generation()