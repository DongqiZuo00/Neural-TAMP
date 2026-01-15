import sys
import os

# 获取当前脚本的目录: .../Neural-TAMP/src/perception
script_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录: .../Neural-TAMP
project_root = os.path.abspath(os.path.join(script_dir, "../.."))

# 将项目根目录加入路径，这样 Python 就能找到 'src' 包了
sys.path.append(project_root)
import torch
import json
import re
import uuid
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# 引入我们刚才定义的通用数据结构
from src.core.graph_schema import SceneGraph, Node, Edge

class VLMInterface:
    def __init__(self, model_path="Neural-TAMP/models/Qwen2-VL-7B-Instruct", device="cuda"):
        print(f"[VLM] Initializing Qwen2-VL from {model_path}...")
        self.device = device
        
        try:
            # 加载模型 (自动使用 bfloat16 节省显存)
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(model_path)
            print("[VLM] Model loaded successfully.")
        except Exception as e:
            print(f"[VLM Error] Failed to load model: {e}")
            print("Tip: Check if the model path is correct and download finished.")
            raise e

    def _build_system_prompt(self, instruction: str) -> str:
        """
        [Prompt Engineering] 
        这是整个感知模块最关键的部分。
        我们必须给 VLM 一个严格的 Schema，否则它会开始讲故事。
        """
        return f"""You are a robot perception system. 
        Task: Analyze the image based on the instruction: "{instruction}".
        Output: A Scene Graph in strict JSON format.
        
        **Requirements:**
        1. **Objects**: Detect the target object, receptacle, and any obstacles/blockers.
        2. **Boxes**: Provide 2D bounding boxes [ymin, xmin, ymax, xmax] (scale 0-1000).
        3. **States**: Infer states like "Open", "Closed", "Empty", "Full".
        4. **Relations**: Identify spatial relations ("inside", "on", "close_to") and logical relations ("blocked_by").
        
        **Output Format (JSON Only):**
        ```json
        {{
          "objects": [
            {{ "label": "apple", "box_2d": [100, 200, 300, 400], "state": "default" }},
            {{ "label": "fridge", "box_2d": [0, 500, 1000, 900], "state": "closed" }}
          ],
          "relations": [
            {{ "source_label": "apple", "target_label": "fridge", "relation": "inside" }}
          ]
        }}
        ```
        Do not output any markdown or explanation. Just the JSON string. """
        
    def parse(self, image_input, instruction: str) -> SceneGraph:
        """
        核心流程: Image -> VLM -> Text -> JSON -> SceneGraph Object
        """
        # 1. 图像加载
        if isinstance(image_input, str):
            image = Image.open(image_input)
        else:
            image = image_input
    
        # 2. 构建输入
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self._build_system_prompt(instruction)},
                ],
            }
        ]
    
        # 3. 预处理
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
    
        # 4. 推理
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
            
        # 5. 解码
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
    
        # 6. 解析并转换为 Graph 对象
        return self._text_to_graph(output_text)

    def _text_to_graph(self, text: str) -> SceneGraph:
        """
        将 VLM 的文本输出清洗并封装成 SceneGraph 对象
        """
        sg = SceneGraph()
    
        try:
            # 提取 JSON 块
            match = re.search(r"```json(.*?)```", text, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
            else:
                # 尝试直接找 {}
                match = re.search(r"\{.*\}", text, re.DOTALL)
                json_str = match.group(0).strip() if match else "{}"
            
            data = json.loads(json_str)
            
            # --- 构建 Node ---
            # 这里的 ID 只是临时的，Global Fusion 时会更新
            label_to_id = {} 
            
            for i, obj in enumerate(data.get("objects", [])):
                label = obj.get("label", "unknown")
                # 生成唯一 ID: label + 序号 (e.g., "apple|0")
                node_id = f"{label}|{i}" 
                label_to_id[label] = node_id # 简单记录，用于处理边的关系
                
                # 创建 Node 对象 (注意：目前 pos 还是空的，spatial_lifter 负责填入)
                node = Node(
                    id=node_id,
                    label=label,
                    pos=(0.0, 0.0, 0.0), # 占位，待 depth 填充
                    bbox=obj.get("box_2d", [0,0,0,0]), # 这里存的是 2D 框
                    state=obj.get("state")
                )
                sg.add_node(node)
                
            # --- 构建 Edge ---
            for rel in data.get("relations", []):
                src_label = rel.get("source_label")
                tgt_label = rel.get("target_label")
                relation = rel.get("relation")
                
                # 尝试找到对应的 ID (简单的模糊匹配)
                # 实际生产中这里需要更复杂的匹配逻辑，防止多个 apple 搞混
                src_id = None
                tgt_id = None
                
                for l, nid in label_to_id.items():
                    if src_label in l: src_id = nid
                    if tgt_label in l: tgt_id = nid
                
                if src_id and tgt_id:
                    edge = Edge(src_id, tgt_id, relation)
                    sg.add_edge(edge)
                    
        except Exception as e:
            print(f"[VLM Parser Error] Could not parse JSON: {e}")
            print(f"[Raw Output] {text}")
            
        return sg

# 修改 src/perception/vlm_interface.py 的底部

if __name__ == "__main__":
    import os
    
    # 1. 指定图片路径
    img_path = "Neural-TAMP/test_image.png" # 确保你把图片放到了这里
    
    # 如果没有图片，先创建一个假的（防止报错，但最好用你的真图）
    if not os.path.exists(img_path):
        print(f"Warning: {img_path} not found. using dummy white image.")
        dummy_img = Image.new('RGB', (640, 480), color='white')
        image_input = dummy_img
    else:
        print(f"Loading real image from {img_path}...")
        image_input = img_path

    # 2. 初始化接口
    try:
        vlm = VLMInterface()
        
        # 3. 发送指令 (我们故意问一个稍微难点的)
        # 指令：找到所有物体，特别是植物
        instruction = "Detect all objects in the room, especially the plant."
        
        print(f"Instruction: {instruction}")
        print("Parsing...")
        
        graph = vlm.parse(image_input, instruction)
        
        print("\n" + "="*40)
        print("🎉 SUCCESS! Generated Scene Graph:")
        print("="*40)
        # 打印生成的 Prompt 文本，检查是否包含 "Plant"
        print(graph.to_prompt_text()) 
        
        print("\n[Raw Node Data]:")
        for node_id, node in graph.nodes.items():
            print(f"- {node.label} (Box: {node.bbox})")
            
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
