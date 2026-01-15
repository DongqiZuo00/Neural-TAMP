import sys
import os

# 获取当前脚本的目录: .../Neural-TAMP/src/perception
script_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录: .../Neural-TAMP
project_root = os.path.abspath(os.path.join(script_dir, "../.."))

# 将项目根目录加入路径，这样 Python 就能找到 'src' 包了
sys.path.append(project_root)
import torch
from sentence_transformers import SentenceTransformer, util

class SemanticMatcher:
    """
    [语义匹配器]
    利用轻量级 BERT 模型将文本转换为向量，实现基于含义的模糊匹配。
    解决 "receptacle" != "container" 但意思相近的问题。
    """
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        print(f"[Matcher] Loading semantic model: {model_name}...")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        self.model = SentenceTransformer(model_name, device=device)
        
        # --- 定义核心概念锚点 (Anchors) ---
        # 我们不再列举几百个单词，而是定义几个核心概念。
        # 模型会自动计算输入词和这些概念的距离。
        
        # 1. 大型家具/固定设施 (不可移动，作为定位锚点)
        self.anchor_concepts = [
            "large furniture", "heavy appliance", "fixed fixture",
            "bed", "table", "sofa", "fridge", "cabinet", "shelf", "door", "window"
        ]
        
        # 2. 容器/收纳用品 (用于 Inside 关系)
        self.container_concepts = [
            "container", "receptacle", "storage", "holder",
            "box", "can", "bin", "drawer", "cup", "bowl", "bag"
        ]
        
        # 预计算锚点的向量 (加速运行时推理)
        self.anchor_embeddings = self.model.encode(self.anchor_concepts, convert_to_tensor=True)
        self.container_embeddings = self.model.encode(self.container_concepts, convert_to_tensor=True)
        
        # 相似度阈值 (0.0 ~ 1.0)
        # 大于此值认为意思相近。0.4 是经验值，对应 MiniLM 模型比较鲁棒。
        self.similarity_threshold = 0.45
        
        print("[Matcher] Ready.")

    def is_anchor(self, label: str) -> bool:
        """判断是否为大型家具/锚点"""
        return self._check_similarity(label, self.anchor_embeddings)

    def is_container(self, label: str) -> bool:
        """判断是否为容器"""
        return self._check_similarity(label, self.container_embeddings)

    def _check_similarity(self, label: str, target_embeddings) -> bool:
        # 1. 将输入标签转为向量
        label_embedding = self.model.encode(label, convert_to_tensor=True)
        
        # 2. 计算与所有目标概念的余弦相似度
        # util.cos_sim 返回一个矩阵，我们取最大值
        cosine_scores = util.cos_sim(label_embedding, target_embeddings)
        max_score = torch.max(cosine_scores).item()
        
        # 3. 调试日志 (可选，用于观察模型在想什么)
        # if max_score > 0.3:
        #     print(f"Debug: '{label}' similarity to category: {max_score:.3f}")
            
        return max_score > self.similarity_threshold