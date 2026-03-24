#!/usr/bin/env python3
"""
嵌入模型模块
用于生成文本的向量表示
"""

import os
import sys
import numpy as np
from typing import List, Union

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from backend.config import settings


class EmbeddingModel:
    """嵌入模型类
    
    使用CPU友好的轻量级模型生成文本嵌入
    """
    
    _instance = None
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化模型"""
        if self._initialized:
            return
        
        self.dimension = getattr(settings, 'VECTOR_DIMENSION', 1024)
        self.model_name = getattr(settings, 'EMBEDDING_MODEL', 'cpu-friendly-model')
        
        # 尝试加载真实模型，如果失败则使用模拟实现
        self._load_model()
        self._initialized = True
    
    def _load_model(self):
        """加载嵌入模型"""
        try:
            # 尝试使用 sentence-transformers
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            self.use_real_model = True
            # 更新维度为实际模型维度
            self.dimension = self.model.get_sentence_embedding_dimension()
            print(f"嵌入模型加载成功: {self.model_name}, 维度: {self.dimension}")
        except ImportError:
            print("警告: sentence-transformers 未安装，使用模拟嵌入")
            self.use_real_model = False
            self.model = None
    
    def embed_text(self, text: str) -> np.ndarray:
        """生成单条文本的嵌入向量
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量 (numpy数组)
        """
        if self.use_real_model and self.model:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)
        else:
            # 模拟嵌入：基于文本哈希生成确定性向量
            return self._mock_embed(text)
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """批量生成文本嵌入
        
        Args:
            texts: 文本列表
            
        Returns:
            嵌入向量矩阵 (n_texts x dimension)
        """
        if self.use_real_model and self.model:
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return embeddings.astype(np.float32)
        else:
            # 批量模拟嵌入
            embeddings = [self._mock_embed(text) for text in texts]
            return np.array(embeddings, dtype=np.float32)
    
    def _mock_embed(self, text: str) -> np.ndarray:
        """生成模拟嵌入向量
        
        使用文本哈希生成确定性向量，确保相同文本产生相同嵌入
        
        Args:
            text: 输入文本
            
        Returns:
            模拟嵌入向量
        """
        # 使用哈希生成种子
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        np.random.seed(hash_val % (2**32))
        
        # 生成随机向量并归一化
        vector = np.random.randn(self.dimension).astype(np.float32)
        vector = vector / np.linalg.norm(vector)
        
        # 重置随机种子
        np.random.seed(None)
        
        return vector


# 便捷函数
def get_embedding_model() -> EmbeddingModel:
    """获取EmbeddingModel单例实例"""
    return EmbeddingModel()


if __name__ == "__main__":
    # 测试
    model = EmbeddingModel()
    
    test_texts = [
        "这是一个测试文本",
        "这是另一个测试文本",
        "完全无关的内容"
    ]
    
    print("测试单条嵌入:")
    embedding = model.embed_text(test_texts[0])
    print(f"  维度: {embedding.shape}")
    print(f"  前5个值: {embedding[:5]}")
    
    print("\n测试批量嵌入:")
    embeddings = model.embed_texts(test_texts)
    print(f"  形状: {embeddings.shape}")
    
    # 测试相似度
    from numpy import dot
    from numpy.linalg import norm
    
    sim_0_1 = dot(embeddings[0], embeddings[1]) / (norm(embeddings[0]) * norm(embeddings[1]))
    sim_0_2 = dot(embeddings[0], embeddings[2]) / (norm(embeddings[0]) * norm(embeddings[2]))
    
    print(f"\n相似度测试:")
    print(f"  文本0 vs 文本1 (相似): {sim_0_1:.4f}")
    print(f"  文本0 vs 文本2 (不相似): {sim_0_2:.4f}")
