import os
import json
import numpy as np
import faiss
from typing import List, Dict, Tuple
from backend.config import settings

class VectorStore:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """初始化向量数据库"""
        self.index_path = os.path.join(settings.VECTOR_DB_PATH, "faiss.index")
        self.metadata_path = os.path.join(settings.VECTOR_DB_PATH, "metadata.json")

        # 加载或创建索引
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self._load_index()
        else:
            self._create_index()

    def _create_index(self):
        """创建新的FAISS索引"""
        dimension = settings.VECTOR_DIMENSION
        # 使用HNSW索引以提高检索速度
        self.index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
        # 存储元数据
        self.metadata: List[Dict] = []
        print(f"创建新的FAISS索引，维度: {dimension}")

    def _load_index(self):
        """加载已存在的索引"""
        self.index = faiss.read_index(self.index_path)
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        print(f"加载FAISS索引成功，包含 {len(self.metadata)} 条记录")

    def _save_index(self):
        """保存索引到磁盘"""
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def add_documents(self, embeddings: np.ndarray, documents: List[Dict]):
        """添加文档到向量数据库"""
        if len(embeddings) != len(documents):
            raise ValueError("嵌入向量数量与文档数量不匹配")

        # 添加到索引
        self.index.add(embeddings)
        # 添加元数据
        self.metadata.extend(documents)
        # 保存到磁盘
        self._save_index()
        print(f"添加了 {len(documents)} 条文档到向量数据库")

    def search(self, query_embedding: np.ndarray, top_k: int = None) -> List[Tuple[Dict, float]]:
        """搜索相似文档"""
        if top_k is None:
            top_k = settings.TOP_K

        if len(self.metadata) == 0:
            return []

        # 确保不超过现有文档数量
        top_k = min(top_k, len(self.metadata))

        # 搜索
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype(np.float32),
            top_k
        )

        # 返回结果（文档元数据和相似度分数）
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                results.append((self.metadata[idx], float(dist)))

        return results

    def delete_by_filename(self, filename: str) -> int:
        """根据文件名删除文档"""
        # 找到需要保留的文档索引
        keep_indices = [i for i, doc in enumerate(self.metadata)
                       if doc.get('metadata', {}).get('filename') != filename]

        if len(keep_indices) == len(self.metadata):
            return 0  # 没有找到要删除的文档

        # 重建索引
        if len(keep_indices) > 0:
            # 获取所有向量
            all_vectors = self.index.reconstruct_n(0, len(self.metadata))
            # 重新创建索引
            dimension = settings.VECTOR_DIMENSION
            self.index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
            self.index.add(all_vectors[keep_indices])
            # 更新元数据
            self.metadata = [self.metadata[i] for i in keep_indices]
        else:
            # 所有文档都被删除，创建新索引
            self._create_index()

        self._save_index()
        deleted_count = len(self.metadata) - len(keep_indices)
        return deleted_count

    def get_all_documents(self) -> List[Dict]:
        """获取所有文档"""
        return self.metadata

    def get_document_count(self) -> int:
        """获取文档数量"""
        return len(self.metadata)

    def clear(self):
        """清空向量数据库"""
        self._create_index()
        self._save_index()
        print("已清空向量数据库")
