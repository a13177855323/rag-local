"""
向量存储模块 - 管理文档嵌入向量的存储和检索
基于FAISS实现高效的相似性搜索，支持增量更新和持久化存储
"""

import os
import json
import shutil
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from threading import Lock
import numpy as np
import faiss

from backend.config import settings


@dataclass
class SearchResult:
    """搜索结果数据结构"""
    document: Dict
    score: float
    rank: int = 0

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "document": self.document,
            "score": float(self.score),
            "rank": self.rank
        }


class VectorStore:
    """
    向量存储管理器 - 单例模式

    基于FAISS实现高效的向量存储和检索，支持：
    1. HNSW索引结构（高速近似最近邻搜索）
    2. 增量更新（支持添加、删除文档）
    3. 持久化存储（自动保存到磁盘）
    4. 线程安全操作
    5. 索引备份和恢复

    Attributes:
        index: FAISS索引实例
        metadata: 文档元数据列表
        index_path: 索引文件路径
        metadata_path: 元数据文件路径
        lock: 线程锁
    """

    _instance: Optional['VectorStore'] = None
    _initialized: bool = False

    def __new__(cls) -> 'VectorStore':
        """创建或获取单例实例"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """初始化向量存储（仅执行一次）"""
        if self._initialized:
            return

        self._initialize()
        self._initialized = True

    def _initialize(self) -> None:
        """初始化向量存储"""
        # 配置路径
        self.index_path = os.path.join(settings.VECTOR_DB_PATH, "faiss.index")
        self.metadata_path = os.path.join(settings.VECTOR_DB_PATH, "metadata.json")
        self.backup_dir = os.path.join(settings.VECTOR_DB_PATH, "backups")

        # 创建必要的目录
        os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)

        # 线程锁
        self.lock = Lock()

        # 索引配置
        self.dimension = settings.VECTOR_DIMENSION
        self.use_hnsw = True  # 使用HNSW索引以获得更高的检索速度

        # 加载或创建索引
        if self._index_exists():
            self._load_index()
        else:
            self._create_index()

        print(f"向量存储初始化完成，当前包含 {len(self.metadata)} 条记录")

    def _create_index(self) -> None:
        """创建新的FAISS索引"""
        with self.lock:
            if self.use_hnsw:
                # HNSW索引 - 适用于高维向量的快速近似搜索
                # M: 每个节点的邻居数 (默认32)
                # efConstruction: 建图时探索的邻居数 (默认40)
                # efSearch: 搜索时探索的邻居数 (默认16)
                self.index = faiss.IndexHNSWFlat(
                    self.dimension,
                    32,  # M参数
                    faiss.METRIC_INNER_PRODUCT
                )
                # 设置HNSW参数
                if hasattr(self.index, 'hnsw'):
                    self.index.hnsw.efConstruction = 40
                    self.index.hnsw.efSearch = 16
            else:
                # 平面索引 - 精确搜索但速度较慢
                self.index = faiss.IndexFlatIP(self.dimension)

            self.metadata: List[Dict] = []
            print(f"创建新的FAISS索引，维度: {self.dimension}, 类型: HNSW")

    def _index_exists(self) -> bool:
        """检查索引是否已存在"""
        return os.path.exists(self.index_path) and os.path.exists(self.metadata_path)

    def _load_index(self) -> None:
        """加载已存在的索引"""
        try:
            with self.lock:
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                print(f"加载FAISS索引成功，包含 {len(self.metadata)} 条记录")
        except Exception as e:
            print(f"加载索引失败，创建新索引: {e}")
            self._create_index()

    def _save_index(self) -> None:
        """保存索引到磁盘（线程安全）"""
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            raise RuntimeError(f"保存索引失败: {str(e)}") from e

    def _backup_index(self) -> None:
        """创建索引备份"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(self.backup_dir, f"backup_{timestamp}")
        os.makedirs(backup_path, exist_ok=True)

        try:
            # 复制索引文件
            if os.path.exists(self.index_path):
                shutil.copy2(
                    self.index_path,
                    os.path.join(backup_path, "faiss.index")
                )
            # 复制元数据文件
            if os.path.exists(self.metadata_path):
                shutil.copy2(
                    self.metadata_path,
                    os.path.join(backup_path, "metadata.json")
                )
            print(f"索引已备份到: {backup_path}")
        except Exception as e:
            print(f"备份索引失败: {e}")

    def add_documents(
        self,
        embeddings: np.ndarray,
        documents: List[Dict],
        create_backup: bool = True
    ) -> None:
        """
        批量添加文档到向量数据库

        Args:
            embeddings: shape为(n_documents, vector_dimension)的嵌入向量矩阵
            documents: 文档元数据列表，与embeddings一一对应
            create_backup: 是否在添加前创建备份

        Raises:
            ValueError: 嵌入向量数量与文档数量不匹配时抛出
            RuntimeError: 添加文档失败时抛出
        """
        if len(embeddings) != len(documents):
            raise ValueError(
                f"嵌入向量数量({len(embeddings)})与文档数量({len(documents)})不匹配"
            )

        if len(embeddings) == 0:
            print("警告：没有要添加的文档")
            return

        # 验证嵌入向量维度
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"嵌入向量维度不匹配: 期望 {self.dimension}, 实际 {embeddings.shape[1]}"
            )

        with self.lock:
            try:
                # 创建备份（可选）
                if create_backup and len(self.metadata) > 0:
                    self._backup_index()

                # 确保向量是float32类型（FAISS要求）
                if embeddings.dtype != np.float32:
                    embeddings = embeddings.astype(np.float32)

                # 添加到索引
                self.index.add(embeddings)

                # 添加元数据
                self.metadata.extend(documents)

                # 保存到磁盘
                self._save_index()

                print(f"成功添加 {len(documents)} 条文档到向量数据库")

            except Exception as e:
                # 尝试恢复（如果有备份）
                raise RuntimeError(f"添加文档失败: {str(e)}") from e

    def add_document(
        self,
        embedding: np.ndarray,
        document: Dict,
        create_backup: bool = True
    ) -> None:
        """
        添加单条文档到向量数据库

        Args:
            embedding: shape为(vector_dimension,)的嵌入向量
            document: 文档元数据
            create_backup: 是否在添加前创建备份
        """
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        self.add_documents(embedding, [document], create_backup)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = None,
        threshold: float = 0.0
    ) -> List[SearchResult]:
        """
        搜索相似文档

        Args:
            query_embedding: 查询向量，shape为(vector_dimension,)或(1, vector_dimension)
            top_k: 返回结果数量，默认使用配置中的TOP_K
            threshold: 相似度阈值（仅返回相似度大于此值的结果）

        Returns:
            List[SearchResult]: 搜索结果列表，按相似度从高到低排序

        Example:
            >>> query_vec = embedding_model.embed_query("什么是RAG？")
            >>> results = vector_store.search(query_vec, top_k=5)
            >>> for result in results:
            ...     print(f"相似度: {result.score:.4f}")
            ...     print(f"内容: {result.document.get('content', '')[:100]}")
        """
        if top_k is None:
            top_k = settings.TOP_K

        if len(self.metadata) == 0:
            return []

        # 确保不超过现有文档数量
        top_k = min(top_k, len(self.metadata))

        # 确保向量形状正确
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # 确保是float32类型
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)

        try:
            with self.lock:
                distances, indices = self.index.search(query_embedding, top_k)

            results = []
            for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
                if idx < 0 or idx >= len(self.metadata):
                    continue

                # 应用相似度阈值过滤
                if dist < threshold:
                    continue

                result = SearchResult(
                    document=self.metadata[idx],
                    score=float(dist),
                    rank=rank
                )
                results.append(result)

            return results

        except Exception as e:
            print(f"搜索失败: {e}")
            return []

    def search_with_scores(
        self,
        query_embedding: np.ndarray,
        top_k: int = None
    ) -> List[Tuple[Dict, float]]:
        """
        搜索相似文档（兼容旧接口）

        Returns:
            List[Tuple[Dict, float]]: (文档元数据, 相似度分数)元组列表
        """
        results = self.search(query_embedding, top_k)
        return [(result.document, result.score) for result in results]

    def delete_by_filename(self, filename: str) -> int:
        """
        根据文件名删除文档

        注意：由于FAISS不支持原地删除，需要重建索引。
        对于大规模数据集，建议定期重建或使用支持删除的索引结构。

        Args:
            filename: 要删除的文件名

        Returns:
            int: 被删除的文档数量
        """
        with self.lock:
            # 找到需要保留的文档索引
            keep_indices = [
                i for i, doc in enumerate(self.metadata)
                if doc.get('metadata', {}).get('filename') != filename
            ]

            deleted_count = len(self.metadata) - len(keep_indices)

            if deleted_count == 0:
                return 0

            print(f"准备删除 {deleted_count} 条文档")

            # 创建备份
            self._backup_index()

            if len(keep_indices) > 0:
                # 重建索引：获取所有向量并重新添加
                try:
                    all_vectors = self.index.reconstruct_n(0, len(self.metadata))
                except Exception as e:
                    print(f"重建索引失败，尝试恢复备份: {e}")
                    raise

                # 创建新索引
                self._create_index()

                # 批量添加保留的向量
                keep_vectors = all_vectors[keep_indices]
                self.metadata = [self.metadata[i] for i in keep_indices]

                if keep_vectors.size > 0:
                    self.index.add(keep_vectors)
            else:
                # 所有文档都被删除，创建新索引
                self._create_index()

            # 保存更改
            self._save_index()
            print(f"成功删除 {deleted_count} 条文档，剩余 {len(self.metadata)} 条")

            return deleted_count

    def get_all_documents(self) -> List[Dict]:
        """获取所有文档"""
        return self.metadata.copy()

    def get_document_count(self) -> int:
        """获取文档总数"""
        return len(self.metadata)

    def get_document_by_index(self, index: int) -> Optional[Dict]:
        """根据索引获取文档"""
        if 0 <= index < len(self.metadata):
            return self.metadata[index].copy()
        return None

    def get_vector_by_index(self, index: int) -> Optional[np.ndarray]:
        """根据索引获取向量"""
        if 0 <= index < len(self.metadata):
            try:
                return self.index.reconstruct(index)
            except Exception as e:
                print(f"获取向量失败: {e}")
                return None
        return None

    def get_all_vectors(self) -> Optional[np.ndarray]:
        """获取所有向量"""
        if len(self.metadata) == 0:
            return None

        try:
            return self.index.reconstruct_n(0, len(self.metadata))
        except Exception as e:
            print(f"获取所有向量失败: {e}")
            return None

    def clear(self) -> None:
        """清空向量数据库"""
        with self.lock:
            # 在清空之前备份
            if len(self.metadata) > 0:
                self._backup_index()

            self._create_index()
            self._save_index()
            print("已清空向量数据库")

    def get_index_stats(self) -> Dict[str, Any]:
        """
        获取索引统计信息

        Returns:
            Dict: 包含索引类型、向量数量、维度等信息的字典
        """
        index_type = type(self.index).__name__
        if hasattr(self.index, 'hnsw'):
            index_type = "HNSWFlat"

        return {
            "index_type": index_type,
            "total_vectors": len(self.metadata),
            "dimension": self.dimension,
            "metric": "Inner Product (Cosine Similarity)",
            "index_path": self.index_path,
            "metadata_path": self.metadata_path,
            "backup_count": len(os.listdir(self.backup_dir)) if os.path.exists(self.backup_dir) else 0
        }

    def rebuild_index(self, new_dimension: Optional[int] = None) -> bool:
        """
        重建索引（用于修复损坏的索引或更改维度）

        Args:
            new_dimension: 新的向量维度，None表示使用原维度

        Returns:
            bool: 重建成功返回True，失败返回False
        """
        if new_dimension is not None:
            self.dimension = new_dimension

        try:
            with self.lock:
                # 获取所有向量和元数据
                all_vectors = self.get_all_vectors()
                all_metadata = self.metadata.copy()

                if all_vectors is None:
                    print("没有向量需要重建，创建空索引")
                    self._create_index()
                    return True

                # 创建新索引
                self._create_index()

                # 重新添加向量
                self.index.add(all_vectors)
                self.metadata = all_metadata

                # 保存
                self._save_index()

                print(f"索引重建成功，包含 {len(self.metadata)} 条记录")
                return True

        except Exception as e:
            print(f"索引重建失败: {e}")
            return False

    def optimize_index(self) -> None:
        """优化索引性能（对于可训练的索引）"""
        try:
            if hasattr(self.index, 'train') and not self.index.is_trained:
                print("训练索引...")
                # 这里需要训练数据
                print("注意：当前索引类型不需要训练或训练需要额外数据")
            else:
                print("索引已经过训练或不支持训练")
        except Exception as e:
            print(f"索引优化失败: {e}")


# 全局便捷函数
_vector_store_instance: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """
    获取VectorStore单例实例

    Returns:
        VectorStore: 全局向量存储实例
    """
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore()
    return _vector_store_instance
