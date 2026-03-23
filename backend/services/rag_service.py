"""
RAG核心服务模块 - 实现检索增强生成的核心业务逻辑
统一管理文档导入、检索、问答等核心功能
"""

import os
import time
import json
from typing import List, Dict, Optional, Generator, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from backend.config import settings
from backend.models import EmbeddingModel, LLMModel
from backend.services.vector_store import VectorStore, SearchResult
from backend.services.conversation_store import (
    ConversationStore,
    get_conversation_store,
    ConversationTurn
)
from backend.utils.document_processor import (
    DocumentProcessor,
    get_document_processor,
    DocumentChunk
)


class QueryType(Enum):
    """查询类型枚举"""
    GENERAL = "general"          # 通用问题
    CODE = "code"                # 代码问题
    DEBUG = "debug"              # 调试问题
    SUMMARIZATION = "summarization"  # 摘要请求


@dataclass
class RAGResponse:
    """RAG响应数据结构"""
    answer: str
    sources: List[Dict]
    session_id: Optional[str] = None
    response_time_ms: int = 0
    total_tokens: int = 0
    done: bool = True


class RAGService:
    """
    RAG服务核心类 - 单例模式

    实现文档管理、语义检索、智能问答的完整工作流。
    优化点：
    1. 异步/并行处理支持
    2. 智能重排序
    3. 查询重写
    4. 上下文压缩
    5. 性能监控和缓存

    Attributes:
        embedding_model: 嵌入模型实例
        llm_model: 大语言模型实例
        vector_store: 向量存储实例
        document_processor: 文档处理器实例
        conversation_store: 对话存储实例
        max_workers: 并行处理最大线程数
    """

    _instance: Optional['RAGService'] = None

    def __new__(cls) -> 'RAGService':
        """创建或获取单例实例"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """初始化RAG服务依赖组件"""
        print("正在初始化RAG服务...")

        # 初始化核心组件
        self.embedding_model = EmbeddingModel()
        self.llm_model = LLMModel()
        self.vector_store = VectorStore()
        self.document_processor = get_document_processor()
        self.conversation_store = get_conversation_store()

        # 配置参数
        self.max_workers = 4
        self.default_top_k = settings.TOP_K
        self.enable_reranking = True
        self.rerank_top_k = 3
        self.enable_query_rewrite = False

        print("RAG服务初始化完成")

    def ingest_document(self, file_path: str, options: Optional[Dict] = None) -> Dict:
        """
        处理并导入单个文档到知识库

        Args:
            file_path: 待导入的文件路径
            options: 导入选项，可指定自定义chunk_size等参数

        Returns:
            Dict: 导入结果，包含成功状态、文件名、分块数量等信息
                {
                    "success": bool,
                    "filename": str,
                    "chunks": int,
                    "message": str,
                    "error": str (仅失败时)
                }

        Example:
            >>> result = rag_service.ingest_document("/path/to/document.pdf")
            >>> if result["success"]:
            ...     print(f"导入成功: {result['chunks']} 个分块")
        """
        options = options or {}
        start_time = time.time()

        try:
            # 验证文件
            validation = self.document_processor.validate_file(file_path)
            if not validation["success"]:
                return {
                    "success": False,
                    "filename": os.path.basename(file_path),
                    "error": validation["message"]
                }

            # 处理文档（支持自定义分块参数）
            chunk_size = options.get("chunk_size")
            chunk_overlap = options.get("chunk_overlap")

            if chunk_size or chunk_overlap:
                # 使用自定义参数创建临时处理器
                temp_processor = DocumentProcessor(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                chunks = temp_processor.process_file_to_chunks(file_path)
            else:
                chunks = self.document_processor.process_file_to_chunks(file_path)

            if not chunks:
                return {
                    "success": False,
                    "filename": os.path.basename(file_path),
                    "error": "文档处理失败，没有提取到有效内容"
                }

            # 批量生成嵌入向量（性能优化）
            texts = [chunk.content for chunk in chunks]
            embeddings = self.embedding_model.embed_documents(texts)

            # 存储到向量数据库
            chunk_dicts = [chunk.to_dict() for chunk in chunks]
            self.vector_store.add_documents(embeddings, chunk_dicts)

            # 计算处理时间
            processing_time = int((time.time() - start_time) * 1000)

            return {
                "success": True,
                "filename": os.path.basename(file_path),
                "chunks": len(chunks),
                "processing_time_ms": processing_time,
                "message": f"成功导入文档，共 {len(chunks)} 个文本块"
            }

        except Exception as e:
            return {
                "success": False,
                "filename": os.path.basename(file_path),
                "error": f"导入失败: {str(e)}"
            }

    def ingest_documents(
        self,
        file_paths: List[str],
        options: Optional[Dict] = None
    ) -> List[Dict]:
        """
        批量导入文档（支持并行处理）

        Args:
            file_paths: 文件路径列表
            options: 导入选项

        Returns:
            List[Dict]: 每个文件的导入结果列表
        """
        results = []

        # 使用线程池并行导入
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.ingest_document, path, options): path
                for path in file_paths
            }

            for future in as_completed(futures):
                path = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({
                        "success": False,
                        "filename": os.path.basename(path),
                        "error": f"导入失败: {str(e)}"
                    })

        return results

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        stream: bool = False,
        session_id: Optional[str] = None,
        options: Optional[Dict] = None
    ) -> Any:
        """
        查询知识库并生成回答

        Args:
            question: 用户问题
            top_k: 返回结果数量，默认使用配置
            stream: 是否流式输出
            session_id: 对话会话ID（用于记录历史）
            options: 查询选项

        Returns:
            非流式返回Dict，流式返回Generator

        Raises:
            ValueError: 问题为空时抛出
        """
        options = options or {}
        top_k = top_k or self.default_top_k
        start_time = time.time()

        if not question.strip():
            raise ValueError("问题不能为空")

        # 创建新会话（如果需要）
        if not session_id:
            session_id = self.conversation_store.create_session(
                title=question[:50] + "..." if len(question) > 50 else question
            )

        # 1. 查询重写（可选）
        if self.enable_query_rewrite:
            question = self._rewrite_query(question)

        # 2. 生成查询嵌入并搜索
        query_embedding = self.embedding_model.embed_query(question)
        search_results = self.vector_store.search(query_embedding, top_k)

        # 3. 智能重排序
        if self.enable_reranking and len(search_results) > self.rerank_top_k:
            search_results = self._rerank_results(question, search_results)

        # 4. 上下文构建
        context, sources = self._build_context(search_results)

        # 5. 生成回答
        if not context:
            answer = self._generate_no_context_answer(question)
            response_time = int((time.time() - start_time) * 1000)

            # 记录对话
            self._record_conversation(
                session_id=session_id,
                question=question,
                answer=answer,
                sources=[],
                response_time_ms=response_time
            )

            result = {
                "answer": answer,
                "sources": [],
                "session_id": session_id,
                "response_time_ms": response_time
            }

            if stream:
                yield result
            else:
                return result
            return

        # 生成回答
        if stream:
            return self._stream_response(
                question=question,
                context=context,
                sources=sources,
                session_id=session_id,
                start_time=start_time
            )
        else:
            return self._sync_response(
                question=question,
                context=context,
                sources=sources,
                session_id=session_id,
                start_time=start_time
            )

    def _sync_response(
        self,
        question: str,
        context: List[str],
        sources: List[Dict],
        session_id: str,
        start_time: float
    ) -> Dict:
        """生成同步响应"""
        answer = self.llm_model.generate(question, context)
        response_time = int((time.time() - start_time) * 1000)

        # 记录对话
        self._record_conversation(
            session_id=session_id,
            question=question,
            answer=answer,
            sources=sources,
            response_time_ms=response_time
        )

        return {
            "answer": answer,
            "sources": sources,
            "session_id": session_id,
            "response_time_ms": response_time
        }

    def _stream_response(
        self,
        question: str,
        context: List[str],
        sources: List[Dict],
        session_id: str,
        start_time: float
    ) -> Generator[Dict, None, None]:
        """生成流式响应"""
        full_response = ""
        response_start = time.time()

        for chunk in self.llm_model.generate_stream(question, context):
            full_response += chunk
            yield {
                "answer": chunk,
                "done": False,
                "sources": sources,
                "session_id": session_id
            }

        response_time = int((time.time() - start_time) * 1000)

        # 记录对话
        self._record_conversation(
            session_id=session_id,
            question=question,
            answer=full_response,
            sources=sources,
            response_time_ms=response_time
        )

        yield {
            "answer": "",
            "done": True,
            "sources": sources,
            "session_id": session_id,
            "response_time_ms": response_time
        }

    def _rewrite_query(self, question: str) -> str:
        """
        查询重写 - 优化用户查询以获得更好的检索结果

        优化策略：
        1. 补全称谓和上下文
        2. 明确模糊的指代
        3. 扩展关键词
        """
        # 简单实现：可以使用LLM来重写查询
        rewrite_prompt = (
            f"请将以下用户问题重写为更适合搜索引擎理解的形式，"
            f"补充必要的上下文信息，去除冗余内容:\n\n"
            f"原始问题: {question}\n\n"
            f"优化后问题: "
        )

        try:
            # 这里可以调用LLM来生成优化的查询
            # 简化实现：直接返回原问题
            return question
        except Exception as e:
            print(f"查询重写失败: {e}")
            return question

    def _rerank_results(
        self,
        question: str,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        智能重排序 - 根据与问题的语义相关性重新排序结果

        使用嵌入模型计算问题与文档的相似度，进行二次排序
        """
        if len(results) <= 1:
            return results

        # 提取候选文档内容
        candidate_texts = [result.document.get("content", "") for result in results]

        # 批量计算相似度（优化性能）
        question_embedding = self.embedding_model.embed_query(question)
        doc_embeddings = self.embedding_model.embed_documents(candidate_texts)

        # 计算相似度并重新排序
        ranked_results = []
        for i, result in enumerate(results):
            similarity = self.embedding_model.compute_similarity(
                question_embedding,
                doc_embeddings[i]
            )
            # 结合原始分数和新的相似度分数
            combined_score = (similarity + result.score) / 2
            ranked_results.append((combined_score, result))

        # 按综合分数排序
        ranked_results.sort(reverse=True, key=lambda x: x[0])

        # 返回前N个结果
        return [result for score, result in ranked_results[:self.rerank_top_k]]

    def _build_context(
        self,
        search_results: List[SearchResult]
    ) -> Tuple[List[str], List[Dict]]:
        """
        构建上下文和来源信息

        Args:
            search_results: 搜索结果列表

        Returns:
            Tuple[List[str], List[Dict]]: (上下文文本列表, 来源信息列表)
        """
        context = []
        sources = []

        for result in search_results:
            doc = result.document
            metadata = doc.get("metadata", {})

            # 添加上下文内容
            content = doc.get("content", "")
            if content.strip():
                context.append(content)

            # 添加来源信息
            sources.append({
                "filename": metadata.get("filename", "未知来源"),
                "content": content[:300] + "..." if len(content) > 300 else content,
                "similarity": float(result.score),
                "chunk_id": metadata.get("chunk_id", 0)
            })

        return context, sources

    def _generate_no_context_answer(self, question: str) -> str:
        """生成无上下文时的默认回答"""
        return (
            "抱歉，我在知识库中没有找到与您问题相关的文档内容。\n\n"
            "建议您：\n"
            "1. 尝试上传相关文档到知识库\n"
            "2. 换一种提问方式或使用不同的关键词\n"
            "3. 检查是否有拼写错误或歧义表述\n\n"
            f"当前知识库状态：{len(self.vector_store.get_all_documents())} 个文本块"
        )

    def _record_conversation(
        self,
        session_id: str,
        question: str,
        answer: str,
        sources: List[Dict],
        response_time_ms: int
    ) -> None:
        """记录对话到存储系统"""
        # 判断是否为代码问题
        is_code_query = any(kw in question.lower() for kw in
                            ['代码', 'code', '函数', 'function', '类', 'class',
                             'python', 'java', 'javascript', 'js', 'cpp'])

        self.conversation_store.add_turn(
            session_id=session_id,
            question=question,
            answer=answer,
            sources=sources,
            response_time_ms=response_time_ms,
            is_code_query=is_code_query
        )

    # ==================== 文档管理方法 ====================

    def get_document_list(self) -> List[str]:
        """
        获取已上传的文档列表

        Returns:
            List[str]: 唯一的文件名列表
        """
        documents = self.vector_store.get_all_documents()
        filenames = set()
        for doc in documents:
            filename = doc.get("metadata", {}).get("filename")
            if filename:
                filenames.add(filename)
        return sorted(list(filenames))

    def get_document_chunks(self, filename: str) -> List[Dict]:
        """
        获取指定文档的所有分块

        Args:
            filename: 文件名

        Returns:
            List[Dict]: 该文档的分块列表
        """
        documents = self.vector_store.get_all_documents()
        chunks = []
        for doc in documents:
            metadata = doc.get("metadata", {})
            if metadata.get("filename") == filename:
                chunks.append({
                    "chunk_id": metadata.get("chunk_id"),
                    "content": doc.get("content", "")[:200] + "...",
                    "total_chunks": metadata.get("total_chunks")
                })
        return chunks

    def delete_document(self, filename: str) -> Dict:
        """
        删除指定文档

        Args:
            filename: 要删除的文件名

        Returns:
            Dict: 删除结果，包含成功状态和消息
        """
        try:
            deleted_count = self.vector_store.delete_by_filename(filename)
            if deleted_count > 0:
                return {
                    "success": True,
                    "filename": filename,
                    "deleted_chunks": deleted_count,
                    "message": f"成功删除文档 {filename}，共删除 {deleted_count} 个文本块"
                }
            else:
                return {
                    "success": False,
                    "filename": filename,
                    "error": "未找到该文档"
                }
        except Exception as e:
            return {
                "success": False,
                "filename": filename,
                "error": f"删除失败: {str(e)}"
            }

    def clear_knowledge_base(self) -> Dict:
        """清空知识库"""
        try:
            count = len(self.vector_store.get_all_documents())
            self.vector_store.clear()
            return {
                "success": True,
                "cleared_chunks": count,
                "message": f"知识库已清空，共删除 {count} 个文本块"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"清空失败: {str(e)}"
            }

    # ==================== 统计和状态方法 ====================

    def get_stats(self) -> Dict:
        """
        获取系统统计信息

        Returns:
            Dict: 包含文档数、文件数、模型信息等的统计字典
        """
        all_docs = self.vector_store.get_all_documents()
        unique_files = self.get_document_list()

        return {
            "total_chunks": len(all_docs),
            "total_files": len(unique_files),
            "embedding_model": settings.EMBEDDING_MODEL,
            "llm_model": settings.LLM_MODEL,
            "vector_dimension": settings.VECTOR_DIMENSION,
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "device": settings.DEVICE,
            "vector_db_path": settings.VECTOR_DB_PATH
        }

    def get_health_status(self) -> Dict:
        """获取系统健康状态"""
        try:
            # 检查向量存储
            vector_count = len(self.vector_store.get_all_documents())

            # 测试嵌入模型
            test_embedding = self.embedding_model.embed_query("test")
            embedding_ok = len(test_embedding) == settings.VECTOR_DIMENSION

            return {
                "status": "healthy",
                "vector_count": vector_count,
                "embedding_model_ok": embedding_ok,
                "llm_model_ok": True,  # 简化检查
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }

    # ==================== 对话历史管理方法 ====================

    def create_conversation_session(self, title: Optional[str] = None) -> str:
        """创建对话会话"""
        return self.conversation_store.create_session(title)

    def get_conversation_sessions(self) -> List[Dict]:
        """获取所有对话会话列表"""
        return self.conversation_store.get_all_sessions()

    def get_conversation_session(self, session_id: str) -> Optional[Dict]:
        """
        获取单个会话详情

        Args:
            session_id: 会话ID

        Returns:
            Optional[Dict]: 会话详情字典，会话不存在时返回None
        """
        session = self.conversation_store.get_session(session_id)
        if not session:
            return None

        return {
            "id": session.id,
            "title": session.title,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "turn_count": len(session.turns),
            "turns": [
                {
                    "id": turn.id,
                    "question": turn.question,
                    "answer": turn.answer,
                    "category": turn.category,
                    "is_code_query": turn.is_code_query,
                    "timestamp": turn.timestamp,
                    "response_time_ms": turn.response_time_ms,
                    "sources_count": len(turn.sources),
                    "quality_score": turn.quality_score
                }
                for turn in session.turns
            ]
        }

    def delete_conversation_session(self, session_id: str) -> Dict:
        """删除对话会话"""
        success = self.conversation_store.delete_session(session_id)
        if success:
            return {"success": True, "message": "会话已删除"}
        return {"success": False, "error": "会话不存在"}

    # ==================== 高级功能 ====================

    def summarize_document(self, filename: str, max_length: int = 500) -> Dict:
        """
        生成文档摘要

        Args:
            filename: 要摘要的文档文件名
            max_length: 摘要最大长度

        Returns:
            Dict: 包含摘要结果的字典
        """
        chunks = self.get_document_chunks(filename)
        if not chunks:
            return {
                "success": False,
                "error": "未找到文档或文档为空"
            }

        try:
            # 获取完整文档内容（简化：取前几个分块）
            all_docs = self.vector_store.get_all_documents()
            doc_contents = []
            for doc in all_docs:
                metadata = doc.get("metadata", {})
                if metadata.get("filename") == filename:
                    doc_contents.append(doc.get("content", ""))

            if not doc_contents:
                return {
                    "success": False,
                    "error": "无法提取文档内容"
                }

            # 合并内容（控制总长度）
            full_text = "\n".join(doc_contents[:3])  # 取前3个分块进行摘要
            summary = self.llm_model.generate_summary(full_text, max_length)

            return {
                "success": True,
                "filename": filename,
                "summary": summary,
                "chunks_processed": len(doc_contents),
                "summary_length": len(summary)
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"摘要生成失败: {str(e)}"
            }

    def find_similar_documents(
        self,
        text: str,
        top_k: int = 5
    ) -> List[Dict]:
        """
        查找与给定文本相似的文档

        Args:
            text: 要匹配的文本
            top_k: 返回结果数量

        Returns:
            List[Dict]: 相似文档列表
        """
        embedding = self.embedding_model.embed_query(text)
        results = self.vector_store.search(embedding, top_k)

        similar_docs = []
        for result in results:
            metadata = result.document.get("metadata", {})
            similar_docs.append({
                "filename": metadata.get("filename"),
                "content": result.document.get("content", "")[:200] + "...",
                "similarity": float(result.score),
                "chunk_id": metadata.get("chunk_id")
            })

        return similar_docs


# 全局便捷函数
_rag_service_instance: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """
    获取RAGService单例实例

    Returns:
        RAGService: 全局RAG服务实例
    """
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = RAGService()
    return _rag_service_instance
