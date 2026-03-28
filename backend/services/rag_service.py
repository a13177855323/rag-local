"""
RAG服务模块

提供文档导入、查询、对话历史管理等核心功能。
遵循PEP8编码规范，包含完整的异常处理和日志记录。

Author: RAG System
Date: 2026-03-23
"""

import logging
import os
import time
from typing import Dict, List, Optional

from backend.config import settings
from backend.models.embedding_model import EmbeddingModel
from backend.models.llm_model import LLMModel
from backend.services.conversation_store import get_conversation_store
from backend.services.vector_store import VectorStore
from backend.utils.document_processor import DocumentProcessor


# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGService:
    """
    RAG服务主类 - 单例模式

    整合文档处理、向量检索、LLM生成、对话历史管理等功能。
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """初始化RAG服务组件"""
        logger.info("初始化RAG服务...")

        self.embedding_model = EmbeddingModel()
        self.llm_model = LLMModel()
        self.vector_store = VectorStore()
        self.document_processor = DocumentProcessor()

        if settings.ENABLE_CONVERSATION_HISTORY:
            self.conversation_store = get_conversation_store()
            logger.info("对话历史功能已启用")
        else:
            self.conversation_store = None
            logger.info("对话历史功能已禁用")

        logger.info("RAG服务初始化完成")

    def ingest_document(self, file_path: str) -> Dict:
        """
        处理并导入单个文档

        Args:
            file_path: 文档文件路径

        Returns:
            处理结果字典
        """
        try:
            documents = self.document_processor.process_file(file_path)

            if not documents:
                logger.warning(f"文档处理失败，没有提取到有效内容: {file_path}")
                return {
                    "success": False,
                    "error": "文档处理失败，没有提取到有效内容"
                }

            texts = [doc["content"] for doc in documents]
            embeddings = self.embedding_model.embed_documents(texts)

            self.vector_store.add_documents(embeddings, documents)

            logger.info(f"成功导入文档: {os.path.basename(file_path)}, 共 {len(documents)} 个文本块")
            return {
                "success": True,
                "filename": os.path.basename(file_path),
                "chunks": len(documents),
                "message": f"成功导入文档，共 {len(documents)} 个文本块"
            }
        except Exception as e:
            logger.error(f"导入文档失败: {e}")
            return {"success": False, "error": str(e)}

    def ingest_documents(self, file_paths: List[str]) -> List[Dict]:
        """
        批量导入文档

        Args:
            file_paths: 文档文件路径列表

        Returns:
            处理结果列表
        """
        results = []
        for file_path in file_paths:
            result = self.ingest_document(file_path)
            results.append(result)
        return results

    def query(
        self,
        question: str,
        top_k: int = None,
        stream: bool = False,
        session_id: str = None
    ):
        """
        查询知识库

        Args:
            question: 用户问题
            top_k: 返回结果数量
            stream: 是否流式输出
            session_id: 对话会话ID

        Returns:
            查询结果
        """
        if top_k is None:
            top_k = settings.TOP_K

        start_time = time.time()

        query_embedding = self.embedding_model.embed_query(question)

        search_results = self.vector_store.search(query_embedding, top_k)

        context = [doc.get("content", "") for doc, score in search_results]
        sources = [
            {
                "filename": doc.get("metadata", {}).get("filename", ""),
                "content": doc.get("content", "")[:200],
                "similarity": float(score)
            }
            for doc, score in search_results
        ]

        if not context:
            response_time = int((time.time() - start_time) * 1000)
            answer = "知识库中没有找到相关文档，请先上传文档。"

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
                "session_id": session_id
            }
            if stream:
                yield result
            else:
                return result
            return

        if stream:
            return self._stream_response(
                question=question,
                context=context,
                sources=sources,
                session_id=session_id,
                start_time=start_time
            )
        else:
            answer = self.llm_model.generate(question, context)
            response_time = int((time.time() - start_time) * 1000)

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
                "session_id": session_id
            }

    def _stream_response(
        self,
        question: str,
        context: List[str],
        sources: List[Dict],
        session_id: str,
        start_time: float
    ):
        """流式响应处理"""
        full_response = ""
        start = time.time()

        for chunk in self.llm_model.generate_stream(question, context):
            full_response += chunk
            yield {
                "answer": chunk,
                "done": False,
                "sources": sources,
                "session_id": session_id
            }

        response_time = int((time.time() - start) * 1000)

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
            "session_id": session_id
        }

    def _record_conversation(
        self,
        session_id: str,
        question: str,
        answer: str,
        sources: List[Dict],
        response_time_ms: int
    ) -> None:
        """记录对话历史"""
        if not settings.ENABLE_CONVERSATION_HISTORY:
            return

        if not self.conversation_store:
            return

        try:
            self.conversation_store.add_turn(
                session_id=session_id,
                question=question,
                answer=answer,
                sources=sources,
                response_time_ms=response_time_ms,
                is_code_query=False
            )
        except Exception as e:
            logger.error(f"记录对话历史失败: {e}")

    def get_document_list(self) -> List[str]:
        """获取已上传的文档列表"""
        documents = self.vector_store.get_all_documents()
        filenames = set()
        for doc in documents:
            filename = doc.get("metadata", {}).get("filename")
            if filename:
                filenames.add(filename)
        return list(filenames)

    def delete_document(self, filename: str) -> Dict:
        """删除指定文档"""
        try:
            deleted_count = self.vector_store.delete_by_filename(filename)
            if deleted_count > 0:
                logger.info(f"删除文档: {filename}, 共 {deleted_count} 个文本块")
                return {
                    "success": True,
                    "message": f"成功删除文档 {filename}，共删除 {deleted_count} 个文本块"
                }
            else:
                return {
                    "success": False,
                    "error": "未找到该文档"
                }
        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            return {"success": False, "error": str(e)}

    def get_stats(self) -> Dict:
        """获取系统统计信息"""
        return {
            "total_documents": len(self.vector_store.get_all_documents()),
            "unique_files": len(self.get_document_list()),
            "embedding_model": settings.EMBEDDING_MODEL,
            "llm_model": settings.LLM_MODEL,
            "vector_dimension": settings.VECTOR_DIMENSION,
            "conversation_history_enabled": settings.ENABLE_CONVERSATION_HISTORY
        }

    def clear_knowledge_base(self) -> Dict:
        """清空知识库"""
        try:
            self.vector_store.clear()
            logger.info("知识库已清空")
            return {
                "success": True,
                "message": "知识库已清空"
            }
        except Exception as e:
            logger.error(f"清空知识库失败: {e}")
            return {"success": False, "error": str(e)}

    def create_conversation_session(self, title: str = None) -> str:
        """创建对话会话"""
        if not self.conversation_store:
            return None
        return self.conversation_store.create_session(title)

    def get_conversation_sessions(self) -> List[Dict]:
        """获取所有对话会话"""
        if not self.conversation_store:
            return []
        return self.conversation_store.get_all_sessions()

    def get_conversation_session(self, session_id: str) -> Optional[Dict]:
        """获取单个会话详情"""
        if not self.conversation_store:
            return None

        session = self.conversation_store.get_session(session_id)
        if not session:
            return None

        return {
            "id": session.id,
            "title": session.title,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "turns": [
                {
                    "id": turn.id,
                    "question": turn.question,
                    "answer": turn.answer,
                    "category": turn.category,
                    "is_code_query": turn.is_code_query,
                    "timestamp": turn.timestamp,
                    "response_time_ms": turn.response_time_ms,
                    "sources": turn.sources,
                    "quality_score": turn.quality_score
                }
                for turn in session.turns
            ]
        }

    def delete_conversation_session(self, session_id: str) -> Dict:
        """删除对话会话"""
        if not self.conversation_store:
            return {"success": False, "error": "对话历史功能未启用"}

        success = self.conversation_store.delete_session(session_id)
        if success:
            return {"success": True, "message": "会话已删除"}
        return {"success": False, "error": "会话不存在"}
