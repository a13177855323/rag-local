from typing import List, Dict, Optional
import os
from backend.config import settings
from backend.models.embedding_model import EmbeddingModel
from backend.models.llm_model import LLMModel
from backend.services.vector_store import VectorStore
from backend.services.chat_history_manager import ChatHistoryManager
from backend.utils.document_processor import DocumentProcessor

class RAGService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """初始化RAG服务"""
        self.embedding_model = EmbeddingModel()
        self.llm_model = LLMModel()
        self.vector_store = VectorStore()
        self.document_processor = DocumentProcessor()
        self.chat_history_manager = ChatHistoryManager()

    def ingest_document(self, file_path: str) -> Dict:
        """处理并导入单个文档"""
        try:
            # 处理文档
            documents = self.document_processor.process_file(file_path)
            
            if not documents:
                return {"success": False, "error": "文档处理失败，没有提取到有效内容"}

            # 生成嵌入向量
            texts = [doc["content"] for doc in documents]
            embeddings = self.embedding_model.embed_documents(texts)

            # 存储到向量数据库
            self.vector_store.add_documents(embeddings, documents)

            return {
                "success": True,
                "filename": os.path.basename(file_path),
                "chunks": len(documents),
                "message": f"成功导入文档，共 {len(documents)} 个文本块"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def ingest_documents(self, file_paths: List[str]) -> List[Dict]:
        """批量导入文档"""
        results = []
        for file_path in file_paths:
            result = self.ingest_document(file_path)
            results.append(result)
        return results

    def query(self, question: str, top_k: int = None, stream: bool = False, session_id: str = None):
        """查询知识库
        Args:
            question: 用户问题
            top_k: 返回结果数量
            stream: 是否流式输出
            session_id: 会话ID，用于关联对话历史
        """
        if top_k is None:
            top_k = settings.TOP_K

        # 记录用户问题到对话历史
        if session_id:
            self.chat_history_manager.add_message(session_id, 'user', question)

        # 生成查询嵌入
        query_embedding = self.embedding_model.embed_query(question)

        # 搜索相关文档
        search_results = self.vector_store.search(query_embedding, top_k)

        # 提取上下文
        context = [doc.get("content", "") for doc, score in search_results]
        sources = [
            {
                "filename": doc.get("metadata", {}).get("filename", ""),
                "content": doc.get("content", "")[:200],  # 只返回前200字符作为预览
                "similarity": float(score)
            } for doc, score in search_results
        ]

        if not context:
            empty_answer = "知识库中没有找到相关文档，请先上传文档。"
            # 记录助手回答
            if session_id:
                self.chat_history_manager.add_message(
                    session_id, 'assistant', empty_answer,
                    metadata={'sources': []}
                )
            if stream:
                yield {
                    "answer": "知识库中没有找到相关文档，请先上传文档。",
                    "sources": []
                }
            else:
                return {
                    "answer": "知识库中没有找到相关文档，请先上传文档。",
                    "sources": []
                }
            return

        # 生成回答
        if stream:
            # 流式输出
            def stream_response():
                full_response = ""
                for chunk in self.llm_model.generate_stream(question, context):
                    full_response += chunk
                    yield {
                        "answer": chunk,
                        "done": False,
                        "sources": sources
                    }
                # 记录完整回答到对话历史
                if session_id:
                    self.chat_history_manager.add_message(
                        session_id, 'assistant', full_response,
                        metadata={'sources': sources}
                    )
                yield {
                    "answer": "",
                    "done": True,
                    "sources": sources
                }
            return stream_response()
        else:
            answer = self.llm_model.generate(question, context)
            # 记录助手回答
            if session_id:
                self.chat_history_manager.add_message(
                    session_id, 'assistant', answer,
                    metadata={'sources': sources}
                )
            return {
                "answer": answer,
                "sources": sources
            }

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
            return {"success": False, "error": str(e)}

    def get_stats(self) -> Dict:
        """获取系统统计信息"""
        return {
            "total_documents": len(self.vector_store.get_all_documents()),
            "unique_files": len(self.get_document_list()),
            "embedding_model": settings.EMBEDDING_MODEL,
            "llm_model": settings.LLM_MODEL,
            "vector_dimension": settings.VECTOR_DIMENSION
        }

    def clear_knowledge_base(self) -> Dict:
        """清空知识库"""
        try:
            self.vector_store.clear()
            return {
                "success": True,
                "message": "知识库已清空"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ==================== 对话历史管理方法 ====================
    
    def create_chat_session(self, title: str = None) -> Dict:
        """创建新的对话会话"""
        session = self.chat_history_manager.create_session(title)
        return {
            "success": True,
            "session_id": session.session_id,
            "title": session.title,
            "created_at": session.created_at
        }
    
    def get_chat_session(self, session_id: str) -> Optional[Dict]:
        """获取指定会话详情"""
        session = self.chat_history_manager.get_session(session_id)
        if not session:
            return None
        return session.to_dict()
    
    def get_all_chat_sessions(self, limit: int = None) -> List[Dict]:
        """获取所有会话列表"""
        sessions = self.chat_history_manager.get_all_sessions(limit)
        return [session.to_dict() for session in sessions]
    
    def delete_chat_session(self, session_id: str) -> Dict:
        """删除指定会话"""
        success = self.chat_history_manager.delete_session(session_id)
        if success:
            return {"success": True, "message": "会话已删除"}
        return {"success": False, "error": "会话不存在"}
    
    def clear_chat_history(self) -> Dict:
        """清空所有对话历史"""
        count = self.chat_history_manager.clear_all()
        return {
            "success": True,
            "message": f"已清空 {count} 个会话",
            "deleted_count": count
        }
    
    def analyze_chat_session(self, session_id: str) -> Optional[Dict]:
        """分析指定会话"""
        return self.chat_history_manager.analyze_session(session_id)
    
    def get_chat_stats(self) -> Dict:
        """获取对话历史整体统计"""
        return self.chat_history_manager.get_overall_stats()
    
    def export_chat_history(self, session_ids: List[str] = None, 
                          export_format: str = 'json',
                          with_analysis: bool = True) -> Dict:
        """导出对话历史
        
        Args:
            session_ids: 要导出的会话ID列表，None表示导出全部
            export_format: 导出格式: 'json', 'markdown', 'csv'
            with_analysis: 是否包含智能分析（仅Markdown格式支持）
        """
        try:
            content = self.chat_history_manager.export_sessions(
                session_ids=session_ids,
                export_format=export_format,
                with_analysis=with_analysis
            )
            return {
                "success": True,
                "format": export_format,
                "content": content
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
