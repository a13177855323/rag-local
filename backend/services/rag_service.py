from typing import List, Dict
import os
import time
from backend.config import settings
from backend.models.embedding_model import EmbeddingModel
from backend.models.llm_model import LLMModel
from backend.services.vector_store import VectorStore
from backend.utils.document_processor import DocumentProcessor
from backend.services.conversation_history import get_conversation_history
from backend.services.conversation_analyzer import get_analyzer

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
        self.conversation_history = get_conversation_history()
        self.analyzer = get_analyzer()

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

    def query(self, question: str, top_k: int = None, stream: bool = False):
        """查询知识库"""
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
            } for doc, score in search_results
        ]

        if not context:
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

        if stream:
            def stream_response():
                full_response = ""
                for chunk in self.llm_model.generate_stream(question, context):
                    full_response += chunk
                    yield {
                        "answer": chunk,
                        "done": False,
                        "sources": sources
                    }
                yield {
                    "answer": "",
                    "done": True,
                    "sources": sources
                }
                
                response_time_ms = (time.time() - start_time) * 1000
                if settings.ENABLE_CONVERSATION_HISTORY:
                    turn = self.conversation_history.add_turn(
                        question=question,
                        answer=full_response,
                        sources=sources,
                        response_time_ms=response_time_ms
                    )
                    if settings.CONVERSATION_AUTO_ANALYZE:
                        category = self.analyzer.classify_question(question)
                        quality = self.analyzer.score_turn(turn)
                        self.conversation_history.update_turn_analysis(
                            turn.turn_id, category, quality
                        )
            return stream_response()
        else:
            answer = self.llm_model.generate(question, context)
            response_time_ms = (time.time() - start_time) * 1000
            
            if settings.ENABLE_CONVERSATION_HISTORY:
                turn = self.conversation_history.add_turn(
                    question=question,
                    answer=answer,
                    sources=sources,
                    response_time_ms=response_time_ms
                )
                if settings.CONVERSATION_AUTO_ANALYZE:
                    category = self.analyzer.classify_question(question)
                    quality = self.analyzer.score_turn(turn)
                    self.conversation_history.update_turn_analysis(
                        turn.turn_id, category, quality
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

    def get_conversation_sessions(self) -> List[Dict]:
        """获取所有对话会话列表"""
        return self.conversation_history.get_all_sessions()

    def get_conversation_session(self, session_id: str) -> Dict:
        """获取指定会话详情"""
        session = self.conversation_history.get_session(session_id)
        if session:
            analysis = self.analyzer.analyze_session(session)
            return {
                "session": session.to_dict(),
                "analysis": {
                    "summary": analysis.summary,
                    "total_turns": analysis.total_turns,
                    "category_stats": analysis.category_stats,
                    "avg_quality_score": analysis.avg_quality_score,
                    "avg_response_time_ms": analysis.avg_response_time_ms,
                    "topics": analysis.topics,
                    "quality_distribution": analysis.quality_distribution
                }
            }
        return {"error": "会话不存在"}

    def get_current_session(self) -> Dict:
        """获取当前会话"""
        session = self.conversation_history.get_current_session()
        if session:
            analysis = self.analyzer.analyze_session(session)
            return {
                "session": session.to_dict(),
                "analysis": {
                    "summary": analysis.summary,
                    "total_turns": analysis.total_turns,
                    "category_stats": analysis.category_stats,
                    "avg_quality_score": analysis.avg_quality_score
                }
            }
        return {"error": "无当前会话"}

    def analyze_conversations(self, session_ids: List[str] = None) -> Dict:
        """分析对话历史"""
        from backend.services.conversation_exporter import get_exporter
        
        if session_ids:
            sessions = []
            for sid in session_ids:
                session = self.conversation_history.get_session(sid)
                if session:
                    sessions.append(session)
        else:
            all_sessions_info = self.conversation_history.get_all_sessions()
            sessions = []
            for info in all_sessions_info:
                session = self.conversation_history.get_session(info["session_id"])
                if session:
                    sessions.append(session)

        return self.analyzer.analyze_multiple_sessions(sessions)

    def export_conversations(
        self,
        format_type: str = "markdown",
        session_ids: List[str] = None
    ) -> Dict:
        """导出对话历史"""
        from backend.services.conversation_exporter import get_exporter
        exporter = get_exporter()

        if session_ids:
            sessions = []
            for sid in session_ids:
                session = self.conversation_history.get_session(sid)
                if session:
                    sessions.append(session)
        else:
            all_sessions_info = self.conversation_history.get_all_sessions()
            sessions = []
            for info in all_sessions_info:
                session = self.conversation_history.get_session(info["session_id"])
                if session:
                    sessions.append(session)

        if not sessions:
            return {"error": "没有可导出的对话记录"}

        results = exporter.export_all_sessions(sessions, self.analyzer, format_type)
        return {
            "success": True,
            "format": format_type,
            "exported_files": results,
            "session_count": len(sessions)
        }

    def delete_conversation_session(self, session_id: str) -> Dict:
        """删除指定会话"""
        success = self.conversation_history.delete_session(session_id)
        if success:
            return {"success": True, "message": f"已删除会话 {session_id}"}
        return {"success": False, "error": "会话不存在"}

    def clear_conversation_history(self) -> Dict:
        """清空所有对话历史"""
        count = self.conversation_history.clear_all_history()
        return {"success": True, "message": f"已清空 {count} 个会话"}
