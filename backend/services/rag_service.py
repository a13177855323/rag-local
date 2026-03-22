from typing import List, Dict
import os
from backend.config import settings
from backend.models.embedding_model import EmbeddingModel
from backend.models.llm_model import LLMModel
from backend.services.vector_store import VectorStore
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
                yield {
                    "answer": "",
                    "done": True,
                    "sources": sources
                }
            return stream_response()
        else:
            answer = self.llm_model.generate(question, context)
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
