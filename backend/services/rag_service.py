from typing import List, Dict, Tuple, Optional
import os
from backend.config import settings
from backend.models.embedding_model import EmbeddingModel
from backend.models.llm_model import LLMModel
from backend.services.vector_store import VectorStore
from backend.utils.document_processor import DocumentProcessor
from backend.utils.code_detector import CodeDetector, detect_code_question
from backend.utils.code_formatter import (
    CodeFormatter, FormattedCodeResult, build_code_prompt
)


class HybridSearchResult:
    def __init__(self):
        self.code_results: List[Tuple[Dict, float]] = []
        self.text_results: List[Tuple[Dict, float]] = []
        self.merged_results: List[Tuple[Dict, float]] = []

    def get_context(self) -> List[str]:
        return [doc.get("content", "") for doc, _ in self.merged_results]

    def get_code_blocks(self) -> List[FormattedCodeResult]:
        results = []
        for doc, score in self.code_results:
            code_content = doc.get("code_content", "")
            if not code_content:
                code_content = doc.get("content", "")
            filename = doc.get("metadata", {}).get("filename", "unknown")
            results.append(FormattedCodeResult(
                code=code_content,
                language="python",
                description="",
                source_file=filename,
                similarity=round(score, 4)
            ))
        return results


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
        self.code_detector = CodeDetector()
        self.code_formatter = CodeFormatter()

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

    def _hybrid_search(
        self,
        query_embedding,
        question: str,
        top_k: int = None,
        is_code_question: bool = False
    ) -> HybridSearchResult:
        """混合检索策略：代码块优先 + 文本补充"""
        if top_k is None:
            top_k = settings.TOP_K

        result = HybridSearchResult()
        all_docs = self.vector_store.get_all_documents()

        if not all_docs:
            return result

        search_results = self.vector_store.search(query_embedding, top_k * 2)

        if is_code_question:
            code_top_k = settings.CODE_SEARCH_TOP_K
            code_results = []
            text_results = []

            for doc, score in search_results:
                metadata = doc.get("metadata", {})
                has_code = metadata.get("has_code", False)
                is_code_chunk = metadata.get("is_code_chunk", False)

                if has_code or is_code_chunk:
                    boosted_score = score * settings.CODE_BOOST_FACTOR
                    code_results.append((doc, boosted_score))
                else:
                    text_results.append((doc, score))

            code_results = sorted(code_results, key=lambda x: x[1], reverse=True)[:code_top_k]
            text_results = text_results[:top_k - len(code_results)]

            result.code_results = code_results
            result.text_results = text_results
            result.merged_results = code_results + text_results
        else:
            result.merged_results = search_results[:top_k]
            for doc, score in result.merged_results:
                if doc.get("metadata", {}).get("has_code", False):
                    result.code_results.append((doc, score))
                else:
                    result.text_results.append((doc, score))

        return result

    def _build_enhanced_prompt(
        self,
        question: str,
        search_result: HybridSearchResult,
        is_code_question: bool
    ) -> str:
        """构建增强的提示词"""
        if is_code_question and search_result.code_results:
            code_blocks = search_result.get_code_blocks()
            text_context = [doc.get("content", "") for doc, _ in search_result.text_results[:2]]
            return build_code_prompt(question, code_blocks, text_context)
        else:
            context = search_result.get_context()
            return None

    def query(self, question: str, top_k: int = None, stream: bool = False):
        """查询知识库（支持代码智能检索）"""
        if top_k is None:
            top_k = settings.TOP_K

        is_code_question = False
        code_confidence = 0.0

        if settings.ENABLE_CODE_DETECTION:
            is_code_question, code_confidence = detect_code_question(question)
            is_code_question = is_code_question and code_confidence >= settings.CODE_CONFIDENCE_THRESHOLD

        query_embedding = self.embedding_model.embed_query(question)

        search_result = self._hybrid_search(query_embedding, question, top_k, is_code_question)

        context = search_result.get_context()

        sources = []
        for doc, score in search_result.merged_results:
            metadata = doc.get("metadata", {})
            source = {
                "filename": metadata.get("filename", ""),
                "content": doc.get("content", "")[:200],
                "similarity": float(score),
                "has_code": metadata.get("has_code", False),
                "is_code_chunk": metadata.get("is_code_chunk", False)
            }
            if metadata.get("has_code", False):
                source["code_languages"] = metadata.get("code_languages", [])
            sources.append(source)

        code_sources = []
        for doc, score in search_result.code_results:
            code_sources.append({
                "filename": doc.get("metadata", {}).get("filename", ""),
                "code": doc.get("code_content", "") or doc.get("content", ""),
                "similarity": float(score)
            })

        if not context:
            if stream:
                yield {
                    "answer": "知识库中没有找到相关文档，请先上传文档。",
                    "sources": [],
                    "code_results": [],
                    "is_code_question": is_code_question
                }
            else:
                return {
                    "answer": "知识库中没有找到相关文档，请先上传文档。",
                    "sources": [],
                    "code_results": [],
                    "is_code_question": is_code_question
                }
            return

        enhanced_prompt = self._build_enhanced_prompt(question, search_result, is_code_question)

        if stream:
            def stream_response():
                if enhanced_prompt:
                    for chunk in self.llm_model.generate_stream(enhanced_prompt, context):
                        yield {
                            "answer": chunk,
                            "done": False,
                            "sources": sources,
                            "code_results": code_sources,
                            "is_code_question": is_code_question
                        }
                else:
                    for chunk in self.llm_model.generate_stream(question, context):
                        yield {
                            "answer": chunk,
                            "done": False,
                            "sources": sources,
                            "code_results": code_sources,
                            "is_code_question": is_code_question
                        }
                yield {
                    "answer": "",
                    "done": True,
                    "sources": sources,
                    "code_results": code_sources,
                    "is_code_question": is_code_question
                }
            return stream_response()
        else:
            if enhanced_prompt:
                answer = self.llm_model.generate(enhanced_prompt, context)
            else:
                answer = self.llm_model.generate(question, context)
            return {
                "answer": answer,
                "sources": sources,
                "code_results": code_sources,
                "is_code_question": is_code_question,
                "code_confidence": code_confidence if settings.ENABLE_CODE_DETECTION else 0.0
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
