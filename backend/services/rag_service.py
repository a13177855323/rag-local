from typing import List, Dict, Tuple, Optional, Union, Generator
import os
import re
from backend.config import settings
from backend.models.embedding_model import EmbeddingModel
from backend.models.llm_model import LLMModel
from backend.services.vector_store import VectorStore
from backend.utils.document_processor import DocumentProcessor
from backend.services.code_rag_handler import CodeRAGHandler
from backend.utils.code_detector import CodeDetector

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
        # 代码RAG相关组件
        self.code_rag_handler = CodeRAGHandler()
        self.code_detector = CodeDetector()

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

    def query(self, question: str, top_k: int = None, stream: bool = False, enable_code_rag: bool = True):
        """查询知识库"""
        if top_k is None:
            top_k = settings.TOP_K

        # 检测是否为代码相关问题
        is_code_question = False
        if enable_code_rag:
            is_code_question = self.code_rag_handler.is_code_question(question)

        # 混合检索策略
        search_results = self._hybrid_search(question, top_k, is_code_question)

        # 提取上下文
        context = [doc.get("content", "") for doc, score in search_results]
        sources = [
            {
                "filename": doc.get("metadata", {}).get("filename", ""),
                "content": doc.get("content", "")[:200],  # 只返回前200字符作为预览
                "similarity": float(score),
                "is_code_related": self.code_rag_handler.is_likely_python_code(doc.get("content", ""))
            } for doc, score in search_results
        ]

        if not context:
            response = {
                "answer": "知识库中没有找到相关文档，请先上传文档。",
                "sources": [],
                "is_code_question": is_code_question
            }
            if stream:
                yield response
            else:
                return response
            return

        # 代码问题处理：增强上下文和提示词
        if is_code_question and enable_code_rag:
            enhanced_prompt, code_sources = self.code_rag_handler.build_code_enhanced_prompt(question, search_results)
            
            # 提取代码源信息
            code_blocks = []
            for src in code_sources:
                for code_block in src.get('python_code_blocks', []):
                    code_blocks.append({
                        'filename': src['filename'],
                        'code': code_block
                    })
            
            if stream:
                # 流式输出
                def stream_response():
                    full_response = ""
                    for chunk in self.llm_model.generate_stream(enhanced_prompt, []):
                        full_response += chunk
                        yield {
                            "answer": chunk,
                            "done": False,
                            "sources": sources,
                            "code_blocks": code_blocks,
                            "is_code_question": is_code_question
                        }
                    # 格式化最终回答
                    formatted_answer = self.code_rag_handler.format_code_answer(full_response)
                    yield {
                        "answer": "",
                        "formatted_answer": formatted_answer,
                        "done": True,
                        "sources": sources,
                        "code_blocks": code_blocks,
                        "is_code_question": is_code_question
                    }
                return stream_response()
            else:
                answer = self.llm_model.generate(enhanced_prompt, [])
                formatted_answer = self.code_rag_handler.format_code_answer(answer)
                return {
                    "answer": formatted_answer,
                    "sources": sources,
                    "code_blocks": code_blocks,
                    "is_code_question": is_code_question
                }
        else:
            # 普通问题处理
            if stream:
                # 流式输出
                def stream_response():
                    full_response = ""
                    for chunk in self.llm_model.generate_stream(question, context):
                        full_response += chunk
                        yield {
                            "answer": chunk,
                            "done": False,
                            "sources": sources,
                            "is_code_question": is_code_question
                        }
                    yield {
                        "answer": "",
                        "done": True,
                        "sources": sources,
                        "is_code_question": is_code_question
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

    def _hybrid_search(self, question: str, top_k: int, is_code_question: bool = False) -> List[Tuple[Dict, float]]:
        """
        混合检索策略：结合语义检索和关键词检索

        Args:
            question: 用户问题
            top_k: 返回结果数量
            is_code_question: 是否为代码相关问题

        Returns:
            检索结果列表，每个元素为(文档, 相似度分数)
        """
        # 1. 语义检索
        query_embedding = self.embedding_model.embed_query(question)
        semantic_results = self.vector_store.search(query_embedding, top_k * 2)  # 扩大检索范围

        if not semantic_results:
            return []

        # 2. 关键词检索评分
        if is_code_question:
            # 代码问题：额外增加代码相关文档的权重
            scored_results = []
            for doc, semantic_score in semantic_results:
                content = doc.get('content', '')
                # 计算关键词匹配分数
                keyword_score = self.code_rag_handler.calculate_keyword_score(content, question)
                # 代码问题中代码相关文档获得额外权重
                code_boost = 1.2 if self.code_rag_handler.is_likely_python_code(content) else 1.0
                # 混合评分
                final_score = self.code_rag_handler.hybrid_score(semantic_score, keyword_score) * code_boost
                scored_results.append((doc, final_score))

            # 按最终分数排序
            scored_results.sort(key=lambda x: x[1], reverse=True)
            results = scored_results[:top_k]
        else:
            # 普通问题：使用语义检索结果，但也进行关键词重排序
            scored_results = []
            for doc, semantic_score in semantic_results:
                content = doc.get('content', '')
                keyword_score = self.code_rag_handler.calculate_keyword_score(content, question)
                final_score = self.code_rag_handler.hybrid_score(semantic_score, keyword_score, alpha=0.8)
                scored_results.append((doc, final_score))

            scored_results.sort(key=lambda x: x[1], reverse=True)
            results = scored_results[:top_k]

        return results

    def code_query(self, question: str, top_k: int = None, stream: bool = False):
        """
        代码专属查询接口 - 专门用于代码相关问题

        Args:
            question: 用户问题
            top_k: 返回结果数量
            stream: 是否流式输出

        Returns:
            包含代码信息的查询结果
        """
        if top_k is None:
            top_k = settings.TOP_K

        # 强制启用代码RAG处理
        is_code_question = True

        # 混合检索，增加代码相关文档的检索数量
        search_results = self._hybrid_search(question, top_k * 2, is_code_question)

        # 提取上下文
        context = [doc.get("content", "") for doc, score in search_results]
        sources = [
            {
                "filename": doc.get("metadata", {}).get("filename", ""),
                "content": doc.get("content", "")[:200],
                "similarity": float(score),
                "is_code_related": self.code_rag_handler.is_likely_python_code(doc.get("content", ""))
            } for doc, score in search_results
        ]

        if not context:
            response = {
                "answer": "知识库中没有找到相关的代码文档，请先上传包含代码的文档。",
                "sources": [],
                "is_code_question": is_code_question,
                "code_blocks": []
            }
            if stream:
                yield response
            else:
                return response
            return

        # 使用代码增强提示词
        enhanced_prompt, code_sources = self.code_rag_handler.build_code_enhanced_prompt(question, search_results)

        # 提取所有代码块
        code_blocks = []
        for src in code_sources:
            for code_block in src.get('python_code_blocks', []):
                code_blocks.append({
                    'filename': src['filename'],
                    'code': code_block
                })

        # 从知识库的文本中额外检测代码块
        for doc, _ in search_results:
            content = doc.get('content', '')
            filename = doc.get('metadata', {}).get('filename', 'unknown')
            # 使用code_detector检测代码
            detected_codes = self.code_detector.extract_python_code(content)
            for code_info in detected_codes:
                if code_info['confidence'] >= 0.7:
                    code_blocks.append({
                        'filename': filename,
                        'code': code_info['code'],
                        'confidence': code_info['confidence']
                    })

        # 去重
        unique_codes = []
        seen_codes = set()
        for cb in code_blocks:
            code_key = cb['code'][:100]  # 使用前100字符作为去重键
            if code_key not in seen_codes:
                seen_codes.add(code_key)
                unique_codes.append(cb)
        code_blocks = unique_codes

        if stream:
            # 流式输出
            def stream_response():
                full_response = ""
                for chunk in self.llm_model.generate_stream(enhanced_prompt, []):
                    full_response += chunk
                    yield {
                        "answer": chunk,
                        "done": False,
                        "sources": sources,
                        "code_blocks": code_blocks,
                        "is_code_question": is_code_question
                    }
                formatted_answer = self.code_rag_handler.format_code_answer(full_response)
                yield {
                    "answer": "",
                    "formatted_answer": formatted_answer,
                    "done": True,
                    "sources": sources,
                    "code_blocks": code_blocks,
                    "is_code_question": is_code_question
                }
            return stream_response()
        else:
            answer = self.llm_model.generate(enhanced_prompt, [])
            formatted_answer = self.code_rag_handler.format_code_answer(answer)
            return {
                "answer": formatted_answer,
                "sources": sources,
                "code_blocks": code_blocks,
                "is_code_question": is_code_question
            }

    def detect_code_in_document(self, content: str) -> Dict:
        """
        检测文档内容中的代码并返回分析结果

        Args:
            content: 文档内容

        Returns:
            代码检测结果
        """
        is_python_code = self.code_rag_handler.is_likely_python_code(content)
        code_blocks = self.code_detector.extract_python_code(content)

        return {
            "has_python_code": is_python_code,
            "code_blocks": code_blocks,
            "code_count": len(code_blocks)
        }
