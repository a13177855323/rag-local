import os
from typing import List, Dict
from PyPDF2 import PdfReader
from docx import Document
import markdown
from langchain.text_splitter import RecursiveCharacterTextSplitter
from backend.config import settings

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        )

    def read_pdf(self, file_path: str) -> str:
        """读取PDF文件"""
        text = ""
        try:
            reader = PdfReader(file_path)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        except Exception as e:
            raise Exception(f"读取PDF文件失败: {str(e)}")
        return text

    def read_docx(self, file_path: str) -> str:
        """读取Word文档"""
        try:
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            raise Exception(f"读取Word文件失败: {str(e)}")
        return text

    def read_markdown(self, file_path: str) -> str:
        """读取Markdown文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
            text = markdown.markdown(md_content)
        except Exception as e:
            raise Exception(f"读取Markdown文件失败: {str(e)}")
        return text

    def read_text(self, file_path: str) -> str:
        """读取纯文本文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            raise Exception(f"读取文本文件失败: {str(e)}")
        return text

    def process_file(self, file_path: str) -> List[Dict]:
        """处理单个文件，返回分块后的文档"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        file_ext = os.path.splitext(file_path)[1].lower()
        filename = os.path.basename(file_path)

        # 根据文件类型选择读取方法
        if file_ext == '.pdf':
            text = self.read_pdf(file_path)
        elif file_ext == '.docx':
            text = self.read_docx(file_path)
        elif file_ext == '.md':
            text = self.read_markdown(file_path)
        elif file_ext in ['.txt', '.text']:
            text = self.read_text(file_path)
        else:
            raise ValueError(f"不支持的文件类型: {file_ext}")

        # 分块处理
        chunks = self.text_splitter.split_text(text)
        documents = []
        for i, chunk in enumerate(chunks):
            documents.append({
                "id": f"{filename}_{i}",
                "content": chunk,
                "metadata": {
                    "filename": filename,
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "file_type": file_ext
                }
            })

        return documents

    def process_files(self, file_paths: List[str]) -> List[Dict]:
        """批量处理文件"""
        all_documents = []
        for file_path in file_paths:
            try:
                documents = self.process_file(file_path)
                all_documents.extend(documents)
            except Exception as e:
                print(f"处理文件 {file_path} 失败: {str(e)}")
                continue
        return all_documents
