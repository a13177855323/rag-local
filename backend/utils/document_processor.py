import os
import re
from typing import List, Dict, Tuple
from PyPDF2 import PdfReader
from docx import Document
import markdown
from langchain.text_splitter import RecursiveCharacterTextSplitter
from backend.config import settings


class CodeAwareTextSplitter(RecursiveCharacterTextSplitter):
    """支持代码块感知的文本分割器"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.code_block_pattern = re.compile(r'```(\w*)\n(.*?)```', re.DOTALL)

    def _split_text_with_code(self, text: str) -> List[Tuple[str, bool, str]]:
        """分割文本，保留代码块完整性，返回 (内容, 是否代码, 语言)"""
        parts = []
        last_end = 0

        for match in self.code_block_pattern.finditer(text):
            if match.start() > last_end:
                text_part = text[last_end:match.start()]
                if text_part.strip():
                    parts.append((text_part, False, ""))

            language = match.group(1) or "unknown"
            code = match.group(2)
            if code.strip():
                parts.append((code, True, language))

            last_end = match.end()

        if last_end < len(text):
            remaining = text[last_end:]
            if remaining.strip():
                parts.append((remaining, False, ""))

        return parts

    def split_text(self, text: str) -> List[str]:
        parts = self._split_text_with_code(text)
        chunks = []

        for content, is_code, language in parts:
            if is_code:
                wrapped = f"```{language}\n{content}\n```"
                if len(wrapped) <= self.chunk_size:
                    chunks.append(wrapped)
                else:
                    code_chunks = self._split_long_code(content, language)
                    chunks.extend(code_chunks)
            else:
                text_chunks = super().split_text(content)
                chunks.extend(text_chunks)

        return chunks

    def _split_long_code(self, code: str, language: str) -> List[str]:
        """分割长代码块，按函数/类边界分割"""
        lines = code.split('\n')
        chunks = []
        current_chunk = []
        current_length = 0

        for line in lines:
            if re.match(r'^(def |class |@|if __name__|$)', line) and current_chunk:
                if current_length > self.chunk_size * 0.5:
                    chunk_text = f"```{language}\n" + '\n'.join(current_chunk) + "\n```"
                    chunks.append(chunk_text)
                    current_chunk = []
                    current_length = 0

            current_chunk.append(line)
            current_length += len(line) + 1

            if current_length >= self.chunk_size:
                chunk_text = f"```{language}\n" + '\n'.join(current_chunk) + "\n```"
                chunks.append(chunk_text)
                current_chunk = []
                current_length = 0

        if current_chunk:
            chunk_text = f"```{language}\n" + '\n'.join(current_chunk) + "\n```"
            chunks.append(chunk_text)

        return chunks


class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        )
        self.code_splitter = CodeAwareTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
        )
        self.code_block_pattern = re.compile(r'```(\w*)\n(.*?)```', re.DOTALL)

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

    def _extract_code_info(self, content: str) -> Dict:
        """提取代码块信息"""
        code_blocks = self.code_block_pattern.findall(content)
        has_code = len(code_blocks) > 0
        languages = [lang or "unknown" for lang, _ in code_blocks]
        code_content = "\n\n".join([code for _, code in code_blocks])
        
        return {
            "has_code": has_code,
            "code_languages": languages,
            "code_content": code_content,
            "code_count": len(code_blocks)
        }

    def _is_code_chunk(self, content: str) -> bool:
        """判断是否为代码块"""
        return bool(self.code_block_pattern.search(content))

    def process_file(self, file_path: str) -> List[Dict]:
        """处理单个文件，返回分块后的文档"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        file_ext = os.path.splitext(file_path)[1].lower()
        filename = os.path.basename(file_path)

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

        has_code_in_doc = bool(self.code_block_pattern.search(text))
        if has_code_in_doc:
            chunks = self.code_splitter.split_text(text)
        else:
            chunks = self.text_splitter.split_text(text)

        documents = []
        for i, chunk in enumerate(chunks):
            code_info = self._extract_code_info(chunk)
            is_code = self._is_code_chunk(chunk)
            
            documents.append({
                "id": f"{filename}_{i}",
                "content": chunk,
                "metadata": {
                    "filename": filename,
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "file_type": file_ext,
                    "has_code": code_info["has_code"],
                    "code_languages": code_info["code_languages"],
                    "code_count": code_info["code_count"],
                    "is_code_chunk": is_code
                },
                "code_content": code_info["code_content"] if code_info["has_code"] else ""
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
