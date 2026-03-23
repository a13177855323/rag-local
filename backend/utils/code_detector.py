import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CodeBlock:
    language: str
    code: str
    start_pos: int
    end_pos: int
    context_before: str = ""
    context_after: str = ""


class CodeDetector:
    CODE_KEYWORDS = [
        "代码", "函数", "方法", "类", "变量", "循环", "条件",
        "实现", "写一个", "编写", "如何实现", "怎么写",
        "python", "代码块", "示例代码", "代码示例",
        "function", "class", "method", "variable",
        "import", "def ", "return", "if ", "for ", "while ",
        "报错", "错误", "异常", "debug", "调试",
        "api", "接口", "模块", "库", "框架"
    ]
    
    CODE_QUESTION_PATTERNS = [
        r"如何(实现|编写|写).{0,20}(代码|函数|类|方法)",
        r"(写|编写|实现).{0,20}(代码|函数|类|方法)",
        r"(代码|函数|方法).{0,20}(怎么|如何)",
        r"(有|提供|给).{0,10}(代码|示例)",
        r"(python|Python).{0,30}(实现|写|代码)",
        r"(解决|修复|处理).{0,20}(错误|异常|报错)",
        r"(调用|使用).{0,20}(api|接口|函数|方法)",
    ]
    
    PYTHON_CODE_PATTERNS = [
        (r'def\s+\w+\s*\([^)]*\)\s*:', 'function'),
        (r'class\s+\w+(\([^)]*\))?\s*:', 'class'),
        (r'import\s+\w+', 'import'),
        (r'from\s+\w+\s+import', 'import'),
        (r'if\s+__name__\s*==\s*[\'"]__main__[\'"]', 'main'),
        (r'@\w+\s*\n\s*def', 'decorator'),
        (r'for\s+\w+\s+in\s+', 'loop'),
        (r'while\s+.+:', 'loop'),
        (r'try\s*:', 'exception'),
        (r'with\s+.+\s+as\s+', 'context'),
    ]

    def __init__(self):
        self.code_block_pattern = re.compile(
            r'```(\w*)\n(.*?)```',
            re.DOTALL
        )
        self.inline_code_pattern = re.compile(r'`([^`]+)`')

    def is_code_related_question(self, question: str) -> Tuple[bool, float]:
        question_lower = question.lower()
        confidence = 0.0
        
        keyword_matches = sum(1 for kw in self.CODE_KEYWORDS if kw.lower() in question_lower)
        confidence += min(keyword_matches * 0.15, 0.45)
        
        for pattern in self.CODE_QUESTION_PATTERNS:
            if re.search(pattern, question, re.IGNORECASE):
                confidence += 0.25
                break
        
        code_keywords = ["def ", "class ", "import ", "return ", "for ", "if ", "while "]
        for kw in code_keywords:
            if kw in question:
                confidence += 0.1
        
        if re.search(r'```(\w*)', question):
            confidence += 0.3
        
        confidence = min(confidence, 1.0)
        return confidence >= 0.3, confidence

    def extract_code_blocks(self, text: str) -> List[CodeBlock]:
        blocks = []
        
        for match in self.code_block_pattern.finditer(text):
            language = match.group(1) or "unknown"
            code = match.group(2)
            start_pos = match.start()
            end_pos = match.end()
            
            context_before = text[max(0, start_pos - 200):start_pos].strip()
            context_after = text[end_pos:min(len(text), end_pos + 200)].strip()
            
            blocks.append(CodeBlock(
                language=language.lower(),
                code=code,
                start_pos=start_pos,
                end_pos=end_pos,
                context_before=context_before,
                context_after=context_after
            ))
        
        return blocks

    def extract_python_blocks(self, text: str) -> List[CodeBlock]:
        all_blocks = self.extract_code_blocks(text)
        python_blocks = []
        
        for block in all_blocks:
            if block.language in ['python', 'py', '']:
                if self._is_python_code(block.code):
                    python_blocks.append(block)
            elif block.language == 'python':
                python_blocks.append(block)
        
        return python_blocks

    def _is_python_code(self, code: str) -> bool:
        if not code.strip():
            return False
        
        matches = 0
        for pattern, _ in self.PYTHON_CODE_PATTERNS:
            if re.search(pattern, code):
                matches += 1
        
        python_indicators = [
            'def ', 'class ', 'import ', 'from ', 'return ',
            'self.', 'print(', 'if __name__', 'raise ',
            'try:', 'except:', 'with ', 'for ', 'while '
        ]
        
        indicator_count = sum(1 for ind in python_indicators if ind in code)
        
        return matches >= 1 or indicator_count >= 2

    def detect_code_in_chunks(self, chunks: List[Dict]) -> List[Dict]:
        enriched_chunks = []
        
        for chunk in chunks:
            content = chunk.get("content", "")
            code_blocks = self.extract_python_blocks(content)
            
            has_code = len(code_blocks) > 0
            code_count = len(code_blocks)
            
            code_content = "\n\n".join([b.code for b in code_blocks])
            
            enriched_chunk = chunk.copy()
            enriched_chunk["metadata"] = chunk.get("metadata", {}).copy()
            enriched_chunk["metadata"]["has_code"] = has_code
            enriched_chunk["metadata"]["code_count"] = code_count
            enriched_chunk["metadata"]["code_languages"] = list(set(b.language for b in code_blocks))
            enriched_chunk["code_content"] = code_content if has_code else ""
            
            enriched_chunks.append(enriched_chunk)
        
        return enriched_chunks

    def get_code_keywords_from_question(self, question: str) -> List[str]:
        keywords = []
        
        code_blocks = self.extract_code_blocks(question)
        for block in code_blocks:
            code = block.code
            func_names = re.findall(r'def\s+(\w+)', code)
            class_names = re.findall(r'class\s+(\w+)', code)
            imports = re.findall(r'(?:from\s+(\w+)|import\s+(\w+))', code)
            
            keywords.extend(func_names)
            keywords.extend(class_names)
            for imp in imports:
                keywords.extend([i for i in imp if i])
        
        important_words = re.findall(r'\b[A-Z][a-z]+\b|\b[a-z_]{3,}\b', question)
        code_related = ['function', 'class', 'method', 'variable', 'loop', 'array', 
                       'list', 'dict', 'string', 'file', 'api', 'http', 'request']
        keywords.extend([w.lower() for w in important_words if w.lower() in code_related])
        
        return list(set(keywords))


code_detector = CodeDetector()


def detect_code_question(question: str) -> Tuple[bool, float]:
    return code_detector.is_code_related_question(question)


def extract_code_from_text(text: str) -> List[CodeBlock]:
    return code_detector.extract_python_blocks(text)


def enrich_chunks_with_code(chunks: List[Dict]) -> List[Dict]:
    return code_detector.detect_code_in_chunks(chunks)
