"""
代码块检测与提取模块
支持Python代码块识别、代码类型判断、混合检索策略
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class CodeType(Enum):
    """代码类型枚举"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    JAVA = "java"
    CPP = "cpp"
    SQL = "sql"
    BASH = "bash"
    UNKNOWN = "unknown"


@dataclass
class CodeBlock:
    """代码块数据结构"""
    content: str
    language: str
    start_line: int
    end_line: int
    context_before: str
    context_after: str


class CodeDetector:
    """代码检测器 - 单例模式"""
    _instance = None

    # 代码相关关键词（用于判断是否为代码相关问题）
    CODE_KEYWORDS = [
        "代码", "code", "函数", "function", "类", "class", "方法", "method",
        "python", "py", "实现", "implement", "算法", "algorithm",
        "报错", "error", "bug", "修复", "fix", "调试", "debug",
        "import", "def ", "class ", "return", "print(", "for ", "while ",
        "语法", "syntax", "怎么写", "how to", "示例", "example"
    ]

    # 语言检测模式
    LANGUAGE_PATTERNS = {
        CodeType.PYTHON: [
            r'def\s+\w+\s*\(',
            r'class\s+\w+[\(:\s]',
            r'import\s+\w+',
            r'from\s+\w+\s+import',
            r'if\s+__name__\s*==\s*["\']__main__["\']',
            r'@\w+\s*\n\s*def',
            r'\s*#.*\n',
            r'print\s*\(',
            r'lambda\s+.*:',
        ],
        CodeType.JAVASCRIPT: [
            r'const\s+\w+\s*=',
            r'let\s+\w+\s*=',
            r'var\s+\w+\s*=',
            r'function\s+\w+\s*\(',
            r'=>\s*\{',
            r'require\s*\(',
            r'module\.exports',
        ],
        CodeType.JAVA: [
            r'public\s+(class|static|void|int|String)',
            r'private\s+\w+',
            r'System\.out\.println',
            r'import\s+java\.',
        ],
        CodeType.SQL: [
            r'SELECT\s+.*\s+FROM',
            r'INSERT\s+INTO',
            r'UPDATE\s+\w+\s+SET',
            r'DELETE\s+FROM',
            r'CREATE\s+TABLE',
        ],
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def is_code_query(self, question: str) -> bool:
        """
        判断用户提问是否与代码相关

        Args:
            question: 用户问题

        Returns:
            是否为代码相关问题
        """
        question_lower = question.lower()
        return any(keyword.lower() in question_lower for keyword in self.CODE_KEYWORDS)

    def detect_language(self, code: str) -> CodeType:
        """
        检测代码语言类型

        Args:
            code: 代码内容

        Returns:
            代码类型枚举
        """
        scores = {}
        for lang, patterns in self.LANGUAGE_PATTERNS.items():
            score = sum(1 for pattern in patterns if re.search(pattern, code, re.IGNORECASE))
            if score > 0:
                scores[lang] = score

        if not scores:
            return CodeType.UNKNOWN

        return max(scores, key=scores.get)

    def extract_code_blocks(self, text: str) -> List[CodeBlock]:
        """
        从文本中提取代码块

        Args:
            text: 原始文本

        Returns:
            代码块列表
        """
        code_blocks = []
        lines = text.split('\n')

        # 匹配Markdown代码块
        md_pattern = r'```(\w*)\n(.*?)```'
        for match in re.finditer(md_pattern, text, re.DOTALL):
            lang = match.group(1).lower() or "text"
            content = match.group(2).strip()

            # 计算行号
            start_pos = match.start()
            start_line = text[:start_pos].count('\n') + 1
            end_line = start_line + content.count('\n')

            # 获取上下文
            context_before = self._get_context(lines, start_line - 1, 3, before=True)
            context_after = self._get_context(lines, end_line, 3, before=False)

            code_blocks.append(CodeBlock(
                content=content,
                language=lang,
                start_line=start_line,
                end_line=end_line,
                context_before=context_before,
                context_after=context_after
            ))

        # 匹配缩进代码块（4个空格或Tab）
        indent_blocks = self._extract_indent_blocks(text)
        code_blocks.extend(indent_blocks)

        return code_blocks

    def _extract_indent_blocks(self, text: str) -> List[CodeBlock]:
        """提取缩进代码块"""
        code_blocks = []
        lines = text.split('\n')
        i = 0

        while i < len(lines):
            line = lines[i]

            # 检测是否是代码行（缩进4个空格或Tab开头）
            if line.startswith('    ') or line.startswith('\t'):
                # 找到代码块的开始
                start_line = i + 1
                code_lines = []

                while i < len(lines) and (lines[i].startswith('    ') or
                                           lines[i].startswith('\t') or
                                           lines[i].strip() == ''):
                    code_lines.append(lines[i])
                    i += 1

                # 过滤掉空行后检查是否有效
                non_empty = [l for l in code_lines if l.strip()]
                if len(non_empty) >= 2:  # 至少2行非空代码
                    content = '\n'.join(code_lines).strip()
                    end_line = start_line + len(code_lines) - 1

                    # 检测语言
                    detected_lang = self.detect_language(content)
                    lang_str = detected_lang.value if detected_lang != CodeType.UNKNOWN else "text"

                    context_before = self._get_context(lines, start_line - 1, 2, before=True)
                    context_after = self._get_context(lines, end_line, 2, before=False)

                    code_blocks.append(CodeBlock(
                        content=content,
                        language=lang_str,
                        start_line=start_line,
                        end_line=end_line,
                        context_before=context_before,
                        context_after=context_after
                    ))
            else:
                i += 1

        return code_blocks

    def _get_context(self, lines: List[str], line_num: int, count: int, before: bool) -> str:
        """获取上下文文本"""
        if before:
            start = max(0, line_num - count)
            context_lines = lines[start:line_num]
        else:
            end = min(len(lines), line_num + count)
            context_lines = lines[line_num:end]

        return '\n'.join(context_lines).strip()

    def enhance_code_query(self, question: str) -> str:
        """
        增强代码相关查询，添加语义提示

        Args:
            question: 原始问题

        Returns:
            增强后的查询
        """
        enhancements = []

        # 检测Python相关
        if any(kw in question.lower() for kw in ["python", "py ", "pytorch", "numpy", "pandas"]):
            enhancements.append("Python code implementation function class method")

        # 检测函数/方法相关
        if any(kw in question.lower() for kw in ["函数", "function", "方法", "method", "def "]):
            enhancements.append("def function definition parameters return")

        # 检测类相关
        if any(kw in question.lower() for kw in ["类", "class", "对象", "object"]):
            enhancements.append("class definition inheritance init method")

        # 检测错误相关
        if any(kw in question.lower() for kw in ["报错", "error", "exception", "bug", "fix"]):
            enhancements.append("error handling exception try except debug")

        if enhancements:
            return f"{question} {' '.join(enhancements)}"
        return question

    def format_code_for_prompt(self, code_blocks: List[CodeBlock]) -> str:
        """
        格式化代码块用于LLM提示

        Args:
            code_blocks: 代码块列表

        Returns:
            格式化后的代码文本
        """
        formatted = []
        for i, block in enumerate(code_blocks, 1):
            formatted.append(f"【代码片段 {i}】")
            formatted.append(f"语言: {block.language}")
            formatted.append(f"位置: 第{block.start_line}-{block.end_line}行")

            if block.context_before:
                formatted.append(f"上下文(前): {block.context_before}")

            formatted.append(f"```\n{block.content}\n```")

            if block.context_after:
                formatted.append(f"上下文(后): {block.context_after}")

            formatted.append("")

        return '\n'.join(formatted)

    def score_code_relevance(self, code_block: CodeBlock, question: str) -> float:
        """
        计算代码块与问题的相关性分数

        Args:
            code_block: 代码块
            question: 问题

        Returns:
            相关性分数 (0-1)
        """
        score = 0.0
        question_lower = question.lower()
        code_lower = code_block.content.lower()

        # 1. 关键词匹配
        keywords = re.findall(r'\b\w+\b', question_lower)
        matched_keywords = sum(1 for kw in keywords if len(kw) > 2 and kw in code_lower)
        score += min(matched_keywords * 0.1, 0.3)

        # 2. 函数名匹配
        func_names = re.findall(r'def\s+(\w+)', code_block.content)
        for func_name in func_names:
            if func_name.lower() in question_lower:
                score += 0.2

        # 3. 类名匹配
        class_names = re.findall(r'class\s+(\w+)', code_block.content)
        for class_name in class_names:
            if class_name.lower() in question_lower:
                score += 0.2

        # 4. 语言匹配
        if "python" in question_lower and code_block.language == "python":
            score += 0.15

        # 5. 代码长度惩罚（太短的代码相关性低）
        lines_count = len(code_block.content.split('\n'))
        if lines_count < 3:
            score *= 0.5
        elif lines_count > 20:
            score *= 0.9  # 太长的代码稍微降权

        return min(score, 1.0)


# 全局单例实例
_code_detector = None


def get_code_detector() -> CodeDetector:
    """获取 CodeDetector 单例实例"""
    global _code_detector
    if _code_detector is None:
        _code_detector = CodeDetector()
    return _code_detector


def is_code_query(question: str) -> bool:
    """便捷函数：判断是否为代码相关问题"""
    detector = get_code_detector()
    return detector.is_code_query(question)


def extract_code_blocks(text: str) -> List[CodeBlock]:
    """便捷函数：提取代码块"""
    detector = get_code_detector()
    return detector.extract_code_blocks(text)
