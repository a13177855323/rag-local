import re
from typing import List, Dict, Tuple, Optional
from backend.config import settings


class CodeRAGHandler:
    """代码块智能检索与问答处理器"""
    _instance = None
    _initialized = False

    # Python代码相关关键词
    PYTHON_KEYWORDS = {
        'def', 'class', 'import', 'from', 'print', 'return', 'if', 'else',
        'elif', 'for', 'while', 'try', 'except', 'finally', 'with', 'as',
        'lambda', 'yield', 'raise', 'assert', 'break', 'continue', 'pass',
        'global', 'nonlocal', 'async', 'await', 'True', 'False', 'None',
        'and', 'or', 'not', 'in', 'is', 'self', '__init__', '__name__',
        'list', 'dict', 'set', 'tuple', 'str', 'int', 'float', 'bool',
        'len', 'range', 'enumerate', 'zip', 'map', 'filter', 'sorted',
        'reversed', 'sum', 'max', 'min', 'abs', 'round', 'open', 'type',
        'isinstance', 'issubclass', 'hasattr', 'getattr', 'setattr',
        'function', 'method', 'module', 'package', 'pip', 'conda',
        'python', 'py', '.py', 'def ', 'class ', 'import ',
        'pandas', 'pd', 'numpy', 'np', 'torch', 'tf', 'tensorflow',
        'sklearn', 'scikit-learn', 'matplotlib', 'plt', 'seaborn',
        'def ', 'lambda ', 'return ', 'async def ', 'await ',
        '@staticmethod', '@classmethod', '@property'
    }

    # 代码问题指示词
    CODE_QUESTION_INDICATORS = {
        '代码', '编程', '程序', '函数', '方法', '类', '接口', 'API',
        '怎么写', '如何写', '写法', '实现', '示例', '例子', 'demo',
        '错误', 'bug', '异常', 'exception', 'error', '报错',
        '运行', '执行', '调用', '返回', '参数', '变量', '循环',
        '条件', '判断', '语法', '格式', '规范', '最佳实践',
        'code', 'python', 'java', 'javascript', 'js', 'c++',
        'function', 'class', 'method', 'how to', 'example of',
        '错误码', 'traceback', 'stacktrace', '堆栈', '调试',
        'debug', '测试', 'test', 'unittest', 'pytest',
        '性能', '优化', 'optimize', '内存泄漏', 'cpu', '内存'
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            # 预编译正则表达式
            self._compile_patterns()
            self._initialized = True

    def _compile_patterns(self):
        """预编译正则表达式模式"""
        # Python代码块模式匹配 (```python ... ```)
        self.python_code_block_pattern = re.compile(
            r'```(?:python|py)\s*\n(.*?)\n```',
            re.DOTALL | re.IGNORECASE
        )

        # 通用代码块模式匹配 (``` ... ```)
        self.generic_code_block_pattern = re.compile(
            r'```(?:\w+)?\s*\n(.*?)\n```',
            re.DOTALL
        )

        # 行内代码模式匹配 (`...`)
        self.inline_code_pattern = re.compile(r'`([^`]+)`')

        # Python函数定义模式
        self.python_function_pattern = re.compile(
            r'\bdef\s+(\w+)\s*\([^)]*\)\s*:|^def\s+(\w+)\s*\([^)]*\)\s*:',
            re.MULTILINE
        )

        # Python类定义模式
        self.python_class_pattern = re.compile(
            r'\bclass\s+(\w+)(?:\([^)]*\))?\s*:|^class\s+(\w+)(?:\([^)]*\))?\s*:',
            re.MULTILINE
        )

        # Python导入语句模式
        self.import_pattern = re.compile(
            r'^(?:from|import)\s+[\w.]+(?:\s+import\s+[\w, ]+)?',
            re.MULTILINE
        )

        # Python变量赋值模式 (简单匹配)
        self.assignment_pattern = re.compile(
            r'^\s*[\w_]+\s*=|\b[\w_]+\s*=',
            re.MULTILINE
        )

    def is_code_question(self, question: str) -> bool:
        """
        判断用户问题是否与代码相关

        Args:
            question: 用户问题

        Returns:
            是否为代码相关问题
        """
        question_lower = question.lower()

        # 检查是否包含代码问题指示词
        for indicator in self.CODE_QUESTION_INDICATORS:
            if indicator.lower() in question_lower:
                return True

        # 检查是否包含Python关键词
        keyword_count = 0
        for keyword in self.PYTHON_KEYWORDS:
            if keyword.lower() in question_lower:
                keyword_count += 1
                if keyword_count >= 2:  # 至少匹配2个关键词才判定为代码问题
                    return True

        # 检查是否包含行内代码
        if self.inline_code_pattern.search(question):
            return True

        return False

    def extract_python_code_blocks(self, text: str) -> List[str]:
        """
        从文本中提取Python代码块

        Args:
            text: 输入文本

        Returns:
            Python代码块列表
        """
        code_blocks = []

        # 首先尝试匹配带python标记的代码块
        python_blocks = self.python_code_block_pattern.findall(text)
        code_blocks.extend([block.strip() for block in python_blocks if block.strip()])

        # 如果没有找到python标记的代码块，尝试匹配通用代码块并验证是否为Python代码
        if not code_blocks:
            generic_blocks = self.generic_code_block_pattern.findall(text)
            for block in generic_blocks:
                block = block.strip()
                if block and self.is_likely_python_code(block):
                    code_blocks.append(block)

        return code_blocks

    def extract_all_code_blocks(self, text: str) -> List[Dict[str, str]]:
        """
        提取所有代码块并尝试识别语言

        Args:
            text: 输入文本

        Returns:
            代码块信息字典列表，包含code和language字段
        """
        code_blocks = []

        # 提取带语言标记的代码块
        pattern = re.compile(r'```(\w*)\s*\n(.*?)\n```', re.DOTALL)
        matches = pattern.finditer(text)

        for match in matches:
            language = match.group(1).lower() or 'unknown'
            code = match.group(2).strip()
            if code:
                # 如果语言未指定，尝试检测
                if language == 'unknown' and self.is_likely_python_code(code):
                    language = 'python'
                code_blocks.append({
                    'code': code,
                    'language': language
                })

        return code_blocks

    def extract_inline_code(self, text: str) -> List[str]:
        """
        提取行内代码片段

        Args:
            text: 输入文本

        Returns:
            行内代码列表
        """
        return [code.strip() for code in self.inline_code_pattern.findall(text) if code.strip()]

    def is_likely_python_code(self, text: str) -> bool:
        """
        判断一段文本是否可能是Python代码

        Args:
            text: 输入文本

        Returns:
            是否可能为Python代码
        """
        if not text or len(text.strip()) < 5:
            return False

        score = 0
        text_lower = text.lower()
        lines = text.strip().split('\n')
        first_line = lines[0].strip() if lines else ''

        # 检查注释和shebang
        if first_line.startswith('#!/') or 'python' in first_line:
            score += 3
        if first_line.startswith('#') or first_line.startswith('"""') or first_line.startswith("'''"):
            score += 1

        # 检查Python关键字密度
        keyword_matches = 0
        for keyword in self.PYTHON_KEYWORDS:
            if keyword in text or keyword.lower() in text_lower:
                keyword_matches += 1

        # 根据文本长度调整分数
        score += min(keyword_matches * 0.5, 5)

        # 检查函数定义
        if self.python_function_pattern.search(text):
            score += 3

        # 检查类定义
        if self.python_class_pattern.search(text):
            score += 3

        # 检查导入语句
        if self.import_pattern.search(text):
            score += 2

        # 检查赋值语句
        if self.assignment_pattern.search(text):
            score += 1

        # 检查Python语法特征
        if ':\n' in text or ':\r\n' in text:  # 代码块开始
            score += 2
        if text.count('(') > 0 and text.count(')') > 0:  # 函数调用
            score += 1
        if '__init__' in text or '__name__' in text or 'self.' in text:  # Python特殊语法
            score += 3
        if text.count('[') > 0 and text.count(']') > 0:  # 列表/索引
            score += 0.5
        if text.count('{') > 0 and text.count('}') > 0:  # 字典/格式化
            score += 0.5

        # 缩进模式（4空格或1制表符）
        indent_count = 0
        for line in lines:
            if line.startswith('    ') or line.startswith('\t'):
                indent_count += 1
        if indent_count >= 2:
            score += 2

        # 检查常见Python库
        common_libs = ['import pandas', 'import numpy', 'import torch', 'import tensorflow',
                       'import sklearn', 'import matplotlib', 'import requests', 'from flask',
                       'import django', 'import fastapi', 'import pygame']
        for lib in common_libs:
            if lib in text or lib.replace('import ', 'import ') in text_lower:
                score += 2
                break

        # 判断非代码特征
        non_code_indicators = ['. ', '。', '？', '！', '，', '；', '：']  # 中文标点
        for indicator in non_code_indicators:
            if text.count(indicator) > 3:  # 自然语言特征
                score -= 2

        # 最终判断阈值
        return score >= 4

    def find_code_in_document(self, document: Dict) -> Optional[Dict]:
        """
        从文档中查找代码信息

        Args:
            document: 文档元数据

        Returns:
            包含代码信息的增强文档，或None
        """
        content = document.get('content', '')
        if not content:
            return None

        # 提取代码块
        python_code_blocks = self.extract_python_code_blocks(content)
        all_code_blocks = self.extract_all_code_blocks(content)
        inline_code = self.extract_inline_code(content)

        # 判断文档是否包含Python代码
        has_python_code = len(python_code_blocks) > 0 or self.is_likely_python_code(content)

        # 增强文档元数据
        enhanced_doc = document.copy()
        enhanced_doc['metadata'] = enhanced_doc.get('metadata', {}).copy()
        enhanced_doc['metadata'].update({
            'has_python_code': has_python_code,
            'python_code_blocks': python_code_blocks,
            'all_code_blocks': all_code_blocks,
            'inline_code': inline_code,
            'is_code_related': has_python_code or len(all_code_blocks) > 0
        })

        return enhanced_doc

    def build_code_enhanced_prompt(self, question: str, context_docs: List[Tuple[Dict, float]]) -> Tuple[str, List[Dict]]:
        """
        构建代码增强的提示词

        Args:
            question: 用户问题
            context_docs: 检索到的上下文文档列表 (文档, 相似度)

        Returns:
            增强后的提示词, 代码源信息列表
        """
        code_sources = []
        code_contexts = []
        text_contexts = []

        for doc, score in context_docs:
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})
            filename = metadata.get('filename', 'unknown')

            # 增强文档代码信息
            enhanced = self.find_code_in_document(doc)
            if enhanced and enhanced['metadata'].get('is_code_related'):
                code_info = {
                    'filename': filename,
                    'similarity': float(score),
                    'python_code_blocks': enhanced['metadata'].get('python_code_blocks', []),
                    'all_code_blocks': enhanced['metadata'].get('all_code_blocks', []),
                    'inline_code': enhanced['metadata'].get('inline_code', [])
                }
                code_sources.append(code_info)

                # 优先添加代码内容
                if enhanced['metadata'].get('python_code_blocks'):
                    for code_block in enhanced['metadata']['python_code_blocks']:
                        code_contexts.append(f"[来自 {filename} 的Python代码]\n```python\n{code_block}\n```")

                # 添加相关上下文文本
                text_contexts.append(f"[来自 {filename} 的相关文本]\n{content[:800]}")
            else:
                # 非代码文档也添加为上下文
                text_contexts.append(f"[来自 {filename} 的相关内容]\n{content[:600]}")

        # 构建提示词
        prompt = self._build_final_prompt(question, code_contexts, text_contexts)

        return prompt, code_sources

    def _build_final_prompt(self, question: str, code_contexts: List[str], text_contexts: List[str]) -> str:
        """
        构建最终的提示词

        Args:
            question: 用户问题
            code_contexts: 代码上下文列表
            text_contexts: 文本上下文列表

        Returns:
            最终提示词
        """
        prompt_parts = []

        # 系统提示
        prompt_parts.append("""你是一个专业的编程助手。请基于以下上下文信息回答用户的问题。
注意事项：
1. 如果提供了代码示例，请基于代码示例进行回答
2. 回答要准确、简洁，代码要规范可读
3. 如果是代码实现问题，优先给出可运行的Python代码
4. 如果问题无法从上下文中得到答案，请说明并给出通用建议
5. 代码输出请使用标准的Markdown代码块格式（```python ... ```）
""")

        # 添加代码上下文（优先）
        if code_contexts:
            prompt_parts.append("\n=== 相关代码示例 ===")
            prompt_parts.extend(code_contexts)

        # 添加文本上下文
        if text_contexts:
            prompt_parts.append("\n=== 相关参考资料 ===")
            prompt_parts.extend(text_contexts)

        # 添加用户问题
        prompt_parts.append(f"\n=== 用户问题 ===")
        prompt_parts.append(f"问题：{question}")
        prompt_parts.append("\n请给出准确、详细的回答：")

        return '\n'.join(prompt_parts)

    def format_code_answer(self, answer: str) -> str:
        """
        格式化代码回答，确保代码块统一输出

        Args:
            answer: 原始回答

        Returns:
            格式化后的回答
        """
        # 确保代码块使用统一的格式
        # 替换 ```py 为 ```python
        answer = answer.replace('```py\n', '```python\n')
        answer = answer.replace('```Py\n', '```python\n')
        answer = answer.replace('```PY\n', '```python\n')

        # 为没有标记的Python代码块添加标记
        lines = answer.split('\n')
        in_code_block = False
        formatted_lines = []

        for line in lines:
            if line.strip() == '```':
                if not in_code_block:
                    # 开始代码块，检查下一行是否可能是Python代码
                    formatted_lines.append('```python')
                else:
                    formatted_lines.append('```')
                in_code_block = not in_code_block
            else:
                formatted_lines.append(line)

        return '\n'.join(formatted_lines)

    def hybrid_score(self, semantic_score: float, keyword_score: float, alpha: float = 0.7) -> float:
        """
        混合评分：语义分数 * alpha + 关键词分数 * (1 - alpha)

        Args:
            semantic_score: 语义相似度分数 (0-1)
            keyword_score: 关键词匹配分数 (0-1)
            alpha: 语义分数权重

        Returns:
            混合分数
        """
        return semantic_score * alpha + keyword_score * (1 - alpha)

    def calculate_keyword_score(self, text: str, question: str) -> float:
        """
        计算文本与问题的关键词匹配分数

        Args:
            text: 待评估文本
            question: 用户问题

        Returns:
            关键词匹配分数 (0-1)
        """
        # 提取问题中的关键词（1-4字符的词）
        question_words = set(re.findall(r'\b\w{1,4}\b', question.lower()))
        text_words = set(re.findall(r'\b\w{1,4}\b', text.lower()))

        # 计算交集
        common_words = question_words.intersection(text_words)

        # 提取代码相关关键词
        code_keywords = {
            'def', 'class', 'import', 'function', 'method', 'code', 'python',
            '错误', '异常', '报错', 'error', 'exception', 'bug',
            '实现', '怎么写', '如何', 'example', 'demo', '示例'
        }
        code_matches = sum(1 for kw in code_keywords if kw in text.lower() or kw in question.lower())

        # 综合分数
        if len(question_words) == 0:
            base_score = 0.0
        else:
            base_score = len(common_words) / len(question_words)

        keyword_boost = min(code_matches * 0.05, 0.3)  # 代码关键词加分

        return min(base_score + keyword_boost, 1.0)
