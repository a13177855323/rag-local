import re
import ast
import tokenize
import io
from typing import List, Dict, Tuple, Optional, Any


class CodeDetector:
    """Python代码块检测器与提取器"""

    def __init__(self):
        """初始化代码检测器"""
        self._compile_patterns()

    def _compile_patterns(self):
        """预编译正则表达式模式"""
        # Markdown代码块模式
        self.code_block_patterns = {
            'python': [
                re.compile(r'```(?:python|py)\s*\n(.*?)\n```', re.DOTALL | re.IGNORECASE),
                re.compile(r'```\s*\n(?:#.*?\n)?(.*?)\n```', re.DOTALL),
            ],
            'generic': [
                re.compile(r'```(?:\w+)?\s*\n(.*?)\n```', re.DOTALL),
            ]
        }

        # 行内代码模式
        self.inline_code_pattern = re.compile(r'`([^`]+)`')

        # Python语法特征模式
        self.syntax_patterns = {
            'function_def': re.compile(r'^def\s+\w+\s*\([^)]*\)\s*:', re.MULTILINE),
            'class_def': re.compile(r'^class\s+\w+(?:\([^)]*\))?\s*:', re.MULTILINE),
            'import': re.compile(r'^(?:from|import)\s+[\w.]+', re.MULTILINE),
            'assignment': re.compile(r'^\s*\w+\s*=', re.MULTILINE),
            'for_loop': re.compile(r'^\s*for\s+\w+\s+in\s+.+:', re.MULTILINE),
            'if_stmt': re.compile(r'^\s*if\s+.+:', re.MULTILINE),
            'comment': re.compile(r'^\s*#', re.MULTILINE),
            'docstring': re.compile(r'^\s*["\']{3}', re.MULTILINE),
        }

        # Python内置函数和特殊名称
        self.python_builtins = {
            'print', 'len', 'range', 'str', 'int', 'float', 'list', 'dict', 'set',
            'tuple', 'bool', 'type', 'isinstance', 'issubclass', 'hasattr',
            'getattr', 'setattr', 'open', 'input', 'super', 'staticmethod',
            'classmethod', 'property', 'enumerate', 'zip', 'map', 'filter',
            'sorted', 'reversed', 'sum', 'max', 'min', 'abs', 'round', 'divmod',
            'pow', 'ord', 'chr', 'bin', 'oct', 'hex', 'all', 'any', 'dir',
            'help', 'id', 'repr', 'str', 'format'
        }

    def is_valid_python_syntax(self, code: str) -> bool:
        """
        检查代码是否具有有效的Python语法

        Args:
            code: 待检查的代码字符串

        Returns:
            是否为有效Python语法
        """
        code = code.strip()
        if not code:
            return False

        try:
            ast.parse(code)
            return True
        except SyntaxError:
            # 尝试修复常见问题（如缺少缩进、缺少冒号等）
            return self._try_fix_and_parse(code)
        except Exception:
            return False

    def _try_fix_and_parse(self, code: str) -> bool:
        """尝试修复代码并解析"""
        # 尝试添加缩进来修复单行函数定义
        if code.startswith('def ') and ':' in code:
            body_indent = '    '
            parts = code.split(':', 1)
            if len(parts) == 2 and not parts[1].strip():
                fixed = parts[0] + ':' + body_indent + 'pass'
                try:
                    ast.parse(fixed)
                    return True
                except Exception:
                    pass

        # 尝试在末尾添加pass来修复不完整的代码块
        if code.rstrip().endswith(':'):
            try:
                ast.parse(code + '\n    pass')
                return True
            except Exception:
                pass

        return False

    def extract_python_code(self, text: str, include_likely: bool = True) -> List[Dict[str, Any]]:
        """
        从文本中提取Python代码

        Args:
            text: 输入文本
            include_likely: 是否包含可能是Python但不确定的代码

        Returns:
            代码片段列表，每个元素包含code、confidence、type字段
        """
        results = []

        # 1. 提取Markdown代码块（优先带python标记的）
        for pattern in self.code_block_patterns['python']:
            matches = pattern.findall(text)
            for match in matches:
                code = match.strip()
                if code:
                    confidence = self.calculate_python_confidence(code)
                    if confidence >= 0.3:  # 阈值过滤
                        results.append({
                            'code': code,
                            'confidence': confidence,
                            'type': 'markdown_python' if confidence >= 0.7 else 'markdown_likely'
                        })

        # 2. 提取通用代码块并检测
        if include_likely:
            for pattern in self.code_block_patterns['generic']:
                matches = pattern.findall(text)
                for match in matches:
                    code = match.strip()
                    if code:
                        confidence = self.calculate_python_confidence(code)
                        if confidence >= 0.5:  # 通用代码块阈值更高
                            # 避免重复
                            if not any(r['code'] == code for r in results):
                                results.append({
                                    'code': code,
                                    'confidence': confidence,
                                    'type': 'generic_code_block'
                                })

        # 3. 提取行内代码中可能的代码片段
        inline_codes = self.inline_code_pattern.findall(text)
        for code in inline_codes:
            code = code.strip()
            if code and len(code) >= 10:  # 只考虑较长的行内代码
                confidence = self.calculate_python_confidence(code)
                if confidence >= 0.6:
                    if not any(r['code'] == code for r in results):
                        results.append({
                            'code': code,
                            'confidence': confidence,
                            'type': 'inline_code'
                        })

        # 按置信度排序
        results.sort(key=lambda x: x['confidence'], reverse=True)

        return results

    def calculate_python_confidence(self, code: str) -> float:
        """
        计算一段文本是Python代码的置信度（0-1）

        Args:
            code: 待评估的文本

        Returns:
            置信度分数（0-1）
        """
        if not code or len(code.strip()) < 3:
            return 0.0

        score = 0.0
        max_score = 0.0
        code_lower = code.lower()

        # 1. 语法验证（权重最高）
        max_score += 4.0
        if self.is_valid_python_syntax(code):
            score += 4.0
        else:
            # 部分匹配的语法特征
            syntax_hits = sum(1 for pattern in self.syntax_patterns.values()
                            if pattern.search(code))
            score += min(syntax_hits * 0.5, 2.0)

        # 2. Python关键字特征
        max_score += 3.0
        keyword_score = 0.0

        # 检查定义语句
        if re.search(r'\bdef\s+\w+\s*\(', code) or 'def ' in code_lower:
            keyword_score += 1.0
        if re.search(r'\bclass\s+\w+', code) or 'class ' in code_lower:
            keyword_score += 1.0
        if 'import ' in code_lower or 'from ' in code_lower and ' import ' in code_lower:
            keyword_score += 0.8
        if '__init__' in code or '__name__' in code or 'self.' in code:
            keyword_score += 1.0

        # 检查内置函数
        builtin_hits = sum(1 for func in self.python_builtins
                          if f'{func}(' in code or f'{func} (' in code)
        keyword_score += min(builtin_hits * 0.2, 0.8)

        score += min(keyword_score, 3.0)

        # 3. 操作符和语法符号
        max_score += 1.5
        symbol_score = 0.0

        # Python特有操作符
        python_operators = ['**', '//', '@', ':=', '->', '==', '!=', '<=', '>=']
        for op in python_operators:
            if op in code:
                symbol_score += 0.15

        # 括号平衡
        if code.count('(') > 0 and code.count(')') > 0:
            symbol_score += 0.2
        if code.count('[') > 0 and code.count(']') > 0:
            symbol_score += 0.15
        if code.count('{') > 0 and code.count('}') > 0:
            symbol_score += 0.15

        # 注释和字符串
        if '#' in code:
            symbol_score += 0.2
        if '"""' in code or "'''" in code:
            symbol_score += 0.2

        score += min(symbol_score, 1.5)

        # 4. 缩进模式
        max_score += 1.0
        lines = code.split('\n')
        indented_lines = sum(1 for line in lines
                           if line.startswith('    ') or line.startswith('\t'))
        if indented_lines >= 2 and len(lines) >= 3:
            score += 1.0
        elif indented_lines >= 1 and len(lines) >= 2:
            score += 0.5

        # 5. 库导入特征
        max_score += 0.5
        libs = ['pandas', 'pd.', 'numpy', 'np.', 'torch', 'tensorflow', 'tf.',
                'sklearn', 'matplotlib', 'plt.', 'seaborn', 'requests', 'flask',
                'django', 'fastapi', 'pygame', 'cv2', 'keras']
        lib_hits = sum(1 for lib in libs if lib in code_lower)
        if lib_hits > 0:
            score += min(lib_hits * 0.2, 0.5)

        # 6. 非代码特征惩罚
        non_code_indicators = ['. ', '。', '？', '！', '，', '；', '：',
                              ' is ', ' are ', ' the ', ' be ', ' to ', ' and ']
        non_code_hits = sum(1 for indicator in non_code_indicators
                          if indicator in code_lower[:200])  # 只检查前200字符
        if non_code_hits > 3:
            score -= min(non_code_hits * 0.3, 2.0)

        # 归一化到0-1范围
        final_score = max(0.0, min(1.0, score / max(max_score, 1.0)))

        # 对于非常短的代码片段降低分数
        if len(code.strip()) < 20:
            final_score *= 0.8
        elif len(code.strip()) < 10:
            final_score *= 0.5

        return round(final_score, 3)

    def find_code_regions(self, text: str) -> List[Tuple[int, int, str]]:
        """
        查找文本中所有代码区域的位置（起始和结束索引）

        Args:
            text: 输入文本

        Returns:
            代码区域列表，每个元素为(start, end, code_type)
        """
        regions = []

        # 查找Markdown代码块
        pattern = re.compile(r'```(\w*)\s*\n(.*?)\n```', re.DOTALL)
        for match in pattern.finditer(text):
            start = match.start()
            end = match.end()
            lang = match.group(1).lower() or 'code'
            regions.append((start, end, f'markdown_{lang}'))

        return regions

    def separate_code_and_text(self, text: str) -> List[Dict[str, Any]]:
        """
        分离文本中的代码块和普通文本段落

        Args:
            text: 输入文本

        Returns:
            分段列表，每个段有type和content字段
        """
        segments = []
        code_regions = self.find_code_regions(text)

        # 如果没有代码区域，直接返回
        if not code_regions:
            if self.calculate_python_confidence(text) >= 0.5:
                return [{'type': 'likely_code', 'content': text, 'confidence': self.calculate_python_confidence(text)}]
            return [{'type': 'text', 'content': text}]

        # 按起始位置排序
        code_regions.sort()

        last_end = 0
        for start, end, code_type in code_regions:
            # 添加之前的文本
            if start > last_end:
                text_segment = text[last_end:start].strip()
                if text_segment:
                    segments.append({'type': 'text', 'content': text_segment})

            # 添加代码块
            code_content = text[start:end]
            segments.append({'type': code_type, 'content': code_content})

            last_end = end

        # 添加最后的文本
        if last_end < len(text):
            text_segment = text[last_end:].strip()
            if text_segment:
                segments.append({'type': 'text', 'content': text_segment})

        return segments

    def analyze_code_complexity(self, code: str) -> Dict[str, Any]:
        """
        分析Python代码的复杂度指标

        Args:
            code: Python代码字符串

        Returns:
            复杂度分析结果
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {'error': 'Invalid Python syntax', 'valid': False}

        # 统计信息
        stats = {
            'valid': True,
            'num_functions': 0,
            'num_classes': 0,
            'num_methods': 0,
            'num_imports': 0,
            'num_assignments': 0,
            'num_conditions': 0,
            'num_loops': 0,
            'num_calls': 0,
            'function_names': [],
            'class_names': [],
            'imported_modules': []
        }

        class StatsVisitor(ast.NodeVisitor):
            def __init__(self, stats_dict):
                self.stats = stats_dict

            def visit_FunctionDef(self, node):
                self.stats['num_functions'] += 1
                self.stats['function_names'].append(node.name)
                self.generic_visit(node)

            def visit_ClassDef(self, node):
                self.stats['num_classes'] += 1
                self.stats['class_names'].append(node.name)
                # 统计方法
                for body_node in node.body:
                    if isinstance(body_node, ast.FunctionDef):
                        self.stats['num_methods'] += 1
                self.generic_visit(node)

            def visit_Import(self, node):
                self.stats['num_imports'] += 1
                for name in node.names:
                    self.stats['imported_modules'].append(name.name)
                self.generic_visit(node)

            def visit_ImportFrom(self, node):
                self.stats['num_imports'] += 1
                if node.module:
                    self.stats['imported_modules'].append(node.module)
                self.generic_visit(node)

            def visit_Assign(self, node):
                self.stats['num_assignments'] += 1
                self.generic_visit(node)

            def visit_For(self, node):
                self.stats['num_loops'] += 1
                self.generic_visit(node)

            def visit_While(self, node):
                self.stats['num_loops'] += 1
                self.generic_visit(node)

            def visit_If(self, node):
                self.stats['num_conditions'] += 1
                self.generic_visit(node)

            def visit_Call(self, node):
                self.stats['num_calls'] += 1
                self.generic_visit(node)

        visitor = StatsVisitor(stats)
        visitor.visit(tree)

        # 计算复杂度评分
        complexity_score = (
            stats['num_functions'] * 2 +
            stats['num_classes'] * 5 +
            stats['num_conditions'] * 1 +
            stats['num_loops'] * 2 +
            stats['num_calls'] * 0.5
        )

        if complexity_score <= 5:
            complexity_level = 'simple'
        elif complexity_score <= 20:
            complexity_level = 'moderate'
        else:
            complexity_level = 'complex'

        stats['complexity_score'] = complexity_score
        stats['complexity_level'] = complexity_level
        stats['total_lines'] = len(code.strip().split('\n'))

        return stats

    def generate_code_preview(self, code: str, max_lines: int = 10) -> str:
        """
        生成代码预览，最多显示指定行数

        Args:
            code: Python代码
            max_lines: 最大显示行数

        Returns:
            代码预览字符串
        """
        lines = code.strip().split('\n')
        if len(lines) <= max_lines:
            return code

        preview = '\n'.join(lines[:max_lines])
        preview += f'\n\n... (共 {len(lines)} 行，省略 {len(lines) - max_lines} 行)'
        return preview
