import re
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class FormattedCodeResult:
    code: str
    language: str
    description: str
    source_file: str
    similarity: float


class CodeFormatter:
    def __init__(self):
        self.max_code_length = 2000
        self.max_description_length = 300

    def format_code_block(self, code: str, language: str = "python") -> str:
        code = code.strip()
        if not code:
            return ""
        
        lines = code.split('\n')
        formatted_lines = []
        
        for line in lines:
            stripped = line.rstrip()
            formatted_lines.append(stripped)
        
        while formatted_lines and not formatted_lines[0].strip():
            formatted_lines.pop(0)
        while formatted_lines and not formatted_lines[-1].strip():
            formatted_lines.pop()
        
        return '\n'.join(formatted_lines)

    def wrap_in_markdown(self, code: str, language: str = "python") -> str:
        formatted = self.format_code_block(code, language)
        if not formatted:
            return ""
        return f"```{language}\n{formatted}\n```"

    def extract_function_signature(self, code: str) -> Optional[str]:
        match = re.search(r'def\s+(\w+)\s*\([^)]*\)', code)
        if match:
            return match.group(0)
        return None

    def extract_class_name(self, code: str) -> Optional[str]:
        match = re.search(r'class\s+(\w+)', code)
        if match:
            return match.group(1)
        return None

    def generate_code_description(self, code: str, context: str = "") -> str:
        description_parts = []
        
        func_sig = self.extract_function_signature(code)
        if func_sig:
            description_parts.append(f"函数: {func_sig}")
        
        class_name = self.extract_class_name(code)
        if class_name:
            description_parts.append(f"类: {class_name}")
        
        imports = re.findall(r'(?:from\s+(\S+)\s+import|import\s+(\S+))', code)
        if imports:
            import_list = [i[0] or i[1] for i in imports[:3]]
            description_parts.append(f"依赖: {', '.join(import_list)}")
        
        if context:
            context_preview = context[:self.max_description_length].strip()
            if context_preview:
                description_parts.append(f"上下文: {context_preview}")
        
        return " | ".join(description_parts) if description_parts else "代码片段"

    def format_retrieved_code(
        self,
        code: str,
        source_file: str,
        similarity: float,
        context: str = "",
        language: str = "python"
    ) -> FormattedCodeResult:
        formatted_code = self.format_code_block(code, language)
        description = self.generate_code_description(code, context)
        
        return FormattedCodeResult(
            code=formatted_code,
            language=language,
            description=description,
            source_file=source_file,
            similarity=round(similarity, 4)
        )

    def format_code_results_for_display(
        self,
        results: List[FormattedCodeResult],
        include_context: bool = True
    ) -> str:
        if not results:
            return "未找到相关代码"
        
        output_parts = ["## 相关代码片段\n"]
        
        for i, result in enumerate(results, 1):
            output_parts.append(f"### 代码片段 {i}")
            output_parts.append(f"- **来源**: {result.source_file}")
            output_parts.append(f"- **相似度**: {result.similarity}")
            output_parts.append(f"- **说明**: {result.description}")
            output_parts.append("")
            output_parts.append(self.wrap_in_markdown(result.code, result.language))
            output_parts.append("")
        
        return '\n'.join(output_parts)

    def format_code_for_llm_context(
        self,
        results: List[FormattedCodeResult]
    ) -> str:
        if not results:
            return ""
        
        context_parts = ["以下是检索到的相关代码片段：\n"]
        
        for i, result in enumerate(results, 1):
            context_parts.append(f"[代码片段 {i}]")
            context_parts.append(f"文件: {result.source_file}")
            context_parts.append(f"说明: {result.description}")
            context_parts.append(self.wrap_in_markdown(result.code, result.language))
            context_parts.append("")
        
        return '\n'.join(context_parts)

    def truncate_code(self, code: str, max_lines: int = 50) -> str:
        lines = code.split('\n')
        if len(lines) <= max_lines:
            return code
        
        truncated = lines[:max_lines]
        truncated.append(f"\n# ... 省略了 {len(lines) - max_lines} 行代码 ...")
        return '\n'.join(truncated)

    def build_code_prompt(
        self,
        question: str,
        code_results: List[FormattedCodeResult],
        text_context: List[str] = None
    ) -> str:
        prompt_parts = []
        
        prompt_parts.append("你是一个专业的编程助手。请根据以下信息回答用户的问题。")
        prompt_parts.append("")
        prompt_parts.append(f"用户问题: {question}")
        prompt_parts.append("")
        
        if code_results:
            prompt_parts.append("=== 相关代码片段 ===")
            for i, result in enumerate(code_results, 1):
                prompt_parts.append(f"\n[代码 {i}] 来源: {result.source_file} (相似度: {result.similarity})")
                prompt_parts.append(self.wrap_in_markdown(result.code, result.language))
        
        if text_context:
            prompt_parts.append("\n=== 相关文本内容 ===")
            for i, ctx in enumerate(text_context[:3], 1):
                prompt_parts.append(f"\n[文本 {i}]")
                prompt_parts.append(ctx[:500])
        
        prompt_parts.append("\n=== 回答要求 ===")
        prompt_parts.append("1. 如果问题涉及代码实现，请提供完整的代码示例")
        prompt_parts.append("2. 代码需要用 ```python 代码块格式包裹")
        prompt_parts.append("3. 解释代码的关键部分和实现思路")
        prompt_parts.append("4. 如果检索到的代码相关，可以参考或改进")
        
        return '\n'.join(prompt_parts)


code_formatter = CodeFormatter()


def format_code(code: str, language: str = "python") -> str:
    return code_formatter.format_code_block(code, language)


def wrap_code_markdown(code: str, language: str = "python") -> str:
    return code_formatter.wrap_in_markdown(code, language)


def create_code_result(
    code: str,
    source_file: str,
    similarity: float,
    context: str = "",
    language: str = "python"
) -> FormattedCodeResult:
    return code_formatter.format_retrieved_code(
        code, source_file, similarity, context, language
    )


def build_code_prompt(
    question: str,
    code_results: List[FormattedCodeResult],
    text_context: List[str] = None
) -> str:
    return code_formatter.build_code_prompt(question, code_results, text_context)
