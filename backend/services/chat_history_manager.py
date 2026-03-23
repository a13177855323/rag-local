#!/usr/bin/env python3
"""对话历史管理核心模块

支持对话记录存储、智能分析、多格式导出等功能。
重构说明：
- 遵循PEP8编码规范
- 增强异常处理机制
- 添加统一日志格式
- 模块化函数设计
- CPU优化参数保留
"""
import json
import csv
import time
import uuid
import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import Counter
import numpy as np

from backend.config import settings

# 日志配置 - 遵循项目统一日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# CPU优化参数保留 - 维持原有性能优化特性
CPU_OPTIMIZATION_PARAMS = {
    'max_workers': 4,
    'chunk_size': 1000,
    'memory_limit_mb': 512,
    'enable_multiprocessing': False  # 保持CPU友好的单进程模式
}


class ChatMessage:
    """对话消息数据模型

    存储单条对话消息的完整信息，包括角色、内容、时间戳和元数据。

    Attributes:
        message_id: 消息唯一标识，UUID格式
        role: 消息角色 ('user' 或 'assistant')
        content: 消息内容
        timestamp: Unix时间戳
        metadata: 额外元数据（引用来源、token使用情况等）
    """

    def __init__(
        self,
        role: str,
        content: str,
        message_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """初始化ChatMessage对象

        Args:
            role: 消息角色，必须是 'user' 或 'assistant'
            content: 消息文本内容
            message_id: 可选，消息唯一ID，不提供则自动生成
            timestamp: 可选，消息时间戳，不提供则使用当前时间
            metadata: 可选，消息元数据字典
        """
        self.message_id = message_id or str(uuid.uuid4())
        self.role = self._validate_role(role)
        self.content = content
        self.timestamp = timestamp or time.time()
        self.metadata = metadata or {}

    def _validate_role(self, role: str) -> str:
        """验证消息角色有效性

        Args:
            role: 待验证的角色字符串

        Returns:
            验证后的角色字符串

        Raises:
            ValueError: 当角色不是 'user' 或 'assistant' 时
        """
        valid_roles = ['user', 'assistant']
        if role not in valid_roles:
            raise ValueError(f"无效的消息角色: {role}, 必须为 {valid_roles} 之一")
        return role

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式用于序列化

        Returns:
            包含消息所有属性的字典
        """
        return {
            'message_id': self.message_id,
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(
                self.timestamp
            ).strftime('%Y-%m-%d %H:%M:%S'),
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatMessage':
        """从字典创建ChatMessage对象

        Args:
            data: 包含消息数据的字典

        Returns:
            新创建的ChatMessage对象

        Raises:
            KeyError: 当缺少必要字段时
        """
        try:
            return cls(
                role=data['role'],
                content=data['content'],
                message_id=data.get('message_id'),
                timestamp=data.get('timestamp'),
                metadata=data.get('metadata', {})
            )
        except KeyError as e:
            logger.error(f"[ChatMessage.from_dict] 缺少必要字段: {e}")
            raise


class ChatSession:
    """对话会话管理类

    管理一组相关的对话消息，支持消息添加、标题生成和序列化。

    Attributes:
        session_id: 会话唯一标识
        title: 会话标题
        created_at: 会话创建时间戳
        updated_at: 会话最后更新时间戳
        messages: 消息列表
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        title: Optional[str] = None,
        created_at: Optional[float] = None,
        updated_at: Optional[float] = None
    ):
        """初始化ChatSession对象

        Args:
            session_id: 可选，会话唯一ID
            title: 可选，会话标题
            created_at: 可选，创建时间戳
            updated_at: 可选，最后更新时间戳
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.title = title or self._generate_default_title()
        self.created_at = created_at or time.time()
        self.updated_at = updated_at or time.time()
        self.messages: List[ChatMessage] = []

    def _generate_default_title(self) -> str:
        """生成默认会话标题

        Returns:
            基于当前时间的默认标题字符串
        """
        return f"对话 {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    def add_message(self, message: ChatMessage) -> None:
        """添加消息到会话

        Args:
            message: 要添加的ChatMessage对象

        Note:
            自动更新最后更新时间
            如果标题为空且是用户消息，自动从内容生成标题
        """
        self.messages.append(message)
        self.updated_at = time.time()

        # 自动更新标题（使用第一条用户消息的前30字符）
        if not self.title and message.role == 'user':
            content = message.content.strip()
            if content:
                self.title = (
                    content[:30] + "..."
                    if len(content) > 30
                    else content
                )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式用于序列化

        Returns:
            包含会话所有属性的字典
        """
        return {
            'session_id': self.session_id,
            'title': self.title,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'created_datetime': datetime.fromtimestamp(
                self.created_at
            ).strftime('%Y-%m-%d %H:%M:%S'),
            'message_count': len(self.messages),
            'messages': [msg.to_dict() for msg in self.messages]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatSession':
        """从字典创建ChatSession对象

        Args:
            data: 包含会话数据的字典

        Returns:
            新创建的ChatSession对象

        Raises:
            Exception: 当反序列化失败时记录错误并继续
        """
        session = cls(
            session_id=data.get('session_id'),
            title=data.get('title'),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at')
        )

        # 反序列化消息列表
        for msg_data in data.get('messages', []):
            try:
                session.messages.append(ChatMessage.from_dict(msg_data))
            except Exception as e:
                logger.error(f"[ChatSession.from_dict] 反序列化消息失败: {e}")

        return session


class ChatHistoryAnalyzer:
    """对话历史智能分析器

    提供对话摘要生成、问题分类、回答质量评分等智能分析功能。
    所有分析均在CPU环境下执行，使用现有嵌入模型兼容方案。

    Attributes:
        CATEGORY_KEYWORDS: 问题分类关键词映射表
        embedding_model: 嵌入模型实例（可选）
    """

    # 问题分类关键词映射表
    CATEGORY_KEYWORDS = {
        '代码开发': [
            '代码', '编程', '函数', '类', 'python', 'java', 'c++',
            'bug', '调试', '算法', '实现', '开发', '程序'
        ],
        '文档查询': [
            '文档', '说明', '怎么用', '使用方法', '参数', '配置',
            '安装', '部署', '教程', '示例'
        ],
        '理论知识': [
            '什么是', '解释', '原理', '概念', '理论', '区别',
            '对比', '为什么', '如何理解'
        ],
        '问题解决': [
            '错误', '问题', '无法', '失败', '异常', '报错',
            '怎么办', '解决', '修复', '卡住'
        ],
        '其他': []
    }

    def __init__(self, embedding_model: Optional[Any] = None):
        """初始化分析器

        Args:
            embedding_model: 可选，嵌入模型实例用于高级分析
        """
        self.embedding_model = embedding_model
        self._stop_words = self._load_stop_words()

    def _load_stop_words(self) -> set:
        """加载停用词列表

        Returns:
            停用词集合
        """
        return {
            '的', '是', '在', '了', '和', '与', '或', '这', '那',
            '有', '能', '会', '要', '什么', '怎么', '如何', '请问',
            'i', 'the', 'a', 'an', 'is', 'are', 'was', 'were', 'been'
        }

    def _tokenize_text(self, text: str) -> List[str]:
        """文本分词处理 - tokenizer独立子函数

        Args:
            text: 输入文本

        Returns:
            分词后的词列表
        """
        import re
        # 使用正则表达式分词，同时支持中文和英文
        words = re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]+', text.lower())
        # 过滤停用词和短词
        return [w for w in words if len(w) > 1 and w not in self._stop_words]

    def _extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """提取关键词

        Args:
            text: 输入文本
            top_n: 返回前N个关键词

        Returns:
            关键词列表，按重要性排序
        """
        words = self._tokenize_text(text)
        if not words:
            return []

        counter = Counter(words)
        return [word for word, _ in counter.most_common(top_n)]

    def generate_summary(self, session: ChatSession, max_sentences: int = 3) -> str:
        """生成对话摘要（抽取式摘要）

        Args:
            session: 对话会话对象
            max_sentences: 最大摘要句子数

        Returns:
            对话摘要字符串
        """
        if not session.messages:
            return "暂无对话内容"

        user_messages = [msg for msg in session.messages if msg.role == 'user']
        if not user_messages:
            return "暂无有效对话内容"

        summary_points = []

        # 1. 第一个问题（核心问题）
        first_question = user_messages[0].content.strip()
        if first_question:
            preview = (
                first_question[:50] + "..."
                if len(first_question) > 50
                else first_question
            )
            summary_points.append(f"用户询问：「{preview}」")

        # 2. 关键主题词
        all_content = " ".join([msg.content for msg in session.messages])
        keywords = self._extract_keywords(all_content, top_n=5)
        if keywords:
            summary_points.append(f"主要涉及主题：{', '.join(keywords)}")

        # 3. 对话轮次统计
        round_count = len(user_messages)
        if round_count > 1:
            summary_points.append(f"共进行了 {round_count} 轮问答互动")

        # 4. 最后回复摘要
        assistant_messages = [msg for msg in session.messages if msg.role == 'assistant']
        if assistant_messages and len(summary_points) < max_sentences:
            last_answer = assistant_messages[-1].content.strip()
            if last_answer:
                preview = (
                    last_answer[:60] + "..."
                    if len(last_answer) > 60
                    else last_answer
                )
                summary_points.append(f"最终解答摘要：{preview}")

        return "；".join(summary_points[:max_sentences])

    def classify_question(self, question: str) -> str:
        """问题分类

        Args:
            question: 用户问题文本

        Returns:
            分类结果标签（代码开发/文档查询/理论知识/问题解决/其他）
        """
        if not question:
            return '其他'

        question_lower = question.lower()

        # 基于关键词匹配的分类
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            if category == '其他':
                continue
            for keyword in keywords:
                if keyword in question_lower:
                    return category

        return '其他'

    def calculate_answer_quality(
        self,
        question: str,
        answer: str
    ) -> Dict[str, Any]:
        """计算回答质量评分

        从多个维度评估回答质量：长度合理性、内容结构化、
        内容相关性、内容帮助性。

        Args:
            question: 用户问题文本
            answer: 助手回答文本

        Returns:
            包含评分结果的字典：
            - total_score: 综合评分(0-1)
            - dimension_scores: 各维度得分
            - level: 质量等级（优秀/良好/一般/待提升）
            - suggestions: 改进建议列表
        """
        scores = {}
        suggestions = []

        # 1. 长度合理性评分
        answer_len = len(answer.strip())
        if answer_len < 20:
            scores['长度合理性'] = 0.3
            suggestions.append("回答内容偏短，建议补充更多细节")
        elif answer_len > 500:
            scores['长度合理性'] = 0.7
        else:
            scores['长度合理性'] = min(1.0, answer_len / 100)

        # 2. 内容结构化评分
        has_code = '```' in answer or '代码' in answer
        has_list = '\n- ' in answer or '\n* ' in answer or '\n1.' in answer
        has_table = '| ' in answer and ' |' in answer
        structure_score = 0.4
        if has_code:
            structure_score += 0.3
        if has_list:
            structure_score += 0.2
        if has_table:
            structure_score += 0.1
        scores['内容结构化'] = min(1.0, structure_score)
        if structure_score < 0.6:
            suggestions.append("建议使用列表、代码块等结构化格式")

        # 3. 内容相关性评分
        question_words = set(self._extract_keywords(question, top_n=10))
        answer_words = set(self._extract_keywords(answer, top_n=20))
        overlap = len(question_words & answer_words)
        relevance_score = overlap / len(question_words) if question_words else 0.5
        scores['内容相关性'] = relevance_score
        if relevance_score < 0.3:
            suggestions.append("回答可能与问题相关性不足")

        # 4. 内容帮助性评分
        helpful_words = ['可以', '需要', '建议', '应该', '推荐', '注意', '例如', '示例']
        helpful_count = sum(1 for word in helpful_words if word in answer)
        helpful_score = min(1.0, 0.5 + helpful_count * 0.1)
        scores['内容帮助性'] = helpful_score

        # 综合评分（加权平均）
        weights = {
            '长度合理性': 0.2,
            '内容结构化': 0.25,
            '内容相关性': 0.35,
            '内容帮助性': 0.2
        }
        total_score = sum(scores[k] * weights[k] for k in scores)

        return {
            'total_score': round(float(total_score), 2),
            'dimension_scores': scores,
            'level': self._score_to_level(float(total_score)),
            'suggestions': suggestions
        }

    def _score_to_level(self, score: float) -> str:
        """将数值分数转换为等级标签

        Args:
            score: 0-1之间的分数值

        Returns:
            质量等级字符串
        """
        if score >= 0.8:
            return '优秀'
        elif score >= 0.6:
            return '良好'
        elif score >= 0.4:
            return '一般'
        else:
            return '待提升'

    def analyze_session(self, session: ChatSession) -> Dict[str, Any]:
        """完整分析对话会话

        Args:
            session: 对话会话对象

        Returns:
            包含完整分析结果的字典
        """
        # 问题分类统计
        categories = []
        for msg in session.messages:
            if msg.role == 'user':
                categories.append(self.classify_question(msg.content))
        category_stats = dict(Counter(categories))

        # 回答质量统计
        quality_scores = []
        user_questions = [msg for msg in session.messages if msg.role == 'user']
        assistant_answers = [msg for msg in session.messages if msg.role == 'assistant']

        for q, a in zip(user_questions, assistant_answers):
            quality = self.calculate_answer_quality(q.content, a.content)
            quality_scores.append(quality['total_score'])

        avg_quality = np.mean(quality_scores) if quality_scores else 0.0

        # 对话时长
        duration = 0.0
        if len(session.messages) >= 2:
            start_time = session.messages[0].timestamp
            end_time = session.messages[-1].timestamp
            duration = end_time - start_time

        # 关键词提取
        all_content = " ".join([msg.content for msg in session.messages])
        keywords = self._extract_keywords(all_content, top_n=8)

        return {
            'summary': self.generate_summary(session),
            'category_distribution': category_stats,
            'avg_answer_quality': round(float(avg_quality), 2),
            'quality_level': self._score_to_level(float(avg_quality)),
            'total_rounds': len(user_questions),
            'duration_seconds': round(duration),
            'duration_formatted': str(
                timedelta(seconds=round(duration))
            ).split('.')[0],
            'keywords': keywords
        }


class ChatHistoryExporter:
    """对话历史导出器

    支持将对话历史导出为多种格式：JSON、Markdown报告、CSV统计。
    保持与原有导出功能的完全兼容。
    """

    def export_to_json(
        self,
        sessions: List[ChatSession],
        filename: Optional[str] = None
    ) -> str:
        """导出为JSON格式

        Args:
            sessions: 要导出的会话列表
            filename: 可选，保存到文件的路径

        Returns:
            JSON格式的字符串

        Raises:
            IOError: 当文件写入失败时
        """
        export_data = {
            'export_time': datetime.now().isoformat(),
            'version': '1.0',
            'total_sessions': len(sessions),
            'cpu_optimization': CPU_OPTIMIZATION_PARAMS,
            'sessions': [session.to_dict() for session in sessions]
        }

        json_str = json.dumps(export_data, ensure_ascii=False, indent=2)

        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(json_str)
                logger.info(f"[ChatHistoryExporter.export_to_json] 成功保存到: {filename}")
            except IOError as e:
                logger.error(f"[ChatHistoryExporter.export_to_json] 保存失败: {e}")
                raise

        return json_str

    def export_to_markdown(
        self,
        sessions: List[ChatSession],
        with_analysis: bool = True,
        filename: Optional[str] = None
    ) -> str:
        """导出为Markdown报告格式

        Args:
            sessions: 要导出的会话列表
            with_analysis: 是否包含智能分析内容
            filename: 可选，保存到文件的路径

        Returns:
            Markdown格式的字符串

        Raises:
            IOError: 当文件写入失败时
        """
        analyzer = ChatHistoryAnalyzer()
        md_content = []

        # 报告头部
        md_content.append("# 对话历史报告")
        md_content.append(
            f"\n**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        md_content.append(f"**会话总数**: {len(sessions)}")

        # 总统计
        if sessions:
            total_messages = sum(len(s.messages) for s in sessions)
            md_content.append(f"**消息总数**: {total_messages}")
            timestamps = [s.created_at for s in sessions]
            md_content.append(
                f"**最早会话**: {datetime.fromtimestamp(min(timestamps)).strftime('%Y-%m-%d %H:%M')}"
            )
            md_content.append(
                f"**最晚会话**: {datetime.fromtimestamp(max(timestamps)).strftime('%Y-%m-%d %H:%M')}"
            )

        md_content.append("\n---\n")

        # 每个会话详情
        for i, session in enumerate(sessions, 1):
            md_content.append(f"## 会话 {i}: {session.title}")
            md_content.append(f"- **会话ID**: {session.session_id}")
            md_content.append(
                f"- **创建时间**: {datetime.fromtimestamp(session.created_at).strftime('%Y-%m-%d %H:%M:%S')}"
            )
            md_content.append(f"- **消息数量**: {len(session.messages)}")

            # 智能分析部分
            if with_analysis:
                analysis = analyzer.analyze_session(session)
                md_content.append(f"- **对话摘要**: {analysis['summary']}")
                md_content.append(
                    f"- **平均回答质量**: {analysis['avg_answer_quality']} ({analysis['quality_level']})"
                )
                md_content.append(
                    f"- **问题分类分布**: {analysis['category_distribution']}"
                )
                md_content.append(f"- **关键词**: {', '.join(analysis['keywords'])}")

            md_content.append("\n### 对话详情\n")

            # 对话内容
            for msg in session.messages:
                role_icon = "👤 用户" if msg.role == 'user' else "🤖 助手"
                time_str = datetime.fromtimestamp(msg.timestamp).strftime('%H:%M:%S')
                md_content.append(f"**{role_icon}** ({time_str})")
                md_content.append(f"\n{msg.content}\n")

            md_content.append("\n---\n")

        result = "\n".join(md_content)

        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(result)
                logger.info(f"[ChatHistoryExporter.export_to_markdown] 成功保存到: {filename}")
            except IOError as e:
                logger.error(f"[ChatHistoryExporter.export_to_markdown] 保存失败: {e}")
                raise

        return result

    def export_to_csv(
        self,
        sessions: List[ChatSession],
        filename: Optional[str] = None
    ) -> str:
        """导出为CSV统计格式

        Args:
            sessions: 要导出的会话列表
            filename: 可选，保存到文件的路径

        Returns:
            CSV格式的字符串

        Raises:
            IOError: 当文件写入失败时
        """
        import io
        output = io.StringIO()
        writer = csv.writer(output)

        # 表头
        writer.writerow([
            '会话ID', '会话标题', '创建时间', '消息数量',
            '用户问题数', '平均回答质量', '主要分类',
            '对话时长(秒)', '关键词'
        ])

        analyzer = ChatHistoryAnalyzer()

        for session in sessions:
            analysis = analyzer.analyze_session(session)
            user_count = sum(1 for m in session.messages if m.role == 'user')

            # 确定主要分类
            cat_dist = analysis['category_distribution']
            main_category = max(
                cat_dist.items(),
                key=lambda x: x[1]
            )[0] if cat_dist else '未知'

            writer.writerow([
                session.session_id,
                session.title,
                datetime.fromtimestamp(session.created_at).strftime('%Y-%m-%d %H:%M:%S'),
                len(session.messages),
                user_count,
                analysis['avg_answer_quality'],
                main_category,
                analysis['duration_seconds'],
                ','.join(analysis['keywords'])
            ])

        result = output.getvalue()

        if filename:
            try:
                with open(filename, 'w', encoding='utf-8-sig', newline='') as f:
                    f.write(result)
                logger.info(f"[ChatHistoryExporter.export_to_csv] 成功保存到: {filename}")
            except IOError as e:
                logger.error(f"[ChatHistoryExporter.export_to_csv] 保存失败: {e}")
                raise

        return result


class ChatHistoryManager:
    """对话历史管理器 - 核心入口类

    提供完整的对话历史管理功能：会话CRUD、智能分析、多格式导出。
    保持与原有接口的完全兼容，内部实现重构优化。

    Attributes:
        storage_path: 会话数据存储目录
        analyzer: 智能分析器实例
        exporter: 导出器实例
    """

    def __init__(self, storage_path: Optional[str] = None):
        """初始化管理器

        Args:
            storage_path: 可选，会话数据存储目录，默认使用settings配置
        """
        self.storage_path = storage_path or settings.CHAT_HISTORY_PATH
        self._sessions: Dict[str, ChatSession] = {}
        self.analyzer = ChatHistoryAnalyzer()
        self.exporter = ChatHistoryExporter()

        self._init_storage()
        self._load_sessions()

    def _init_storage(self) -> None:
        """初始化存储目录

        Raises:
            OSError: 当目录创建失败时记录错误但不抛出异常
        """
        try:
            os.makedirs(self.storage_path, exist_ok=True)
            logger.info(f"[ChatHistoryManager] 存储目录已准备: {self.storage_path}")
        except OSError as e:
            logger.error(f"[ChatHistoryManager] 创建存储目录失败: {e}")
            raise

    def _load_sessions(self) -> None:
        """从磁盘加载所有会话数据

        异常处理：单个文件加载失败不影响其他会话
        """
        if not os.path.exists(self.storage_path):
            return

        loaded_count = 0
        error_count = 0

        for filename in os.listdir(self.storage_path):
            if not filename.endswith('.json'):
                continue

            filepath = os.path.join(self.storage_path, filename)

            try:
                session = self._load_session_file(filepath)
                if session:
                    self._sessions[session.session_id] = session
                    loaded_count += 1
            except (json.JSONDecodeError, KeyError, IOError) as e:
                error_count += 1
                logger.error(
                    f"[ChatHistoryManager._load_sessions] "
                    f"加载文件失败 {filename}: {e}"
                )

        logger.info(
            f"[ChatHistoryManager] 会话加载完成: "
            f"成功 {loaded_count} 个, 失败 {error_count} 个"
        )

    def _load_session_file(self, filepath: str) -> Optional[ChatSession]:
        """加载单个会话文件

        Args:
            filepath: 会话文件路径

        Returns:
            成功返回ChatSession对象，失败返回None

        Raises:
            json.JSONDecodeError: JSON格式错误
            KeyError: 缺少必要字段
            IOError: 文件读取失败
        """
        if not os.path.exists(filepath):
            logger.error(f"[ChatHistoryManager._load_session_file] 文件不存在: {filepath}")
            return None

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(
                f"[ChatHistoryManager._load_session_file] "
                f"JSON格式错误 {filepath}: {e}"
            )
            raise
        except IOError as e:
            logger.error(
                f"[ChatHistoryManager._load_session_file] "
                f"文件读取失败 {filepath}: {e}"
            )
            raise

        try:
            return ChatSession.from_dict(data)
        except KeyError as e:
            logger.error(
                f"[ChatHistoryManager._load_session_file] "
                f"数据格式错误，缺少字段 {filepath}: {e}"
            )
            raise

    def _save_session(self, session: ChatSession) -> bool:
        """保存会话到磁盘

        Args:
            session: 要保存的会话对象

        Returns:
            保存成功返回True，失败返回False
        """
        filepath = os.path.join(self.storage_path, f"{session.session_id}.json")

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)
            return True
        except IOError as e:
            logger.error(
                f"[ChatHistoryManager._save_session] "
                f"保存会话失败 {session.session_id}: {e}"
            )
            return False

    def create_session(self, title: Optional[str] = None) -> ChatSession:
        """创建新会话

        Args:
            title: 可选，会话标题

        Returns:
            新创建的ChatSession对象
        """
        session = ChatSession(title=title)
        self._sessions[session.session_id] = session

        if self._save_session(session):
            logger.info(f"[ChatHistoryManager.create_session] 成功创建会话: {session.session_id}")
        else:
            logger.warning(f"[ChatHistoryManager.create_session] 会话创建但未持久化: {session.session_id}")

        return session

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """获取指定会话

        Args:
            session_id: 会话ID

        Returns:
            找到返回ChatSession对象，未找到返回None
        """
        return self._sessions.get(session_id)

    def get_all_sessions(self, limit: Optional[int] = None) -> List[ChatSession]:
        """获取所有会话，按更新时间倒序排列

        Args:
            limit: 可选，限制返回数量

        Returns:
            会话列表，按最后更新时间从新到旧排序
        """
        sessions = sorted(
            self._sessions.values(),
            key=lambda s: s.updated_at,
            reverse=True
        )
        if limit:
            sessions = sessions[:limit]
        return sessions

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[ChatMessage]:
        """添加消息到会话

        Args:
            session_id: 目标会话ID
            role: 消息角色
            content: 消息内容
            metadata: 可选，元数据

        Returns:
            成功返回ChatMessage对象，失败返回None
        """
        session = self._sessions.get(session_id)
        if not session:
            logger.warning(
                f"[ChatHistoryManager.add_message] "
                f"会话不存在，创建新会话: {session_id}"
            )
            session = self.create_session()

        try:
            message = ChatMessage(
                role=role,
                content=content,
                metadata=metadata
            )
            session.add_message(message)

            if self._save_session(session):
                logger.debug(
                    f"[ChatHistoryManager.add_message] "
                    f"消息已添加: {session_id}, {role}"
                )
            return message
        except ValueError as e:
            logger.error(
                f"[ChatHistoryManager.add_message] "
                f"创建消息失败: {e}"
            )
            return None

    def delete_session(self, session_id: str) -> bool:
        """删除指定会话

        Args:
            session_id: 要删除的会话ID

        Returns:
            删除成功返回True，失败返回False
        """
        if session_id not in self._sessions:
            logger.warning(
                f"[ChatHistoryManager.delete_session] "
                f"尝试删除不存在的会话: {session_id}"
            )
            return False

        # 从内存删除
        del self._sessions[session_id]

        # 从磁盘删除
        filepath = os.path.join(self.storage_path, f"{session_id}.json")
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                logger.info(
                    f"[ChatHistoryManager.delete_session] "
                    f"会话已删除: {session_id}"
                )
            except OSError as e:
                logger.error(
                    f"[ChatHistoryManager.delete_session] "
                    f"删除文件失败 {filepath}: {e}"
                )
                # 文件删除失败不返回失败，因为内存已删除

        return True

    def clear_all(self) -> int:
        """清空所有会话

        Returns:
            删除的会话数量
        """
        count = len(self._sessions)

        # 从磁盘删除所有文件
        for session_id in list(self._sessions.keys()):
            filepath = os.path.join(self.storage_path, f"{session_id}.json")
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except OSError as e:
                    logger.error(
                        f"[ChatHistoryManager.clear_all] "
                        f"删除文件失败 {filepath}: {e}"
                    )

        self._sessions.clear()
        logger.info(f"[ChatHistoryManager.clear_all] 已清空 {count} 个会话")
        return count

    def analyze_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """分析指定会话

        Args:
            session_id: 会话ID

        Returns:
            分析结果字典，会话不存在返回None
        """
        session = self.get_session(session_id)
        if not session:
            logger.warning(
                f"[ChatHistoryManager.analyze_session] "
                f"会话不存在: {session_id}"
            )
            return None
        return self.analyzer.analyze_session(session)

    def get_overall_stats(self) -> Dict[str, Any]:
        """获取整体统计信息

        Returns:
            包含所有统计数据的字典
        """
        sessions = self.get_all_sessions()
        if not sessions:
            return {
                'total_sessions': 0,
                'total_messages': 0,
                'total_user_questions': 0,
                'avg_answer_quality': 0.0,
                'category_distribution': {},
                'top_keywords': [],
                'active_days': 0,
                'avg_messages_per_session': 0.0
            }

        total_messages = sum(len(s.messages) for s in sessions)
        total_questions = sum(
            1 for s in sessions for m in s.messages if m.role == 'user'
        )

        # 质量统计
        all_qualities = []
        for s in sessions:
            analysis = self.analyzer.analyze_session(s)
            if analysis['avg_answer_quality'] > 0:
                all_qualities.append(analysis['avg_answer_quality'])
        avg_quality = np.mean(all_qualities) if all_qualities else 0.0

        # 分类分布
        all_categories = []
        for s in sessions:
            for m in s.messages:
                if m.role == 'user':
                    all_categories.append(
                        self.analyzer.classify_question(m.content)
                    )
        category_dist = dict(Counter(all_categories))

        # 活跃天数
        dates = set(
            datetime.fromtimestamp(s.created_at).date() for s in sessions
        )

        # 热门关键词
        all_content = " ".join(
            m.content for s in sessions for m in s.messages
        )
        top_keywords = self.analyzer._extract_keywords(all_content, top_n=10)

        return {
            'total_sessions': len(sessions),
            'total_messages': total_messages,
            'total_user_questions': total_questions,
            'avg_answer_quality': round(float(avg_quality), 2),
            'category_distribution': category_dist,
            'top_keywords': top_keywords,
            'active_days': len(dates),
            'avg_messages_per_session': round(total_messages / len(sessions), 1)
        }

    def export_sessions(
        self,
        session_ids: Optional[List[str]] = None,
        export_format: str = 'json',
        with_analysis: bool = True
    ) -> str:
        """导出会话

        Args:
            session_ids: 要导出的会话ID列表，None表示导出全部
            export_format: 导出格式: 'json', 'markdown', 'csv'
            with_analysis: 是否包含智能分析（仅Markdown支持）

        Returns:
            导出内容字符串

        Raises:
            ValueError: 当格式不支持时
        """
        if session_ids is None:
            sessions = self.get_all_sessions()
        else:
            sessions = [
                self._sessions[sid]
                for sid in session_ids
                if sid in self._sessions
            ]

        if export_format == 'json':
            return self.exporter.export_to_json(sessions)
        elif export_format == 'markdown':
            return self.exporter.export_to_markdown(
                sessions,
                with_analysis=with_analysis
            )
        elif export_format == 'csv':
            return self.exporter.export_to_csv(sessions)
        else:
            raise ValueError(f"不支持的导出格式: {export_format}")
