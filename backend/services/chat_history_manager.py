#!/usr/bin/env python3
"""对话历史管理核心模块
支持对话记录存储、智能分析、多格式导出等功能
"""
import json
import csv
import time
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import numpy as np
from backend.config import settings


class ChatMessage:
    """对话消息类"""
    
    def __init__(self, 
                 role: str, 
                 content: str, 
                 message_id: str = None,
                 timestamp: float = None,
                 metadata: Dict[str, Any] = None):
        self.message_id = message_id or str(uuid.uuid4())
        self.role = role  # 'user' or 'assistant'
        self.content = content
        self.timestamp = timestamp or time.time()
        self.metadata = metadata or {}  # 存储额外信息：tokens、sources等
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'message_id': self.message_id,
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M:%S'),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatMessage':
        """从字典创建对象"""
        return cls(
            role=data['role'],
            content=data['content'],
            message_id=data.get('message_id'),
            timestamp=data.get('timestamp'),
            metadata=data.get('metadata', {})
        )


class ChatSession:
    """对话会话类"""
    
    def __init__(self, 
                 session_id: str = None,
                 title: str = None,
                 created_at: float = None,
                 updated_at: float = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.title = title or f"对话 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        self.created_at = created_at or time.time()
        self.updated_at = updated_at or time.time()
        self.messages: List[ChatMessage] = []
    
    def add_message(self, message: ChatMessage):
        """添加消息"""
        self.messages.append(message)
        self.updated_at = time.time()
        # 自动更新标题（使用第一条用户消息）
        if not self.title and message.role == 'user':
            self.title = message.content[:30] + ("..." if len(message.content) > 30 else "")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'session_id': self.session_id,
            'title': self.title,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'created_datetime': datetime.fromtimestamp(self.created_at).strftime('%Y-%m-%d %H:%M:%S'),
            'message_count': len(self.messages),
            'messages': [msg.to_dict() for msg in self.messages]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatSession':
        """从字典创建对象"""
        session = cls(
            session_id=data.get('session_id'),
            title=data.get('title'),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at')
        )
        session.messages = [ChatMessage.from_dict(msg) for msg in data.get('messages', [])]
        return session


class ChatHistoryAnalyzer:
    """对话历史智能分析器"""
    
    # 问题分类关键词
    CATEGORY_KEYWORDS = {
        '代码开发': ['代码', '编程', '函数', '类', 'python', 'java', 'c++', 'bug', '调试', '算法', '实现', '开发', '程序'],
        '文档查询': ['文档', '说明', '怎么用', '使用方法', '参数', '配置', '安装', '部署', '教程', '示例'],
        '理论知识': ['什么是', '解释', '原理', '概念', '理论', '区别', '对比', '为什么', '如何理解'],
        '问题解决': ['错误', '问题', '无法', '失败', '异常', '报错', '怎么办', '解决', '修复', '卡住'],
        '其他': []
    }
    
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model
    
    def generate_summary(self, session: ChatSession, max_sentences: int = 3) -> str:
        """生成对话摘要
        使用抽取式摘要方法，基于关键词和位置权重
        """
        if not session.messages:
            return "暂无对话内容"
        
        # 获取所有用户问题和助手回答
        user_messages = [msg for msg in session.messages if msg.role == 'user']
        assistant_messages = [msg for msg in session.messages if msg.role == 'assistant']
        
        if not user_messages:
            return "暂无有效对话内容"
        
        # 收集重要句子
        important_points = []
        
        # 1. 第一个问题（通常是核心问题）
        first_question = user_messages[0].content
        important_points.append(f"用户询问：「{first_question[:50]}{'...' if len(first_question) > 50 else ''}」")
        
        # 2. 关键主题词提取
        all_content = " ".join([msg.content for msg in session.messages])
        keywords = self._extract_keywords(all_content, top_n=5)
        
        if keywords:
            important_points.append(f"主要涉及主题：{', '.join(keywords)}")
        
        # 3. 对话轮次统计
        round_count = len(user_messages)
        if round_count > 1:
            important_points.append(f"共进行了 {round_count} 轮问答互动")
        
        # 4. 最后一个重要回复
        if assistant_messages:
            last_answer = assistant_messages[-1].content
            if len(last_answer) > 30:
                important_points.append(f"最终解答摘要：{last_answer[:60]}{'...' if len(last_answer) > 60 else ''}")
        
        return "；".join(important_points[:max_sentences])
    
    def _extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """简单关键词提取"""
        # 分词（简单按空格和标点分割）
        import re
        words = re.findall(r'[\u4e00-\u9fa5a-zA-Z]+', text.lower())
        
        # 停用词过滤
        stop_words = {'的', '是', '在', '了', '和', '与', '或', '这', '那', '有', '能', '会', '要', '什么', '怎么', '如何', '请问', 'i', 'the', 'a', 'an', 'is', 'are'}
        filtered = [w for w in words if len(w) > 1 and w not in stop_words]
        
        # 词频统计
        counter = Counter(filtered)
        return [word for word, _ in counter.most_common(top_n)]
    
    def classify_question(self, question: str) -> str:
        """问题分类"""
        question_lower = question.lower()
        
        # 基于关键词匹配
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            if category == '其他':
                continue
            for keyword in keywords:
                if keyword in question_lower:
                    return category
        
        return '其他'
    
    def calculate_answer_quality(self, question: str, answer: str) -> Dict[str, Any]:
        """回答质量评分
        返回：分数(0-1)、评分维度、建议
        """
        scores = {}
        
        # 1. 长度得分（避免过短或过长）
        answer_len = len(answer)
        if answer_len < 20:
            len_score = 0.3
        elif answer_len > 500:
            len_score = 0.7
        else:
            len_score = min(1.0, answer_len / 100)
        scores['长度合理性'] = len_score
        
        # 2. 内容丰富度（代码块、列表等结构化内容）
        has_code = '```' in answer or '代码' in answer
        has_list = '\n- ' in answer or '\n* ' in answer or '\n1.' in answer
        structure_score = 0.4 + (0.3 if has_code else 0) + (0.3 if has_list else 0)
        scores['内容结构化'] = structure_score
        
        # 3. 相关性（简单检查是否包含问题中的关键词）
        question_words = set(self._extract_keywords(question, top_n=10))
        answer_words = set(self._extract_keywords(answer, top_n=20))
        overlap = len(question_words & answer_words)
        relevance_score = overlap / len(question_words) if question_words else 0.5
        scores['内容相关性'] = relevance_score
        
        # 4. 帮助性词汇检测
        helpful_words = ['可以', '需要', '建议', '应该', '推荐', '注意', '例如', '示例']
        helpful_count = sum(1 for word in helpful_words if word in answer)
        helpful_score = min(1.0, 0.5 + helpful_count * 0.1)
        scores['内容帮助性'] = helpful_score
        
        # 综合得分
        weights = {'长度合理性': 0.2, '内容结构化': 0.25, '内容相关性': 0.35, '内容帮助性': 0.2}
        total_score = sum(scores[k] * weights[k] for k in scores)
        
        # 生成建议
        suggestions = []
        if len_score < 0.5:
            suggestions.append("回答内容偏短，建议补充更多细节")
        if structure_score < 0.6:
            suggestions.append("建议使用列表、代码块等结构化格式")
        if relevance_score < 0.3:
            suggestions.append("回答可能与问题相关性不足")
        
        return {
            'total_score': round(total_score, 2),
            'dimension_scores': scores,
            'level': self._score_to_level(total_score),
            'suggestions': suggestions
        }
    
    def _score_to_level(self, score: float) -> str:
        """分数转等级"""
        if score >= 0.8:
            return '优秀'
        elif score >= 0.6:
            return '良好'
        elif score >= 0.4:
            return '一般'
        else:
            return '待提升'
    
    def analyze_session(self, session: ChatSession) -> Dict[str, Any]:
        """完整分析对话会话"""
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
        
        avg_quality = np.mean(quality_scores) if quality_scores else 0
        
        # 对话时长
        duration = 0
        if len(session.messages) >= 2:
            start_time = session.messages[0].timestamp
            end_time = session.messages[-1].timestamp
            duration = end_time - start_time
        
        return {
            'summary': self.generate_summary(session),
            'category_distribution': category_stats,
            'avg_answer_quality': round(float(avg_quality), 2),
            'quality_level': self._score_to_level(float(avg_quality)),
            'total_rounds': len(user_questions),
            'duration_seconds': round(duration),
            'duration_formatted': str(timedelta(seconds=round(duration))).split('.')[0],
            'keywords': self._extract_keywords(
                " ".join([msg.content for msg in session.messages]), 
                top_n=8
            )
        }


class ChatHistoryExporter:
    """对话历史导出器"""
    
    def export_to_json(self, sessions: List[ChatSession], filename: str = None) -> str:
        """导出为JSON格式"""
        data = {
            'export_time': datetime.now().isoformat(),
            'version': '1.0',
            'total_sessions': len(sessions),
            'sessions': [session.to_dict() for session in sessions]
        }
        
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        
        return json.dumps(data, ensure_ascii=False, indent=2)
    
    def export_to_markdown(self, sessions: List[ChatSession], 
                           with_analysis: bool = True,
                           filename: str = None) -> str:
        """导出为Markdown报告格式"""
        analyzer = ChatHistoryAnalyzer()
        
        md_content = []
        md_content.append("# 对话历史报告")
        md_content.append(f"\n**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        md_content.append(f"**会话总数**: {len(sessions)}")
        
        # 总统计
        if sessions:
            total_messages = sum(len(s.messages) for s in sessions)
            md_content.append(f"**消息总数**: {total_messages}")
            md_content.append(f"**最早会话**: {datetime.fromtimestamp(min(s.created_at for s in sessions)).strftime('%Y-%m-%d %H:%M')}")
            md_content.append(f"**最晚会话**: {datetime.fromtimestamp(max(s.updated_at for s in sessions)).strftime('%Y-%m-%d %H:%M')}")
        
        md_content.append("\n---\n")
        
        # 每个会话详情
        for i, session in enumerate(sessions, 1):
            md_content.append(f"## 会话 {i}: {session.title}")
            md_content.append(f"- **会话ID**: {session.session_id}")
            md_content.append(f"- **创建时间**: {datetime.fromtimestamp(session.created_at).strftime('%Y-%m-%d %H:%M:%S')}")
            md_content.append(f"- **消息数量**: {len(session.messages)}")
            
            if with_analysis:
                analysis = analyzer.analyze_session(session)
                md_content.append(f"- **对话摘要**: {analysis['summary']}")
                md_content.append(f"- **平均回答质量**: {analysis['avg_answer_quality']} ({analysis['quality_level']})")
                md_content.append(f"- **问题分类分布**: {analysis['category_distribution']}")
                md_content.append(f"- **关键词**: {', '.join(analysis['keywords'])}")
            
            md_content.append("\n### 对话详情\n")
            
            for msg in session.messages:
                role_icon = "👤 用户" if msg.role == 'user' else "🤖 助手"
                md_content.append(f"**{role_icon}** ({datetime.fromtimestamp(msg.timestamp).strftime('%H:%M:%S')})")
                md_content.append(f"\n{msg.content}\n")
            
            md_content.append("\n---\n")
        
        result = "\n".join(md_content)
        
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(result)
        
        return result
    
    def export_to_csv(self, sessions: List[ChatSession], filename: str = None) -> str:
        """导出为CSV统计格式"""
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # 统计表头
        writer.writerow([
            '会话ID', '会话标题', '创建时间', '消息数量',
            '用户问题数', '平均回答质量', '主要分类',
            '对话时长(秒)', '关键词'
        ])
        
        analyzer = ChatHistoryAnalyzer()
        
        for session in sessions:
            analysis = analyzer.analyze_session(session)
            user_count = sum(1 for m in session.messages if m.role == 'user')
            main_category = max(analysis['category_distribution'].items(), 
                              key=lambda x: x[1])[0] if analysis['category_distribution'] else '未知'
            
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
            with open(filename, 'w', encoding='utf-8-sig', newline='') as f:
                f.write(result)
        
        return result


class ChatHistoryManager:
    """对话历史管理器 - 核心入口类"""
    
    def __init__(self, storage_path: str = None):
        self.storage_path = storage_path or settings.CHAT_HISTORY_PATH
        self._sessions: Dict[str, ChatSession] = {}  # session_id -> ChatSession
        self.analyzer = ChatHistoryAnalyzer()
        self.exporter = ChatHistoryExporter()
        self._load_sessions()
    
    def _load_sessions(self):
        """加载已存储的会话"""
        import os
        os.makedirs(self.storage_path, exist_ok=True)
        
        for filename in os.listdir(self.storage_path):
            if filename.endswith('.json'):
                filepath = os.path.join(self.storage_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        session = ChatSession.from_dict(data)
                        self._sessions[session.session_id] = session
                except Exception as e:
                    print(f"加载会话失败 {filename}: {e}")
    
    def _save_session(self, session: ChatSession):
        """保存会话到文件"""
        import os
        os.makedirs(self.storage_path, exist_ok=True)
        
        filepath = os.path.join(self.storage_path, f"{session.session_id}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)
    
    def create_session(self, title: str = None) -> ChatSession:
        """创建新会话"""
        session = ChatSession(title=title)
        self._sessions[session.session_id] = session
        self._save_session(session)
        return session
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """获取会话"""
        return self._sessions.get(session_id)
    
    def get_all_sessions(self, limit: int = None) -> List[ChatSession]:
        """获取所有会话，按更新时间倒序"""
        sessions = sorted(self._sessions.values(), 
                         key=lambda s: s.updated_at, 
                         reverse=True)
        if limit:
            sessions = sessions[:limit]
        return sessions
    
    def add_message(self, session_id: str, role: str, content: str, 
                   metadata: Dict[str, Any] = None) -> Optional[ChatMessage]:
        """添加消息到会话"""
        session = self._sessions.get(session_id)
        if not session:
            session = self.create_session()
        
        message = ChatMessage(role=role, content=content, metadata=metadata)
        session.add_message(message)
        self._save_session(session)
        return message
    
    def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            import os
            filepath = os.path.join(self.storage_path, f"{session_id}.json")
            if os.path.exists(filepath):
                os.remove(filepath)
            return True
        return False
    
    def clear_all(self) -> int:
        """清空所有会话"""
        count = len(self._sessions)
        import os
        for session_id in list(self._sessions.keys()):
            filepath = os.path.join(self.storage_path, f"{session_id}.json")
            if os.path.exists(filepath):
                os.remove(filepath)
        self._sessions.clear()
        return count
    
    def analyze_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """分析指定会话"""
        session = self.get_session(session_id)
        if not session:
            return None
        return self.analyzer.analyze_session(session)
    
    def get_overall_stats(self) -> Dict[str, Any]:
        """获取整体统计信息"""
        sessions = self.get_all_sessions()
        if not sessions:
            return {
                'total_sessions': 0,
                'total_messages': 0,
                'total_user_questions': 0,
                'avg_answer_quality': 0,
                'category_distribution': {},
                'top_keywords': [],
                'active_days': 0
            }
        
        total_messages = sum(len(s.messages) for s in sessions)
        total_questions = sum(1 for s in sessions for m in s.messages if m.role == 'user')
        
        # 质量统计
        all_qualities = []
        for s in sessions:
            analysis = self.analyzer.analyze_session(s)
            if analysis['avg_answer_quality'] > 0:
                all_qualities.append(analysis['avg_answer_quality'])
        avg_quality = np.mean(all_qualities) if all_qualities else 0
        
        # 分类分布
        all_categories = []
        for s in sessions:
            for m in s.messages:
                if m.role == 'user':
                    all_categories.append(self.analyzer.classify_question(m.content))
        category_dist = dict(Counter(all_categories))
        
        # 活跃天数
        dates = set(datetime.fromtimestamp(s.created_at).date() for s in sessions)
        
        # 热门关键词
        all_content = " ".join(m.content for s in sessions for m in s.messages)
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
    
    def export_sessions(self, session_ids: List[str] = None, 
                       export_format: str = 'json',
                       with_analysis: bool = True) -> str:
        """导出会话
        
        Args:
            session_ids: 要导出的会话ID列表，None表示导出全部
            export_format: 导出格式: 'json', 'markdown', 'csv'
            with_analysis: 是否包含智能分析
        """
        if session_ids is None:
            sessions = self.get_all_sessions()
        else:
            sessions = [self._sessions[sid] for sid in session_ids if sid in self._sessions]
        
        if export_format == 'json':
            return self.exporter.export_to_json(sessions)
        elif export_format == 'markdown':
            return self.exporter.export_to_markdown(sessions, with_analysis=with_analysis)
        elif export_format == 'csv':
            return self.exporter.export_to_csv(sessions)
        else:
            raise ValueError(f"不支持的导出格式: {export_format}")
