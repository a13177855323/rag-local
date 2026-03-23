"""
对话功能配置管理模块
集中管理对话历史相关的所有配置参数
支持配置验证和默认值管理
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class ConversationConfig:
    """对话功能配置类"""

    # 存储配置
    storage_dir: str = "./data/conversations"
    sessions_filename: str = "sessions.json"

    # 分类配置
    code_keywords: List[str] = field(default_factory=lambda: [
        '代码', 'code', '函数', 'function', '类', 'class', '方法', 'method',
        'python', 'java', 'javascript', 'js', 'py', 'cpp', 'c++',
        'bug', '报错', 'error', 'exception', 'debug', '调试'
    ])

    how_to_keywords: List[str] = field(default_factory=lambda: [
        '如何', '怎么', 'how to', '步骤', '教程', 'guide', '步骤', '怎样做'
    ])

    debug_keywords: List[str] = field(default_factory=lambda: [
        '报错', '错误', 'error', 'exception', '失败', '无法', '不能', 'debug', '调试', 'fix'
    ])

    compare_keywords: List[str] = field(default_factory=lambda: [
        '区别', '对比', '比较', 'difference', 'compare', 'vs', 'versus', '哪个好'
    ])

    concept_keywords: List[str] = field(default_factory=lambda: [
        '什么是', '概念', '原理', '机制', '什么是', 'what is', 'explain', '定义'
    ])

    # 质量评分配置
    quality_base_score: float = 70.0
    quality_length_threshold_1: int = 200
    quality_length_threshold_2: int = 500
    quality_length_bonus_1: float = 10.0
    quality_length_bonus_2: float = 10.0
    quality_sources_bonus: float = 10.0
    quality_slow_response_threshold: int = 5000
    quality_very_slow_response_threshold: int = 10000
    quality_slow_penalty: float = 10.0

    # 分析配置
    summary_max_questions: int = 5
    summary_max_topics: int = 3
    min_code_block_lines: int = 3
    max_code_block_lines: int = 20

    # 导出配置
    export_max_sources: int = 3
    export_answer_preview_length: int = 100

    def __post_init__(self):
        """初始化后处理，确保存储目录存在"""
        os.makedirs(self.storage_dir, exist_ok=True)

    @property
    def sessions_file_path(self) -> str:
        """获取会话文件完整路径"""
        return os.path.join(self.storage_dir, self.sessions_filename)

    def get_category_keywords(self, category: str) -> List[str]:
        """
        获取指定分类的关键词列表

        Args:
            category: 分类名称

        Returns:
            关键词列表
        """
        keyword_map = {
            'code': self.code_keywords,
            'how_to': self.how_to_keywords,
            'debug': self.debug_keywords,
            'compare': self.compare_keywords,
            'concept': self.concept_keywords
        }
        return keyword_map.get(category, [])

    def validate(self) -> bool:
        """
        验证配置有效性

        Returns:
            配置是否有效
        """
        # 验证存储目录可写
        if not os.access(self.storage_dir, os.W_OK):
            return False

        # 验证数值参数
        if self.quality_base_score < 0 or self.quality_base_score > 100:
            return False

        if self.quality_length_threshold_1 >= self.quality_length_threshold_2:
            return False

        return True


class ConfigManager:
    """配置管理器 - 单例模式"""
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """初始化配置"""
        self._config = ConversationConfig()

    @property
    def config(self) -> ConversationConfig:
        """获取配置对象"""
        return self._config

    def reload_config(self, **kwargs):
        """
        重新加载配置

        Args:
            **kwargs: 配置参数
        """
        self._config = ConversationConfig(**kwargs)


# 全局单例实例
_config_manager = None


def get_config_manager() -> ConfigManager:
    """获取 ConfigManager 单例实例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config() -> ConversationConfig:
    """便捷函数：获取配置对象"""
    return get_config_manager().config
