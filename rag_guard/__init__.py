# This file makes the directory a Python package

from .masker import DataMasker
from .pipeline import RAGGuard
from .guard import RAGGuardian
from .config import GuardConfig
from .ner import BaseNER, RegexNER, LLMNER, HybridNER, Entity
from .exceptions import (
    RAGGuardError, 
    MaskingError, 
    UnmaskingError, 
    ConfigError, 
    LLMCallError,
    NERError
)

# 主要导出的类
__all__ = [
    'RAGGuardian',  # 主要入口类
    'DataMasker', 
    'RAGGuard', 
    'GuardConfig',
    'BaseNER',
    'RegexNER',
    'LLMNER', 
    'HybridNER',
    'Entity',
    'RAGGuardError', 
    'MaskingError', 
    'UnmaskingError', 
    'ConfigError', 
    'LLMCallError',
    'NERError'
]

# 版本信息
__version__ = '1.1.0'

# 便捷函数
def create_guardian(salt: str = None, ner_strategy: str = 'regex_only', **kwargs) -> RAGGuardian:
    """
    快速创建 RAGGuardian 实例的便捷函数
    
    Args:
        salt: 加密盐值
        ner_strategy: NER策略，'regex_only', 'llm_only', 'hybrid'
        **kwargs: 其他配置参数
    
    Returns:
        RAGGuardian 实例
    """
    config_params = kwargs.copy()
    if salt:
        config_params['salt'] = salt
    config_params['ner_strategy'] = ner_strategy
    
    config = GuardConfig(**config_params)
    return RAGGuardian(config)