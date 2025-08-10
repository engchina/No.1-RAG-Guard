from typing import Dict, Optional, List, Callable
from dataclasses import dataclass, field
import logging
import os


@dataclass
class GuardConfig:
    """
    RAG Guard 配置类
    """
    # 核心配置
    salt: str = field(default_factory=lambda: os.getenv('RAG_GUARD_SALT', 'default_salt_change_me'))
    custom_patterns: Optional[Dict[str, str]] = None
    
    # NER配置
    ner_strategy: str = 'regex_only'  # 'regex_only', 'llm_only', 'hybrid'
    use_llm_ner: bool = False
    llm_entity_types: Optional[List[str]] = None
    llm_confidence_threshold: float = 0.7
    
    # 提示模板配置
    prompt_template: Optional[str] = None
    
    # 日志配置
    enable_logging: bool = True
    log_level: int = logging.INFO
    
    # 调试配置
    include_debug_info: bool = False
    
    # 性能配置
    max_chunk_length: int = 10000
    max_chunks_count: int = 50
    
    def __post_init__(self):
        """初始化后处理"""
        if self.custom_patterns is None:
            self.custom_patterns = {}
        
        if self.llm_entity_types is None:
            self.llm_entity_types = [
                "PERSON",        # 人名
                "EMAIL",         # 邮箱
                "PHONE",         # 电话号码
                "ID_NUMBER",     # 身份证号
                "IDCN",          # 中国身份证
                "CREDIT_CARD",   # 信用卡号
                "BANK_ACCOUNT",  # 银行账号
                "ADDRESS",       # 地址
                "ORGANIZATION",  # 组织机构
                "COMPANY",       # 公司名称
                "LICENSE_PLATE", # 车牌号
                "IP_ADDRESS",    # IP地址
                "URL",           # 网址
                "LOCATION",      # 地理位置
                "FINANCIAL_INFO" # 其他金融信息
            ]
        
        # 验证配置
        if len(self.salt) < 8:
            raise ValueError("盐值长度至少需要8个字符")
        
        if self.ner_strategy not in ['regex_only', 'llm_only', 'hybrid']:
            raise ValueError(f"不支持的NER策略: {self.ner_strategy}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'GuardConfig':
        """从字典创建配置"""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'salt': '[HIDDEN]',
            'custom_patterns': list(self.custom_patterns.keys()) if self.custom_patterns else [],
            'ner_strategy': self.ner_strategy,
            'use_llm_ner': self.use_llm_ner,
            'llm_entity_types': self.llm_entity_types,
            'llm_confidence_threshold': self.llm_confidence_threshold,
            'enable_logging': self.enable_logging,
            'log_level': self.log_level,
            'include_debug_info': self.include_debug_info,
            'max_chunk_length': self.max_chunk_length,
            'max_chunks_count': self.max_chunks_count
        }