import re
import hashlib
import base64
from typing import Dict, Tuple, List, Pattern, Optional, Callable
from .exceptions import MaskingError, UnmaskingError, NERError
from .ner import BaseNER, RegexNER, LLMNER, HybridNER, Entity


class DataMasker:
    """
    将文本中的敏感信息替换为稳定的占位符，并记录映射以便后续回填。
    支持正则表达式和LLM两种实体识别方式。
    占位符格式：<RG:{KIND}:{SHORT_HASH}>
    """
    def __init__(self, 
                 salt: str = "your_salt_here", 
                 custom_patterns: Optional[Dict[str, str]] = None,
                 llm_caller: Optional[Callable[[str], str]] = None,
                 use_llm_ner: bool = False,
                 llm_entity_types: Optional[List[str]] = None,
                 ner_strategy: str = 'regex_only'):
        """
        初始化DataMasker
        
        Args:
            salt: 加密盐值
            custom_patterns: 自定义正则表达式模式
            llm_caller: LLM调用函数
            use_llm_ner: 是否使用LLM进行实体识别
            llm_entity_types: LLM识别的实体类型列表
            ner_strategy: 实体识别策略，'regex_only', 'llm_only', 'hybrid'
        """
        self.salt = salt
        self.ner_strategy = ner_strategy
        self._validate_salt()
        
        # 默认正则表达式模式
        default_patterns = {
            "EMAIL": r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
            "PHONE": r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}\b|\b1[3-9]\d{9}\b",
            "IDCN": r"\b\d{17}[\dXx]\b|\b\d{15}\b",  # 中国身份证（18位或15位）
            "IPV4": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
            "BANK_ACCOUNT": r"\b\d{16,19}\b",  # 银行账号（16-19位数字）
            "URL": r"https?://[^\s<>\"]+",
        }
        
        if custom_patterns:
            default_patterns.update(custom_patterns)
        
        self.last_entities: List[Entity] = []  # 添加这行
        
        # 初始化实体识别器
        self._init_ner(default_patterns, llm_caller, use_llm_ner, llm_entity_types)
    
    def _init_ner(self, patterns: Dict[str, str], 
                  llm_caller: Optional[Callable], 
                  use_llm_ner: bool,
                  llm_entity_types: Optional[List[str]]):
        """初始化实体识别器"""
        try:
            if self.ner_strategy == 'regex_only':
                self.ner = RegexNER(patterns)
            elif self.ner_strategy == 'llm_only':
                if not llm_caller:
                    raise NERError("使用LLM策略时必须提供llm_caller")
                self.ner = LLMNER(llm_caller, llm_entity_types)
            elif self.ner_strategy == 'hybrid':
                regex_ner = RegexNER(patterns)
                llm_ner = LLMNER(llm_caller, llm_entity_types) if llm_caller else None
                self.ner = HybridNER(regex_ner, llm_ner)
            else:
                raise NERError(f"不支持的NER策略: {self.ner_strategy}")
        except Exception as e:
            raise MaskingError(f"初始化实体识别器失败: {e}")
    
    def _validate_salt(self):
        """验证盐值"""
        if not self.salt or len(self.salt) < 8:
            raise MaskingError("盐值长度至少需要8个字符")
    
    def add_pattern(self, name: str, pattern: str) -> None:
        """动态添加正则表达式模式（仅适用于包含RegexNER的策略）"""
        if hasattr(self.ner, 'patterns'):
            try:
                self.ner.patterns[name] = re.compile(pattern)
            except re.error as e:
                raise MaskingError(f"添加模式 {name} 失败: {e}")
        elif hasattr(self.ner, 'regex_ner') and self.ner.regex_ner:
            try:
                self.ner.regex_ner.patterns[name] = re.compile(pattern)
            except re.error as e:
                raise MaskingError(f"添加模式 {name} 失败: {e}")
        else:
            raise MaskingError("当前NER策略不支持添加正则表达式模式")
    
    def _short_hash(self, kind: str, value: str) -> str:
        try:
            h = hashlib.sha1((self.salt + kind + value).encode("utf-8")).digest()
            return base64.b32encode(h).decode("ascii").rstrip("=").lower()[:8]
        except Exception as e:
            raise MaskingError(f"生成哈希失败: {e}")

    def _token(self, kind: str, value: str) -> str:
        return f"<RG:{kind}:{self._short_hash(kind, value)}>"

    def mask(self, text: str) -> Tuple[str, Dict[str, str], List[Entity]]:
        """
        将文本中的敏感信息替换为占位符。
        返回 (masked_text, mapping, entities)，其中：
        - masked_text: 脱敏后的文本
        - mapping: {token: original_value} 映射
        - entities: 处理的实体列表
        """
        if not isinstance(text, str):
            raise MaskingError("输入必须是字符串类型")
        
        try:
            # 使用NER提取实体
            entities = self.ner.extract_entities(text)
            self.last_entities = entities  # 保存实体信息
            
            # 按位置倒序排序，避免替换时位置偏移
            entities.sort(key=lambda x: x.start, reverse=True)
            
            mapping: Dict[str, str] = {}
            masked_text = text
            
            # 替换实体为占位符
            for entity in entities:
                token = self._token(entity.label, entity.text)
                if token not in mapping:
                    mapping[token] = entity.text
                
                # 替换文本
                masked_text = (
                    masked_text[:entity.start] + 
                    token + 
                    masked_text[entity.end:]
                )
            
            return masked_text, mapping, entities
            
        except Exception as e:
            raise MaskingError(f"脱敏处理失败: {e}")

    def get_last_entities(self) -> List[Entity]:
        """获取最后一次脱敏处理的实体"""
        return self.last_entities

    def unmask(self, text: str, mapping: Dict[str, str]) -> str:
        """
        将 text 中出现的占位符替换回原文。仅替换 mapping 中存在的 token，避免误替换。
        """
        if not isinstance(text, str):
            raise UnmaskingError("输入必须是字符串类型")
        
        if not isinstance(mapping, dict):
            raise UnmaskingError("映射表必须是字典类型")
        
        try:
            # 构造安全的替换（按 token 降序长度排序，避免子串影响）
            tokens_sorted = sorted(mapping.keys(), key=len, reverse=True)
            result = text
            for tk in tokens_sorted:
                # 使用 re.escape 确保尖括号等安全替换
                result = re.sub(re.escape(tk), mapping[tk], result)
            return result
        except Exception as e:
            raise UnmaskingError(f"回填处理失败: {e}")
    
    def get_entities(self, text: str) -> List[Entity]:
        """获取文本中的实体（不进行脱敏）"""
        return self.ner.extract_entities(text)
    
    def get_ner_info(self) -> Dict[str, any]:
        """获取当前NER配置信息"""
        info = {
            'strategy': self.ner_strategy,
            'ner_type': type(self.ner).__name__
        }
        
        if hasattr(self.ner, 'patterns'):
            info['regex_patterns'] = list(self.ner.patterns.keys())
        
        if hasattr(self.ner, 'entity_types'):
            info['llm_entity_types'] = self.ner.entity_types
        
        return info