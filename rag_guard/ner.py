from typing import List, Dict, Tuple, Optional, Callable, Any
from abc import ABC, abstractmethod
import json
import re
from .exceptions import NERError


class Entity:
    """实体类，表示识别出的敏感实体"""
    def __init__(self, text: str, label: str, start: int, end: int, confidence: float = 1.0):
        self.text = text
        self.label = label  # 实体类型，如 PERSON, EMAIL, PHONE 等
        self.start = start  # 在原文中的起始位置
        self.end = end     # 在原文中的结束位置
        self.confidence = confidence  # 置信度
    
    def __repr__(self):
        return f"Entity(text='{self.text}', label='{self.label}', start={self.start}, end={self.end}, confidence={self.confidence})"


class BaseNER(ABC):
    """实体识别基类"""
    
    @abstractmethod
    def extract_entities(self, text: str) -> List[Entity]:
        """提取文本中的实体"""
        pass


class RegexNER(BaseNER):
    """基于正则表达式的实体识别器"""
    
    def __init__(self, patterns: Dict[str, str]):
        self.patterns = {}
        for label, pattern in patterns.items():
            try:
                self.patterns[label] = re.compile(pattern)
            except re.error as e:
                raise NERError(f"正则表达式编译失败 {label}: {e}")
    
    def extract_entities(self, text: str) -> List[Entity]:
        entities = []
        for label, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                entity = Entity(
                    text=match.group(0),
                    label=label,
                    start=match.start(),
                    end=match.end(),
                    confidence=1.0
                )
                entities.append(entity)
        return entities


class LLMNER(BaseNER):
    """基于LLM的实体识别器"""
    
    def __init__(self, 
                 llm_caller: Callable[[str], str],
                 entity_types: Optional[List[str]] = None,
                 confidence_threshold: float = 0.7,
                 max_text_length: int = 2000):
        """
        初始化LLM实体识别器
        
        Args:
            llm_caller: LLM调用函数，接收prompt返回response
            entity_types: 要识别的实体类型列表，如果为None则使用默认类型
            confidence_threshold: 置信度阈值，低于此值的实体将被过滤
            max_text_length: 最大文本长度，超过将被截断
        """
        self.llm_caller = llm_caller
        self.confidence_threshold = confidence_threshold
        self.max_text_length = max_text_length
        
        # 默认实体类型
        self.entity_types = entity_types or [
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
            "FINANCIAL_INFO" # 金融信息
        ]
    
    def _build_prompt(self, text: str) -> str:
        """构建LLM提示词"""
        entity_types_str = "、".join(self.entity_types)
        
        prompt = f"""请识别以下文本中的敏感实体信息。

需要识别的实体类型：{entity_types_str}

实体类型说明和识别要求：
- PERSON: 人名、姓名（如：张三、李四、王五）
- EMAIL: 电子邮箱地址（如：user@example.com、test@company.cn）
- PHONE: 电话号码、手机号码、座机号码，包括：
  * 手机号：13800138000、138-0013-8000
  * 座机号：010-62345678、021-58888888
  * 带区号的完整号码
- ID_NUMBER/IDCN: 身份证号、护照号等身份证件（18位或15位数字）
- BANK_ACCOUNT: 银行账户号码、卡号（16-19位数字）
- CREDIT_CARD: 信用卡号（带空格或连字符的16位数字）
- ADDRESS: 详细地址、街道地址（如：北京市朝阳区建国路88号）
- ORGANIZATION/COMPANY: 公司名称、组织机构名称（如：北京科技有限公司）
- IP_ADDRESS: IP地址（如：192.168.1.1）
- URL: 网址链接（如：https://example.com）
- LOCATION: 地理位置、地名
- LICENSE_PLATE: 车牌号
- FINANCIAL_INFO: 其他金融敏感信息

特别注意：
1. EMAIL必须包含@符号和域名
2. PHONE包括所有数字组合的电话号码（10位以上）
3. 银行账号通常是连续的16-19位数字
4. 地址信息要完整识别
5. 公司名称通常包含"公司"、"企业"、"集团"等关键词

文本内容：
{text}

请仔细分析文本，以JSON格式返回识别结果：
{{
  "entities": [
    {{
      "text": "实体的完整文本",
      "label": "实体类型",
      "start": 起始位置,
      "end": 结束位置,
      "confidence": 置信度
    }}
  ]
}}

要求：
1. 只返回JSON格式，不要其他解释
2. 起始和结束位置是字符索引（从0开始）
3. 置信度请根据识别的确定性给出0.7-1.0之间的数值
4. 如果没有找到实体，返回空数组
5. 必须识别所有EMAIL和PHONE类型的实体
6. 确保位置索引准确对应文本内容"""
        
        return prompt
    
    def extract_entities(self, text: str) -> List[Entity]:
        """使用LLM提取实体"""
        if len(text) > self.max_text_length:
            text = text[:self.max_text_length] + "...[截断]"
        
        try:
            prompt = self._build_prompt(text)
            response = self.llm_caller(prompt)
            
            # 解析LLM返回的JSON
            entities = self._parse_llm_response(response, text)
            
            # 过滤低置信度实体
            filtered_entities = [
                entity for entity in entities 
                if entity.confidence >= self.confidence_threshold
            ]
            
            return filtered_entities
            
        except Exception as e:
            raise NERError(f"LLM实体识别失败: {str(e)}")
    
    def _parse_llm_response(self, response: str, original_text: str) -> List[Entity]:
        """解析LLM返回的JSON响应"""
        try:
            # 尝试提取JSON部分
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response
            
            data = json.loads(json_str)
            entities = []
            
            for item in data.get('entities', []):
                # 验证实体位置的准确性
                start = item.get('start', 0)
                end = item.get('end', 0)
                expected_text = item.get('text', '')
                
                # 检查位置是否正确
                if 0 <= start < end <= len(original_text):
                    actual_text = original_text[start:end]
                    if actual_text == expected_text:
                        entity = Entity(
                            text=expected_text,
                            label=item.get('label', 'UNKNOWN'),
                            start=start,
                            end=end,
                            confidence=float(item.get('confidence', 0.5))
                        )
                        entities.append(entity)
            
            return entities
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # 如果JSON解析失败，尝试简单的文本解析
            return self._fallback_parse(response, original_text)
    
    def _fallback_parse(self, response: str, original_text: str) -> List[Entity]:
        """备用解析方法，当JSON解析失败时使用"""
        entities = []
        # 简单的文本解析逻辑，可以根据需要扩展
        # 这里只是一个示例，实际使用中可能需要更复杂的解析
        return entities


class HybridNER(BaseNER):
    """混合实体识别器，结合正则表达式和LLM"""
    
    def __init__(self, 
                 regex_ner: Optional[RegexNER] = None,
                 llm_ner: Optional[LLMNER] = None,
                 merge_strategy: str = 'union'):
        """
        初始化混合实体识别器
        
        Args:
            regex_ner: 正则表达式识别器
            llm_ner: LLM识别器
            merge_strategy: 合并策略，'union'(并集) 或 'intersection'(交集)
        """
        self.regex_ner = regex_ner
        self.llm_ner = llm_ner
        self.merge_strategy = merge_strategy
        
        if not regex_ner and not llm_ner:
            raise NERError("至少需要提供一个识别器")
    
    def extract_entities(self, text: str) -> List[Entity]:
        """提取实体，合并多个识别器的结果"""
        all_entities = []
        
        # 使用正则表达式识别
        if self.regex_ner:
            regex_entities = self.regex_ner.extract_entities(text)
            all_entities.extend(regex_entities)
        
        # 使用LLM识别
        if self.llm_ner:
            llm_entities = self.llm_ner.extract_entities(text)
            all_entities.extend(llm_entities)
        
        # 去重和合并
        merged_entities = self._merge_entities(all_entities)
        
        return merged_entities
    
    def _merge_entities(self, entities: List[Entity]) -> List[Entity]:
        """合并重叠的实体"""
        if not entities:
            return []
        
        # 按起始位置排序
        sorted_entities = sorted(entities, key=lambda x: x.start)
        merged = []
        
        for entity in sorted_entities:
            # 检查是否与已有实体重叠
            overlapped = False
            for i, existing in enumerate(merged):
                if self._is_overlap(entity, existing):
                    # 选择置信度更高的实体
                    if entity.confidence > existing.confidence:
                        merged[i] = entity
                    overlapped = True
                    break
            
            if not overlapped:
                merged.append(entity)
        
        return merged
    
    def _is_overlap(self, entity1: Entity, entity2: Entity) -> bool:
        """检查两个实体是否重叠"""
        return not (entity1.end <= entity2.start or entity2.end <= entity1.start)