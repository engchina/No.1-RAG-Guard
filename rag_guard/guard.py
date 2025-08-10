from typing import List, Dict, Optional, Callable, Any, Union
from .masker import DataMasker
from .pipeline import RAGGuard
from .config import GuardConfig
from .exceptions import RAGGuardError
import logging


class RAGGuardian:
    """
    RAG Guard 统一入口类，提供简化的API接口。
    支持正则表达式和LLM两种实体识别方式。
    """
    
    def __init__(self, config: Optional[GuardConfig] = None, 
                 llm_caller: Optional[Callable[[str], str]] = None,
                 **kwargs):
        """
        初始化 RAG Guardian
        
        Args:
            config: 配置对象，如果为None则使用默认配置
            llm_caller: LLM调用函数（用于实体识别）
            **kwargs: 快速配置参数，会覆盖config中的对应设置
        """
        self.config = config or GuardConfig()
        self.llm_caller = llm_caller
        
        # 应用快速配置参数
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # 初始化组件
        self.masker = DataMasker(
            salt=self.config.salt,
            custom_patterns=self.config.custom_patterns,
            llm_caller=self.llm_caller,
            use_llm_ner=self.config.use_llm_ner,
            llm_entity_types=self.config.llm_entity_types,
            ner_strategy=self.config.ner_strategy
        )
        self.pipeline = RAGGuard(self.masker)
        
        # 设置日志
        if self.config.enable_logging:
            self._setup_logging()
    
    def _setup_logging(self):
        """设置日志记录"""
        self.logger = logging.getLogger('rag_guard')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(self.config.log_level)
    
    def protect_and_query(
        self,
        chunks: List[str],
        question: str,
        llm_caller: Callable[[str], str],
        unmask_result: bool = True,
        **llm_kwargs
    ) -> Dict[str, Any]:
        """
        一键完成：脱敏 -> 调用LLM -> 回填的完整流程
        
        Args:
            chunks: RAG检索到的文档片段
            question: 用户问题
            llm_caller: 外部LLM调用函数，接收prompt返回response
            unmask_result: 是否在结果中回填敏感信息
            **llm_kwargs: 传递给llm_caller的额外参数
        
        Returns:
            完整的处理结果字典
        """
        try:
            if self.config.enable_logging:
                self.logger.info(f"开始处理查询，文档片段数: {len(chunks)}，NER策略: {self.config.ner_strategy}")
            
            # 1. 脱敏处理
            masked_chunks, mapping = self.pipeline.prepare_chunks(chunks)
            
            # 2. 构建prompt
            prompt = self.pipeline.build_prompt(
                question, masked_chunks, template=self.config.prompt_template
            )
            
            # 3. 调用外部LLM
            if self.config.enable_logging:
                self.logger.info("调用外部LLM")
            
            raw_response = llm_caller(prompt, **llm_kwargs)
            
            # 4. 后处理
            final_response = self.pipeline.postprocess(
                raw_response, mapping, need_unmask=unmask_result
            )
            
            # 5. 统计信息
            stats = {
                'chunks_count': len(chunks),
                'masked_entities': len(mapping),
                'prompt_length': len(prompt),
                'response_length': len(raw_response),
                'ner_strategy': self.config.ner_strategy
            }
            
            if self.config.enable_logging:
                self.logger.info(f"处理完成，脱敏实体数: {len(mapping)}")
            
            return {
                'answer': final_response,  # 修复变量名
                'masked_chunks': masked_chunks,
                'prompt': prompt,
                'raw_response': raw_response,
                'mapping': mapping if self.config.include_debug_info else {},
                'stats': stats
            }
            
        except Exception as e:
            if self.config.enable_logging:
                self.logger.error(f"处理过程中出错: {str(e)}")
            raise RAGGuardError(f"RAG Guard处理失败: {str(e)}") from e
    
    def mask_text(self, text: str) -> Dict[str, Any]:
        """
        单独的文本脱敏功能
        """
        masked_text, mapping, entities = self.masker.mask(text)
        
        # 如果启用日志且使用LLM NER，输出实体识别信息
        if self.config.enable_logging and 'llm' in self.config.ner_strategy:
            self.logger.info(f"LLM识别到 {len(entities)} 个实体:")
            for entity in entities:
                self.logger.info(f"  - {entity.label}: '{entity.text}' (置信度: {entity.confidence}, 位置: {entity.start}-{entity.end})")
        
        # 构建实体信息
        entities_info = []
        for entity in entities:
            entities_info.append({
                'text': entity.text if self.config.include_debug_info else '[MASKED]',
                'label': entity.label,
                'start': entity.start,
                'end': entity.end,
                'confidence': entity.confidence
            })
        
        return {
            'masked_text': masked_text,
            'mapping': mapping if self.config.include_debug_info else {},
            'entities_found': entities_info,
            'ner_info': self.masker.get_ner_info()
        }
    
    def add_custom_pattern(self, name: str, pattern: str) -> None:
        """动态添加自定义敏感信息模式"""
        self.masker.add_pattern(name, pattern)
        if self.config.enable_logging:
            self.logger.info(f"添加自定义模式: {name}")
    
    def set_llm_caller(self, llm_caller: Callable[[str], str]) -> None:
        """设置LLM调用函数（用于实体识别）"""
        self.llm_caller = llm_caller
        # 重新初始化masker以使用新的LLM调用函数
        self.masker = DataMasker(
            salt=self.config.salt,
            custom_patterns=self.config.custom_patterns,
            llm_caller=self.llm_caller,
            use_llm_ner=self.config.use_llm_ner,
            llm_entity_types=self.config.llm_entity_types,
            ner_strategy=self.config.ner_strategy
        )
        self.pipeline = RAGGuard(self.masker)