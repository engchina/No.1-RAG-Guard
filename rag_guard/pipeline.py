from typing import List, Tuple, Dict, Optional
from .masker import DataMasker
from .exceptions import RAGGuardError


class RAGGuard:
    """
    典型集成方式：
    - prepare_chunks: 对 RAG 检索出的上下文块逐块脱敏，合并映射
    - build_prompt:   构造要发给外网模型的 Prompt（支持自定义模板）
    - postprocess:    外网回复回填（如需要）
    """
    def __init__(self, masker: DataMasker):
        self.masker = masker
        self.default_template = (
            "你将看到已脱敏的上下文和问题。\n"
            "占位符格式为 <RG:{KIND}:{HASH}>，表示某类实体（如 EMAIL、PHONE、IDCN、IPV4 等）。\n"
            "请忽略占位符的具体值，只需利用上下文语义回答问题。\n"
            "如果需要引用具体值，请直接引用占位符原样（不要猜测还原）。\n"
        )

    def prepare_chunks(self, chunks: List[str], max_chunk_length: Optional[int] = None) -> Tuple[List[str], Dict[str, str]]:
        """
        对RAG检索出的上下文块进行脱敏处理
        """
        if max_chunk_length is None:
            max_chunk_length = 10000
        
        masked_chunks = []
        combined_mapping = {}
        all_entities = []
        
        for i, chunk in enumerate(chunks):
            if len(chunk) > max_chunk_length:
                chunk = chunk[:max_chunk_length] + "...[截断]"
            
            # 使用新的mask方法
            masked_chunk, chunk_mapping, entities = self.masker.mask(chunk)
            masked_chunks.append(masked_chunk)
            combined_mapping.update(chunk_mapping)
            all_entities.extend(entities)
        
        return masked_chunks, combined_mapping

    def build_prompt(
        self, 
        question: str, 
        masked_chunks: List[str], 
        template: Optional[str] = None,
        context_format: str = "[CTX#{i}]\n{content}"
    ) -> str:
        """
        构造要发给外网模型的 Prompt
        
        Args:
            question: 用户问题
            masked_chunks: 脱敏后的文档片段
            template: 自定义指令模板
            context_format: 上下文格式化模板
        
        Returns:
            完整的prompt
        """
        if not isinstance(question, str):
            raise RAGGuardError("问题必须是字符串类型")
        
        # 使用自定义模板或默认模板
        instructions = template or self.default_template
        
        # 格式化上下文
        if masked_chunks:
            ctx_parts = []
            for i, chunk in enumerate(masked_chunks):
                formatted = context_format.format(i=i+1, content=chunk)
                ctx_parts.append(formatted)
            ctx = "\n\n".join(ctx_parts)
        else:
            ctx = "[无相关上下文]"
        
        prompt = f"{instructions}\n{ctx}\n\n[QUESTION]\n{question}"
        return prompt

    def postprocess(
        self, 
        external_answer: str, 
        mapping: Dict[str, str], 
        need_unmask: bool = True,
        partial_unmask: Optional[List[str]] = None
    ) -> str:
        """
        对外网回答进行后处理
        
        Args:
            external_answer: 外网模型的原始回答
            mapping: 脱敏映射表
            need_unmask: 是否需要回填
            partial_unmask: 仅回填指定类型的实体（如 ['EMAIL', 'PHONE']）
        
        Returns:
            处理后的答案
        """
        if not need_unmask:
            return external_answer
        
        if partial_unmask:
            # 部分回填：只回填指定类型的实体
            filtered_mapping = {}
            for token, original in mapping.items():
                # 解析token格式获取类型
                parts = token.strip('<>').split(':')
                if len(parts) == 3 and parts[1] in partial_unmask:
                    filtered_mapping[token] = original
            return self.masker.unmask(external_answer, filtered_mapping)
        else:
            # 完全回填
            return self.masker.unmask(external_answer, mapping)
    
    def get_template_variables(self) -> List[str]:
        """获取模板中可用的变量"""
        return ['{i}', '{content}']