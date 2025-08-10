class RAGGuardError(Exception):
    """RAG Guard 基础异常类"""
    pass


class MaskingError(RAGGuardError):
    """脱敏处理异常"""
    pass


class UnmaskingError(RAGGuardError):
    """回填处理异常"""
    pass


class ConfigError(RAGGuardError):
    """配置错误异常"""
    pass


class LLMCallError(RAGGuardError):
    """LLM调用异常"""
    pass


class NERError(RAGGuardError):
    """实体识别异常"""
    pass