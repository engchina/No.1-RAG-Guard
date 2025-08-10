from typing import List, Dict, Any
import time
from functools import wraps

def timing_decorator(func):
    """性能计时装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} 执行时间: {end_time - start_time:.4f}秒")
        return result
    return wrapper

def validate_chunks(chunks: List[str], max_length: int = 10000) -> List[str]:
    """验证和清理文档片段"""
    cleaned = []
    for i, chunk in enumerate(chunks):
        if not isinstance(chunk, str):
            raise ValueError(f"第{i+1}个片段不是字符串")
        
        # 移除多余空白
        chunk = chunk.strip()
        if not chunk:
            continue
        
        # 截断过长内容
        if len(chunk) > max_length:
            chunk = chunk[:max_length] + "...[截断]"
        
        cleaned.append(chunk)
    
    return cleaned

def estimate_tokens(text: str) -> int:
    """粗略估算token数量（中英文混合）"""
    # 简单估算：中文字符按1个token，英文单词按0.75个token
    chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
    english_words = len(text.replace(' ', '').replace('\n', '')) - chinese_chars
    return int(chinese_chars + english_words * 0.75)

def batch_process(items: List[Any], batch_size: int = 10):
    """批量处理生成器"""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]