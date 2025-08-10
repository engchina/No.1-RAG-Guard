from rag_guard import RAGGuardian, GuardConfig
from openai import OpenAI
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 创建OpenAI客户端
client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)

def create_llm_caller():
    """创建LLM调用函数"""
    def llm_caller(prompt: str) -> str:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0  # 降低温度以获得更稳定的实体识别结果
        )
        return response.choices[0].message.content
    return llm_caller

def demo_regex_only():
    """演示仅使用正则表达式的脱敏"""
    print("=== 演示1: 仅使用正则表达式 ===")
    
    config = GuardConfig(
        salt="demo_salt_123",
        ner_strategy="regex_only",
        enable_logging=True
    )
    
    guardian = RAGGuardian(config)
    
    # 添加自定义模式
    guardian.add_custom_pattern("BANK_ACCOUNT", r"\b\d{16,19}\b")
    guardian.add_custom_pattern("IDCN_FULL", r"\b\d{17}[\dXx]\b")
    
    text = "张三的邮箱是 zhangsan@example.com，电话 138-0013-8000，身份证号 110101199001011234"
    result = guardian.mask_text(text)
    
    print(f"原文: {text}")
    print(f"脱敏后: {result['masked_text']}")
    print(f"发现的实体: {len(result['entities_found'])}个")
    for entity in result['entities_found']:
        print(f"  - {entity['label']}: [MASKED] (置信度: {entity['confidence']})")

def demo_llm_only():
    """演示仅使用LLM的实体识别"""
    print("\n=== 演示2: 仅使用LLM实体识别 ===")
    
    llm_caller = create_llm_caller()
    
    config = GuardConfig(
        salt="demo_salt_123",
        ner_strategy="llm_only",
        use_llm_ner=True,
        llm_entity_types=["PERSON", "EMAIL", "PHONE", "ID_NUMBER", "ADDRESS", "ORGANIZATION", "COMPANY"],
        llm_confidence_threshold=0.6,  # 降低阈值以提高召回率
        enable_logging=True
    )
    
    guardian = RAGGuardian(config, llm_caller=llm_caller)
    
    text = "李四在北京科技有限公司工作，住址是北京市朝阳区建国路88号，联系方式：lisi@company.com，手机13800138000"
    result = guardian.mask_text(text)
    
    print(f"原文: {text}")
    print(f"脱敏后: {result['masked_text']}")
    print(f"发现的实体: {len(result['entities_found'])}个")
    for entity in result['entities_found']:
        print(f"  - {entity['label']}: [MASKED] (置信度: {entity['confidence']})")

def demo_hybrid():
    """演示混合模式：正则表达式 + LLM"""
    print("\n=== 演示3: 混合模式（正则表达式 + LLM） ===")
    
    llm_caller = create_llm_caller()
    
    config = GuardConfig(
        salt="demo_salt_123",
        ner_strategy="hybrid",
        use_llm_ner=True,
        custom_patterns={
            "BANK_CARD": r"\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}",
            "ORDER_ID": r"ORDER-\d{8}",
            "EMAIL": r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
            "PHONE": r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}\b|\b1[3-9]\d{9}\b"
        },
        llm_entity_types=["PERSON", "ORGANIZATION", "ADDRESS", "EMAIL", "PHONE", "COMPANY"],
        llm_confidence_threshold=0.6,
        enable_logging=True
    )
    
    guardian = RAGGuardian(config, llm_caller=llm_caller)
    
    text = """王五在上海浦东新区陆家嘴金融中心的华夏银行工作，
    订单号ORDER-12345678，银行卡号6222 0202 0000 0000，
    邮箱wangwu@huaxia-bank.com，电话021-58888888"""
    
    result = guardian.mask_text(text)
    
    print(f"原文: {text}")
    print(f"脱敏后: {result['masked_text']}")
    print(f"发现的实体: {len(result['entities_found'])}个")
    print(f"NER信息: {result['ner_info']}")
    
    for entity in result['entities_found']:
        print(f"  - {entity['label']}: [MASKED] (置信度: {entity['confidence']})")

def demo_full_pipeline():
    """演示完整的RAG流程"""
    print("\n=== 演示4: 完整RAG流程（混合NER） ===")
    
    # 用于实体识别的LLM调用函数
    ner_llm_caller = create_llm_caller()
    
    # 用于回答问题的LLM调用函数
    def qa_llm_caller(prompt: str) -> str:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    
    config = GuardConfig(
        salt="demo_salt_123",
        ner_strategy="hybrid",
        use_llm_ner=True,
        llm_entity_types=[
            "PERSON", "EMAIL", "PHONE", "ID_NUMBER", "IDCN", 
            "BANK_ACCOUNT", "CREDIT_CARD", "ADDRESS", 
            "ORGANIZATION", "COMPANY", "FINANCIAL_INFO"
        ],
        llm_confidence_threshold=0.6,
        enable_logging=True
    )
    
    guardian = RAGGuardian(config, llm_caller=ner_llm_caller)
    
    # 模拟RAG检索到的文档
    chunks = [
        "客户赵六，身份证号110101199005051234，在中国工商银行开户，账号6222020200000000",
        "联系地址：北京市海淀区中关村大街1号，邮箱zhaoliu@icbc.com.cn，电话010-62345678",
        "最近一笔交易：向招商银行转账50000元，交易时间2024-01-15 14:30:00"
    ]
    
    question = "请总结这个客户的基本信息和最近交易情况"
    
    # 执行完整流程 - 使用正确的参数名
    result = guardian.protect_and_query(
        chunks=chunks,
        question=question,
        llm_caller=qa_llm_caller,
        unmask_result=True
    )
    
    print("\n=== 发送给外部AI的脱敏内容 ===")
    print(result['prompt'])
    
    print("\n=== 外部AI原始回复 ===")
    print(result['raw_response'])
    
    print("\n=== 最终答案（已回填） ===")
    print(result['answer'])
    
    print("\n=== 统计信息 ===")
    for key, value in result['stats'].items():
        print(f"{key}: {value}")

def demo_custom_patterns():
    """演示自定义模式的使用"""
    print("\n=== 演示5: 自定义模式测试 ===")
    
    config = GuardConfig(
        salt="demo_salt_123",
        ner_strategy="regex_only",
        enable_logging=True
    )
    
    guardian = RAGGuardian(config)
    
    # 添加自定义模式
    guardian.add_custom_pattern("BANK_ACCOUNT", r"\b\d{16,19}\b")
    guardian.add_custom_pattern("IDCN_FULL", r"\b\d{17}[\dXx]\b")
    
    text = "张三的邮箱是 zhangsan@example.com，电话 138-0013-8000，身份证号 110101199001011234"
    result = guardian.mask_text(text)
    
    print(f"原文: {text}")
    print(f"脱敏后: {result['masked_text']}")
    print(f"发现的实体: {len(result['entities_found'])}个")
    for entity in result['entities_found']:
        print(f"  - {entity['label']}: [MASKED] (置信度: {entity['confidence']})")

if __name__ == "__main__":
    demo_regex_only()
    demo_llm_only()
    demo_hybrid()
    demo_full_pipeline()
    # 注意：不要在这里使用未定义的 guardian 变量进行 add_custom_pattern 操作
    # 如需测试自定义模式，请在具体的 demo 函数中创建 guardian 后调用 guardian.add_custom_pattern(...)