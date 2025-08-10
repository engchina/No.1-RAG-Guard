# RAG-Guard

RAG-Guard 是一个专为检索增强生成（RAG）场景设计的安全防护与脱敏库。它能在对外调用大模型前，对上下文中的敏感数据进行识别与替换（脱敏），并在需要时将模型回复中的占位符安全地回填为原文，避免敏感数据外泄。

占位符格式：<RG:{KIND}:{HASH}>
- KIND 为实体类别（如 EMAIL、PHONE、IDCN、IPV4 等）
- HASH 为基于盐值的稳定短哈希，确保相同原文在同一配置下映射一致

目录
- 特性
- 安装
- 快速开始
- 统一入口：RAGGuardian
- 进阶用法
  - 仅正则模式（regex_only）
  - 仅 LLM 模式（llm_only）
  - 混合模式（hybrid）
  - 自定义正则模式
  - 自定义 Prompt 模板与上下文格式
- 配置项 GuardConfig
- 错误类型
- 示例与开发
- 许可证

特性
- 多种敏感实体识别
  - 内置正则：EMAIL, PHONE, IDCN（中国身份证）, IPV4, CREDIT_CARD, BANK_ACCOUNT, URL
  - 可对接 LLM 做更丰富、更语义化的实体识别
- 稳定的占位符生成
  - 使用盐（salt）与稳定哈希确保同一值映射一致
- 灵活的 NER 策略
  - regex_only、llm_only、hybrid 三种策略可选
- 安全回填
  - 支持完整或按实体类型的部分回填
- 可配置、可扩展
  - 支持自定义正则模式、模板、日志、阈值等

安装
本项目基于 Python 3.11+ 和 setuptools 打包。你可以直接以开发模式安装：

Windows PowerShell:
```powershell
pip install -e .
pip install -r requirements.txt
# pip list --format=freeze > requirements.txt
```

如果你使用 uv 进行依赖/环境管理，也可以：
Windows PowerShell:
```powershell
uv venv
```

Windows PowerShell:
```powershell
uv pip install -e .
uv pip install -r requirements.txt
# uv pip list --format=freeze > requirements.txt
```

快速开始
以下示例展示典型的 RAG 前置脱敏、Prompt 构造、外部 LLM 调用及回填的流程。

Python:
```python
from rag_guard import RAGGuardian, GuardConfig

# 配置（请务必使用长度>=8的自定义盐）
config = GuardConfig(
    salt="your_strong_salt_here",
    ner_strategy="regex_only",   # 'regex_only' | 'llm_only' | 'hybrid'
    enable_logging=True
)

guardian = RAGGuardian(config)

# 模拟 RAG 检索到的上下文片段
chunks = [
    "用户张三，邮箱 zhangsan@example.com，最近登录 IP 10.2.3.4，手机 138-0013-8000。",
    "身份证号 110101199001011234，工资卡 6222020200000000。"
]

question = "请总结该用户的基本信息。"

# 你的 LLM 调用函数（示例：直接回显）
def dummy_llm(prompt: str) -> str:
    # 实际请改为调用你的大模型，如 OpenAI、Azure OpenAI、通义等
    return f"[LLM 回复示例] 已阅读：\n{prompt[:200]}..."

# 一键完成：脱敏 -> 构建 Prompt -> 调用 LLM -> 回填
result = guardian.protect_and_query(
    chunks=chunks,
    question=question,
    llm_caller=dummy_llm,
    unmask_result=True
)

print("最终答案：", result["answer"])
print("统计信息：", result["stats"])
```

统一入口：RAGGuardian
RAGGuardian 是推荐的高层 API，隐藏了 DataMasker 与 RAGGuard 组合细节，提供更直观的一键式方法。

核心方法
- protect_and_query(chunks, question, llm_caller, unmask_result=True, **llm_kwargs)
  - 输入 RAG 检索片段和问题，自动完成脱敏、Prompt 构造、外部 LLM 调用与回填
  - 返回字典：answer、masked_chunks、prompt、raw_response、mapping、stats
- mask_text(text)
  - 对单段文本进行脱敏，返回 masked_text、mapping、entities_found、ner_info
- add_custom_pattern(name, pattern)
  - 动态新增正则模式（仅 regex_only 或包含正则的策略可用）
- set_llm_caller(llm_caller)
  - 设置用于 LLM 实体识别的调用函数（llm_only/hybrid 下有效）

进阶用法

仅正则模式（regex_only）
Python:
```python
from rag_guard import RAGGuardian, GuardConfig

guardian = RAGGuardian(GuardConfig(
    salt="demo_salt_123",
    ner_strategy="regex_only",
    enable_logging=True
))

# 可动态增加自定义模式
guardian.add_custom_pattern("BANK_ACCOUNT", r"\b\d{16,19}\b")

text = "邮箱 a@b.com，电话 13800138000，卡号 6222020200000000"
res = guardian.mask_text(text)
print(res["masked_text"])        # 脱敏结果
print(res["entities_found"])     # 识别到的实体
```

仅 LLM 模式（llm_only）
需要提供 llm_caller，并启用 use_llm_ner/配置实体类型。

Python:
```python
from rag_guard import RAGGuardian, GuardConfig
from openai import OpenAI
import os

client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)

def llm_caller(prompt: str) -> str:
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    return resp.choices[0].message.content

guardian = RAGGuardian(
    GuardConfig(
        salt="demo_salt_123",
        ner_strategy="llm_only",
        use_llm_ner=True,
        llm_entity_types=["PERSON", "EMAIL", "PHONE", "ADDRESS", "COMPANY"],
        enable_logging=True
    ),
    llm_caller=llm_caller
)

print(guardian.mask_text("李四在北京科技有限公司工作，邮箱 lisi@company.com"))
```

混合模式（hybrid）
同时结合正则与 LLM 的优势（更鲁棒、更全面）。

Python:
```python
from rag_guard import RAGGuardian, GuardConfig

guardian = RAGGuardian(
    GuardConfig(
        salt="demo_salt_123",
        ner_strategy="hybrid",
        use_llm_ner=True,
        custom_patterns={
            "ORDER_ID": r"ORDER-\d{8}"
        },
        llm_entity_types=["PERSON", "ORGANIZATION", "ADDRESS", "EMAIL", "PHONE", "COMPANY"],
        enable_logging=True
    ),
    llm_caller=llm_caller  # 见上文 llm_caller 定义
)

text = "王五 邮箱 wangwu@bank.com，订单 ORDER-12345678，电话 021-58888888"
print(guardian.mask_text(text))
```

自定义正则模式
- 方式一：通过 GuardConfig.custom_patterns 在初始化时注入
- 方式二：运行时调用 guardian.add_custom_pattern(name, pattern)

注意：仅 regex_only 或包含 RegexNER 的策略可用（hybrid 也支持）。

自定义 Prompt 模板与上下文格式
- 可通过 GuardConfig.prompt_template 覆盖默认提示词
- 也可直接使用底层 RAGGuard.build_prompt 的 context_format 参数自定义上下文块格式
- 模板变量：{i}, {content}

Python:
```python
from rag_guard import RAGGuardian, GuardConfig

tpl = (
    "你将看到已脱敏的上下文，请基于语义回答问题。\n"
    "如需引用具体值，请原样引用占位符，不要猜测还原。"
)

guardian = RAGGuardian(GuardConfig(
    salt="demo_salt_123",
    ner_strategy="regex_only",
    prompt_template=tpl
))

chunks = ["邮箱 a@b.com", "IP 10.0.0.1"]
masked_chunks, mapping = guardian.pipeline.prepare_chunks(chunks)
prompt = guardian.pipeline.build_prompt(
    question="邮箱和IP是什么？",
    masked_chunks=masked_chunks,
    context_format="[CHUNK {i}]\n{content}"
)
print(prompt)
```

配置项 GuardConfig
- salt: 脱敏盐值（必需，长度>=8）
- custom_patterns: 自定义正则模式字典
- ner_strategy: 'regex_only' | 'llm_only' | 'hybrid'
- use_llm_ner: 是否使用 LLM 实体识别
- llm_entity_types: LLM 识别的实体类型列表
- llm_confidence_threshold: LLM 识别的最低置信度阈值（默认 0.7）
- prompt_template: 自定义提示词模板
- enable_logging: 启用日志
- log_level: 日志级别（logging.INFO 等）
- include_debug_info: 是否在返回中包含 mapping 等调试信息
- max_chunk_length: 单片段最大长度（超出截断）
- max_chunks_count: 最大片段数量

错误类型
- RAGGuardError: 基础异常
- MaskingError: 脱敏处理异常
- UnmaskingError: 回填处理异常
- ConfigError: 配置错误
- LLMCallError: LLM 调用异常
- NERError: 实体识别异常

示例与开发
示例脚本
- examples/llm_ner_example.py 包含以下演示：
  - 仅正则
  - 仅 LLM
  - 混合模式
  - 完整 RAG 流程（脱敏 -> Prompt -> LLM -> 回填）

环境变量
- 你可以复制 .env.example 为 .env 并填入你的 OpenAI 配置：
  - OPENAI_BASE_URL
  - OPENAI_API_KEY
  - OPENAI_MODEL

运行示例（Windows PowerShell）
Windows PowerShell:
```powershell
python examples\llm_ner_example.py
```

脱敏与回填的工作方式
- 对上下文进行实体识别并将敏感内容替换为占位符 <RG:KIND:HASH>
- 保存 token->原文 的 mapping（回填时使用）
- 构造 Prompt 时仅包含脱敏后的上下文
- 收到外部 LLM 回复后，按需回填：
  - 完全回填：将出现的占位符全部替换为原文
  - 部分回填：仅回填指定类型（如 ['EMAIL', 'PHONE']）

注意事项
- 请务必自定义盐（salt），并确保长度>=8，以保证占位符稳定性与安全性
- 若使用 LLM 实体识别，请提供可靠的 llm_caller，并按需配置 llm_entity_types 与阈值
- include_debug_info 默认关闭，生产环境建议保持关闭，避免意外泄露 mapping

版本与 API
- 包导出（from rag_guard import ...）：
  - RAGGuardian, DataMasker, RAGGuard, GuardConfig
  - BaseNER, RegexNER, LLMNER, HybridNER, Entity
  - RAGGuardError, MaskingError, UnmaskingError, ConfigError, LLMCallError, NERError
- 版本：1.1.0（库内声明）

许可证
本项目采用 Apache-2.0 许可证。详见 LICENSE。