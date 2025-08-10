# RAG-Guard

RAG-Guard 是一个用于在检索增强生成 (RAG) 应用中对敏感数据进行脱敏处理的库。它能够识别文本中的敏感信息（如邮箱、电话号码、IP地址等），将其替换为占位符，并在需要时进行回填。

## 特性

- 识别多种类型的敏感信息（邮箱、电话、IP地址、身份证号等）
- 使用稳定的哈希算法生成占位符，确保相同内容始终映射到相同占位符
- 支持自定义敏感信息识别模式
- 提供完整的脱敏和回填流程

## 安装要求

- Python 3.11 或更高版本
- [uv](https://github.com/astral-sh/uv) - 极速Python包和虚拟环境管理器

## 快速开始

### 1. 安装 uv

根据你的操作系统，选择以下方式之一安装 uv：

**使用 pip 安装：**
```bash
pip install uv
```

**在 macOS/Linux 上使用 curl 安装：**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**在 Windows 上使用 powershell 安装：**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

更多安装选项请参考 [uv 官方文档](https://docs.astral.sh/uv/)。

### 2. 创建虚拟环境并安装项目

```bash
# 克隆项目
git clone <repository-url>
cd No.1-RAG-Guard

# 使用 uv 创建虚拟环境并安装依赖
uv venv
uv pip install -e .
```

### 3. 运行示例

```bash
# 激活虚拟环境
source .venv/bin/activate  # Linux/macOS
# 或
.venv\Scripts\activate      # Windows

# 运行示例代码
python examples/demo.py
```

## 使用方法

```python
from rag_guard.masker import DataMasker
from rag_guard.pipeline import RAGGuard

# 创建数据脱敏器
masker = DataMasker(salt="your_salt_here")

# 创建 RAG-Guard 实例
guard = RAGGuard(masker)

# 准备需要脱敏的文本块
chunks = [
    "工单 1024 关联用户：Alice，邮箱 alice@example.com，最近登录 IP 10.2.3.4。",
    "联系电话：+86 138-0013-8000。用户反馈：无法重置密码。"
]

# 对文本块进行脱敏处理
masked_chunks, mapping = guard.prepare_chunks(chunks)

# 构建发送给外部模型的 Prompt
question = "请基于上下文给出排查建议，并指出涉及的用户邮箱和 IP。"
prompt = guard.build_prompt(question, masked_chunks)

# 处理外部模型的回复并进行回填
external_answer = call_your_llm_api(prompt)  # 你的外部模型调用
final_answer = guard.postprocess(external_answer, mapping, need_unmask=True)
```

## 许可证

本项目采用 Apache-2.0 许可证。详细信息请查看 [LICENSE](LICENSE) 文件。