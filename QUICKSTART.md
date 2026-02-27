# LayeredRetriever 快速启动指南

## 🚀 5分钟快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 设置 API Key

创建 `.env` 文件（或设置环境变量）：

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your-api-key-here"

# Linux/Mac
export OPENAI_API_KEY="your-api-key-here"
```

### 3. 运行基础测试

```bash
python test_basic.py
```

如果所有测试通过，说明安装成功！

### 4. 添加文档并查询

#### 方式 A: 使用示例脚本

```bash
python example_usage.py
```

#### 方式 B: 在代码中使用

```python
from pipeline.orchestrator import LayeredRetrieverPipeline
from pathlib import Path

# 初始化
pipeline = LayeredRetrieverPipeline(
    config_path="config/retriever.yaml"
)

# 查询
result = pipeline.process("What is multi-stage retrieval?")
print(result["answer"]["answer"])
```

#### 方式 C: 使用 API

```bash
# 启动 API 服务器
cd api
python app.py

# 在另一个终端发送请求
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is multi-stage retrieval?"}'
```

## 📝 添加自己的文档

```python
from storage.vector_store import VectorStore
from storage.doc_store import DocumentStore, Document
import yaml

# 加载配置
with open("config/retriever.yaml", "r") as f:
    config = yaml.safe_load(f)

# 初始化存储
vector_store = VectorStore(config["storage"]["vector_store"])
doc_store = DocumentStore(config["storage"]["doc_store"])

# 准备文档
documents = [
    {
        "doc_id": "my_doc_1",
        "content": "Your document content here...",
        "metadata": {"source": "my_source"}
    },
    # 添加更多文档...
]

# 添加到向量存储
vector_store.add_documents(documents)

# 添加到文档存储
docs = [Document(**doc) for doc in documents]
doc_store.add_documents(docs)

print(f"✓ Added {len(documents)} documents")
```

## 🔧 常见问题

### Q: 提示找不到 OpenAI API Key

**A:** 确保设置了环境变量：
```bash
# 检查是否设置
echo $OPENAI_API_KEY  # Linux/Mac
echo $env:OPENAI_API_KEY  # Windows PowerShell
```

### Q: 导入错误

**A:** 确保安装了所有依赖：
```bash
pip install -r requirements.txt
```

### Q: ChromaDB 初始化失败

**A:** 确保有写入权限，或修改 `config/retriever.yaml` 中的 `persist_directory` 路径。

### Q: 如何调试特定阶段？

**A:** 使用 dry-run：
```python
# 只运行到 Stage 1
result = pipeline.process(query, dry_run_stage="stage1")
print(result["candidates"])

# 只运行到 Stage 2
result = pipeline.process(query, dry_run_stage="stage2")
print(result["contexts"])
```

## 📚 下一步

- 阅读 [README.md](README.md) 了解详细设计理念
- 查看 [example_usage.py](example_usage.py) 了解更多示例
- 修改 `config/retriever.yaml` 调整系统参数

## 🆘 需要帮助？

- 检查 `test_basic.py` 的输出，定位问题
- 查看 `data/traces/` 目录下的执行轨迹
- 阅读各模块的文档字符串
