# 🚀 LayeredRetriever — Multi-Stage Retrieval Assistant

<div align="center">

**中文名：层级检索助手** | **多阶段、可解释、可调度的认知检索系统**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**🎯 不是简单的 RAG，而是模拟人类专家思维的认知检索系统**

[快速开始](#-快速开始) • [架构设计](#-系统架构) • [功能特性](#-核心功能) • [代码示例](#-代码示例) • [API 文档](#-api-接口)

</div>

---

## 📖 项目简介

### 🧠 核心理念

LayeredRetriever **不是**一个简单的"问一句 → 查一下 → 拼答案"的 RAG 工具。

> **我们的目标是：用多阶段、可解释、可调度的检索链路，模拟"人类专家逐步收敛问题空间"的过程。**

想象一下，当人类专家回答复杂问题时，他们不会：
- ❌ 只看前 5 个最相似的文档
- ❌ 把所有结果拼在一起
- ❌ 只依赖单一的相似度分数

而是会：
- ✅ **先广泛收集**可能相关的信息（不遗漏）
- ✅ **再仔细筛选**真正相关的部分（去伪存真）
- ✅ **最后精准合成**答案（有理有据）

这就是 LayeredRetriever 的设计哲学！🎯

### 🌟 为什么需要多阶段检索？

传统 RAG 的问题：
- 🔴 **召回不足**：只取 top-k，可能漏掉关键信息
- 🔴 **精度不够**：相似度高 ≠ 真正相关
- 🔴 **不可解释**：不知道为什么选这些文档
- 🔴 **难以调试**：出错了不知道是哪个环节的问题

LayeredRetriever 的解决方案：
- 🟢 **Stage 1 高召回**：先用多种策略收集 50-200 个候选
- 🟢 **Stage 2 高精度**：用 LLM 深度判断，筛选到 5-10 个
- 🟢 **全程可追踪**：每个阶段都有详细的 trace
- 🟢 **支持 dry-run**：可以停在任意阶段进行调试

---

## 🎯 核心功能

### 1️⃣ Query Analyzer（查询分析器）🔍

**功能**：深度理解用户查询，制定检索策略

**能力**：
- 🎯 **意图识别**：factual（事实性）、comparative（比较性）、procedural（程序性）、exploratory（探索性）、causal（因果性）、temporal（时间性）
- 🔑 **概念提取**：自动提取查询中的关键概念
- 📋 **约束检测**：识别时间、领域、数值等约束条件
- 🔗 **多跳检测**：判断是否需要多跳推理
- 📊 **复杂度评估**：计算查询复杂度分数

**输出示例**：
```python
QueryPlan(
    intent="comparative",
    needs_multi_hop=True,
    key_concepts=["transformer", "retrieval", "multi-stage"],
    constraints=["recent", "research"],
    complexity_score=0.75
)
```

### 2️⃣ Stage 1: Broad Retrieval（广泛检索）🌐

**目标**：高召回，不漏关键信息

**策略组合**：
- 🔷 **向量检索**：语义相似度搜索（k=200）
- 🔷 **关键词检索**：BM25 算法（k=100）
- 🔷 **元数据过滤**：基于约束条件过滤
- 🔷 **智能合并**：多源结果去重和分数融合

**特点**：
- ⚡ 快速、低成本
- 📈 容忍噪声（宁可多收，不可遗漏）
- 🎯 最终输出 50-200 个候选文档

### 3️⃣ Stage 2: Semantic Refinement（语义精炼）✨

**目标**：高精度，剔除假阳性

**精炼流程**：
1. 🔄 **交叉编码器重排序**（可选）：更精确的相关性计算
2. 🧩 **子问题分解**：复杂查询拆解为子问题
3. 🤖 **LLM 相关性判断**：深度语义理解，判断是否真正回答子问题
4. 🗑️ **硬负样本剔除**：移除"看起来像但不相关"的候选

**特点**：
- 🎯 从 50-200 个候选精炼到 5-10 个
- 📝 每个选中的上下文都有明确的理由
- 🔍 支持子问题匹配追踪

### 4️⃣ Stage 3: Answer Synthesis（答案合成）📝

**目标**：基于精选上下文生成高质量答案

**要求**：
- ✅ 仅使用 Stage 2 选中的上下文
- ✅ 强制引用来源
- ✅ 输出结构化 reasoning
- ✅ 计算置信度分数

**输出结构**：
```python
{
    "answer": "Multi-stage retrieval is...",
    "citations": [
        {
            "doc_id": "doc_12",
            "content_snippet": "...",
            "relevance_reason": "Directly explains..."
        }
    ],
    "confidence": 0.82,
    "reasoning_trace": "..."
}
```

### 5️⃣ Pipeline Orchestrator（管道编排）🎼

**功能**：协调整个多阶段流程

**特性**：
- 🔄 自动执行完整流程
- 🐛 支持 dry-run 调试（可停在任意阶段）
- 📊 自动生成执行轨迹
- ⏱️ 性能监控和计时

---

## 🏗️ 系统架构

### 整体流程图

```
┌─────────────────────────────────────────────────────────────┐
│                    User Query Input                          │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   Query Analyzer & Planner    │
        │  • Intent Detection            │
        │  • Concept Extraction          │
        │  • Constraint Analysis         │
        │  • Execution Plan Generation   │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   Stage 1: Broad Retrieval    │
        │  • Vector Search (k=200)     │
        │  • Keyword Search (k=100)     │
        │  • Metadata Filtering         │
        │  • Candidate Merging          │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   Candidate Pool (50-200)     │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   Stage 2: Semantic Refine    │
        │  • Cross-Encoder Reranking     │
        │  • Sub-question Decomposition  │
        │  • LLM Relevance Judgment      │
        │  • Hard Negative Elimination   │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │  High-Confidence Context (5-10)│
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   Stage 3: Answer Synthesis   │
        │  • Context Preparation         │
        │  • LLM Answer Generation      │
        │  • Citation Building          │
        │  • Confidence Calculation     │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   Final Answer + Trace        │
        │  • Answer Text                │
        │  • Citations                  │
        │  • Confidence Score            │
        │  • Execution Trace             │
        └───────────────────────────────┘
```

### 模块依赖关系

```
pipeline/
  └── orchestrator.py
       ├── query/
       │    ├── analyzer.py
       │    └── planner.py
       ├── retrieval/
       │    ├── stage1_broad.py ──┐
       │    ├── stage2_refine.py ─┤── storage/
       │    └── ranker.py          │    ├── vector_store.py
       └── synthesis/              │    └── doc_store.py
            ├── answer_generator.py │
            └── citation_builder.py │
                                    │
                            LLM Client
```

---

## 💻 代码示例

### 示例 1: 基础使用 🎯

```python
from pipeline.orchestrator import LayeredRetrieverPipeline
from pathlib import Path

# 初始化管道
pipeline = LayeredRetrieverPipeline(
    config_path="config/retriever.yaml"
)

# 处理查询
query = "What is multi-stage retrieval and how does it work?"
result = pipeline.process(query)

# 查看结果
answer = result["answer"]
print("📝 Answer:", answer["answer"])
print("🎯 Confidence:", answer["confidence"])
print("📚 Citations:", len(answer["citations"]))

# 查看执行轨迹
trace = result["trace"]
print(f"🔍 Stage 1 Candidates: {len(trace['stage1_candidates'])}")
print(f"✨ Stage 2 Contexts: {len(trace['stage2_contexts'])}")
print(f"⏱️  Execution Time: {trace['execution_time']:.2f}s")
```

**输出示例**：
```
📝 Answer: Multi-stage retrieval is a technique that uses multiple 
stages to improve both recall and precision. Stage 1 focuses on 
high recall using fast methods, while Stage 2 refines results 
for high precision...

🎯 Confidence: 0.85
📚 Citations: 3
🔍 Stage 1 Candidates: 156
✨ Stage 2 Contexts: 7
⏱️  Execution Time: 2.34s
```

### 示例 2: 查询分析深度解析 🔬

```python
from query.analyzer import QueryAnalyzer

analyzer = QueryAnalyzer()

# 分析不同类型的查询
queries = [
    "What is transformer architecture?",  # 事实性
    "Compare transformer and BERT",       # 比较性
    "How to implement multi-stage retrieval?",  # 程序性
    "Tell me about recent research on retrieval systems",  # 探索性
]

for query in queries:
    plan = analyzer.analyze(query)
    print(f"\n🔍 Query: {query}")
    print(f"   Intent: {plan.intent.value}")
    print(f"   Key Concepts: {plan.key_concepts}")
    print(f"   Constraints: {plan.constraints}")
    print(f"   Complexity: {plan.complexity_score:.2f}")
    print(f"   Multi-hop: {plan.needs_multi_hop}")
```

**输出**：
```
🔍 Query: What is transformer architecture?
   Intent: factual
   Key Concepts: ['transformer', 'architecture']
   Constraints: []
   Complexity: 0.35
   Multi-hop: False

🔍 Query: Compare transformer and BERT
   Intent: comparative
   Key Concepts: ['compare', 'transformer', 'bert']
   Constraints: []
   Complexity: 0.45
   Multi-hop: False

🔍 Query: How to implement multi-stage retrieval?
   Intent: procedural
   Key Concepts: ['implement', 'multi', 'stage', 'retrieval']
   Constraints: []
   Complexity: 0.50
   Multi-hop: False

🔍 Query: Tell me about recent research on retrieval systems
   Intent: exploratory
   Key Concepts: ['recent', 'research', 'retrieval', 'systems']
   Constraints: ['recent', 'research']
   Complexity: 0.60
   Multi-hop: False
```

### 示例 3: Stage 1 检索策略详解 🎯

```python
from retrieval.stage1_broad import Stage1BroadRetriever
from storage.vector_store import VectorStore
from storage.doc_store import DocumentStore

# 初始化组件
vector_store = VectorStore(config={"type": "chroma"})
doc_store = DocumentStore(config={"type": "memory"})

# 创建 Stage 1 检索器
retriever = Stage1BroadRetriever(
    vector_store=vector_store,
    doc_store=doc_store,
    config={
        "strategy": "hybrid",  # 混合策略
        "vector_k": 200,
        "keyword_k": 100,
        "final_k": 200
    }
)

# 执行检索
query = "multi-stage retrieval techniques"
candidates = retriever.retrieve(
    query=query,
    key_concepts=["retrieval", "multi-stage"],
    constraints=["recent"]
)

# 分析结果
print(f"📊 Total Candidates: {len(candidates)}")
print(f"\n🔷 By Source:")
sources = {}
for c in candidates:
    source = c.source.split("+")[0]  # 处理合并来源
    sources[source] = sources.get(source, 0) + 1
for source, count in sources.items():
    print(f"   {source}: {count}")

print(f"\n📈 Top 5 Candidates:")
for i, candidate in enumerate(candidates[:5], 1):
    print(f"   {i}. {candidate.doc_id} (score: {candidate.score:.3f}, source: {candidate.source})")
```

### 示例 4: Stage 2 精炼过程追踪 🔬

```python
from retrieval.stage2_refine import Stage2Refiner
from retrieval.stage1_broad import Candidate

# 假设我们已经有了 Stage 1 的候选
candidates = [
    Candidate(doc_id="doc1", score=0.85, source="vector", content="..."),
    Candidate(doc_id="doc2", score=0.78, source="keyword", content="..."),
    # ... 更多候选
]

# 创建 Stage 2 精炼器
refiner = Stage2Refiner(
    llm_client=llm_client,
    ranker=ranker,
    config={
        "use_llm_judge": True,
        "max_contexts": 10,
        "min_relevance_score": 0.5,
        "enable_subquestion_decomposition": True
    }
)

# 执行精炼
query = "How does multi-stage retrieval improve precision and recall?"
query_plan = {
    "needs_multi_hop": True,
    "complexity_score": 0.7,
    "key_concepts": ["multi-stage", "retrieval", "precision", "recall"]
}

contexts = refiner.refine(
    query=query,
    candidates=candidates,
    query_plan=query_plan
)

# 查看精炼结果
print(f"✨ Refined Contexts: {len(contexts)}")
print(f"📉 Reduction: {len(candidates)} → {len(contexts)} ({len(contexts)/len(candidates)*100:.1f}%)")

for i, ctx in enumerate(contexts, 1):
    print(f"\n{i}. Doc ID: {ctx.doc_id}")
    print(f"   Relevance: {ctx.relevance_score:.3f}")
    print(f"   Reason: {ctx.reason}")
    if ctx.sub_question_match:
        print(f"   Matches Sub-question: {ctx.sub_question_match}")
```

### 示例 5: Dry Run 调试 🐛

```python
# 只运行到 Stage 1，查看候选
result = pipeline.process(
    "Compare transformer and BERT architectures",
    dry_run_stage="stage1"
)

print("🔍 Stage 1 Results:")
candidates = result["candidates"]
print(f"   Total: {len(candidates)}")
print(f"\n📊 Score Distribution:")
scores = [c["score"] for c in candidates]
print(f"   Min: {min(scores):.3f}")
print(f"   Max: {max(scores):.3f}")
print(f"   Avg: {sum(scores)/len(scores):.3f}")

# 只运行到 Stage 2，查看精炼结果
result = pipeline.process(
    "Compare transformer and BERT architectures",
    dry_run_stage="stage2"
)

print("\n✨ Stage 2 Results:")
contexts = result["contexts"]
for ctx in contexts:
    print(f"   {ctx['doc_id']}: {ctx['relevance_score']:.3f} - {ctx['reason']}")
```

### 示例 6: 完整流程追踪 📊

```python
result = pipeline.process("What are the advantages of multi-stage retrieval?")

trace = result["trace"]

print("=" * 60)
print("📊 Complete Execution Trace")
print("=" * 60)

# Query Plan
print("\n🔍 Query Plan:")
qp = trace["query_plan"]
print(f"   Intent: {qp['intent']}")
print(f"   Key Concepts: {', '.join(qp['key_concepts'])}")
print(f"   Constraints: {', '.join(qp['constraints']) if qp['constraints'] else 'None'}")
print(f"   Complexity: {qp['complexity_score']:.2f}")
print(f"   Multi-hop: {qp['needs_multi_hop']}")

# Stage 1
print(f"\n🌐 Stage 1: Broad Retrieval")
print(f"   Candidates: {len(trace['stage1_candidates'])}")
if trace['stage1_candidates']:
    top = trace['stage1_candidates'][0]
    print(f"   Top Candidate: {top['doc_id']} (score: {top['score']:.3f}, source: {top['source']})")

# Stage 2
print(f"\n✨ Stage 2: Semantic Refinement")
print(f"   Contexts: {len(trace['stage2_contexts'])}")
if trace['stage2_contexts']:
    top = trace['stage2_contexts'][0]
    print(f"   Top Context: {top['doc_id']} (relevance: {top['relevance_score']:.3f})")
    print(f"   Reason: {top['reason']}")

# Stage 3
if trace['stage3_answer']:
    print(f"\n📝 Stage 3: Answer Synthesis")
    answer = trace['stage3_answer']
    print(f"   Answer Length: {len(answer['answer'])} chars")
    print(f"   Citations: {len(answer['citations'])}")
    print(f"   Confidence: {answer['confidence']:.2f}")
    print(f"   Used Contexts: {len(answer['used_contexts'])}")

# Performance
print(f"\n⏱️  Performance:")
print(f"   Total Time: {trace['execution_time']:.2f}s")
print(f"   Stage 1 Time: ~{trace['execution_time']*0.3:.2f}s (estimated)")
print(f"   Stage 2 Time: ~{trace['execution_time']*0.5:.2f}s (estimated)")
print(f"   Stage 3 Time: ~{trace['execution_time']*0.2:.2f}s (estimated)")
```

---

## 🚀 快速开始

### 1. 安装依赖 📦

```bash
# 克隆仓库
git clone <repository-url>
cd layered-retriever

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量 🔐

```bash
# 创建 .env 文件
echo "OPENAI_API_KEY=your-api-key-here" > .env

# 或直接设置环境变量
export OPENAI_API_KEY="your-api-key-here"  # Linux/Mac
$env:OPENAI_API_KEY="your-api-key-here"   # Windows PowerShell
```

### 3. 运行测试 ✅

```bash
python test_basic.py
```

### 4. 添加文档并查询 📚

```python
from pipeline.orchestrator import LayeredRetrieverPipeline
from storage.vector_store import VectorStore
from storage.doc_store import DocumentStore, Document

# 初始化存储
vector_store = VectorStore(config={"type": "chroma"})
doc_store = DocumentStore(config={"type": "memory"})

# 添加文档
documents = [
    {
        "doc_id": "doc1",
        "content": "Multi-stage retrieval improves both recall and precision...",
        "metadata": {"topic": "retrieval"}
    }
]

vector_store.add_documents(documents)
doc_store.add_documents([Document(**d) for d in documents])

# 初始化管道并查询
pipeline = LayeredRetrieverPipeline()
result = pipeline.process("What is multi-stage retrieval?")
print(result["answer"]["answer"])
```

---

## 📁 项目结构

```
layered-retriever/
├── 📄 README.md                 # 项目文档（本文件）
├── 📄 QUICKSTART.md             # 快速启动指南
├── 📄 requirements.txt          # Python 依赖
├── 📄 setup.py                  # 安装脚本
├── 📄 test_basic.py             # 基础测试
├── 📄 example_usage.py          # 使用示例
│
├── 📁 config/                   # 配置文件
│   └── retriever.yaml          # 主配置文件
│
├── 📁 query/                    # 查询分析模块
│   ├── __init__.py
│   ├── analyzer.py             # 查询分析器
│   └── planner.py              # 查询规划器
│
├── 📁 retrieval/                # 检索模块
│   ├── __init__.py
│   ├── stage1_broad.py         # Stage 1: 广泛检索
│   ├── stage2_refine.py        # Stage 2: 语义精炼
│   └── ranker.py               # 重排序器
│
├── 📁 synthesis/                # 答案合成模块
│   ├── __init__.py
│   ├── answer_generator.py      # 答案生成器
│   └── citation_builder.py     # 引用构建器
│
├── 📁 storage/                  # 存储模块
│   ├── __init__.py
│   ├── vector_store.py         # 向量存储（ChromaDB/FAISS）
│   └── doc_store.py            # 文档存储
│
├── 📁 pipeline/                 # 管道编排
│   ├── __init__.py
│   └── orchestrator.py         # 主编排器
│
└── 📁 api/                      # API 层
    ├── __init__.py
    └── app.py                  # FastAPI 应用
```

---

## ⚙️ 配置说明

### LLM 配置

```yaml
llm:
  provider: "openai"              # LLM 提供商
  model: "gpt-4-turbo-preview"    # 模型名称
  temperature: 0.1                # 温度参数（低=更确定）
  max_tokens: 2000                # 最大生成 token 数
  api_key_env: "OPENAI_API_KEY"   # API Key 环境变量名
```

### Stage 1 配置

```yaml
stage1:
  strategy: "hybrid"               # 策略：hybrid/vector/keyword/metadata
  vector_k: 200                   # 向量检索返回数量
  keyword_k: 100                  # 关键词检索返回数量
  final_k: 200                    # 最终候选池大小
  min_score: 0.0                  # 最低分数阈值
```

### Stage 2 配置

```yaml
stage2:
  use_llm_judge: true             # 使用 LLM 进行相关性判断
  use_cross_encoder: false        # 使用交叉编码器重排序
  cross_encoder_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  max_contexts: 10                # 最大上下文数量
  min_relevance_score: 0.5       # 最低相关性分数
  enable_subquestion_decomposition: true  # 启用子问题分解
```

### Stage 3 配置

```yaml
stage3:
  max_context_tokens: 4000        # 最大上下文 token 数
  require_citations: true         # 要求引用
  output_format: "structured"     # 输出格式
  confidence_threshold: 0.6       # 置信度阈值
```

---

## 🔌 API 接口

### 启动 API 服务器

```bash
cd api
python app.py
# 或
uvicorn api.app:app --reload
```

### 查询接口

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is multi-stage retrieval?",
    "dry_run_stage": null
  }'
```

**响应示例**：
```json
{
  "answer": {
    "answer": "Multi-stage retrieval is...",
    "citations": [...],
    "confidence": 0.85,
    "used_contexts": ["doc1", "doc2"]
  },
  "trace": {
    "query": "...",
    "query_plan": {...},
    "stage1_candidates": [...],
    "stage2_contexts": [...],
    "execution_time": 2.34
  },
  "success": true
}
```

---

## 🎓 使用场景

### 1. 学术研究 📚
- 文献综述和比较分析
- 研究问题探索
- 概念解释和定义

### 2. 企业知识库 💼
- 内部文档检索
- 技术问题解答
- 最佳实践查询

### 3. 客户支持 🤝
- FAQ 检索
- 问题诊断
- 解决方案推荐

### 4. 代码文档 📖
- API 文档查询
- 代码示例搜索
- 最佳实践查找

---

## 🔧 高级功能

### 自定义检索策略

```python
# 只使用向量检索
retriever = Stage1BroadRetriever(
    vector_store, doc_store,
    config={"strategy": "vector", "vector_k": 200}
)

# 只使用关键词检索
retriever = Stage1BroadRetriever(
    vector_store, doc_store,
    config={"strategy": "keyword", "keyword_k": 100}
)
```

### 自定义精炼逻辑

```python
# 禁用 LLM 判断，只用分数过滤
refiner = Stage2Refiner(
    llm_client, ranker,
    config={"use_llm_judge": False, "min_relevance_score": 0.7}
)
```

### 保存和加载执行轨迹

```python
# 轨迹自动保存到 data/traces/
# 可以后续分析检索效果
import json
with open("data/traces/trace_20240101_120000.json") as f:
    trace = json.load(f)
    # 分析 trace...
```

---

## 🧪 测试

```bash
# 运行基础测试
python test_basic.py

# 运行示例
python example_usage.py
```

---

## 📊 性能优化建议

1. **向量存储选择**：
   - ChromaDB：适合开发和小规模部署
   - FAISS：适合大规模生产环境

2. **LLM 调用优化**：
   - 启用缓存（如果支持）
   - 批量处理候选（Stage 2）
   - 使用更快的模型进行初步判断

3. **检索参数调优**：
   - 根据文档库大小调整 k 值
   - 根据查询复杂度调整精炼阈值

---

## 🚧 后续扩展方向

- [ ] 🔗 Multi-hop reasoning（多跳推理）
- [ ] 🤖 Agent-based retrieval（基于 Agent 的检索）
- [ ] 📊 Uncertainty estimation（不确定性估计）
- [ ] 👥 Human-in-the-loop feedback（人机协作反馈）
- [ ] 🌐 多语言支持
- [ ] 📈 性能监控和可视化
- [ ] 🔄 增量更新支持

---

## 📝 重要说明

> **LayeredRetriever 不是"检索增强生成"，而是"逐层压缩问题空间的认知检索系统"。**

我们模拟的是人类专家的思维过程：
1. **先广泛收集**可能相关的信息（Stage 1）
2. **再仔细筛选**真正相关的部分（Stage 2）
3. **最后精准合成**答案（Stage 3）

每一步都是可解释、可调试、可优化的。🎯

---

## 📄 许可证

MIT License

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给个 Star！⭐**

Made with ❤️ by LayeredRetriever Team

</div>
