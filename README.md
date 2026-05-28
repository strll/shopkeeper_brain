# 掌柜智库 (Shopkeeper Brain)

基于 RAG（检索增强生成）的知识库问答系统，专为产品技术文档场景设计。支持将 PDF/ Markdown 产品手册自动导入向量知识库，并提供智能化的多路检索问答服务。

## 目录

- [核心功能](#核心功能)
- [系统架构](#系统架构)
- [技术栈](#技术栈)
- [项目结构](#项目结构)
- [导入流水线](#导入流水线)
- [查询流水线](#查询流水线)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [API 接口](#api-接口)
- [前端页面](#前端页面)

---

## 核心功能

### 1. 文档智能导入

- **PDF 解析**：通过 MinerU 将 PDF 文档转换为 Markdown，保留表格、标题等结构化信息
- **图片理解**：使用 VLM（视觉语言模型）为文档中的图片自动生成中文摘要
- **智能分块**：基于标题层级进行文档分割，自动合并过短段落，拆分过长段落
- **表格处理**：将 HTML/Markdown 表格转换为自然语言键值对格式，提升检索效果
- **产品识别**：通过 LLM 自动提取文档中的核心产品名称（品牌+型号+品类）
- **向量化存储**：使用 BGE-M3 模型生成稠密+稀疏混合向量，存入 Milvus

### 2. 智能问答检索

- **产品名称消歧**：从用户问题中提取产品名称，并在产品名称库中进行模糊匹配和消歧
- **多路检索融合**：
  - 混合向量搜索（稠密 + 稀疏）
  - HyDE 搜索（生成假设性文档增强召回）
  - Web 搜索（通过 DashScope MCP 联网搜索）
- **RRF 融合**：使用倒数排序融合算法合并多路检索结果
- **BGE 重排序**：使用 BGE-reranker 对候选文档精排，结合断崖截断策略过滤低相关文档
- **流式输出**：支持 SSE（Server-Sent Events）流式推送答案
- **对话历史**：基于 MongoDB 存储多轮对话历史

---

## 系统架构

```
┌──────────────────────────────────────────────────────────┐
│                      前端 (HTML/CSS/JS)                    │
│              import.html  │  chat.html                     │
└────────────────────────┬─────────────────────────────────┘
                         │ HTTP/SSE
┌────────────────────────┴─────────────────────────────────┐
│                   FastAPI 网关层                            │
│        import_router (导入服务)  │  query_router (查询服务)   │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────┴─────────────────────────────────┐
│                 LangGraph 工作流编排                        │
│  ┌──────────────────┐    ┌──────────────────┐             │
│  │  导入流水线       │    │  查询流水线       │             │
│  │  (8个节点)       │    │  (6个节点)        │             │
│  └──────────────────┘    └──────────────────┘             │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────┴─────────────────────────────────┐
│                      基础服务层                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────────┐  │
│  │  Milvus  │ │ MongoDB  │ │  MinIO   │ │  DashScope   │  │
│  │ 向量数据库│ │ 对话历史  │ │ 对象存储  │ │  LLM API     │  │
│  └──────────┘ └──────────┘ └──────────┘ └─────────────┘  │
└──────────────────────────────────────────────────────────┘
```

### 设计亮点

- **混合向量检索**：稠密向量负责语义匹配，稀疏向量负责关键词匹配，两者互补提升召回率
- **HyDE（假设文档嵌入）**：让 LLM 先生成一个假设性答案段落，用其进行向量检索，缓解用户查询与文档之间的语义鸿沟
- **断崖截断**：重排序后检测相邻文档分数骤降点，自动截断低质量结果，减少噪声干扰
- **产品名称前置过滤**：在检索前先锁定目标产品，避免跨产品的文档干扰
- **双重锁单例模式**：客户端管理器使用线程安全的双重检查锁定，确保全局唯一实例

---

## 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| **Web 框架** | FastAPI + Uvicorn | 异步 REST API 服务 |
| **工作流引擎** | LangGraph | 有状态 DAG 流水线编排 |
| **LLM 集成** | langchain-openai / DashScope | Qwen 系列模型（Qwen-Flash, Qwen3-VL-Flash） |
| **嵌入模型** | BGE-M3 | 稠密 1024 维 + 稀疏向量混合嵌入 |
| **重排序模型** | BGE-reranker-large | FlagEmbedding 本地部署 |
| **向量数据库** | Milvus | 存储文档块向量和产品名称向量 |
| **文档数据库** | MongoDB | 对话历史持久化 |
| **对象存储** | MinIO | 源文档和图片存储 |
| **PDF 解析** | MinerU | 布局感知的 PDF 转 Markdown |
| **Web 搜索** | DashScope MCP | 阿里云百炼联网搜索 |
| **前端** | 原生 HTML/CSS/JS | 无框架单页应用 |

### 外部服务依赖

| 服务 | 用途 |
|------|------|
| Milvus (`192.168.200.148:19530`) | 向量存储与混合检索 |
| MongoDB (`192.168.200.148:27017`) | 对话历史 |
| MinIO (`192.168.200.148:9000`) | 文件/图片对象存储 |
| DashScope API | LLM 推理、VLM 图片理解、Web 搜索 |

---

## 项目结构

```
shopkeeper_brain/
├── knowledge/                              # 主应用包
│   ├── api/                                # FastAPI 路由层
│   │   ├── import_router.py                # 导入服务（文件上传、任务状态查询）
│   │   └── query_router.py                 # 查询服务（问答接口、对话历史）
│   ├── core/                               # 核心模块
│   │   ├── deps.py                         # 依赖注入（缓存化服务工厂）
│   │   └── path.py                         # 路径常量
│   ├── front/                              # 前端页面
│   │   ├── import.html                     # 文件导入页面
│   │   └── chat.html                       # 问答聊天页面
│   ├── processor/                          # 流水线处理器
│   │   ├── import_process/                 # 导入流水线
│   │   │   ├── main_graph.py               # LangGraph 状态图定义
│   │   │   ├── state.py                    # 状态类型定义
│   │   │   ├── config.py                   # 配置数据类
│   │   │   ├── base.py                     # 节点基类
│   │   │   └── nodes/                      # 处理节点
│   │   │       ├── entry_node.py           # 入口：文件类型校验
│   │   │       ├── pdf_to_md_node.py       # PDF → Markdown
│   │   │       ├── md_to_img_node.py       # 图片摘要 + MinIO 上传
│   │   │       ├── document_split_node.py  # 文档分块
│   │   │       ├── item_name_recognition_node.py  # 产品名称提取
│   │   │       ├── embedding_chunks_node.py       # 向量嵌入
│   │   │       └── import_milvus_node.py          # Milvus 写入
│   │   └── query_process/                  # 查询流水线
│   │       ├── main_graph.py               # LangGraph 查询图定义
│   │       ├── state.py                    # 状态类型定义
│   │       ├── config.py                   # 查询配置
│   │       ├── base.py                     # 节点基类
│   │       └── nodes/                      # 查询节点
│   │           ├── item_name_confirmed_node.py    # 产品名称确认
│   │           ├── hybrid_vector_search_node.py   # 混合向量搜索
│   │           ├── hyde_vector_search_node.py     # HyDE 搜索
│   │           ├── web_mcp_search_node.py         # Web MCP 搜索
│   │           ├── rrf_merge_node.py              # RRF 融合
│   │           └── answer_output_node.py          # 答案生成
│   ├── prompt/                             # 提示词模板
│   │   ├── import_prompt.py                # 导入相关提示词
│   │   └── query/                          # 查询相关提示词
│   ├── schema/                             # Pydantic 数据模型
│   │   ├── upload_schema.py
│   │   ├── query_schema.py
│   │   └── task_schema.py
│   ├── services/                           # 业务服务层
│   │   ├── upload_service.py               # 上传服务
│   │   ├── import_file_service.py          # 导入编排
│   │   ├── query_service.py                # 查询编排
│   │   └── task_service.py                 # 任务状态管理
│   └── utils/                              # 工具模块
│       ├── client/                         # 客户端管理器
│       │   ├── base.py                     # 双重锁单例基类
│       │   ├── ai_clients.py               # AI 客户端（LLM/VLM/嵌入/重排序）
│       │   └── storage_clients.py          # 存储客户端（MinIO/Milvus）
│       ├── task_util.py                    # 任务状态追踪
│       ├── sse_util.py                     # SSE 推送工具
│       ├── mongo_history_util.py           # MongoDB 历史管理
│       ├── milvus_util.py                  # Milvus 混合搜索
│       ├── markdown_util.py                # Markdown 表格处理
│       └── bge_rerank_util.py             # BGE 重排序工具
├── .env                                    # 环境变量配置
└── requirements.txt                        # Python 依赖
```

---

## 导入流水线

将 PDF/Markdown 产品文档自动处理并存入向量知识库：

```
文件上传 → 入口校验 → PDF转MD → 图片摘要 → 文档分块 → 产品识别 → 向量嵌入 → Milvus写入
```

### 各节点说明

| 步骤 | 节点 | 说明 |
|------|------|------|
| 1 | **EntryNode** | 校验文件类型（PDF/MD），提取文件名，设置处理标志 |
| 2 | **PdfToMdNode** | 调用 MinerU CLI 将 PDF 转为 Markdown，含 10 分钟超时保护 |
| 3 | **MdToImgNode** | 提取 MD 中的图片 → VLM 生成中文摘要 → 上传至 MinIO → 替换图片链接为 URL+摘要 |
| 4 | **DocumentSplitNode** | 按标题层级切分，合并短段落，拆分长段落，表格转自然语言 |
| 5 | **ItemNameRecognitionNode** | LLM 提取产品名称（品牌+型号+品类）→ BGE-M3 向量化 → 存入 Milvus 产品名称库 |
| 6 | **EmbeddingChunksNode** | 批量生成 BGE-M3 稠密+稀疏向量（batch_size=8） |
| 7 | **ImportMilvusNode** | 创建/复用 `kb_chunks_v1` 集合，批量写入向量数据 |

---

## 查询流水线

用户提问后，经过多路检索和精排，生成带引用的答案：

```
用户提问 → 产品名称确认 → ┬─ 混合向量搜索 ─┐
                          ├─ HyDE 搜索 ─────┼─ RRF融合 → BGE重排序 → 答案生成 → SSE流式输出
                          └─ Web MCP 搜索 ─┘
```

### 各节点说明

| 步骤 | 节点 | 说明 |
|------|------|------|
| 1 | **ItemNameConfirmedNode** | LLM 提取查询中的产品名称 + 改写查询 → Milvus 产品名称库匹配 → 消歧确认 |
| 2 | **HybridVectorSearchNode** | 使用产品名称过滤，在 `kb_chunks_v1` 中进行稠密+稀疏混合检索 |
| 3 | **HydeVectorSearchNode** | LLM 生成假设性文档段落 → 拼接到查询 → 混合向量检索 |
| 4 | **WebMcpSearchNode** | 通过 DashScope MCP 协议进行联网搜索 |
| 5 | **RRFMergeNode** | 倒数排序融合：`weight / (k + rank)`，去重合并多路结果 |
| 6 | **AnswerOutputNode** | BGE-reranker 重排序 + 断崖截断 → LLM 生成答案（含图片URL）→ SSE 推送 → 写入 MongoDB 历史 |

### 检索策略

- **混合搜索参数**：稠密向量使用 COSINE 距离，稀疏向量使用 IP（内积）
- **产品过滤**：仅在产品置信度 ≥ 0.75 时使用精确过滤，0.55-0.75 之间提示用户确认
- **RRF 权重**：混合搜索权重 1.0，HyDE 权重 0.8，Web 搜索权重 0.6
- **断崖截断**：检测相邻文档分数骤降点，自动截断低相关噪声

---

## 快速开始

### 环境要求

- Python 3.12+
- Milvus 向量数据库（已部署）
- MongoDB（已部署）
- MinIO 对象存储（已部署）
- MinerU PDF 解析工具

### 安装

```bash
cd knowledge
pip install -r requirements.txt
```

### 配置

编辑 `knowledge/.env` 文件，配置以下关键参数：

```env
# DashScope API
DASHSCOPE_API_KEY=your_api_key
DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1

# Milvus
MILVUS_URI=http://192.168.200.148:19530
MILVUS_TOKEN=

# MongoDB
MONGO_URI=mongodb://admin:123456@192.168.200.148:27017

# MinIO
MINIO_ENDPOINT=192.168.200.148:9000
MINIO_ACCESS_KEY=admin
MINIO_SECRET_KEY=12345678

# 模型路径
BGE_M3_MODEL_PATH=E:/ai_models/BAAI/bge-m3
BGE_RERANKER_MODEL_PATH=E:/ai_models/BAAI/bge-reranker-large
```

### 启动服务

```bash
# 启动导入服务（端口 8001）
python -m knowledge.api.import_router

# 启动查询服务（端口 8001）
python -m knowledge.api.query_router
```

### 使用流程

1. 打开 `http://localhost:8001/front/import.html` — 上传产品 PDF/Markdown 文档
2. 等待导入流水线完成（观察进度条和日志）
3. 打开 `http://localhost:8001/front/chat.html` — 输入问题进行智能问答

---

## API 接口

### 导入服务

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/upload` | 上传文件，启动导入流水线 |
| `GET` | `/status/{task_id}` | 查询任务状态和节点进度 |

### 查询服务

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/query/stream` | 流式问答（SSE） |
| `POST` | `/query` | 非流式问答 |
| `GET` | `/history` | 获取对话历史 |
| `DELETE` | `/history` | 清空对话历史 |

---

## 前端页面

- **import.html**：深色主题文件上传页，支持拖拽上传、实时进度条、展开式日志面板
- **chat.html**：深色主题聊天界面，支持 Markdown 渲染、图片 URL 自动解析、流式/非流式输出切换、对话历史管理

---

## AI 模型清单

| 模型 | 部署方式 | 用途 |
|------|----------|------|
| BGE-M3 | 本地（`E:\ai_models\BAAI\bge-m3`） | 文本嵌入（稠密+稀疏） |
| BGE-reranker-large | 本地（`E:\ai_models\BAAI\bge-reranker-large`） | 文档重排序 |
| Qwen-Flash | DashScope API | LLM 推理（名称提取、查询改写、答案生成、HyDE 生成） |
| Qwen3-VL-Flash | DashScope API | 图片理解与摘要生成 |
| MinerU | 本地 CLI | PDF 布局解析与 OCR |