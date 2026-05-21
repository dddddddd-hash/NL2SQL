# HomeRAG NL2SQL

HomeRAG NL2SQL 是一个面向数据分析场景的自然语言转 SQL 项目。系统接收中文自然语言问题，通过关键词抽取、向量检索、Schema Linking、SQL 生成与校验、SQL 执行、结果总结等步骤，将问题转换为 MySQL 查询并返回结构化结果和自然语言回答。

## 功能特性

- 自然语言生成 MySQL `SELECT` 查询
- 基于 `BAAI/bge-m3` 的本地 Embedding 检索数据库 Schema
- 使用 Qdrant 存储表结构和 SQL 示例向量
- 基于 LangGraph 编排完整 NL2SQL 工作流
- 支持 SQL 安全校验、语法校验和失败重试修复
- 提供 FastAPI 接口、SSE 流式查询接口和静态查询页面
- 内置评估脚本，可生成 CSV、Markdown、HTML 和图表报告

## 技术栈

- Python 3.10+
- FastAPI / Uvicorn
- LangGraph
- OpenAI SDK（默认接入 DeepSeek API）
- Sentence Transformers
- Qdrant
- MySQL / PyMySQL
- Jieba
- Matplotlib
<img width="795" height="1190" alt="3661c78c66534fb553858c93de707d9f" src="https://github.com/user-attachments/assets/2a45ecf7-b872-4eb8-979a-e8ea093d758a" />

## 项目结构

```text
.
├── api/                  # FastAPI 服务入口和模型下载脚本
├── config/               # 项目配置
├── core/                 # NL2SQL 核心逻辑
├── eval/                 # 评估数据集与评估脚本
├── graph/                # LangGraph 工作流
├── utils/                # 数据库连接工具
├── web/                  # 前端查询页面
├── requirements.txt      # Python 依赖
└── README.md
```

## 环境准备

1. 创建并激活虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 启动 Qdrant：

```bash
docker run -p 6333:6333 qdrant/qdrant
```

4. 准备 MySQL 数据库，并设置必要环境变量：

```bash
export DEEPSEEK_API_KEY="your_api_key"
export MYSQL_HOST="localhost"
export MYSQL_PORT="3306"
export MYSQL_USER="root"
export MYSQL_PASSWORD="your_password"
export MYSQL_DB="your_database"
```

Windows PowerShell：

```powershell
$env:DEEPSEEK_API_KEY="your_api_key"
$env:MYSQL_HOST="localhost"
$env:MYSQL_PORT="3306"
$env:MYSQL_USER="root"
$env:MYSQL_PASSWORD="your_password"
$env:MYSQL_DB="your_database"
```

## 模型下载

首次运行前可以提前下载本地 Embedding 模型：

```bash
python api/download.py
```

模型会缓存到 `models/` 目录。该目录体积通常较大，已在 `.gitignore` 中排除。

## 启动服务

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

启动后访问：

- 查询页面：http://127.0.0.1:8000/
- API 文档：http://127.0.0.1:8000/docs
- 健康检查：http://127.0.0.1:8000/api/v1/health

## 初始化向量索引

服务启动并连接 MySQL、Qdrant 后，先索引数据库 Schema：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/admin/index-schema
```

也可以索引 SQL 示例：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/admin/index-examples \
  -H "Content-Type: application/json" \
  -d '{"sql_examples":[{"query":"统计每年的销售额","sql":"SELECT `Year`, SUM(`Revenue`) FROM `bike_sales` GROUP BY `Year`","description":"按年份统计销售额"}]}'
```

## 接口示例

同步查询：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query":"统计哪一年的销售额最高"}'
```

流式查询：

```text
GET /api/v1/query/stream?q=统计哪一年的销售额最高
```

## 评估

评估样例位于 `eval/dataset.sample.jsonl`。服务启动后执行：

```bash
python eval/run_eval.py
```

评估结果会输出到 `reports/eval/`，包括原始结果、CSV、Markdown、HTML 和指标图表。`reports/` 已在 `.gitignore` 中排除。

