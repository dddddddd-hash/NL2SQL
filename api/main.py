from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, AsyncGenerator
import asyncio
import json
import uuid
import logging
import time
from pathlib import Path
import decimal
import datetime

from graph.analysis_graph import analysis_app, AnalysisState
from core.vector_store import vector_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HomeRAG - 数据智能分析管理",
    description="NL2SQL",
    version="1.0.0",
)

reports_dir = Path("reports")
reports_dir.mkdir(parents=True, exist_ok=True)
app.mount("/reports", StaticFiles(directory=str(reports_dir)), name="reports")

web_dir = Path("web")
web_dir.mkdir(parents=True, exist_ok=True)
app.mount("/web", StaticFiles(directory=str(web_dir)), name="web")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    session_id: str
    query: str
    sql: Optional[str]
    results: Optional[list]
    answer: str
    steps: list
    elapsed_ms: int


class IndexRequest(BaseModel):
    sql_examples: Optional[list] = None


async def sse_event_generator(
    query: str, session_id: str
) -> AsyncGenerator[str, None]:

    NODE_NAME_MAP = {
        "keyword_extraction": "关键词提取",
        "vector_retrieval":   "向量检索",
        "schema_linking":     "Schema链接",
        "sql_generation":     "SQL生成",
        "sql_validation":     "SQL校验",
        "sql_execution":      "SQL执行",
        "answer_generation":  "回答生成",
        "error_handler":      "错误处理",
    }

    def json_default(obj):
        if isinstance(obj, decimal.Decimal):
            return float(obj)
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return str(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    def send_event(event_type: str, data: dict) -> str:
        payload = json.dumps(
            {"type": event_type, "data": data},
            ensure_ascii=False,
            default=json_default,
        )
        return f"data: {payload}\n\n"

    try:
        yield send_event("start", {"message": "开始处理查询...", "query": query})
        await asyncio.sleep(0.1)

        initial_state: AnalysisState = {
            "user_query": query,
            "session_id": session_id,
            "keyword_analysis": None,
            "schema_context": None,
            "sql_examples": None,
            "schema_linking_result": None,
            "error_type": None,
            "generated_sql": None,
            "validation_result": None,
            "last_validation_error": None,
            "query_results": None,
            "final_answer": None,
            "error": None,
            "steps": [],
            "retry_count": 0,
        }

        config = {"configurable": {"thread_id": session_id}}

        async for event in analysis_app.astream(initial_state, config=config):
            for node_name, node_state in event.items():
                if node_name == "__end__":
                    continue

                step_msg = ""
                extra_data = {}

                if node_name == "keyword_extraction":
                    kw = node_state.get("keyword_analysis", {}) or {}
                    step_msg = f"关键词提取: {', '.join(kw.get('keywords', []))}"
                    extra_data = {"keywords": kw.get("keywords", [])}

                elif node_name == "vector_retrieval":
                    schemas = node_state.get("schema_context", []) or []
                    step_msg = f"检索到 {len(schemas)} 个相关表"
                    extra_data = {"tables": [s["table_name"] for s in schemas]}

                elif node_name == "schema_linking":
                    result = node_state.get("schema_linking_result", {}) or {}
                    tables = result.get("linked_tables", [])
                    step_msg = f"确认 {len(tables)} 个关联表: {', '.join(tables)}"
                    extra_data = {"linked_tables": tables}

                elif node_name == "sql_generation":
                    sql = node_state.get("generated_sql", "")
                    retry = node_state.get("retry_count", 0)
                    step_msg = f"SQL生成完成（第{retry}次）"
                    extra_data = {"sql": sql}
                    yield send_event("sql_generated", {"sql": sql})

                elif node_name == "sql_validation":
                    val = node_state.get("validation_result", {}) or {}
                    is_valid = val.get("is_valid", False)
                    error_type = node_state.get("error_type") or ""
                    if is_valid:
                        step_msg = "SQL语法校验通过"
                    else:
                        step_msg = f"SQL校验失败 [{error_type}]: {val.get('message', '')}"
                    extra_data = {"valid": is_valid, "error_type": error_type}

                elif node_name == "sql_execution":
                    results = node_state.get("query_results", []) or []
                    step_msg = f"查询返回 {len(results)} 条数据"
                    extra_data = {
                        "row_count": len(results),
                        "results_preview": results[:20],
                    }

                elif node_name == "answer_generation":
                    answer = node_state.get("final_answer", "") or ""
                    step_msg = "回答生成完成"
                    for char in answer:
                        yield send_event("answer_stream", {"char": char})
                        await asyncio.sleep(0.02)

                elif node_name == "error_handler":
                    step_msg = f"处理错误: {node_state.get('error', '未知错误')}"

                else:
                    step_msg = "处理中"

                yield send_event(
                    "step",
                    {
                        "node": NODE_NAME_MAP.get(node_name, node_name),
                        "message": step_msg,
                        **extra_data,
                    },
                )
                await asyncio.sleep(0.05)

        final_state = analysis_app.get_state(config).values
        yield send_event(
            "complete",
            {
                "sql": final_state.get("generated_sql"),
                "row_count": len(final_state.get("query_results") or []),
                "answer": final_state.get("final_answer"),
                "results_preview": (final_state.get("query_results") or [])[:20],
            },
        )

    except Exception as e:
        logger.exception(f"SSE error: {e}")
        yield send_event("error", {"message": str(e)})

    finally:
        yield "data: [DONE]\n\n"


@app.get("/api/v1/query/stream", summary="流式查询", description="SSE实时流式自然语言查询接口")
async def query_stream(
    q: str = Query(..., description="自然语言查询"),
    session_id: Optional[str] = Query(None, description="会话ID，不传则自动生成"),
):
    """SSE流式查询接口"""
    if not q.strip():
        raise HTTPException(status_code=400, detail="查询内容不能为空")

    sid = session_id or str(uuid.uuid4())

    return StreamingResponse(
        sse_event_generator(q, sid),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/api/v1/query", response_model=QueryResponse, summary="同步查询", description="同步自然语言转SQL查询接口，返回完整分析结果")
async def query(request: QueryRequest):
    """同步查询接口"""
    start_time = time.time()
    session_id = request.session_id or str(uuid.uuid4())

    initial_state: AnalysisState = {
        "user_query": request.query,
        "session_id": session_id,
        "keyword_analysis": None,
        "schema_context": None,
        "sql_examples": None,
        "schema_linking_result": None,
        "error_type": None,
        "generated_sql": None,
        "validation_result": None,
        "last_validation_error": None,
        "query_results": None,
        "final_answer": None,
        "error": None,
        "steps": [],
        "retry_count": 0,
    }

    config = {"configurable": {"thread_id": session_id}}
    final_state = await analysis_app.ainvoke(initial_state, config=config)

    elapsed = int((time.time() - start_time) * 1000)

    return QueryResponse(
        session_id=session_id,
        query=request.query,
        sql=final_state.get("generated_sql"),
        results=final_state.get("query_results"),
        answer=final_state.get("final_answer", "处理失败"),
        steps=final_state.get("steps", []),
        elapsed_ms=elapsed,
    )


@app.post("/api/v1/admin/index-schema", summary="索引数据库结构", description="重新扫描并索引数据库Schema到向量库")
async def index_schema():
    """重新索引数据库Schema"""
    try:
        vector_store.index_schema()
        return {"status": "success", "message": "Schema索引完成"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/admin/index-examples", summary="索引SQL示例", description="将用户提供的SQL示例存入向量库，用于增强查询生成质量")
async def index_examples(request: IndexRequest):
    """索引SQL示例"""
    if not request.sql_examples:
        raise HTTPException(status_code=400, detail="请提供SQL示例")
    try:
        vector_store.index_sql_examples(request.sql_examples)
        return {"status": "success", "count": len(request.sql_examples)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/health", summary="服务器状态检查", description="检查服务运行状态")
async def health():
    return {"status": "healthy", "service": "HomeRAG"}


@app.get("/", include_in_schema=False)
async def home():
    """查询页面入口"""
    index_file = web_dir / "query.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="查询页面不存在")
    return FileResponse(index_file)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
