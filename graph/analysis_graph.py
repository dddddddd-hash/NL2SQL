from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Optional
import operator
import logging

from core.keyword_extractor import keyword_extractor
from core.vector_store import vector_store
from core.schema_linker import schema_linker
from core.sql_generator import sql_generator
from utils.db_connector import db_connector

logger = logging.getLogger(__name__)


class AnalysisState(TypedDict):
    """LangGraph状态定义"""
    user_query: str
    session_id: str

    keyword_analysis: Optional[dict]
    schema_context: Optional[list]
    sql_examples: Optional[list]

    schema_linking_result: Optional[dict]

    error_type: Optional[str]

    generated_sql: Optional[str]
    validation_result: Optional[dict]
    last_validation_error: Optional[str]
    query_results: Optional[list]

    final_answer: Optional[str]
    error: Optional[str]

    steps: Annotated[list, operator.add]
    retry_count: int


def node_keyword_extraction(state: AnalysisState) -> dict:
    """关键词提取与意图分析"""
    logger.info("=== Node: keyword_extraction ===")
    try:
        analysis = keyword_extractor.analyze(state["user_query"])
        return {
            "keyword_analysis": analysis,
            "steps": [f"关键词提取完成: {analysis['keywords']}"],
        }
    except Exception as e:
        return {"error": f"关键词提取失败: {str(e)}", "steps": ["关键词提取失败"]}


def node_vector_retrieval(state: AnalysisState) -> dict:
    """向量检索"""
    logger.info("=== Node: vector_retrieval ===")
    try:
        query = state["user_query"]
        schema_context = vector_store.search_schema(query)
        sql_examples = vector_store.search_examples(query)
        logger.info(f"Retrieved {len(schema_context)} schemas, {len(sql_examples)} examples")
        logger.info(f"SQL Examples: {sql_examples}")
        return {
            "schema_context": schema_context,
            "sql_examples": sql_examples,
            "steps": [f"向量检索: {len(schema_context)}个相关表, {len(sql_examples)}个示例"],
        }
    except Exception as e:
        return {"error": f"向量检索失败: {str(e)}", "steps": ["向量检索失败"]}


def node_schema_linking(state: AnalysisState) -> dict:
    """显式字段级Schema匹配"""
    logger.info("=== Node: schema_linking ===")
    try:
        keywords = state.get("keyword_analysis", {}).get("keywords", [])
        vector_results = state.get("schema_context", [])

        result = schema_linker.link(
            user_query=state["user_query"],
            vector_results=vector_results,
            keywords=keywords,
        )
        return {
            "schema_linking_result": result,
            "schema_context": result["schema_context"],
            "steps": [
                f"Schema链接: 确认表{result['linked_tables']}, "
                f"命中字段来源: {len(result['linked_tables'])}个表"
            ],
        }
    except Exception as e:
        logger.warning(f"Schema Linking 失败，使用原向量结果: {e}")
        return {"steps": [f"Schema Linking 跳过: {e}"]}


def node_sql_generation(state: AnalysisState) -> dict:
    """SQL生成（支持分级修复）"""
    logger.info("=== Node: sql_generation ===")
    try:
        last_error = state.get("last_validation_error")
        error_type = state.get("error_type")

        if last_error:
            sql = sql_generator.generate_with_fix(
                user_query=state["user_query"],
                schema_context=state["schema_context"],
                sql_examples=state["sql_examples"],
                keyword_analysis=state["keyword_analysis"],
                last_validation_error=last_error,
                error_type=error_type,
            )
        else:
            sql = sql_generator.generate(
                user_query=state["user_query"],
                schema_context=state["schema_context"],
                sql_examples=state["sql_examples"],
                keyword_analysis=state["keyword_analysis"],
                last_validation_error=last_error,
            )

        logger.info(f"Generated SQL: {sql}")
        return {
            "generated_sql": sql,
            "last_validation_error": None,
            "error_type": None,
            "retry_count": state.get("retry_count", 0) + 1,
            "steps": [f"SQL生成: {sql[:80]}..."],
        }
    except Exception as e:
        return {"error": f"SQL生成失败: {str(e)}", "steps": ["SQL生成失败"]}


def node_sql_validation(state: AnalysisState) -> dict:
    """SQL语法校验"""
    logger.info("=== Node: sql_validation ===")
    try:
        is_valid, msg = sql_generator.validate_syntax(state["generated_sql"])
        validation_result = {"is_valid": is_valid, "message": msg}
        logger.info(f"SQL validation: {validation_result}")

        error_type = None
        if not is_valid:
            error_type = sql_generator.classify_error(msg)
            logger.info(f"错误分类: {error_type}")

        return {
            "validation_result": validation_result,
            "last_validation_error": None if is_valid else msg,
            "error_type": error_type,
            "steps": [
                f"SQL校验: {'通过' if is_valid else f'失败[{error_type}]'} - {msg}"
            ],
        }
    except Exception as e:
        return {
            "validation_result": {"is_valid": False, "message": str(e)},
            "last_validation_error": str(e),
            "error_type": "unknown",
            "steps": ["SQL校验异常"],
        }


def node_sql_execution(state: AnalysisState) -> dict:
    """执行SQL查询"""
    logger.info("=== Node: sql_execution ===")
    try:
        results = db_connector.execute_query(state["generated_sql"])
        logger.info(f"查询返回 {len(results)} 条记录")
        return {
            "query_results": results,
            "steps": [f"SQL执行: 返回{len(results)}条记录"],
        }
    except Exception as e:
        return {"error": f"SQL执行失败: {str(e)}", "steps": ["SQL执行失败"]}


def node_answer_generation(state: AnalysisState) -> dict:
    """自然语言回答生成"""
    logger.info("=== Node: answer_generation ===")
    try:
        answer = sql_generator.format_result(
            results=state["query_results"],
            user_query=state["user_query"],
        )
        return {"final_answer": answer, "steps": ["回答生成完成"]}
    except Exception as e:
        logger.error(f"answer_generation 失败: {e}", exc_info=True)
        return {"error": f"回答生成失败: {str(e)}", "steps": ["回答生成失败"]}


def node_error_handler(state: AnalysisState) -> dict:
    """错误处理节点"""
    logger.error(f"Error in pipeline: {state.get('error')}")
    return {
        "final_answer": f"抱歉，处理您的查询时遇到问题：{state.get('error', '未知错误')}",
        "steps": ["错误处理完成"],
    }


def route_after_extraction(state: AnalysisState) -> str:
    return "error" if state.get("error") else "continue"


def route_after_vector(state: AnalysisState) -> str:
    return "error" if state.get("error") else "continue"


def route_after_execution(state: AnalysisState) -> str:
    return "error" if state.get("error") else "answer"


def route_after_validation(state: AnalysisState) -> str:
    """SQL校验后路由：通过 -> 执行，失败 -> 重试或报错"""
    if state.get("error"):
        return "error"
    if state.get("validation_result", {}).get("is_valid"):
        return "execute"
    retry_count = state.get("retry_count", 0)
    if retry_count >= 2:
        return "error"
    return "regenerate"


def build_analysis_graph():
    """构建LangGraph分析工作流"""

    workflow = StateGraph(AnalysisState)

    workflow.add_node("keyword_extraction", node_keyword_extraction)
    workflow.add_node("vector_retrieval", node_vector_retrieval)
    workflow.add_node("schema_linking", node_schema_linking)
    workflow.add_node("sql_generation", node_sql_generation)
    workflow.add_node("sql_validation", node_sql_validation)
    workflow.add_node("sql_execution", node_sql_execution)
    workflow.add_node("answer_generation", node_answer_generation)
    workflow.add_node("error_handler", node_error_handler)

    workflow.set_entry_point("keyword_extraction")

    workflow.add_conditional_edges(
        "keyword_extraction",
        route_after_extraction,
        {"continue": "vector_retrieval", "error": "error_handler"},
    )
    workflow.add_conditional_edges(
        "vector_retrieval",
        route_after_vector,
        {"continue": "schema_linking", "error": "error_handler"},
    )
    workflow.add_edge("schema_linking", "sql_generation")
    workflow.add_edge("sql_generation", "sql_validation")
    workflow.add_conditional_edges(
        "sql_validation",
        route_after_validation,
        {
            "execute": "sql_execution",
            "regenerate": "sql_generation",
            "error": "error_handler",
        },
    )
    workflow.add_conditional_edges(
        "sql_execution",
        route_after_execution,
        {
            "answer": "answer_generation",
            "error": "error_handler",
        },
    )
    workflow.add_edge("answer_generation", END)
    workflow.add_edge("error_handler", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


analysis_app = build_analysis_graph()
