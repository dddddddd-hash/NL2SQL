"""
Schema Linking 模块
功能：
1. 显式字段级匹配（关键词 -> 表.字段）
2. 关键词精确匹配兜底（补充向量检索盲区）
3. 可解释性输出（Used tables / columns）
"""

import re
import logging
from utils.db_connector import db_connector

logger = logging.getLogger(__name__)


class SchemaLinker:
    def __init__(self):
        self.field_synonyms: dict[str, list[str]] = {
            # bike_sales
            "利润": ["Profit"],
            "收入": ["Revenue"],
            "成本": ["Cost"],
            "数量": ["Order_Quantity"],
            "单价": ["Unit_Price"],
            "单成本": ["Unit_Cost"],
            "日期": ["Date", "date"],
            "月份": ["Month"],
            "年份": ["Year"],
            "国家": ["Country"],
            "州": ["State"],
            "性别": ["Customer_Gender"],
            "年龄": ["Customer_Age"],
            "年龄组": ["Age_Group"],
            "产品": ["Product", "name"],
            "产品类别": ["Product_Category"],
            "子类别": ["Sub_Category"],
            # detail
            "游戏": ["name"],
            "游戏名": ["name"],
            "分类": ["kind"],
            "类型": ["kind"],
            "上架": ["date"],
            "上线": ["date"],
        }

    def _get_full_schema(self) -> dict[str, list[str]]:
        """获取完整数据库结构（带缓存避免重复查询）"""
        if not hasattr(self, "_schema_cache"):
            self._schema_cache = db_connector.get_schema_info()
        return self._schema_cache

    def link(
        self,
        user_query: str,
        vector_results: list[dict],
        keywords: list[str],
    ) -> dict:
        """
        主入口：对向量检索结果做字段级精确匹配增强
        
        返回结构：
        {
            "linked_tables": [...],        # 确认使用的表
            "linked_columns": {...},       # 表 -> 字段列表
            "schema_context": [...],       # 增强后的 schema_context（传给 SQL 生成）
            "explanation": "..."           # 可解释性说明
        }
        """
        full_schema = self._get_full_schema()

        vector_tables = {s["table_name"]: s for s in vector_results}

        keyword_matched: dict[str, set[str]] = {}

        all_terms = set(keywords) | set(re.findall(r"[\u4e00-\u9fa5]+|[a-zA-Z_]+", user_query))

        for term in all_terms:
            matched_fields = self.field_synonyms.get(term, [])
            for field in matched_fields:
                for table, cols in full_schema.items():
                    if field in cols:
                        if table not in keyword_matched:
                            keyword_matched[table] = set()
                        keyword_matched[table].add(field)

        all_tables = set(vector_tables.keys()) | set(keyword_matched.keys())

        linked_tables = []
        linked_columns: dict[str, list[str]] = {}
        schema_context: list[dict] = []
        explanation_parts = []

        for table in all_tables:
            cols = full_schema.get(table, [])
            if not cols:
                continue

            matched_cols = list(keyword_matched.get(table, set()))
            source = []

            if table in vector_tables:
                source.append(f"向量检索(score={vector_tables[table]['score']:.2f})")
            if matched_cols:
                source.append(f"关键词匹配字段:{matched_cols}")

            linked_tables.append(table)
            linked_columns[table] = cols

            schema_context.append({
                "table_name": table,
                "columns": cols,
                "schema_text": f"表名: {table}, 字段: {', '.join(cols)}",
                "score": vector_tables.get(table, {}).get("score", 0.0),
                "matched_columns": matched_cols,
                "source": ", ".join(source),
            })

            explanation_parts.append(
                f"表[{table}] 来源:{', '.join(source)} 命中字段:{matched_cols or '无精确匹配'}"
            )

        explanation = "\n".join(explanation_parts) if explanation_parts else "未匹配到任何表"
        logger.info(f"Schema Linking 结果:\n{explanation}")

        return {
            "linked_tables": linked_tables,
            "linked_columns": linked_columns,
            "schema_context": schema_context,
            "explanation": explanation,
        }


schema_linker = SchemaLinker()
