from openai import OpenAI
from config.settings import settings
from utils.db_connector import db_connector
import re
import json
import logging

logger = logging.getLogger(__name__)

# SQL注入关键词黑名单
SQL_BLACKLIST = [
    "DROP", "DELETE", "UPDATE", "INSERT", "ALTER",
    "TRUNCATE", "CREATE", "REPLACE", "EXEC", "EXECUTE",
]


class SQLGenerator:
    def __init__(self):
        self.client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
        )

    def build_prompt(
        self,
        user_query: str,
        schema_context: list[dict],
        sql_examples: list[dict],
        keyword_analysis: dict,
        last_validation_error: str | None = None,
    ) -> str:
        """构建SQL生成Prompt"""

        if not schema_context:
            logger.warning("向量检索为空，使用数据库完整Schema兜底")
            db_schema = db_connector.get_schema_info()
            schema_context = [
                {
                    "table_name": table_name,
                    "columns": columns,
                }
                for table_name, columns in db_schema.items()
            ]

        FIELD_DESCRIPTIONS = {
            # detail 表字段
            "detail": {
                "name": "游戏名称（字符串）",
                "date": "游戏上架日期（格式：YYYY-MM-DD）",
                "kind": "游戏分类/类型（例如：动作、射击、休闲、益智等，是具体分类名，不是固定值'游戏'）",
            },
            # bike_sales 表字段
            "bike_sales": {
                "Date": "订单日期（格式：YYYY-MM-DD）",
                "Day": "订单日期中的天（数字1-31）",
                "Month": "订单月份（英文名称，如 November）",
                "Year": "订单年份（如 2013、2015）",
                "Customer_Age": "客户年龄（数字）",
                "Age_Group": "客户年龄分组（Youth (<25) / Adults (25-64) / Seniors (64+)）",
                "Customer_Gender": "客户性别（M=男性，F=女性）",
                "Country": "客户所在国家",
                "State": "客户所在州或省份",
                "Product_Category": "产品大类（如 Accessories、Bikes、Clothing）",
                "Sub_Category": "产品子类（如 Bike Racks、Helmets）",
                "Product": "具体产品名称",
                "Order_Quantity": "订单购买数量",
                "Unit_Cost": "单个产品的成本价",
                "Unit_Price": "单个产品的售价",
                "Profit": "该订单的利润（Revenue - Cost）",
                "Cost": "该订单的总成本（Unit_Cost × Order_Quantity）",
                "Revenue": "该订单的总收入（Unit_Price × Order_Quantity）",
            },
            # chocolate_sales 表字段
            "chocolate_sales": {
                "order_id": "订单唯一编号（如 ORD00000001）",
                "order_date": "订单日期（格式：YYYY-MM-DD）",
                "product_id": "产品ID，关联 chocolate_products 表",
                "store_id": "门店ID，关联 chocolate_stores 表",
                "customer_id": "客户ID，关联 chocolate_customers 表",
                "quantity": "购买数量",
                "unit_price": "单价",
                "discount": "折扣（0表示无折扣，0.1表示9折）",
                "revenue": "销售收入",
                "cost": "成本",
                "profit": "利润（revenue - cost）",
            },
            # chocolate_products 表字段
            "chocolate_products": {
                "product_id": "产品唯一编号（如 P0001）",
                "product_name": "产品名称（如 White Chocolate 80%、Dark Chocolate 70%，包含巧克力类型和可可含量）",
                "brand": "品牌名称（如 Mars、Cadbury、Hershey、Ferrero、Godiva、Lindt）",
                "category": "产品类别（如 Truffle、Praline、White、Dark、Milk）",
                "cocoa_percent": "可可含量百分比（数字，如 80 表示80%）",
                "weight_g": "产品重量（克）",
            },
            # chocolate_stores 表字段
            "chocolate_stores": {
                "store_id": "门店唯一编号（如 S001）",
                "store_name": "门店名称（如 Chocolate Store 1）",
                "city": "门店所在城市（如 New York、Melbourne、Berlin）",
                "country": "门店所在国家（如 Canada、France、UK、USA、Australia、Germany）",
                "store_type": "门店类型（Retail=零售店、Mall=商场、Airport=机场、Online=线上）",
            },
            # chocolate_customers 表字段
            "chocolate_customers": {
                "customer_id": "客户唯一编号（如 C000001）",
                "age": "客户年龄",
                "gender": "客户性别（Male=男性、Female=女性）",
                "loyalty_member": "是否忠诚会员（1=是、0=否）",
                "join_date": "入会/注册日期（格式：YYYY-MM-DD）",
            },
            # chocolate_calendar 表字段
            "chocolate_calendar": {
                "date": "日期（格式：YYYY-MM-DD）",
                "year": "年份（如 2023）",
                "month": "月份（数字1-12）",
                "day": "日（数字1-31）",
                "week": "该日期所在的第几周（数字）",
                "day_of_week": "星期几（数字，0=周一，6=周日）",
            },
        }

        def describe_columns(table_name, columns):
            parts = []
            table_desc = FIELD_DESCRIPTIONS.get(table_name, {})
            for col in columns:
                desc = table_desc.get(col, col)
                parts.append(f"`{col}`（{desc}）")
            return ", ".join(parts)

        schema_str = "\n".join([
            f"- 表 `{s['table_name']}`: 字段 {describe_columns(s['table_name'], s['columns'])}"
            for s in schema_context
        ])

        schema_note = """
## 重要说明
- `detail` 表存储的全部是游戏数据，无需用 `kind='游戏'` 来过滤
- `kind` 字段表示游戏的具体分类，如"动作"、"射击"、"休闲"等
- `bike_sales` 表存储自行车销售订单数据
- `bike_sales` 的 Profit = Revenue - Cost，已是计算好的字段，直接用即可
- `bike_sales` 的 Month 是英文字符串，按月份查询时注意用英文（如 'November'）
- 查询销售额用 Revenue，查询利润用 Profit，查询成本用 Cost
"""

        examples_str = ""
        if sql_examples:
            examples_str = "\n\n## 参考示例\n"
            for ex in sql_examples:
                examples_str += f"问题: {ex['query']}\nSQL: {ex['sql']}\n\n"

        validation_feedback = ""
        if last_validation_error:
            validation_feedback = (
                "\n## 上一轮SQL校验反馈\n"
                f"- 上一轮失败原因: {last_validation_error}\n"
                "- 请严格使用上方表结构中的真实表名和列名，修正后重新生成。"
            )

        intents_str = ", ".join(keyword_analysis.get("sql_intents", []))
        time_str = ", ".join(keyword_analysis.get("time_conditions", []))
        keywords_str = ", ".join(keyword_analysis.get("keywords", []))

        prompt = f"""你是一个专业的SQL生成专家，将自然语言查询转换为准确的MySQL SQL语句。

## 数据库表结构
{schema_str}
{schema_note}
{examples_str}
{validation_feedback}
## 关键词分析
- 核心关键词: {keywords_str}
- SQL操作意图: {intents_str}
- 时间条件: {time_str if time_str else "无"}

## 用户查询
{user_query}

## 生成规则
1. 只生成SELECT语句，禁止生成任何修改数据的SQL
2. 使用标准MySQL语法
3. 字段和表名使用反引号包裹
4. 结果只返回SQL语句，不要解释，不要markdown代码块，不要换行符
5. 如果无法确定，使用最合理的推断
6. 注意数据类型匹配，字符串用单引号
7. 不要添加任何注释

SQL:"""

        return prompt

    def generate(
        self,
        user_query: str,
        schema_context: list[dict],
        sql_examples: list[dict],
        keyword_analysis: dict,
        last_validation_error: str | None = None,
    ) -> str:
        """调用GPT生成SQL"""
        prompt = self.build_prompt(
            user_query,
            schema_context,
            sql_examples,
            keyword_analysis,
            last_validation_error=last_validation_error,
        )

        response = self.client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "你是专业的数据分析SQL专家，只生成安全的SELECT查询。",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=500,
        )

        sql = response.choices[0].message.content.strip()
        return self._clean_sql(sql)

    def generate_with_fix(
        self,
        user_query: str,
        schema_context: list[dict],
        sql_examples: list[dict],
        keyword_analysis: dict,
        last_validation_error: str | None = None,
        error_type: str | None = None,
    ) -> str:
        """增强版SQL生成：支持分级错误修复，重试时调用"""
        fix_context = ""
        if last_validation_error and error_type:
            fix_hint = self._get_fix_hint(error_type, last_validation_error)
            fix_context = f"\n## 修复指令\n{fix_hint}\n"

        prompt = self.build_prompt(
            user_query=user_query,
            schema_context=schema_context,
            sql_examples=sql_examples,
            keyword_analysis=keyword_analysis,
            last_validation_error=fix_context if fix_context else last_validation_error,
        )

        response = self.client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "你是专业的数据分析SQL专家，只生成安全的SELECT查询。",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=500,
        )

        sql = response.choices[0].message.content.strip()
        return self._clean_sql(sql)

    def _clean_sql(self, sql: str) -> str:
        """清洗SQL输出"""
        sql = re.sub(r"```sql\n?|```\n?", "", sql).strip()
        sql = sql.replace("\n", " ").replace("\r", " ")
        sql = re.sub(r" {2,}", " ", sql).strip()

        if not sql.upper().startswith("SELECT"):
            match = re.search(r"(SELECT.+?)(?:;|$)", sql, re.IGNORECASE | re.DOTALL)
            if match:
                sql = match.group(1)

        if "LIMIT" not in sql.upper():
            sql = sql.rstrip(";") + " LIMIT 1000"

        return sql.strip()

    def validate_safety(self, sql: str) -> tuple[bool, str]:
        """安全校验：防止危险操作"""
        sql_upper = sql.upper()
        for keyword in SQL_BLACKLIST:
            if re.search(rf"\b{keyword}\b", sql_upper):
                return False, f"危险操作: 检测到 {keyword}"
        if not sql_upper.strip().startswith("SELECT"):
            return False, "仅允许SELECT查询"
        return True, "安全"

    def validate_syntax(self, sql: str) -> tuple[bool, str]:
        """通过EXPLAIN进行语法校验"""
        is_safe, msg = self.validate_safety(sql)
        if not is_safe:
            return False, msg
        return db_connector.validate_sql(sql)

    def format_result(self, results: list[dict], user_query: str) -> str:
        """将查询结果格式化为自然语言"""
        if not results:
            return "查询结果为空，未找到相关数据。"

        import decimal
        import datetime

        def default_serializer(obj):
            if isinstance(obj, decimal.Decimal):
                return float(obj)
            if isinstance(obj, (datetime.date, datetime.datetime)):
                return str(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        results_str = json.dumps(
            results[:20], ensure_ascii=False, indent=2, default=default_serializer
        )

        prompt = f"""将以下数据库查询结果转换为自然语言回答。

用户问题: {user_query}
查询结果 (共{len(results)}条):
{results_str}

请用简洁清晰的中文回答用户问题，重点突出关键数据。"""

        response = self.client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800,
        )

        logger.info(f"finish_reason: {response.choices[0].finish_reason}")
        logger.info(f"content原始值: {repr(response.choices[0].message.content)}")

        return response.choices[0].message.content

    ERROR_PATTERNS = {
        "column_not_found": [
            r"Unknown column",
            r"Column.*not found",
            r"不存在的列",
        ],
        "table_not_found": [
            r"Table.*doesn't exist",
            r"Table.*not found",
            r"1146",
        ],
        "syntax_error": [
            r"You have an error in your SQL syntax",
            r"syntax error",
            r"1064",
        ],
        "ambiguous_column": [
            r"Column.*in field list is ambiguous",
            r"1052",
        ],
        "type_mismatch": [
            r"Incorrect.*value",
            r"Data truncated",
            r"1292", r"1366",
        ],
    }

    def classify_error(self, error_msg: str) -> str:
        """将原始错误信息分类为结构化错误类型"""
        for error_type, patterns in self.ERROR_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, error_msg, re.IGNORECASE):
                    return error_type
        return "unknown"

    def _get_fix_hint(self, error_type: str, error_msg: str) -> str:
        """根据错误类型生成针对性修复提示"""
        hints = {
            "column_not_found": (
                "错误类型：字段不存在。\n"
                "修复要求：严格检查字段名大小写，只使用上方表结构中列出的真实字段名，"
                "不要自行推断或缩写字段名。\n"
                f"原始错误: {error_msg}"
            ),
            "table_not_found": (
                "错误类型：表不存在。\n"
                "修复要求：只使用上方表结构中列出的真实表名，检查表名拼写和大小写。\n"
                f"原始错误: {error_msg}"
            ),
            "syntax_error": (
                "错误类型：SQL语法错误。\n"
                "修复要求：检查括号是否匹配、关键字是否完整、逗号是否多余或缺失。\n"
                f"原始错误: {error_msg}"
            ),
            "ambiguous_column": (
                "错误类型：字段名歧义（多个表有同名字段）。\n"
                "修复要求：为所有字段添加表名前缀，如 `table_name`.`column_name`。\n"
                f"原始错误: {error_msg}"
            ),
            "type_mismatch": (
                "错误类型：数据类型不匹配。\n"
                "修复要求：检查WHERE条件中字符串字段是否用了单引号，数值字段是否误用了引号。\n"
                f"原始错误: {error_msg}"
            ),
            "unknown": (
                f"SQL执行出错，请根据以下错误信息修复：{error_msg}"
            ),
        }
        return hints.get(error_type, hints["unknown"])


sql_generator = SQLGenerator()
