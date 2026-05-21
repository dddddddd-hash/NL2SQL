import pymysql
from pymysql.cursors import DictCursor
from contextlib import contextmanager
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


class MySQLConnector:
    def __init__(self):
        self.config = {
            "host": settings.MYSQL_HOST,
            "port": settings.MYSQL_PORT,
            "user": settings.MYSQL_USER,
            "password": settings.MYSQL_PASSWORD,
            "database": settings.MYSQL_DB,
            "charset": "utf8mb4",
            "cursorclass": DictCursor,
        }

    @contextmanager
    def get_connection(self):
        conn = pymysql.connect(**self.config)
        try:
            yield conn
        finally:
            conn.close()

    def execute_query(self, sql: str, params: tuple = None) -> list[dict]:
        """执行查询并返回结果"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, params)
                results = cursor.fetchall()
                logger.info(f"Query returned {len(results)} rows")
                return results

    def get_schema_info(self) -> dict[str, list[str]]:
        """获取数据库所有表结构，用于向量化索引"""
        schema = {}
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SHOW TABLES")
                tables = [list(row.values())[0] for row in cursor.fetchall()]
                for table in tables:
                    cursor.execute(f"DESCRIBE `{table}`")
                    columns = [row["Field"] for row in cursor.fetchall()]
                    schema[table] = columns
        return schema

    def validate_sql(self, sql: str) -> tuple[bool, str]:
        """通过EXPLAIN校验SQL语法"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"EXPLAIN {sql}")
                    return True, "SQL valid"
        except pymysql.err.ProgrammingError as e:
            return False, str(e)


db_connector = MySQLConnector()