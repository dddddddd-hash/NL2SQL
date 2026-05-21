import jieba
import jieba.analyse
import jieba.posseg as pseg
from typing import List, Tuple
import re

# SQL关键字映射（中文意图 -> SQL关键词）
INTENT_SQL_MAP = {
    "最多": "MAX",
    "最少": "MIN",
    "最大": "MAX",
    "最小": "MIN",
    "平均": "AVG",
    "总数": "COUNT",
    "总量": "SUM",
    "数量": "COUNT",
    "排名": "ORDER BY",
    "前几": "LIMIT",
    "分组": "GROUP BY",
    "大于": ">",
    "小于": "<",
    "等于": "=",
    "包含": "LIKE",
    "不等于": "!=",
}

# 时间关键词
TIME_PATTERNS = {
    r"今天|今日": "CURDATE()",
    r"本月|这个月": "DATE_FORMAT(NOW(), '%Y-%m')",
    r"本年|今年": "YEAR(NOW())",
    r"最近(\d+)天": "DATE_SUB(CURDATE(), INTERVAL {n} DAY)",
    r"最近(\d+)月": "DATE_SUB(CURDATE(), INTERVAL {n} MONTH)",
}


class KeywordExtractor:
    def __init__(self):
        pass

    def extract_keywords(self, text: str, topK: int = 10) -> List[str]:
        """TF-IDF提取关键词"""
        keywords = jieba.analyse.extract_tags(text, topK=topK, withWeight=False)
        return keywords

    def extract_with_pos(self, text: str) -> List[Tuple[str, str]]:
        """带词性标注的分词，用于识别实体"""
        words = pseg.cut(text)
        result = []
        for word, flag in words:
            # 保留名词(n)、动词(v)、数词(m)、时间词(t)
            if flag.startswith(("n", "v", "m", "t", "eng")):
                result.append((word, flag))
        return result

    def extract_sql_intents(self, text: str) -> List[str]:
        """提取SQL意图关键词"""
        intents = []
        for cn_keyword, sql_keyword in INTENT_SQL_MAP.items():
            if cn_keyword in text:
                intents.append(sql_keyword)
        return list(set(intents))

    def extract_time_conditions(self, text: str) -> List[str]:
        """提取时间条件"""
        conditions = []
        for pattern, sql_expr in TIME_PATTERNS.items():
            match = re.search(pattern, text)
            if match:
                if "{n}" in sql_expr and match.lastindex:
                    sql_expr = sql_expr.format(n=match.group(1))
                conditions.append(sql_expr)
        return conditions

    def extract_numbers(self, text: str) -> List[str]:
        """提取数字（用于LIMIT等）"""
        numbers = re.findall(r"\d+\.?\d*", text)
        return numbers

    def analyze(self, text: str) -> dict:
        """综合分析用户查询"""
        return {
            "keywords": self.extract_keywords(text),
            "pos_words": self.extract_with_pos(text),
            "sql_intents": self.extract_sql_intents(text),
            "time_conditions": self.extract_time_conditions(text),
            "numbers": self.extract_numbers(text),
            "raw_text": text,
        }


keyword_extractor = KeywordExtractor()
