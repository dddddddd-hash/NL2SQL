from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchParams,
)
from config.settings import settings
from utils.db_connector import db_connector
from core.embedder import embedder
import logging
import uuid

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self):
        self.client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
        )
        self.collection_name = settings.QDRANT_COLLECTION
        self._ensure_collection()

    def _ensure_collection(self):
        """确保集合存在，维度改为 1024（bge-m3）"""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=settings.EMBEDDING_DIMENSION,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created collection: {self.collection_name}")

    def index_schema(self):
        """将数据库 Schema 批量向量化并入库"""
        schema = db_connector.get_schema_info()

        texts = []
        payloads = []
        for table_name, columns in schema.items():
            schema_text = f"表名: {table_name}, 字段: {', '.join(columns)}"
            texts.append(schema_text)
            payloads.append({
                "table_name": table_name,
                "columns": columns,
                "schema_text": schema_text,
                "type": "schema",
            })

        vectors = embedder.embed_batch(texts, is_query=False)

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload=payload,
            )
            for vec, payload in zip(vectors, payloads)
        ]

        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info(f"Indexed {len(points)} table schemas")

    def index_sql_examples(self, examples: list[dict]):
        """批量索引 SQL 示例"""
        texts = [
            f"{ex['query']} {ex.get('description', '')}"
            for ex in examples
        ]

        vectors = embedder.embed_batch(texts, is_query=False)

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "query": ex["query"],
                    "sql": ex["sql"],
                    "description": ex.get("description", ""),
                    "type": "sql_example",
                },
            )
            for vec, ex in zip(vectors, examples)
        ]

        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info(f"Indexed {len(points)} SQL examples")

    def search_schema(self, query: str, top_k: int = None) -> list[dict]:
        """检索相关表结构"""
        top_k = top_k or settings.TOP_K

        query_vector = embedder.embed(query)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=Filter(
                must=[FieldCondition(key="type", match=MatchValue(value="schema"))]
            ),
            limit=top_k,
            score_threshold=0.3,
            search_params=SearchParams(hnsw_ef=128),
        )

        return [
            {
                "table_name": hit.payload["table_name"],
                "columns": hit.payload["columns"],
                "schema_text": hit.payload["schema_text"],
                "score": hit.score,
            }
            for hit in results
        ]

    def search_examples(self, query: str, top_k: int = 3) -> list[dict]:
        """检索相似 SQL 示例"""
        query_vector = embedder.embed(query)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=Filter(
                must=[FieldCondition(key="type", match=MatchValue(value="sql_example"))]
            ),
            limit=top_k,
            score_threshold=0.60,
        )

        return [
            {
                "query": hit.payload["query"],
                "sql": hit.payload["sql"],
                "score": hit.score,
            }
            for hit in results
        ]


vector_store = VectorStore()
