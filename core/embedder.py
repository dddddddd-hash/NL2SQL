from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np
import torch
import logging
import os

from config.settings import settings

logger = logging.getLogger(__name__)


class LocalEmbedder:
    """
    本地 Embedding 模型封装（基于 sentence-transformers）
    默认使用 BAAI/bge-m3，支持中英文，维度 1024
    """

    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL_NAME
        self.device = settings.EMBEDDING_DEVICE
        self.batch_size = settings.EMBEDDING_BATCH_SIZE
        self.max_length = settings.EMBEDDING_MAX_LENGTH

        logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
        self.model = self._load_model()
        logger.info("Embedding model loaded successfully")

    def _load_model(self) -> SentenceTransformer:
        """加载模型，优先从本地缓存读取"""
        local_path = settings.EMBEDDING_MODEL_PATH

        if os.path.exists(local_path):
            logger.info(f"Loading from local path: {local_path}")
            model = SentenceTransformer(
                    "BAAI/bge-m3",
                    cache_folder="./models/bge-m3",
                    device=self.device
                    )
        else:
            logger.info(f"Downloading model from HuggingFace: {self.model_name}")
            os.makedirs(local_path, exist_ok=True)
            model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=local_path,
            )

        model.max_seq_length = self.max_length
        return model

    def embed(self, text: str) -> List[float]:
        """
        单文本向量化
        bge 系列模型：查询文本加前缀可提升召回效果
        """
        processed = f"Represent this sentence for searching relevant passages: {text}"
        vector = self.model.encode(
            processed,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.tolist()

    def embed_document(self, text: str) -> List[float]:
        """
        文档向量化（入库时使用，不加查询前缀）
        """
        vector = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.tolist()

    def embed_batch(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """
        批量向量化，适合大量文档入库
        """
        if is_query:
            texts = [
                f"Represent this sentence for searching relevant passages: {t}"
                for t in texts
            ]

        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 10,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def similarity(self, text1: str, text2: str) -> float:
        """计算两段文本的余弦相似度（调试用）"""
        vec1 = np.array(self.embed(text1))
        vec2 = np.array(self.embed(text2))
        return float(np.dot(vec1, vec2))


embedder = LocalEmbedder()
