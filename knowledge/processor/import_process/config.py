"""
导入流程配置管理模块

集中管理所有配置项，支持环境变量覆盖
"""

import os
from dataclasses import dataclass, field
from typing import Set, Optional

from dotenv import load_dotenv

load_dotenv(override=True)


@dataclass
class ImportConfig:
    """导入流程配置"""

    # ==================== 文档处理配置 ====================
    max_content_length: int = 2000  # 切片最大长度
    img_content_length: int = 200  # 图片上下文最大长度
    min_content_length: int = 500  # 合并短内容的最小长度
    overlap_sentences: int = 1  # 句子级切分时的重叠句数
    item_name_chunk_k: int = 3  # 商品名识别时使用的切片数量
    item_name_chunk_size: int = 2500  # 商品名识别时使用的切片内容长度

    image_extensions: Set[str] = field(
        default_factory=lambda: {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    )



    mcp_dashscope_base_url: str = field(
        default_factory=lambda: os.getenv("MCP_DASHSCOPE_BASE_URL", "")
    )

    # ==================== LLM 配置 ====================
    openai_api_base: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_BASE", "")
    )
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    vl_model: str = field(
        default_factory=lambda: os.getenv("VL_MODEL", "")
    )
    item_model: str = field(
        default_factory=lambda: os.getenv("ITEM_MODEL", "")
    )
    default_model: str = field(
        default_factory=lambda: os.getenv("MODEL", "")
    )

    # ==================== Milvus 配置 ====================
    milvus_url: str = field(
        default_factory=lambda: os.getenv("MILVUS_URL", "")
    )
    chunks_collection: str = field(
        default_factory=lambda: os.getenv("CHUNKS_COLLECTION", "")
    )
    item_name_collection: str = field(
        default_factory=lambda: os.getenv("ITEM_NAME_COLLECTION", "")
    )
    entity_name_collection: str = field(
        default_factory=lambda: os.getenv("ENTITY_NAME_COLLECTION", "")
    )


    # ==================== MinIO 配置 ====================
    minio_endpoint: str = field(
        default_factory=lambda: os.getenv("MINIO_ENDPOINT", "")
    )
    minio_access_key: str = field(
        default_factory=lambda: os.getenv("MINIO_ACCESS_KEY", "")
    )
    minio_secret_key: str = field(
        default_factory=lambda: os.getenv("MINIO_SECRET_KEY", "")
    )
    minio_bucket: str = field(
        default_factory=lambda: os.getenv("MINIO_BUCKET_NAME", "")
    )
    minio_secure: bool = False

    # ==================== 向量配置 ====================
    embedding_dim: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "1024"))
    )
    embedding_batch_size: int = 8

    rrf_max_results: int = field(
        default_factory=lambda: int(os.getenv("RRF_MAX_RESULTS", "10"))
    )
    rrf_k: int = field(
        default_factory=lambda: int(os.getenv("RRF_K", "60"))
    )
    # ==================== 速率限制 ====================
    requests_per_minute: int = 15  # 图片总结 API 速率限制


    bge_reranker_large: str = field(
        default_factory=lambda: os.getenv("BGE_RERANKER_LARGE", "")
    )

    # ==================== 知识图谱 (Neo4j) 配置 ====================
    neo4j_uri: str = field(
        default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687")
    )
    neo4j_user: str = field(
        default_factory=lambda: os.getenv("NEO4J_USER", "neo4j")
    )
    neo4j_password: str = field(
        default_factory=lambda: os.getenv("NEO4J_PASSWORD", "")
    )

    # 实体与图谱检索控制
    entity_alignment_min_score: float = field(
        default_factory=lambda: float(os.getenv("ENTITY_ALIGNMENT_MIN_SCORE", "0.8"))
    )
    kg_max_seeds: int = field(
        default_factory=lambda: int(os.getenv("KG_MAX_SEEDS", "5"))
    )
    kg_max_triplets: int = field(
        default_factory=lambda: int(os.getenv("KG_MAX_TRIPLETS", "15"))
    )
    kg_max_chunks: int = field(
        default_factory=lambda: int(os.getenv("KG_MAX_CHUNKS", "5"))
    )

    # ==================== 融合与重排补充配置 ====================
    rerank_max_top_k: int = field(
        default_factory=lambda: int(os.getenv("RERANK_MAX_TOP_K", "5"))
    )
    rrf_kg_weight: float = field(
        default_factory=lambda: float(os.getenv("RRF_KG_WEIGHT", "0.3"))
    )

    # ==================== 商品确认节点配置 ====================
    item_name_high_confidence: float = field(
        default_factory=lambda: float(os.getenv("ITEM_NAME_HIGH_CONFIDENCE", "0.85"))
    )
    dense_weight: float = field(
        default_factory=lambda: float(os.getenv("DENSE_WEIGHT", "0.6"))
    )
    sparse_weight: float = field(
        default_factory=lambda: float(os.getenv("SPARSE_WEIGHT", "0.4"))
    )


    # ==================== Rerank 配置 ====================

    rerank_min_top_k: int = field(
        default_factory=lambda: int(os.getenv("RERANK_MIN_TOP_K", "3"))
    )
    rerank_gap_ratio: float = field(
        default_factory=lambda: float(os.getenv("RERANK_GAP_RATIO", "0.25"))
    )
    rerank_gap_abs: float = field(
        default_factory=lambda: float(os.getenv("RERANK_GAP_ABS", "0.5"))
    )









    # ==================== 文本处理配置 ====================
    max_context_chars: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONTEXT_CHARS", "12000"))
    )

    # ==================== Rerank 配置 ====================
    rerank_max_top_k: int = field(
        default_factory=lambda: int(os.getenv("RERANK_MAX_TOP_K", "10"))
    )
    rerank_min_top_k: int = field(
        default_factory=lambda: int(os.getenv("RERANK_MIN_TOP_K", "3"))
    )
    rerank_gap_ratio: float = field(
        default_factory=lambda: float(os.getenv("RERANK_GAP_RATIO", "0.25"))
    )
    rerank_gap_abs: float = field(
        default_factory=lambda: float(os.getenv("RERANK_GAP_ABS", "0.5"))
    )

    # ==================== RRF 配置 ====================
    rrf_k: int = field(
        default_factory=lambda: int(os.getenv("RRF_K", "60"))
    )
    rrf_kg_weight: float = field(
        default_factory=lambda: float(os.getenv("RRF_KG_WEIGHT", "0.7"))
    )
    rrf_max_results: int = field(
        default_factory=lambda: int(os.getenv("RRF_MAX_RESULTS", "10"))
    )

    # ==================== 检索配置 ====================
    embedding_search_limit: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_SEARCH_LIMIT", "10"))
    )
    hyde_search_limit: int = field(
        default_factory=lambda: int(os.getenv("HYDE_SEARCH_LIMIT", "5"))
    )

    # ==================== 商品确认节点配置 ====================
    item_name_high_confidence: float = field(
        default_factory=lambda: float(os.getenv("ITEM_NAME_HIGH_CONFIDENCE", "0.7")) # 直接给的（压测给到）--->RAG评估（了解）
    )
    item_name_mid_confidence: float = field(
        default_factory=lambda: float(os.getenv("ITEM_NAME_MID_CONFIDENCE", "0.6"))  # 直接给的（压测给到）--->RAG评估（了解）
    )
    item_name_max_options: int = field(
        default_factory=lambda: int(os.getenv("ITEM_NAME_MAX_OPTIONS", "5"))
    )
    item_name_dense_weight: float = field(
        default_factory=lambda: float(os.getenv("ITEM_NAME_DENSE_WEIGHT", "0.5"))
    )
    item_name_sparse_weight: float = field(
        default_factory=lambda: float(os.getenv("ITEM_NAME_SPARSE_WEIGHT", "0.5"))
    )

    # ==================== 知识图谱配置 ====================
    kg_entity_align_min_score: Optional[float] = field(
        default_factory=lambda: (
            float(os.getenv("KG_ENTITY_ALIGN_MIN_SCORE"))
            if os.getenv("KG_ENTITY_ALIGN_MIN_SCORE")
            else None
        )
    )
    kg_max_seed_candidates: int = field(
        default_factory=lambda: int(os.getenv("KG_MAX_SEED_CANDIDATES", "3"))
    )
    kg_max_total_seeds: int = field(
        default_factory=lambda: int(os.getenv("KG_MAX_TOTAL_SEEDS", "30"))
    )
    kg_max_triples_per_seed: int = field(
        default_factory=lambda: int(os.getenv("KG_MAX_TRIPLES_PER_SEED", "50"))
    )
    kg_max_total_triples: int = field(
        default_factory=lambda: int(os.getenv("KG_MAX_TOTAL_TRIPLES", "50"))
    )
    kg_max_total_chunks: int = field(
        default_factory=lambda: int(os.getenv("KG_MAX_TOTAL_CHUNKS", "50"))
    )





    neo4j_username: str = field(
        default_factory=lambda: os.getenv("NEO4J_USERNAME", "")
    )

    neo4j_database: str = field(
        default_factory=lambda: os.getenv("NEO4J_DATABASE", "neo4j")
    )





    @classmethod
    def from_env(cls) -> "ImportConfig":
        """从环境变量加载配置"""
        return cls()

    # http://192.168.200.130:9000/
    def get_minio_base_url(self):
        base_protocol = "https://" if self.minio_secure else "http://"
        return base_protocol + f"{self.minio_endpoint}"


# ==================== 全局单例 ====================
_config: Optional[ImportConfig] = None


def get_config() -> ImportConfig:
    """获取配置单例"""
    global _config
    if _config is None:
        _config = ImportConfig.from_env()
    return _config
