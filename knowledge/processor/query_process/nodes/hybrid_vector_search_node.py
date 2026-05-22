import json
from typing import Tuple, Any, Dict

from knowledge.processor.import_process.exceptions import StateFieldError
from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.utils.bge_m3_embedding_util import *
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.milvus_util import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridVectorSearch(BaseNode):
    name = "hybrid_vector_search"

    def process(self, state: QueryGraphState) -> QueryGraphState:

        rewritten_query, item_names = self._validate_state(state)
        try:
            bge_m3_client = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            logger.error(f"获取嵌入模型失败{str(e)}")
            return state
        try:
            milvue_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            logger.error(f"获取milvus客户端失败{str(e)}")
            return state
        try:
            embed_query_vector = generate_hybrid_embeddings(embedding_model=bge_m3_client,
                                                            embedding_documents=[rewritten_query])
            expr, expr_params = self._item_names_filter(item_names)

        except Exception as e:
            logger.error(f"用户的问题是{rewritten_query} 生成嵌入向量失败{str(e)}")
            return state

        try:
            hybrid_search_req = create_hybrid_search_requests(dense_vector=embed_query_vector["dense"][0],
                                                              sparse_vector=embed_query_vector['sparse'][0],
                                                              expr=expr,
                                                              expr_params=expr_params,
                                                              limit=5
                                                              )

            hybrid_search_res = execute_hybrid_search_query(milvus_client=milvue_client,
                                                            collection_name=self.config.chunks_collection,
                                                            search_requests=hybrid_search_req,
                                                            norm_score=True,
                                                            output_fields=["chunk_id", "content", "item_name"]
                                                            )
            if not hybrid_search_res or not hybrid_search_res[0]:
                return state

            state['embedding_chunks'] = hybrid_search_res[0]
            return state

        except Exception as e:
            logger.error(f"用户的问题是{rewritten_query} 创建向量搜索请求失败{str(e)}")
            return state

    def _validate_state(self, state: QueryGraphState) -> Tuple[str, list[str]]:
        rewritten_query = state.get("rewritten_query")

        item_names = state.get('item_names')
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name,
                                  field_name="rewritten_query",
                                  expected_type=str)

        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(node_name=self.name,
                                  field_name="item_names",
                                  expected_type=list)
        return rewritten_query, item_names

    def _item_names_filter(self, item_names: List[str]) -> Tuple[str, Dict[str, Any]]:
        """
        对商品名称进行过滤（绕过 Milvus 模板解析 Bug，直接拼接字符串）
        """
        # 利用 Python 的推导式，把列表变成带引号的字符串格式
        # 例如: ["RS-12", "RS-15"] 会变成 '"RS-12", "RS-15"'
        formatted_names = ", ".join([f'"{name}"' for name in item_names])

        # 直接把最终的值拼到表达式里
        expr = f"item_name in [{formatted_names}]"

        # expr_params 直接返回 None 即可
        return expr, None

if __name__ == '__main__':
    state = {
        "rewritten_query": "万用表如何测试电阻",
        "item_names": ["RS PRO RS-12 数字万用表"]
    }
    vector_search = HybridVectorSearch()
    result = vector_search.process(state)
    for f in result.get("embedding_chunks"):
        print(json.dumps(f, ensure_ascii=False, indent=2))
