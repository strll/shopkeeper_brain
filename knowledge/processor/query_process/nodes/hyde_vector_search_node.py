from typing import Tuple, Dict, Any

from knowledge.processor.import_process.exceptions import StateFieldError
from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompt.query.query_prompt import USER_HYDE_PROMPT_TEMPLATE
from knowledge.utils.bge_m3_embedding_util import generate_hybrid_embeddings
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.milvus_util import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HydeVectorSearchNode(BaseNode):
    name = "hybrid_vector_search_node"
    def process(self, state: QueryGraphState) -> QueryGraphState:
        rewritten_query, item_names = self._validate_state(state)
        #生成假设性文档
        hy_document:str=self._generate_hy_document(rewritten_query,item_names)

        if hy_document is None:
            return state

        try:
            milvue_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            logger.error(f"获取milvus客户端失败{str(e)}")
            return state

        try:
            bge_m3_client = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            logger.error(f"获取嵌入模型失败{str(e)}")
            return state
        try:
            embed_hy_vector=generate_hybrid_embeddings(embedding_model=bge_m3_client,embedding_documents=[ f"{rewritten_query}\n{hy_document}" ])

        except Exception as e:
            logger.error(f"获取假设性文档嵌入向量失败{str(e)}")
            return state



        #进行向量检索
        try:
            expr,expr_params = self._item_names_filter(item_names)

            hybrid_search_req = create_hybrid_search_requests(dense_vector=embed_hy_vector["dense"][0],
                                                              sparse_vector=embed_hy_vector['sparse'][0],
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
            logger.error(f"用户的问题是{rewritten_query} 对应的假设性文档是{hy_document} 创建向量搜索请求失败{str(e)}")
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

    def _generate_hy_document(self, rewritten_query:str, item_names:list[str])->str:
        try:
            llm_client=AIClients.get_llm_client(response_format=False)
        except ConnectionError as e:
            logger.error(f"获取嵌入模型失败{str(e)}")
            return None
        system_prompt=(f"您是一位{item_names}方面的助手，主要擅长编写技术文档,操作手册,文档规格说明")
        user_prompt=USER_HYDE_PROMPT_TEMPLATE.format(file_title=rewritten_query,
                                                  item_names=item_names)
        try:
            llm_response=llm_client.invoke([system_prompt,user_prompt])
        except Exception as e:
            logger.error(f"用户的问题是{rewritten_query} 获取LLM结果失败{str(e)}")
            return None
        return str(llm_response.content)

    def _item_names_filter(self, item_names: list[str]) -> Tuple[str, Dict[str, Any]]:
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

if __name__ == "__main__":
    from knowledge.processor.query_process.base import setup_logging

    setup_logging()

    print("=" * 60)
    print("开始测试: HyDE 检索节点 (HydeSearchNode)")
    print("=" * 60)

    mock_state = {
        "rewritten_query": "RS-12 数字万用表如何测量直流电压？",
        "item_names": ["RS PRO RS-12 数字万用表"],
    }

    print("【输入状态】:")
    print(f"  查询: {mock_state['rewritten_query']}")
    print(f"  商品: {mock_state['item_names']}")
    print("-" * 60)

    node = HydeVectorSearchNode()
    result = node.process(mock_state)

    chunks = result.get("hyde_embedding_chunks", [])
    print(f"\n【HyDE 检索结果】: {len(chunks)} 条")
    for i, chunk in enumerate(chunks, 1):
        entity = chunk.get("entity", {})
        print(f"  [{i}] chunk_id={entity.get('chunk_id')} "
              f"item_name={entity.get('item_name')} "
              f"distance={chunk.get('distance', 'N/A')}")
        content = entity.get("content", "")
        print(f"      内容: {content[:80]}...")

    print("-" * 60)
    print("测试完成")