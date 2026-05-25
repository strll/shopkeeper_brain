import logging
from typing import Any, Tuple, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState

class RrfMergeNode(BaseNode):
    name = "rrf_merge_node"
    def __init__(self):
        super().__init__()
        self._top_k = self.config.rrf_max_results
        self._rrf_k = self.config.rrf_k

    def process(self, state: QueryGraphState) -> QueryGraphState:
        embedding_chunks=state.get("embedding_chunks") or []
        hyde_embedding_chunks=state.get("hyde_embedding_chunks") or []

        search_result_weight={
            "embedding_search_chunks":(self._validate_search_result(embedding_chunks)  ,1.0),
            "hyde_embedding_search_chunks":(self._validate_search_result(hyde_embedding_chunks),1.0),

        }
        rrf_input=list(search_result_weight.values())

        merged_rrf_results:list[Tuple[Dict[str,Any]]]=self._merge_rrf_docs(rrf_input,self._rrf_k,self._top_k)

        state['rrf_chunks']=merged_rrf_results
        return state


    def _merge_rrf_docs(self,rrf_inputs:list[Tuple[list[Dict[str,Any]],float]],rrf_k:int,rrf_max_results:int)\
            ->list[Tuple[Dict[str,Any]],float]:
        """
        合并多个rrf结果
        Args:
            rrf_inputs: 多个rrf结果
            rrf_k: rrf的k值
            rrf_max_results: 最大返回结果数
        Returns:

        """
        chunk_source={}
        chunk_data={}
        for search_result,weight in rrf_inputs:
            for rank,res in enumerate(search_result,1):
                chunk_id=res.get("chunk_id")
                if not chunk_id:
                    continue
                chunk_source[chunk_id]=chunk_source.get("chunk_id",float(0))  +weight/(rrf_k+rank)
                chunk_data.setdefault(chunk_id,res)

        final_rrf_result=sorted([ (chunk_data.get(chunk_id),score)  for chunk_id,score in chunk_source.items()],key=lambda x:x[1],reverse=True)

        return final_rrf_result[:rrf_max_results] if rrf_max_results else final_rrf_result

    def _validate_search_result(self, search_chunks:list[Dict[str,Any]]):
        if not  search_chunks:
            return []
        search_result=[]
        for chunk in search_chunks:
            if not chunk or not isinstance(chunk,dict):
                continue

            entity=chunk.get('entity')
            if not entity or not isinstance(entity,dict):
                continue
            search_result.append(entity)

        return search_result





if __name__ == "__main__":


    print("=" * 60)
    print("开始测试: RRF 融合节点")
    print("=" * 60)

    # 模拟三路检索结果
    # chunk_1 命中 3 路（预期最高分）
    # chunk_2 命中 2 路
    # chunk_3, chunk_4, chunk_5 各命中 1 路
    mock_state = {
        "embedding_chunks": [
            {"entity": {"chunk_id": "chunk_1", "content": "向量搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_2", "content": "向量搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_3", "content": "向量搜索结果#3"}},
        ],
        "hyde_embedding_chunks": [
            {"entity": {"chunk_id": "chunk_2", "content": "HyDE搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_1", "content": "HyDE搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_4", "content": "HyDE搜索结果#3"}},
        ],
        "kg_chunks": [
            {"id": None, "distance": 2.0, "entity": {"chunk_id": "chunk_5", "content": "知识图谱结果#1"}},
            {"id": None, "distance": 1.0, "entity": {"chunk_id": "chunk_1", "content": "知识图谱结果#2"}},
        ],
    }

    print("【输入状态】:")
    print(f"  embedding_chunks: {len(mock_state['embedding_chunks'])} 条")
    print(f"  hyde_embedding_chunks: {len(mock_state['hyde_embedding_chunks'])} 条")
    print(f"  kg_chunks: {len(mock_state['kg_chunks'])} 条")
    print("-" * 60)

    rrf_node = RrfMergeNode()
    result = rrf_node.process(mock_state)

    print("\n【融合结果】:")
    for i, chunk in enumerate(result["rrf_chunks"], 1):
        print(f"[{i}] {chunk[0].get('chunk_id')} - {chunk[0].get('content')}")

    print("-" * 60)
    print("测试完成")
