import logging
from typing import Any, Dict

from knowledge.utils.client.ai_clients import AIClients

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from FlagEmbedding import FlagReranker

class RerankerNode(BaseNode):
    name = "reranker_node"
    def process(self, state: QueryGraphState) -> QueryGraphState:

        user_query=state.get('rewritten_query') or state.get('original_query')

        rerank_outputs:list[Dict[str,Any]]=self._collect_rerank_input(state)

        #利用模型
        refine_docs:list[Dict[str,Any]]=self._refine_rank(user_query,rerank_outputs)
        #断崖检测
        refine_docs=self._cliff_cutoff(refine_docs,self.config.rerank_min_top_k,self.config.rerank_max_top_k)

        state['reranked_docs'] = refine_docs

        return state

    def _collect_rerank_input(self, state:QueryGraphState)->list[Dict[str,Any]]:
        rrf_chunks=state.get('rrf_chunks') or []
        final_docs=[]
        for chunk in rrf_chunks:
            if not chunk or not isinstance(chunk,dict):
                continue

            content=chunk.get('content','')

            if not content:
                continue

            title=chunk.get('title','')

            chunk_id=chunk.get('chunk_id')

            # 格式化文档
            format_doc=self._format_doc(content=content,chunk_id=chunk_id,title=title,source='local')


            final_docs.append(format_doc)
            logger.info(f"获取Reranbker阶段需要的文档个数为{len(rrf_chunks)}")


        web_search_docs=state.get('web_search_docs') or []

        for doc in web_search_docs:
            if not doc or not isinstance(doc,dict):
                continue

            content=doc.get('snippet','')
            title=doc.get('title','')
            url=doc.get('url','')
            format_web_doc=self._format_doc(content=content,

                                            title=title,url=url,source='web')
            logger.info(f"获取Reranbker阶段需要的搜索结果个数为{len(web_search_docs)}")
            final_docs.append(format_web_doc)
        logger.info(f"格式化后的文档数量为{len(final_docs)}")
        return final_docs



    def _format_doc(self,
                    content:str,
                    chunk_id:int=None,
                    title:str="",
                    url:str="",
                    source: str = ""
                    ):

        return {
            'content':content,
            'chunk_id':chunk_id,
            'title':title,
            'source':source,
            'url':url
        }

    def _refine_rank(self, user_query:str, rerank_outputs:str)->list[Dict[str,Any]]:

        """
        利用模型对文档进行排序
        :param user_query: 用户查询
        :param rerank_outputs:本地和远程融合后的结果
        :return:

        """

        try:
            rerank_client:FlagReranker=AIClients.get_bge_m3_rerank_client()
        except ConnectionError as e:
            self.logger.error(f"Reranker模型连接失败:{str(e)}")
            return []

        query_doc_pairs=[(user_query,d.get('content')) for d in rerank_outputs ]

        try:
            rerank_scores=rerank_client.compute_score(sentence_pairs=query_doc_pairs)

            res=  [{**d, 'score': float(score)} for d, score in zip(rerank_outputs, rerank_scores)]
            res = sorted(res, key=lambda x: x.get('score') if x.get('score') is not None else -999,
                                   reverse=True)

            return res

        except Exception as e:
            self.logger.error(f"Reranker模型计算分数失败:{str(e)}")
            return [{**d,'score':None} for d in rerank_outputs ]

    def _cliff_cutoff(self, refine_docs:list[Dict[str,Any]], rerank_min_top_k:int, rerank_max_top_k:int)->list[Dict[str,Any]]:

        """
        断崖检测
        :param refine_docs: 排序后的文档
        :param rerank_min_top_k: 最小返回文档数
        :param rerank_max_top_k: 最大返回文档数
        :return:
        """
        upper_bound=min(rerank_max_top_k,len(refine_docs))
        low_bound=min(rerank_min_top_k,len(refine_docs))
        cut_off=upper_bound
        max_gap=0
        for i in range(low_bound-1,upper_bound-1):
            current_doc_source=refine_docs[i].get("score")
            next_doc_source=refine_docs[i+1].get("score")

            if not current_doc_source or not next_doc_source:
                 continue
            #获取差值
            abs_gap=current_doc_source-next_doc_source
            need_cutoff=False
            if abs_gap>= self.config.rerank_gap_abs:
                need_cutoff=True

            if need_cutoff and max_gap<abs_gap:
                max_gap=abs_gap
                cut_off=i+1
                self.logger.info(f"断崖检测: 断崖位置为{i+1}，当前文档来源为{current_doc_source}，下一文档来源为{next_doc_source}")


            cut_off_docs=refine_docs[:cut_off]
            # 绝对分数底线过滤
            rerank_min_score = getattr(self.config, 'rerank_min_score', None)
            if rerank_min_score is not None:
                filtered_docs = [d for d in cut_off_docs if (d.get("score") or 0) >= rerank_min_score]
                if len(filtered_docs) < low_bound:
                    self.logger.warning(
                        f"绝对分数过滤后仅剩 {len(filtered_docs)} 篇，回退保留前 {low_bound} 篇"
                    )
                    cut_off_docs = cut_off_docs[:low_bound]
                else:
                    cut_off_docs = filtered_docs



        return cut_off_docs









if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    print("=" * 60)
    print("开始测试: 重排序节点 (RerankNode)")
    print("=" * 60)

    mock_state = {
        "rewritten_query": "怎么测这块主板的短路问题？",
        "rrf_chunks": [
            {"chunk_id": "local_1", "title": "主板维修手册",
             "content": "主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。"},
            {"chunk_id": "local_2", "title": "闲聊",
             "content": "今天中午去吃猪脚饭吧，这块主板外观很漂亮。"},
        ],
        "web_search_docs": [
            {"url": "https://example.com/repair", "title": "短路查修指南",
             "snippet": "主板通电前先打各主供电电感的对地阻值，阻值偏低就是短路。"},
            {"url": "https://example.com/news", "title": "科技新闻",
             "snippet": "苹果发布新款手机，A系列芯片性能提升20%。"},
        ],
    }

    print("【输入状态】:")
    print(f"  查询: {mock_state['rewritten_query']}")
    print(f"  本地文档: {len(mock_state['rrf_chunks'])} 篇")
    print(f"  网络文档: {len(mock_state['web_search_docs'])} 篇")
    print("-" * 60)

    node = RerankerNode()
    result = node.process(mock_state)

    print("\n【重排序结果】:")
    res:list =result["reranked_docs"]
  #  res = sorted(res, key=lambda x: x.get('score') if x.get('score') is not None else -999, reverse=True)

    for i, doc in enumerate(res, 1):
        score = doc.get('score')
        score_str = f"{score:.4f}" if score is not None else "N/A"
        print(f"[{i}] score={score_str} | {doc['source']:5} | {doc['content'][:50]}...")

    print("-" * 60)
    print("测试完成")