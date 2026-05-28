import json
import re
from json import JSONDecodeError
from typing import *

from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.config import get_config
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompt.query.query_prompt import ITEM_NAME_EXTRACT_TEMPLATE
from knowledge.utils.bge_m3_embedding_util import *
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query
from knowledge.utils.mongo_history_util import get_recent_messages

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class _ItemNameExtractor:
    pass


class _ItemNameAligner:

    def __init__(self):
        self._config = get_config()

    def search_and_align(self, item_names: List[str]):
        """
        检索数据库 并且和向量数据的商品名称对齐 最终返回确定的商品名列表
        Args:
            item_names: LLM提取的商品名列表

        Returns:

        """

        search_result: List[Dict[str, Any]] = self._search_vector(item_names)
        if not search_result:
            return [], []

        confirmed,options=self._align(search_result)

        if len(confirmed) >1:
            confirmed=self._item_name_score_filter(confirmed,search_result)

        return confirmed,options

    def _align(self, search_result: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        # 准一点的
        confirmed = []
        # 不太准的
        options = []

        for item_sea_res in search_result:
            # 获取extracted_name
            llm_extract_item_name = item_sea_res.get("extracted_name")
            item_name_matches = item_sea_res.get("matches")

            item_name_matchs_sorted = sorted(item_name_matches, key=lambda x: x['score'], reverse=True )

            high=[h for h in item_name_matchs_sorted if h.get("score")> self._config.item_name_high_confidence]
            if high:
                extract=next((h for h in high if h.get("item_name")==llm_extract_item_name),None )
                if extract:
                    picked=extract.get('item_name')
                    if picked not in confirmed:
                        confirmed.append(picked)
                elif len(high)==1:
                    picked=high[0].get('item_name')
                    if picked not in confirmed:
                        confirmed.append(picked)
                else:
                    top_score=high[0]['score']
                    if top_score-high[1]['score']>self._config.item_name_score_gap:
                        picked=high[0].get('item_name')
                        if picked not in confirmed:
                            confirmed.append(picked)



            else:
                mid=[
                   m for m in item_name_matchs_sorted if
                    m.get("score")> self._config.item_name_mid_confidence
                    and m.get("item_name") not in confirmed
                    and m.get("item_name") not in options
                ]
                if mid:
                    for m in mid[:self._config.item_name_max_options]:
                        options.append(m.get('item_name'))

        return confirmed, options[:self._config.item_name_max_options]

    def _search_vector(self, item_names: List[str]) -> List[Dict[str, Any]]:
        """
        对LLM提取到所有的商品名进行向量检索
        Args:
            self:
            item_names: LLM提取商品名列表

        Returns:

        """
        final_search_result = []
        try:
            milvus_client = StorageClients.get_milvus_client()

        except ConnectionError as e:
            logger.error(f"获取milvus客户端失败,原因是{str(e)}")
            return []
        embedding_model = get_beg_m3_embedding_model()
        if embedding_model is None:
            logger.error(f"获取嵌入模型失败")

            return final_search_result

        try:
            hybrid_vector_result =generate_hybrid_embeddings(embedding_model, item_names)
        except Exception as e:
            logger.error(f"商品列表 向量检索失败,原因是{str(e)}")
            return []

        for index, item_name in enumerate(item_names):
            # 构建稠密以及混合向量的检索需求
            hybrid_requests = create_hybrid_search_requests(hybrid_vector_result["dense"][index],
                                                            hybrid_vector_result["sparse"][index])
            hybrid_search_result = execute_hybrid_search_query(milvus_client=milvus_client,
                                                               collection_name=os.getenv("ITEM_NAME_COLLECTION", ""),
                                                               search_requests=hybrid_requests,
                                                               ranker_weights=(0.5, 0.5),
                                                               limit=5,
                                                               output_fields=["item_name"]
                                                               )

            matches = [{"score": item_search_res['distance'],
                        "item_name": item_search_res['entity']['item_name']}
                       for item_search_res in
                       (hybrid_search_result[0] if hybrid_search_result else [])]

            final_search_result.append({
                "extracted_name": item_name,
                "matches": matches
            })
        return final_search_result

    def _item_name_score_filter(self, confirmed:list[str], search_results:list[Dict[str,Any]]):
        """
        对商品名称进行分数过滤
        Args:
            confirmed:
            search_result:

        Returns:

        """
        item_name_score={}
        for search_result in search_results:
            matches=search_result.get("matches",[])
            for match in matches:
                item_name=match.get("item_name")
                score=match.get("score",0)
                if item_name not in confirmed:
                    item_name_score[item_name]=max(item_name_score.get(item_name,0),score)
        if not item_name_score:
            return confirmed
        max_score=max(item_name_score.values())
        return [item_name for item_name,score in item_name_score.items() if max_score-score<=self._config.item_name_score_gap]




class ItemNameConfirmedNode(BaseNode):
    name = "item_name_confirmed_node"

    def __init__(self):
        super().__init__()
        self._extractor = _ItemNameExtractor()
        self._aligner = _ItemNameAligner()

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 获取用户原始问题
        original_query = state.get("original_query")
        # TODO 获取历史对话
        history_context = get_recent_messages(state.get('session_id'),limit=10 )
        format_history=[]
        for history in history_context:
            role=history.get("role",'')
            text=history.get("text",'')
            formatted_context=f"角色:{role} 内容{text}"
            format_history.append(formatted_context)
        format_history_str=" ".join(format_history)



        llm_result: Dict[str, Any] = self._extractor_extract_item_name(original_query, format_history_str)

        item_names = llm_result.get("item_names")
        rewritten_query = llm_result.get("rewritten_query")

        if item_names:
            confirmed, options = self._aligner.search_and_align(item_names)
        else:
            confirmed, options = [], []
        self._decide(confirmed, options, state, rewritten_query, item_names)
        state['history']=history_context
        return state

    def _extractor_extract_item_name(self, original_query: str, history_context: str) -> Dict[str, Any]:
        """
        抽取商品名称
        Args:
            original_query: 用户原始问题
            history_context: 历史对话

        Returns:

        """
        # 获取llm客户端
        llm_result = {"item_names": [], "rewritten_query": original_query}
        try:
            llm_client = AIClients.get_llm_client(response_format=True)
        except ConnectionError as e:
            logger.error(f"获取llm客户端失败,原因是{str(e)}")
            return llm_result
        # 获取商品名提取的提示词
        item_name_system_prompt = "你是一名商品提取专家,请根据用户的问题以及历史对话种提取相关的商品名以改写原始的查询"
        item_name_user_prompt = ITEM_NAME_EXTRACT_TEMPLATE.format(
            history_text=history_context.split() if history_context.split() else "暂无历史上下文",
            query=original_query
        )
        llm_respons = llm_client.invoke(
            [SystemMessage(content=item_name_system_prompt), HumanMessage(content=item_name_user_prompt)])

        llm_response_context = llm_respons.content

        if not llm_response_context:
            return llm_result

        parsed_result: Dict[str, Any] = self._clean_and_parse(llm_response_context)
        llm_result['item_names'] = parsed_result.get("item_names")

        llm_result['rewritten_query'] = parsed_result.get("rewritten_query") if parsed_result.get(
            "rewritten_query") else original_query

        return llm_result

    def _clean_and_parse(self, llm_response_context: str) -> Dict[str, Any]:
        """
        清洗和解析llm返回的文本
        Args:
            llm_response_context: llm返回的文本

        Returns:

        """
        llm_response_result = {"item_names": [], "rewritten_query": ""}

        # 去除json的围栏标记
        cleaned = re.sub(r"^```(?:json)?\s*", "", llm_response_context.strip())
        content = re.sub(r"\s*```$", "", cleaned)
        try:
            llm_content_obj: Dict[str, Any] = json.loads(str(content))
            raw_item_name = llm_content_obj.get("item_names")

            if not isinstance(raw_item_name, list):
                item_names = []
            else:
                item_names = [name.strip() for name in raw_item_name
                              if isinstance(name, str) and name.strip()]
            raw_rewritten_query: str = llm_content_obj.get("rewritten_query")

            if not raw_rewritten_query:
                rewritten_query = ""
            else:
                rewritten_query = str(raw_rewritten_query).strip()


        except JSONDecodeError as e:
            logger.error(f"{llm_response_context} 解析json失败 报错是{str(e)}")
            raise JSONDecodeError(msg=e.msg,
                                  doc=e.doc,
                                  pos=e.pos
                                  )

        return {
            "item_names": item_names,
            "rewritten_query": rewritten_query
        }

    def _decide(self, confirmed: List[str],
                options: List[str],
                state: QueryGraphState,
                rewritten_query: str,
                item_names: List[str]):
        """

        Args:
            confirmed: 确认的商品列表
            options: 可能商品列表
            state: 查询状态
            rewritten_query: 重写后的问题
            item_names: LLM提取到的商品列表

        Returns:

        """
        if confirmed:
            state["item_names"] = item_names
            state["rewritten_query"] = rewritten_query
        elif options:
            state["answer"] = (
                f"我不确定您指的是那个产品",
                f"您是在询问以下的产品吗:{"、".join(options)}?"
            )
        else:
            state["answer"] = (
                f"我无法确定您所提问的是哪个产品",
                f"请重新提问"
            )




if __name__ == '__main__':
    item_name_confirmed_node = ItemNameConfirmedNode()
    init_state = {
        "original_query": "我准备使用RS-12数字万用表测量电阻"
    }
    llm_result = item_name_confirmed_node.process(init_state)
    print(llm_result)
