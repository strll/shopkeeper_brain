import json
import logging
import re
from json import JSONDecodeError
from typing import *
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from langchain_core.messages import SystemMessage, HumanMessage
from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.prompt.import_prompt import ITEM_NAME_USER_PROMPT_TEMPLATE
from knowledge.utils.client.ai_clients import AIClients



class _ItemNameExtractor:
    pass


class _ItemNameAligner:
    pass


class ItemNameConfirmedNode(BaseNode):
    name = "item_name_confirmed_node"

    def __init__(self):
        self._extractor = _ItemNameExtractor()
        self._aligner = _ItemNameAligner()

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 获取用户原始问题
        original_query = state.get("original_query")
        # TODO 获取历史对话
        history_context = ""

        llm_result: Dict[str, Any] = self._extractor_extract_item_name(original_query, history_context)

        pass

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
        item_name_user_prompt = ITEM_NAME_USER_PROMPT_TEMPLATE.format(
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
            llm_content_obj: Dict[str, Any] = json.load(content)
            raw_item_name = llm_content_obj.get("item_names")

            if not isinstance(raw_item_name, list):
                item_names = []
            else:
                item_names = [item_names.strip() for item_names in raw_item_name
                              if isinstance(raw_item_name, str)
                              and raw_item_name.split()]
            raw_rewritten_query: str = llm_content_obj.get("rewritten_query")

            if not raw_rewritten_query:
                rewritten_query = ""
            else:
                rewritten_query = raw_rewritten_query.split()

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

if __name__ == '__main__':
    item_name_confirmed_node = ItemNameConfirmedNode()
    init_state={
        "original_query":"我准备使用RS-12数字万用表测量电阻"
    }
    llm_result=item_name_confirmed_node.process(init_state)
    print(llm_result)

