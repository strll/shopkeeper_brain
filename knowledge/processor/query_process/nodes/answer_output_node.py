import logging
from typing import Dict, Any
from typing import Tuple

from langchain_openai import ChatOpenAI

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompt.query.query_prompt import ANSWER_PROMPT
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.mongo_history_util import save_chat_message
from knowledge.utils.sse_util import push_sse_event, SSEEvent
from knowledge.utils.task_util import set_task_result

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnswerOutputNode(BaseNode):
    name = "answer_output_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        is_stream:bool = state.get('is_stream')
        task_id=state.get('task_id')
        answer=state.get('answer')
        is_streamed=False
        if is_stream:

            self._push_exist_answer(task_id,state)

            is_streamed=False
        else:
            prompt=self._build_prompt(task_id,state)
            state['prompt']=prompt

            self._generate_answer(prompt,state)
            is_streamed = True

        self.save_history(state)

        if is_stream:

            if  is_streamed:
                push_sse_event(task_id=task_id, event=SSEEvent.FINAL, data={})
            else:
                push_sse_event(task_id=task_id, event=SSEEvent.FINAL, data={"answer": state.get('answer')})

        return state


    def _push_exist_answer(self, task_id, state: QueryGraphState):
        is_stream:bool=state.get('is_stream')


        if not is_stream:
            set_task_result(task_id=task_id, key="answer",
                            value=state.get('answer'))



    def _build_prompt(self, task_id: str, state: QueryGraphState)->str:
        user_query=state.get('rewritten_query')
        item_names=state.get('item_names') or []

        reranked_docs=state.get('reranked_docs') or []
        max_context_charts=self.config.max_context_chars
        formatted_context,usage_chars=self._format_retrieved_docs(reranked_docs,max_context_charts)

        chat_history_context=state.get('history') or []


        formatted_history=self._format_chat_history(chat_history_context,usage_chars)
        char_budget = self.config.max_context_chars
        # 4. 格式化图谱关系
        graph_str, char_budget = self._format_kg_triples(
            state.get("kg_trip les") or [], char_budget
        )
        question=user_query or "暂无用户问题"
        if question is None:
            question="暂无用户问题"

        prompt_format = ANSWER_PROMPT.format(context=formatted_context or "暂无检索到上下文",
                                             history=formatted_history or "暂无历史对话",
                                             item_names=",".join(item_names) or "暂无物品名称",
                                             graph_relation_description=graph_str or "无图谱关系",
                                             question=question, )
        return prompt_format

    @staticmethod
    def _format_kg_triples(kg_triples: list, char_budget: int) -> Tuple[str, int]:
        formatted_lines = []
        used_chars = 0
        for triple in kg_triples:
            triple_text = (str(triple) if triple is not None else "").strip()
            if not triple_text:
                continue
            if used_chars + len(triple_text) > char_budget:
                break
            formatted_lines.append(triple_text)
            used_chars += len(triple_text) + 1
        return "\n".join(formatted_lines), char_budget - used_chars

    # def _format_retrieved_docs(self, retrieval_context:list[Dict[str,Any]], max_context_charts:int):
    #     formatted_lines = []
    #     used_chars = 0
    #     for index,context in enumerate(retrieval_context,1) :
    #         context=context.get('content','')
    #
    #         if not context:
    #             continue
    #
    #         metedata_content=[f"[文档:{index}]"]
    #
    #
    #         for meta_field, template in [("chunk_id", "[chunk_id={}]"),
    #                                      ("title", "[title={}]"),
    #                                      ("source", "[source={}]"),
    #                                      ("url", "[url={}]")]:
    #             # a. 获取各个元数据字段的值
    #             filed_value = str(context.get(meta_field, "")).strip()
    #
    #             # b.格式化模版中的占位符
    #             if filed_value:
    #                 metadata_content.append(template.format(filed_value))
    #
    #         #
    #         # for meta_field,template in  [("chunk_id","[chunk_id={}]"),
    #         #  ("title","[title={}]"),
    #         #  ("source","[source=[{}]"),
    #         #  ("url","[url=[]]")]:
    #         #     filed_value=context.get(meta_field).split()
    #         #     if filed_value:
    #         #         metedata_content.append(template.format(filed_value))
    #         #
    #         # score=context.get('score',float(0))
    #         doc_score = context.get('score')
    #         if doc_score is not None:
    #
    #             metedata_content.append(f"[score={doc_score:.6f}]")
    #
    #         formatted_line=" ".join(metedata_content)+"\n"+context
    #
    #
    #
    #         if used_chars + len(formatted_line) > max_context_charts:
    #             break
    #
    #         formatted_lines.append(formatted_line)
    #         used_chars += len(formatted_line) + 2
    #
    #     return "\n\n".join(formatted_lines), max_context_charts - used_chars
    def _format_retrieved_docs(self, retrieval_context: list[Dict[str, Any]], max_context_charts: int):
        formatted_lines = []
        used_chars = 0

        for index, context in enumerate(retrieval_context, 1):
            # 修复 1：使用新的变量名 doc_content，避免覆盖 context 字典
            doc_content = context.get('content', '')

            if not doc_content:
                continue

            metedata_content = [f"[文档:{index}]"]

            for meta_field, template in [("chunk_id", "[chunk_id={}]"),
                                         ("title", "[title={}]"),
                                         ("source", "[source={}]"),
                                         ("url", "[url={}]")]:
                # a. 获取各个元数据字段的值 (此时 context 依然是字典，.get() 方法可以正常工作)



                filed_value = str(context.get(meta_field, "")).strip()

                # b. 格式化模版中的占位符
                if filed_value:
                    metedata_content.append(template.format(filed_value))

            doc_score = context.get('score')
            if doc_score is not None:
                metedata_content.append(f"[score={doc_score:.6f}]")

            # 修复 2：拼接最终字符串时，使用提取出来的 doc_content
            formatted_line = " ".join(metedata_content) + "\n" + doc_content

            if used_chars + len(formatted_line) > max_context_charts:
                break

            formatted_lines.append(formatted_line)
            used_chars += len(formatted_line) + 2

        return "\n\n".join(formatted_lines), max_context_charts - used_chars
    def _generate_answer(self, prompt:str, state:QueryGraphState):
        try:
            llm_client=AIClients.get_llm_client(response_format=False)
        except ConnectionError as e:
            logger.error(f"获取LLM客户端失败{str(e)}")
            state['answer']="LLM暂无回答任何内容 请检查链接是否正常"
            return "LLM暂无回答任何内容 请检查链接是否正常"

        if state.get('is_stream'):
            #获取llm结果
            state['answer']= self._stream_llm(state.get('task_id'),prompt,llm_client)

        else:
            # llm_result=self._invoke_llm(prompt,llm_client)
            # set_task_result(task_id=state.get('task_id'), key="answer", value=llm_result)
            state['answer'] = self._invoke_llm(prompt, llm_client)
            set_task_result(task_id=state.get('task_id'), key="answer", value=state['answer'])

    def _invoke_llm(self, prompt:str,llm_client:ChatOpenAI)->str:

        llm_res=llm_client.invoke(prompt)

        llm_content=getattr(llm_res,'content',"") or ""

        if not llm_content:

            return "LLM暂无回答任何内容"



        return llm_content

    def _stream_llm(self, task_id,prompt:str,llm_client:ChatOpenAI):

        accelerate_data=""
        for chunk in llm_client.stream(prompt):
            delta=getattr(chunk,'content',"") or ""
            if delta:
                push_sse_event(task_id=task_id,
                               event=SSEEvent.DELTA,
                               data={"answer":accelerate_data+delta})
                accelerate_data+=delta

        return accelerate_data

    def save_history(self, state: QueryGraphState):

        session_id=state.get('session_id')
        user_query=state.get('original_query')
        rewritten_query=state.get('rewritten_query')
        item_names=state.get('item_names') or []

        if state.get('answer'):
            try:
                #用户消息
                save_chat_message(session_id=session_id,
                              role="user",
                              text=user_query,
                                  rewritten_query=rewritten_query,
                                  item_names=item_names
                              )

                #AI角色的消息
                save_chat_message(session_id=session_id,
                                  role="assistant",
                                  text=state.get('answer'),
                                  rewritten_query=rewritten_query,
                                  item_names=item_names
                                  )
            except Exception as e:
                logger.error(f"保存历史对话到mongodb失败{str(e)}")

    def _format_chat_history(self, chat_history_context:list[Dict[str,Any]], usage_chars:int):
        """
         格式化历史对话
         Args:
             history: 历史对话
             char_budget:

         Returns:

         """

        formatted_lines = []
        used_chars = 0
        # 1. 遍历格式化后的文档
        role_map = {"user": "用户", "assistant": "助手"}
        for msg in chat_history_context:
            # 1.1 获取消息角色
            role = msg.get('role', '')

            # 1.2 获取消息内容
            text = msg.get('text', '')

            # 1.3 获取格式化后的行
            if not text or role not in role_map:
                continue

            formatted_line = f"{role_map[role]}: {text}"

            # 1.4 计算分割符长度
            seperator_usage = 1 if formatted_lines else 0

            # 1.5 计算总长度
            total_usage = seperator_usage + len(formatted_line)

            if used_chars + total_usage > usage_chars:
                break

            formatted_lines.append(formatted_line)
            used_chars += total_usage

        return "\n".join(formatted_lines), usage_chars - used_chars














