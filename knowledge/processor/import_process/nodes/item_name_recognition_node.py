import json
from pathlib import Path
from typing import Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pymilvus import MilvusClient, DataType

from knowledge.processor.import_process.base import *
from knowledge.processor.import_process.exceptions import *
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.prompt.import_prompt import ITEM_NAME_SYSTEM_PROMPT, ITEM_NAME_USER_PROMPT_TEMPLATE
from knowledge.utils.back_state_util import BackStateUtil
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients


class ItemNameRecognitionNode(BaseNode):
    name = "item_name_recognition_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:

        self.logger.info(f"开始执行节点:{self.name}")
        # 参数校验
        file_title, chunks, item_name_chunk_k, item_name_chunk_size = self._validate_state(state)

        # 构建上下文
        content = self._prepare_llm_context(chunks, item_name_chunk_k)

        # 调用llm大模型提取商品名称
        item_name = self._recognition_item_name(content, file_title)

        # 向量化
        dense_vector, sparse_vector = self._embedding_item_name(item_name)
        #入库
        self._insert_milvus(dense_vector, sparse_vector, file_title, item_name)
        # 回填
        self._fill_item_name(state, item_name)

        BackStateUtil.back_up(self, state)

        return state

    def _validate_state(self, state: ImportGraphState) -> tuple[str, list, int, int]:
        """
        参数校验
        Args:
            state:

        Returns:

        """
        # 商品名兜底
        file_title = state.get("file_title")

        if not file_title:
            raise StateFieldError(node_name=self.name, field_name="file_title", expected_type=str)

        chunks = state.get("chunks")
        if not chunks:
            raise StateFieldError(node_name=self.name,
                                  field_name="chunks",
                                  expected_type=list)
        item_name_chunk_k = self.config.item_name_chunk_k
        item_name_chunk_size = self.config.item_name_chunk_size
        if not item_name_chunk_k or item_name_chunk_k < 0:
            raise ValidationError(message="商品名识别的辅助切片数不合法")
        if not item_name_chunk_size or item_name_chunk_size < 0:
            raise ValidationError(message="商品名识别的辅助切片长度不合法")

        return file_title, chunks, item_name_chunk_k, item_name_chunk_size

    def _prepare_llm_context(self, chunks: list[Dict], item_name_chunk_k: int) -> str:

        final_context = []
        for index, chunk in enumerate(chunks[:item_name_chunk_k]):
            if not isinstance(chunk, dict):
                continue

            content = chunk.get("content")

            splice_context = f"[切片]-f{index}-{content}"

            final_context.append(splice_context)

        return "\n".join(final_context)

    def _recognition_item_name(self, item_name_content: str, file_title: str) -> str:
        # 调用llm客户端
        try:
            llm_client: ChatOpenAI = AIClients.get_llm_client(response_format=False)
        except ConnectionError as e:
            self.logger.error(f"Openai的LLM客户端创建失败 :{str(e)}")
            raise LLMError(message=f"LLM服务调用失败:{str(e)}")

        system_prompt = ITEM_NAME_SYSTEM_PROMPT

        user_prompt = ITEM_NAME_USER_PROMPT_TEMPLATE.format(file_title=file_title,
                                                            context=item_name_content)
        try:
            llm_response = llm_client.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]
            )
            llm_result = llm_response.content.strip('')
            self.logger.info(f"LLM文档名称是{file_title} 商品名称识别结果:{llm_result}")
            return llm_result
        except Exception as e:
            self.logger.error(f"LLM服务调用失败:{str(e)},进行降级使用文件名称{file_title}")

            return file_title

    def _embedding_item_name(self, item_name: str) -> tuple[Optional[list], Optional[dict[Any, Any]]]:

        try:
            bge_m3_ef = AIClients.get_bge_m3_client()
            vector_result = bge_m3_ef.encode_queries(queries=[item_name])
            # 稠密向量
            dense_vector = vector_result.get('dense')[0].tolist()
            # 稀疏向量
            sparse_csr = vector_result.get('sparse')

            start_index = sparse_csr.indptr[0]
            end_index = sparse_csr.indptr[1]
            token_id = sparse_csr.indices[start_index:end_index].tolist()
            weights = sparse_csr.data[start_index:end_index].tolist()

            sparse_vector = dict(zip(token_id, weights))
            self.logger.info(f"商品名称向量化结果:{item_name} "
                             f"稠密向量:{dense_vector} "
                             f"稀疏向量:{sparse_vector}")
            return dense_vector, sparse_vector

        except ConnectionError as e:
            self.logger.error(f"BGE_M3嵌入模型客户端创建失败 :{str(e)}")
            return None, None

    def _insert_milvus(self, dense_vector: list, sparse_vector: Dict[str, Any], file_title: str,
                       item_name: str) -> None:
        """
        向 milvus 插入向量
        Args:
            dense_vector:
            sparse_vector:
            file_title:
            item_name:

        Returns:

        """
        try:

            if not dense_vector or not sparse_vector:
                self.logger.error(f"商品名称向量化结果为空,商品名称:{item_name}")
                return
            milvus_client = StorageClients.get_milvus_client()

            item_name_collection_name = self.config.item_name_collection
            if not milvus_client.has_collection(collection_name=item_name_collection_name):
                self._create_item_name_collection(item_name_collection_name, milvus_client)

            #构建数据

            item_name_data_row={
                "file_title":file_title,
                "item_name":item_name,
                "dense_vector":dense_vector,
                "sparse_vector":sparse_vector
            }
            insert_result= milvus_client.insert(item_name_collection_name,item_name_data_row)
            self.logger.info(f"向量插入结果:{insert_result}  主键值是:{insert_result.get("ids")}")
        except Exception as e:
            self.logger.error(f"向 milvus 插入向量失败:{str(e)}")



    # def _create_item_name_collection(self, item_name_collection_name: str,
    #                                  milvus_client: MilvusClient):
    #     """
    #     创建商品名称向量集合
    #
    #
    #     """
    #     schema = milvus_client.create_schema()
    #     # 1.1 创建主键字段约束
    #     schema.add_field(field_name="pk", datatype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=10)
    #
    #     # 1.2 创建标量字段的约束
    #     schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
    #     schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
    #
    #     # 1.3 创建向量字段的约束
    #     schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    #     schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
    #
    #     index_params = milvus_client.prepare_index_params()
    #     index_params.add_index(
    #         field_name="dense_vector",
    #         index_name="dense_vector_index",
    #         index_type="AUTOINDEX",
    #         metric_type="COSINE"  # IP COSINE L2
    #         # Milvus计算出来的稠密向量已经进行了归一化处理 所以度量类型选择COSINE或者IP效果一样，但是如果使用别的方式计算出来的稠密向量没有经过归一化处理 那么COSINE和IP就不相等
    #     )
    #     index_params.add_index(
    #         field_name="sparse_vector",
    #         index_name="sparse_vector_index",
    #         index_type="SPARSE_INVERTED_INDEX",
    #         metric_type="IP"  # 只有IP 和BM25
    #     )
    #     milvus_client.create_collection(collection_name=item_name_collection_name,
    #                                     schema=schema, index_params=index_params)
    #     self.logger.info(f"创建商品名称向量集合成功:{item_name_collection_name}")
    def _create_item_name_collection(self, item_name_collection_name: str,
                                     milvus_client: MilvusClient):
        """
        创建商品名称向量集合
        """
        schema = milvus_client.create_schema()

        # 1.1 创建主键字段约束 (修复：最大长度提升到64，以容纳36位的自动生成UUID)
        schema.add_field(field_name="pk", datatype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=64)

        # 1.2 创建标量字段的约束
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)

        # 1.3 创建向量字段的约束
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        # 2. 创建索引参数
        index_params = milvus_client.prepare_index_params()

        # 稠密向量索引
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            # 如果你确认你的 Embedding 模型输出已经归一化，建议改为 "IP" 以提升性能
            metric_type="COSINE"
        )

        # 稀疏向量索引
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP"
        )

        # 3. 创建 Collection 并同时绑定 Schema 和 Index
        milvus_client.create_collection(
            collection_name=item_name_collection_name,
            schema=schema,
            index_params=index_params
        )
        self.logger.info(f"创建商品名称向量集合成功:{item_name_collection_name}")


    def _fill_item_name(self, state, item_name:str):

        chunks=state.get("chunks")
        for chunk in chunks:
            chunk["item_name"]=item_name
        state["item_name"]=item_name
        try:
            output_path = Path(state['file_dir']) / "chunks_vector.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(state.get("chunks"), f, ensure_ascii=False, indent=4)
            output_path = Path(state['file_dir']) / "chunks.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(state.get("chunks"), f, ensure_ascii=False, indent=4)
        except Exception as e:
            self.logger.error(f"向文件写入数据失败:{str(e)}")


if __name__ == '__main__':

    setup_logging()

    # 1. 读取chunk.json
    chunk_json_path = r"D:\pycharm-workspace\shopkeeper_brain\knowledge\temp_data\20260518\chunks.json"
    with open(chunk_json_path, "r", encoding="utf-8") as f:
        chunk_content = json.load(f)

    # 2. 构建state
    state = {
        "file_title": "中华人民共和国刑法（2023年修正）.pdf",
        "chunks": chunk_content
    }

    # 3. 实例化节点
    node = ItemNameRecognitionNode()

    # 4. 调用process
    result = node.process(state)

    # 5. 输出结果
    print(f"商品名: {result.get('item_name')}")
    print(f"chunks数量: {len(result.get('chunks', []))}")
    print(f"首个chunk是否含item_name: {'item_name' in result['chunks'][0]}")

