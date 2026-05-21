import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Sequence

from pymilvus import MilvusClient, DataType

from knowledge.processor.import_processor.base import *
from knowledge.processor.import_processor.exceptions import *
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.utils.back_state_util import BackStateUtil
from knowledge.utils.client.storage_clients import StorageClients


@dataclass
class _SCARLR_FIELD_SPC:
    field_name:str
    datatype:DataType
    max_length:Optional[int]=None

_SCALAR_FIELDS:Sequence[_SCARLR_FIELD_SPC] = [
    _SCARLR_FIELD_SPC(field_name="content", datatype=DataType.VARCHAR, max_length=65535),
    _SCARLR_FIELD_SPC(field_name="title", datatype=DataType.VARCHAR, max_length=65535),
    _SCARLR_FIELD_SPC(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535),
    _SCARLR_FIELD_SPC(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535),
    _SCARLR_FIELD_SPC(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535),
]

class _MilvusSchemaBuilder():
    """
    Milvus集合结构生成器
    """

    @staticmethod
    def build_schema(milvus_client: MilvusClient, dim: int):
        schema = milvus_client.create_schema(enable_dynamic_field=True)



        schema.add_field(field_name="chunk_id",
                         datatype=DataType.INT64,
                         is_primary=True,
                         auto_id=True
                         )

        schema.add_field(field_name="dense_vector",
                         datatype=DataType.FLOAT_VECTOR,
                         dim=dim)
        schema.add_field(field_name="sparse_vector",
                         datatype=DataType.SPARSE_FLOAT_VECTOR,max_length=100)

        for spec in _SCALAR_FIELDS:
            kwargs:Dict={
                "field_name":spec.field_name,
                "datatype":spec.datatype,

            }
            if spec.max_length:
                kwargs["max_length"] = spec.max_length

            schema.add_field(**kwargs)
        return schema


class _MilvusIndexBuilder:
    @staticmethod
    def build(milvus_client, collection_name):
        index = milvus_client.prepare_index_params(collection_name=collection_name)

        # 稠密向量：AUTOINDEX + COSINE（BGE-M3 已归一化，COSINE ≡ IP）
        index.add_index(field_name="dense_vector", index_name="dense_vector_index",
                        index_type="AUTOINDEX", metric_type="COSINE")

        # 稀疏向量：倒排索引 + 内积（token 权重累加）
        index.add_index(field_name="sparse_vector", index_name="sparse_vector_index",
                        index_type="SPARSE_INVERTED_INDEX", metric_type="IP")

        return index
class _MilvusInserter:
    def __init__(self,milvus_client:MilvusClient,collection_name:str):
        self.milvus_client = milvus_client
        self.collection_name = collection_name

    def insert_rows(self,data:List[Dict[str,Any]]):
        try:
         instert_result= self.milvus_client.insert(collection_name=self.collection_name, data=data)

         chunk_ids = instert_result.get("ids", [])

         for chunk_id,chunk in zip(chunk_ids, data):
             chunk["chunk_id"] = chunk_id

        except Exception as e:
            self.logger.error(f"向milvus插入数据失败 错误是{str(e)}")
            raise MilvusError(node_name=self.name, message=f"向milvus插入数据失败 错误是{str(e)}")

        return data



class ImportMilvusNode(BaseNode):
    name = "import_milvus_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 校验
        validated_chunks, dim = self._validate_state(state)
        try:
            milvus_client:MilvusClient = StorageClients.get_milvus_client()
        except Exception as e:
            self.logger.error(f"获取milvus客户端失败 报错是{str(e)}")
            raise MilvusError(node_name=self.name, message=f"Milvus客户端创建失败报错原因是{e}")

        chunks_collection = self.config.chunks_collection

        self.create_chunks_collection(chunks_collection, milvus_client,dim)

        inserter = _MilvusInserter(milvus_client=milvus_client,collection_name=chunks_collection)

        final_chunks = inserter.insert_rows(data=validated_chunks)

        state['chunks'] = final_chunks

        BackStateUtil.back_up(self, state)

        return state

    def _validate_state(self, state: ImportGraphState) -> tuple[list[Dict[str, Any]], int]:
        self.log_step("步骤一", f"校验状态")

        if  isinstance(state.get("chunks"),list):
            chunks: list = state.get("chunks")
        else:
         chunks: list = state.get("chunks").get("chunks")


        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(node_name=self.name, field_name="chunks", message="chunks字段不存在或者格式错误")

        validated_chunks = []
        for index, chunk in enumerate(chunks):

            if not chunk or not isinstance(chunk, dict):
                raise StateFieldError(node_name=self.name, field_name="chunks", message="chunks字段不存在或者格式错误")

            if chunk.get("dense_vector") and chunk.get("sparse_vector"):
                validated_chunks.append(chunk)
            else:
                self.logger.warning(f"第{index}个chunk没有向量信息 已经跳过")

        if not validated_chunks:
            raise StateFieldError(node_name=self.name, field_name="chunks",
                                  message="所有从chunks均没有有效向量 不能入库")
        dim = len(validated_chunks[0]["dense_vector"])
        self.logger.info(f"有效的chunks:{len(validated_chunks)} 向量维度是{dim}")

        return validated_chunks, dim

    def create_chunks_collection(self, chunks_collection: str, milvus_client: MilvusClient,dim:int):

        if milvus_client.has_collection(chunks_collection):
            self.logger.info(f"已存在{chunks_collection}集合")
            return

        scheam=_MilvusSchemaBuilder.build_schema(milvus_client,dim)
        index_params=_MilvusIndexBuilder.build(milvus_client,chunks_collection)

        milvus_client.create_collection(collection_name=chunks_collection, schema=scheam, index_params=index_params)

def _cli_main() -> None:

    import  json
    setup_logging()

    temp_dir = Path(r"D:\pycharm-workspace\shopkeeper_brain\knowledge\temp_data\20260519\state_back\embedding_chunks_node_state.json" )

    output_path =Path(r"D:\pycharm-workspace\shopkeeper_brain\knowledge\temp_data\20260519\state_back\chunks_vector_ids.json" )


    if not temp_dir.exists():
        raise FileNotFoundError(f"找不到输入文件: {temp_dir}")

    with open(temp_dir, "r", encoding="utf-8") as f:
        state = json.load(f)


    node = ImportMilvusNode()
    result_state = node.process(state)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_state.get("chunks"), f, ensure_ascii=False, indent=4)

    print(f"结果已保存至: {output_path}")


if __name__ == "__main__":
    _cli_main()