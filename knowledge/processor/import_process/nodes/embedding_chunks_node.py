import json
from pathlib import Path
from typing import Dict, Any

from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from knowledge.processor.import_process.base import *
from knowledge.processor.import_process.exceptions import *
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.back_state_util import BackStateUtil
from knowledge.utils.client.ai_clients import AIClients


class EmbeddingChunksNode(BaseNode):
    name = "embedding_chunks_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        self.log_step("步骤一",f"校验状态")
        validated_chunks: list = self._validate_state(state)
        self.log_step("步骤二",f"开始处理 获取BGE-M3嵌入模型客户端")
        try:
            bge_m3_client = AIClients.get_bge_m3_client()
        except Exception as e:
            self.logger.info(f"获取BGE-M3嵌入模型客户端失败:{str(e)}")
            raise EmbeddingError(node_name=self.name,
                               message=f"获取BGE-M3嵌入模型客户端失败:{str(e)}")

        batch_size=self.config.embedding_batch_size

        total=len(validated_chunks)
        final_chunks=[]
        for index in range(0,total,batch_size):
            bath_chunks=validated_chunks[index:index+batch_size]
            batch_end =index+len(bath_chunks)
            self.log_step("步骤三",f"开始处理 批量处理[{index+1}-{batch_end}]/{total}")
            current_chunks=self._embed_chunks(bath_chunks,bge_m3_client)
            final_chunks.extend(current_chunks)
        state["chunks"]=final_chunks
        output_path =Path(state['file_dir']) / "chunks_vector.json"
        # output_path = Path(r"../temp_dir") / "chunks_vector.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(state["chunks"], f, ensure_ascii=False, indent=4)

        BackStateUtil.back_up(self, state)

        return state





    def _validate_state(self, state: ImportGraphState)->list[Dict[str,Any]]:
        chunks: list = state.get("chunks")
        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(node_name=self.name,
                                  field_name="chunks",
                                  expected_type=list, )
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                raise ValidationError(node_name=self.name,
                                      message=f"chunk的第[{index + 1}]的内容不是dict: chunk的类型是 {type(chunk)}")

        return chunks

    def _embed_chunks(self, bath_chunks:list[Dict[str,Any]], embed_model:BGEM3EmbeddingFunction)->list[Dict[str,Any]]:

        embedding_documents=[f"{chunk.get("content")}\n{chunk.get('item_name')}"  for chunk in bath_chunks ]

        embed_vector=embed_model.encode_documents(embedding_documents)
        sparse_scr=embed_vector.get("sparse")
        for i,chunk in enumerate(bath_chunks):
            chunk["dense_vector"]=embed_vector.get('dense')[i].tolist()
            chunk["sparse_vector"]=self._extract_sparse_vector(sparse_scr,i)
        return bath_chunks

    #从稀疏矩阵提取当前chunk对象提取稀疏向量
    def _extract_sparse_vector(self, sparse_scr, index:int):

        start_index=sparse_scr.indptr[index]
        end_index=sparse_scr.indptr[index+1]
        #获取token_id
        token_id=sparse_scr.indices[start_index:end_index].tolist()
        #获取权重
        weight=sparse_scr.data[start_index:end_index].tolist()

        return dict(zip(token_id,weight))


if __name__ == '__main__':
    # 注意：确保 setup_logging 在你的 base 模块或当前上下文中已定义
    setup_logging()

    base_temp_dir = Path(
        r"/knowledge/processor/import_process\temp_dir")

    input_path = base_temp_dir / "chunks.json"
    output_path = base_temp_dir / "chunks_vector.json"

    # 1. 读取上游状态
    if not input_path.exists():
        print(f"找不到输入文件: {input_path}")
        # 补充：如果找不到文件应该退出测试，避免后续报错
        exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        content = json.load(f)

    # 2. 构建模拟的图状态 (Graph State)
    state = {
        "chunks": content

    }

    # 3. 触发节点执行
    node_bge_embedding = EmbeddingChunksNode()
    # 补充：如果你的 BaseNode 强依赖配置注入，请在测试时模拟注入，否则 self.config 会报错
    # node_bge_embedding.config = MockConfig(embedding_batch_size=10)

    proceed_result = node_bge_embedding.process(state)

    # 4. 结果落盘
    # with open(output_path, "w", encoding="utf-8") as f:
    #     json.dump(proceed_result, f, ensure_ascii=False, indent=4)

    print(f" 向量生成测试完成！结果已成功备份至:\n{output_path}")