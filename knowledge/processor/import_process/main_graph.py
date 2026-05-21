import json

from langgraph.graph import END
from langgraph.graph.state import *

from knowledge.processor.import_processor.nodes.document_split_node import DocumentSplitNode
from knowledge.processor.import_processor.nodes.embedding_chunks_node import EmbeddingChunksNode
from knowledge.processor.import_processor.nodes.entry_node import EntryNode
from knowledge.processor.import_processor.nodes.import_milvus_node import ImportMilvusNode
from knowledge.processor.import_processor.nodes.item_name_recognition_node import ItemNameRecognitionNode
from knowledge.processor.import_processor.nodes.md_to_img_node import *
from knowledge.processor.import_processor.nodes.pdf_to_md_node import PdfToMdNode


def import_router(state: ImportGraphState):
    # 设置路由规则
    if state.get("is_pdf_read_enabled"):
        return "pdf_to_md_node"
    elif state.get("is_md_read_enabled"):
        return END


# 定义运行的节点


def import_graph() -> CompiledStateGraph:
    work_flow = StateGraph(ImportGraphState)  # type:ignore
    # 入口节点
    work_flow.set_entry_point("entry_node")
    # 其他节点
    node_name_obj = {
        "entry_node": EntryNode(),
        "pdf_to_md_node": PdfToMdNode(),
        "md_to_img_node": MarkDownToImgNode(),
        "document_split_node": DocumentSplitNode(),
        "item_name_recognition_node": ItemNameRecognitionNode(),
        "embedding_chunks_node": EmbeddingChunksNode(),
        "import_milvus_node": ImportMilvusNode(),
    }

    # 添加节点
    for node_name, node_obj in node_name_obj.items():
        work_flow.add_node(node_name, node_obj)

    # 定义边
    work_flow.add_conditional_edges("entry_node", import_router, {
        "pdf_to_md_node": "pdf_to_md_node",  # key:路由函数的返回值 value:节点的名字
        "md_to_img_node": "md_to_img_node",
        END: END
    })

    # 5.2 定义业务边
    work_flow.add_edge("pdf_to_md_node", "md_to_img_node")  # pdf到md

    work_flow.add_edge("md_to_img_node", "document_split_node")  # md到文档切分

    work_flow.add_edge("document_split_node", "item_name_recognition_node")  # 文档切分到商品名识别

    work_flow.add_edge("item_name_recognition_node", "embedding_chunks_node")  # 商品名识别到嵌入
    work_flow.add_edge("embedding_chunks_node", "import_milvus_node")  # 嵌入到导入
    work_flow.add_edge("import_milvus_node", END)  # 导入到结束

    compile_state_graph = work_flow.compile()
    return compile_state_graph


import_app = import_graph()


def run_import_graph():
    grap_state = {
        "import_file_path": r"D:\pycharm-workspace\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\万用表的使用.pdf",
        "file_dir": r"D:\pycharm-workspace\shopkeeper_brain\knowledge\processor\import_processor\temp_dir"
    }
    final_state = {}
    for event in import_app.stream(grap_state):
        for key, value in event.items():
            print(f"当前正在执行的节点：{key}")
            final_state = value
        return final_state


if __name__ == '__main__':
    setup_logging()
    final_state = run_import_graph()
    print(json.dumps(final_state, ensure_ascii=False, indent=4))

    # 整个执行的状态图(方便观察) ascii
    print(import_app.get_graph().print_ascii())
