from pathlib import Path

from knowledge.processor.import_processor.base import *
from knowledge.processor.import_processor.exceptions import *
from knowledge.processor.import_processor.state import *
from knowledge.utils.back_state_util import BackStateUtil


class EntryNode(BaseNode):
    name="entry_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        import_file_path=state.get("import_file_path","")
        file_dir=state.get("file_dir","")
        if not import_file_path:
            raise StateFieldError(node_name=self.name,
                                  field_name="import_file_path",
                                  expected_type=str
                                  )
        import_file_path_obj = Path(import_file_path)
        if not import_file_path_obj.exists():
            raise StateFieldError(node_name=self.name,
                                    field_name="import_file_path",
                                  expected_type=Path
                                    )

        if import_file_path_obj.suffix ==".pdf":
            state["is_pdf_read_enabled"]=True
            state["pdf_path"]=str(import_file_path_obj)
        elif import_file_path_obj.suffix ==".md":
            state["is_md_read_enabled"]=True
            state["pdf_path"] = str(import_file_path_obj)
        else:
            self.logger.error(f"不支持的文件格式: {import_file_path_obj.suffix}")
            raise ValidationError(node_name=self.name,
                                  message="不支持的文件格式: {import_file_path.suffix}")

        state["file_title"]=import_file_path_obj.stem

        BackStateUtil.back_up(self, state)
        return state


if __name__ == '__main__':
    import json
    entry_node=EntryNode()
    init_state={
        "import_file_path":r"D:\pycharm-workspace\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\万用表的使用\hybrid_auto\万用表的使用.md",
        "file_dir":r"D:\pycharm-workspace\shopkeeper_brain\knowledge\processor\import_processor\temp_dir"
    }
    result=entry_node.process(state=init_state)
    print(json.dumps(result,ensure_ascii=False,indent=4))
