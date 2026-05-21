import json
import logging
from pathlib import Path

from knowledge.processor.import_process.state import ImportGraphState

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

class BackStateUtil:
    @staticmethod
    def back_up(node_instance,state: ImportGraphState):
        try:
            output_path = Path(state['file_dir']) / "state_back" / f"{node_instance.name}_state.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=4)
            logger.info(f"{node_instance.name}节点备份完成 备份地址是 {str(output_path) }")
        except Exception as e:
            logger.error(f"{node_instance.name}备份失败: {str(e)}")
