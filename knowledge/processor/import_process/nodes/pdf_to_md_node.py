import json
import os  # 新增：用于操作环境变量
import subprocess
from pathlib import Path
from typing import Tuple

from knowledge.processor.import_processor.base import *
from knowledge.processor.import_processor.exceptions import *
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.utils.back_state_util import BackStateUtil


class PdfToMdNode(BaseNode):
    """
    节点的逻辑处理入口
    """
    name = "pdf_to_md_node"
    def process(self, state: ImportGraphState) -> ImportGraphState:
        import_file_path, file_dir = self._validate_state(state)

        # 获取处理后的code码
        process_code = self._excute_mineru_parse(import_file_path, file_dir)
        if process_code != 0:
            raise PdfConversionError(message="mineru转换pdf失败", node_name=self.name)
        else:
            self.log_step("step2", "mineru转换pdf成功")

        md_path = self.get_md_parth(import_file_path, file_dir)
        state['md_path'] = md_path




        BackStateUtil.back_up(self, state)


        return state

    def _validate_state(self, state: ImportGraphState) -> Tuple[Path, Path]:
        """
        验证状态
        """
        import_file_path = state.get("import_file_path", "")
        self.log_step("step1", "准备校验和获取解析文件的路径和输出的目录")
        if not import_file_path:
            raise StateFieldError(node_name=self.name,
                                  field_name="import_file_path",
                                  message="导入文件路径不能为空")
        import_file_path_obj = Path(import_file_path)
        if not import_file_path_obj.exists():
            raise StateFieldError(node_name=self.name,
                                  field_name="import_file_path",
                                  message="解析的文件不存在,请检查路径是否有误")

        file_dir = state.get("file_dir", "")
        file_dir_obj=Path(file_dir)
        # 输出文件的目录
        if not file_dir:
            file_dir = import_file_path_obj.parent
        if not file_dir_obj.exists():
            raise StateFieldError(node_name=self.name,
                                  field_name="file_dir",
                                  message="输出目录不存在,请检查路径是否有误")

        file_dir_obj = Path(file_dir)
        self.logger.info(f"解析的文件路径{import_file_path},输出的文件目录是{file_dir_obj}")
        return import_file_path_obj, file_dir_obj

    def _excute_mineru_parse(self, import_file_path: Path, file_dir_path: Path) -> int:
        """
        以独立运行模式执行 MinerU 解析（自动启动引擎，已加入防卡死机制）
        """
        self.log_step("step2", "执行MinerU解析PDF (独立运行模式)")

        # 1. 核心修复：禁用本地回环代理，防止内部临时 API 请求被系统代理黑洞化导致卡死
        env = os.environ.copy()
        env["NO_PROXY"] = "127.0.0.1,localhost"

        # 2. 构建命令行 (不加 --api-url，让它自己拉起临时服务)
        cmd = [
            "mineru",
            "-p", str(import_file_path),
            "-o", str(file_dir_path),
            "--source", "local"
        ]

        process_start_time = time.time()

        # 3. 执行命令行
        proc = subprocess.Popen(
            args=cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            errors="replace",
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=env  # 注入修改后的环境变量
        )

        # 4. 安全地获取日志信息并防止 IO 管道阻塞
        try:
            while True:
                line = proc.stdout.readline()
                if not line:
                    if proc.poll() is not None:
                        break
                    # 短暂休眠防止死循环导致 CPU 占用过高
                    time.sleep(0.1)
                    continue

                # 打印去除换行符的日志
                self.logger.info(f"执行MinerU产生的日志：{line.strip()}")

            # 核心修复：增加 600 秒 (10分钟) 超时机制。如果 PDF 极大，可以把 600 改得更大
            processed_code = proc.wait(timeout=600)

        except subprocess.TimeoutExpired:
            self.logger.error(f"MinerU 进程运行超时 (10分钟)，强制结束子进程以防止死锁：{import_file_path.name}")
            proc.kill()
            processed_code = -1
        except Exception as e:
            self.logger.error(f"执行 MinerU 时发生未知异常: {str(e)}")
            proc.kill()
            processed_code = -1

        process_end_time = time.time()

        # 5. 结果校验与日志输出
        if processed_code == 0:
            self.logger.info(
                f"MinerU 成功解析PDF：{import_file_path.name} 耗时: {process_end_time - process_start_time:.2f}s")
        else:
            self.logger.error(f"MinerU 解析PDF失败：{import_file_path.name}，退出码: {processed_code}")

        return processed_code

    # 获取md文件路径
    def get_md_parth(self, import_file_path_obj: Path, file_dir_obj: Path) -> str:
        file_name = import_file_path_obj.stem
        return str(file_dir_obj / file_name / "hybrid_auto" / f"{file_name}.md")


#########
# 测试
##########
if __name__ == '__main__':
    setup_logging()
    pdf_to_md_load = PdfToMdNode()
    result = pdf_to_md_load.process(state={
        "import_file_path": r"D:\pycharm-workspace\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\万用表的使用.pdf",
        "file_dir": r"D:\pycharm-workspace\shopkeeper_brain\knowledge\processor\import_processor\temp_dir"
    })
    result_str = json.dumps(result, indent=4, ensure_ascii=False)
    print(result_str)