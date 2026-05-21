import base64
import re
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import *

from openai import OpenAI
from pypdfium2._helpers.pageobjects import ImageInfo

from knowledge.processor.import_process.base import *
from knowledge.processor.import_process.exceptions import *
from knowledge.processor.import_process.state import *
from knowledge.utils.back_state_util import BackStateUtil
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients


@dataclass
class ImageContext:
    head: str  # 标题内容
    pre_text: str  # 上文内容
    post_text: str  # 下文内容


@dataclass
class ImageInfo:
    # 存放图片的完整信息
    name: str  # 名字
    path: str  # 地址
    imag_context: ImageContext  # 图片上下文信息


# 读取md的内容路径 以及图片的目录
# 备份新的md_content
class _MdFileHandler:
    def __init__(self, logger: Logger, node_name: str):
        self.logger = logger
        self.node_name = node_name


    def backup(self, md_path_obj: Path, new_md_content: str) -> str:
        self.logger.info("【step_5】备份新文件")

        new_file_path = md_path_obj.with_name(
            f"{md_path_obj.stem}_new{md_path_obj.suffix}"
        )
        new_file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(new_file_path, "w", encoding="utf-8") as f:
                f.write(new_md_content)
            self.logger.info(f"处理后的文件已备份至: {new_file_path}")
        except IOError as e:
            self.logger.error(f"写入新文件失败 {new_file_path}: {e}")
            raise FileProcessingError(
                f"文件写入失败: {e}", node_name="md_img_node"
            )
        return str(new_file_path)

    def validate_and_read_md(self, state: ImportGraphState) -> Tuple[str, Path, Path]:
        # 读取md的内容
        # 读取md路径
        # 读取图片目录
        md_path = state.get("md_path", "")
        if not md_path:
            # 非空判断
            raise StateFieldError(message="md文档的路径为空",
                                  field_name="md_path",
                                  expected_type=str
                                  )
        md_path_obj = Path(md_path)
        if not md_path_obj.exists():
            # 文件不存在
            raise StateFieldError(message="md文档的路径不存在",
                                  field_name="md_path",
                                  expected_type=str
                                  )
        try:
            with open(md_path_obj, "r", encoding="utf-8") as f:
                md_content = f.read()
        except IOError as expected:
            self.logger.error(f"{self.node_name} 读取md文档失败: {expected}")
            raise FileProcessingError(message="md文档的读取失败",
                                      field_name="md_path",
                                      expected_type=str
                                      )
        img_dir = md_path_obj.parent / "images"

        return md_content, md_path_obj, img_dir


# 图片扫描器
# 根据图片目录得到有效的图片文件
class _ImageScanner:
    def __init__(self, logger: Logger):
        self.logger = logger

    def scan_imgs_dir(self, img_dir_obj: Path, md_content: str, image_extensions: Set[str], img_content_length: int) -> \
            List[ImageInfo]:
        """
            1.扫描指定图片目录下的所有的图片文件
            2.遍历每一个图片文件去MD中获取到位置上下文
            3.获取 上文中的 上文标题和上文内容
            4.下文信息(下文的内容)
            5.把每一个图片的上下文放到每个图片的完整图片信息的容器中
        """
        # 遍历图片目录
        img_info_list = []
        for img_path in img_dir_obj.iterdir():
            # 不看子目录
            if not img_path.is_file():
                self.logger.info(f"{img_path}不是一个有效的文件")
                continue

            if not img_path.suffix in image_extensions:
                self.logger.info(f"{img_path}不是一个有效的图片文件,请检查是否是允许的图片后缀格式")
                continue

            # 找上下文
            cxt = self._find_context(img_path.name, md_content, img_content_length)
            if not cxt:
                self.logger.info(f"这个图片{img_path}没有引用")
                continue
            # 封装ImageInfo对象 放到容器里面
            img_info_list.append(ImageInfo(name=str(img_path.name),
                                           path=str(img_path),
                                           imag_context=cxt))

        return img_info_list

    def _find_context(self, img_name: str,
                      md_content: str,
                      img_content_length: int) -> Optional[ImageContext]:
        """
        根据图片名称 找到图片的上下文
        """
        # 预编译一个正则规则 从md中抓取到图片
        pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(img_name) + r".*?\)")
        md_lines = md_content.split("\n")

        for md_indx, md_line in enumerate(md_lines):
            # 如果这一行不匹配我们设定的规则，就跳过
            if not pattern.match(md_line):
                continue

            head, prev_index = self._find_heading_up(md_lines=md_lines, from_idx=md_indx)
            pre_lines = md_lines[prev_index + 1:md_indx]
            pre_context = self._extract_limited_context(pre_lines, img_content_length, direction="front")

            next_index = self._find_heading_down(md_lines, md_indx)
            next_lines = md_lines[md_indx + 1:next_index]
            post_context = self._extract_limited_context(next_lines, img_content_length, direction="back")

            return ImageContext(head=head,
                                pre_text=pre_context,
                                post_text=post_context)
        return None

    def _find_heading_up(self, md_lines: List[str], from_idx: int) -> Tuple[str, int]:
        """
        根据图片名称 找到图片的上下文
        """
        for i in range(from_idx - 1, -1, -1):
            if re.match(r"^#{1,6}\s+", md_lines[i]):
                return md_lines[i], i
        return "", -1

    def _find_heading_down(self, md_lines: List[str], from_idx: int) -> int:
        """
        根据图片名称 找到图片的上下文
        """
        for i in range(from_idx + 1, len(md_lines)):
            if re.match(r"^#{1,6}\s+", md_lines[i]):
                return i
        return len(md_lines)

    def _extract_limited_context(self,
                                 extracted_md_lines: List[str],
                                 img_content_length: int,
                                 direction: str = "front"
                                 ) -> str:
        """
        从上下文里提取 Limited_context_length 个字符
        """
        current_paragraph = []
        paragraphs = []
        for line in extracted_md_lines:
            # 1.1 定义自然而然段落的规则
            is_blank_line = not line.strip()
            # 1.2 定义人为设计的图片段落规则
            is_other_image = re.match(r"^!\[.*?\]\(.*?\)$", line.strip())

            if is_blank_line or is_other_image:
                # 如果是空行或者其他行
                if current_paragraph:
                    paragraphs.append("\n".join(str(current_paragraph)))
                    current_paragraph = []
                continue

            current_paragraph.append(line)

        if direction == "front":
            paragraphs.reverse()

        total = 0
        selected = []
        for paragraph in paragraphs:
            if total + len(paragraph) > img_content_length and selected:
                break

            selected.append(paragraph)
            total += len(paragraph)

        if direction == "front":
            selected.reverse()
        return "\n\n".join(selected)


class _VLMSummarizer:
    # 根据图片信息和图片的上下文 生成对应的图片的摘要信息
    def __init__(self, logger: Logger):
        self.logger = logger

    def _summary_all(self, document_name: str,
                     img_info_list: List[ImageInfo],
                     vl_model: str):
        summaries = {}
        try:
           vlm_client = AIClients.get_vlm_client()
        except Exception as e:
            self.logger.error(f"获取VLM客户端失败: {e}")
            for img_info in img_info_list:
                summaries[img_info.name] = "暂无摘要"
            return summaries

        # 调用VLM 给图片生成摘要
        for img_info in img_info_list:
            summaries[img_info.name] = self._summary_one(document_name,
                                                         img_info,
                                                         vlm_client,
                                                         vl_model)

        return summaries

    # 给图片生成摘要信息
    def _summary_one(self,
                     document_name: str,
                     img_info: ImageInfo,
                     vlm_client: OpenAI,
                     vl_model: str):
        """
        Args:
            img_info: 当前图片信息
            vlm_client: 链接
            vl_model: vlm名称
        """
        parts = [p for p in (img_info.imag_context.head,
                             img_info.imag_context.pre_text,
                             img_info.imag_context.post_text)
                 if p
                 ]
        final_context = "\n".join(parts) if parts else "暂无上下文"
        try:
            with open(img_info.path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")
        except IOError as e:
            self.logger.error(f"图片 {img_info.name} 打开失败: {e}")
            return "暂无图片"

        try:
            resp = vlm_client.chat.completions.create(
                model=vl_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"任务：为Markdown文档中的图片生成一个简短的中文标题。\n"
                                f"背景信息：\n"
                                f"  1. 所属文档标题：\"{document_name}\"\n"
                                f"  2. 图片上下文：{final_context}\n"
                                f"请结合图片内容和上述上下文信息，"
                                f"用中文简要总结这张图片的内容，"
                                f"生成一个精准的中文标题摘要（不要包含图片二字）。"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_data}"
                            },
                        },
                    ],
                }],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"图片摘要生成失败 {img_info.path}: {e}")
            return "暂无图片描述"


class _ImageUploader:
    # 把本地图片上传到minio上 然后把minio的图片地址进行替换
    def __init__(self, logger: Logger):
        self.logger = logger

    def upload_and_replace(self, object_dir_name: str,
                                    md_content: str,
                                    img_info_list: List[ImageInfo],
                                    summaries: Dict[str, str],
                                    minio_url: str,
                                    minio_bucket_name: str) -> str:
        # 吧图片上传到minio  然后替换图片地址 修改图片的名称
        """

        Args:
            object_dir_name: minio对象目录
            md_content: md内容
            img_info_list: 图片列表
            summaries: 图片摘要
            minio_url: minio地址
            minio_bucket_name: 桶的名字

        Returns: 更新后的md内容

        """
        #上传
        remote_urls=self._upload_all(object_dir_name, img_info_list,minio_url,minio_bucket_name)

        #更新
        md_content=self._update_md(md_content,summaries, remote_urls)

        return  md_content






    def _upload_all(self, object_dir_name:str, img_info_list:List[ImageInfo],
                    minio_url: str, minio_bucket_name:str)->Dict[str, str]:

        remote_urls = {}

        try:
            minio_client =StorageClients.get_minio_client()
        except Exception as e:
            for img_info in img_info_list:
                remote_urls[img_info.name] = img_info.path
            return remote_urls

        for img_info in img_info_list:
            object_name = f"{object_dir_name}/{img_info.name}"

            try:
                minio_client.fput_object(bucket_name=minio_bucket_name,
                                         object_name=object_name,
                                         file_path=img_info.path)
                self.logger.info(f"成功将图片{img_info.name}上传到MinIO中")
                remote_urls[img_info.name] = f"{minio_url}/{minio_bucket_name}/{object_name}"
            except Exception as e:
                self.logger.info(f"图片上传失败 {img_info.path}: 使用本地图片兜底")
                remote_urls[img_info.name]=img_info.path

        self.logger.info(f"获取到远程的{len(remote_urls)}")
        return remote_urls

    def _update_md(self,md_content:str, summaries:Dict[str, str], remote_urls:Dict[str,str])->str:

        pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")

        def replacer(match:re.Match) -> str:
            for img_name, img_summary in summaries.items():
                origin_img_path = match.group(2)
                img_name_in_md = Path(origin_img_path).name
                if img_name == img_name_in_md:
                    return f"![{img_summary}]({remote_urls[img_name]})"
            return match.group(0)

        return pattern.sub(replacer, md_content)




# 分别调用设计的四个类对应的方法
class MarkDownToImgNode(BaseNode):
    name = "md_to_img_node"

    def __init__(self):
        super().__init__()
        self._md_file_handler = _MdFileHandler(logger=self.logger, node_name="md_file_handler")
        self._image_scanner = _ImageScanner(self.logger)
        self._image_uploader = _ImageUploader(self.logger)
        self._vlm_summarizer = _VLMSummarizer(self.logger)

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 调用各个类 入口逻辑
        config = self.config
        self.log_step("step1", "读取MD内容、路径以及图片的目录")
        md_content, md_path_obj, img_dir_obj = self._md_file_handler.validate_and_read_md(state)

        if not img_dir_obj.exists():
            state["md_content"] = md_content
            return state

        self.log_step("step2", "准备开始扫描图片目录")
        img_info_list: List[ImageInfo] = self._image_scanner.scan_imgs_dir(img_dir_obj,
                                                                           md_content,
                                                                           config.image_extensions,
                                                                           config.img_content_length)
        self.log_step("step3", "利用VLM提取摘要")
        summaries: Dict[str, str] = self._vlm_summarizer._summary_all(md_path_obj.stem, img_info_list, config.vl_model)
        self.log_step("step4", "上传文件到MinIO,且更新MD")


        new_md_content = self._image_uploader.upload_and_replace(md_path_obj.stem, md_content, img_info_list,
                                                               summaries,
                                                               config.get_minio_base_url(),
                                                               config.minio_bucket)

        self._md_file_handler.backup(md_path_obj, new_md_content)

        state['md_content'] = new_md_content
        BackStateUtil.back_up(self, state)
        return state


if __name__ == '__main__':
    setup_logging()
    md_to_img_node = MarkDownToImgNode()
    init_state = {
        "md_path": r"D:\pycharm-workspace\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\万用表的使用\hybrid_auto\万用表的使用.md"
    }
    importGraphState = md_to_img_node.process(init_state)
    print(importGraphState)
