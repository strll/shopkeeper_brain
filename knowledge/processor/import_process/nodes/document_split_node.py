import json
import os
import re
from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from knowledge.processor.import_processor.base import *
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.utils.back_state_util import BackStateUtil
from knowledge.utils.markdown_util import MarkdownTableLinearizer


class DocumentSplitNode(BaseNode):
    name = "document_split_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        config = self.config
        md_content, file_title, max_content_length, min_content_length = self._validate_state(state, config)
        #一次切分
        sections: List[Dict[str, Any]] = self._split_by_headings(md_content, file_title)

        #二次切分或者合并
        final_sections=self._split_and_merge(sections,max_content_length,min_content_length)

        #组装成chunk对象
        final_chunks=self._assemble_chunks(final_sections)
        #备份
        self._back_up(final_chunks,state)
        state["chunks"]=final_chunks

        BackStateUtil.back_up(self, state)


        return state
    def _split_and_merge(self,sections: List[Dict[str, Any]],
                         max_content_length,
                         min_content_length):
        current_sections=[] #切分列表
        for section in sections:
            current_section=self._split_long_section(section,max_content_length)
            current_sections.extend(current_section)

        final_sections=self._merger_short_section(current_sections,min_content_length)


        return  final_sections


    # 参数校验
    def _validate_state(self, state: ImportGraphState, config) -> tuple[str | None, str | None, int, int]:
        self.log_step("step1", "节分文档的目录获取")

        md_content = state.get("md_content")

        if md_content:
            md_content = (md_content.replace("\r\n", "\n")
                          .replace("\r", "\n"))

        file_title = state.get("file_title")

        if config.max_content_length <= 0 or config.min_content_length < 0:
            raise ValueError(f"config配置文件的切片长度校验有问题")

        return (md_content,
                file_title,
                config.max_content_length, config.min_content_length)

    # 根据标题去切分
    def _split_by_headings(self, md_content: str, file_title: str) -> List[Dict[str, Any]]:
        """
        对所有标题都进行切分
        Args:
            md_content:
            file_title:

        Returns:

        """
        in_fence = False  # 是否在代码块内
        body_liens = []
        sections = []  # 最终收集到的章节对象
        current_title = ""
        hierarchy = [""] * 7  # （数组）存储所有标题内容（作为section的父标题使用） 标题层级追踪数组
        current_level = 0



        def _flush() -> List[Dict[str, Any]]:
            """
            打包标题
            Returns:

            """
           # body = md_content.replace("\r\n", "\n").replace("\r", "\n")
            body = "\n".join(body_liens)
            if current_title or body:
                parent_title = ""
                for i in range(current_level - 1, 0, -1):
                    if hierarchy[i]:  # 找父标题的时候 排除某一个位置的空值
                        parent_title = hierarchy[i]  # 读取操作
                        break

                if not parent_title:
                    parent_title = current_title if current_title else file_title

                sections.append({
                    "body": body,
                    "title": current_title if current_title else file_title,  # 内容标题
                    "parent_title": parent_title,  # 内容父标题
                    "file_title": file_title,
                })




        heading_re = re.compile(r"^\s*(#{1,6})\s+(.+)")
        md_lines = md_content.split("\n")
        for md_line in md_lines:
            if md_line.strip().startswith("```") or md_line.strip().startswith("~~~"):
                in_fence = not in_fence

            match = heading_re.match(md_line)

            if match and not in_fence:
                # 吧标题行封装到section对象

                # 将 body_liens中收集到的行封装到section对象
                _flush()

                current_title = md_line  # 当前标题
                level = len(match.group(1))  # 当前标题的层级（# {1,6}）
                current_level = level
                hierarchy[level] = current_title  # 写入操作

                for i in range(level + 1, 7):
                    hierarchy[i] = ""  # 下面的清空
                # 没有匹配到标题[普通行] 或者是代码块（加入）
                body_liens = []
            else:
                body_liens.append(md_line)

        _flush()
        return sections



    def _split_long_section(self, section:Dict[str,Any], max_content_length:int)->List[Dict[str,Any]]:
        body=section.get("body")
        title=section.get("title")
        parent_title=section.get("parent_title")
        file_title=section.get("file_title")
        if len(title)>80:
            title=title[:80]

        if "<table>" in body:
            self.logger.info("检查到section中有表格")
            body = MarkdownTableLinearizer.process(body)
            section["body"]=body

        #标题前缀
        title_prefix=f"{title}\n\n"
        #总长度
        total_length=len(title_prefix)+len(body)
        if total_length<=max_content_length:
            return [section]


        body_length = max_content_length - len(title_prefix)
        if  body_length <=0:
            return [section]

        text_spliter=RecursiveCharacterTextSplitter(chunk_size=body_length,
                                                    chunk_overlap=0,
                                                    separators=[
                                                        "\n\n",
                                                        "\n",
                                                        "。",
                                                        "！",
                                                        "？",
                                                        "；",
                                                        "，",
                                                        " ",
                                                        ""
                                                    ],
                                                    keep_separator=True
                                                    )

        sections=text_spliter.split_text(body)
        if len(sections)==1:
            return [section]

        sub_sections=[]
        for i,section in enumerate(sections):
            sub_sections.append({
                "body":section,
                "title":f"{title}_{i+1}",
                "parent_title":parent_title,
                "file_title":file_title
            })


        return sub_sections

    def _merger_short_section(self, current_sections, min_content_length)->List[Dict[str,Any]]:
        current_section = current_sections[0]
        final_sections=[]
        for next_sections in current_sections[1:]:
            same_parrent=(current_section["parent_title"]==next_sections["parent_title"])
            if same_parrent and len(current_section["body"]) <min_content_length:
                current_section["body"]=(
                    current_section.get("body").rstrip()+"\n\n"+next_sections.get("body").lstrip()

                )

                current_section['title'] = current_section['parent_title']
            else:
                final_sections.append(current_section)
                current_section = next_sections

        final_sections.append(current_section)
        return final_sections

    def _assemble_chunks(self,final_sections:List[Dict[str,Any]])->List[Dict[str,Any]]:
        final_chunks=[]
        for section in final_sections:
            body=section.get("body")
            title=section.get("title")
            parent_title=section.get("parent_title")
            file_title=section.get("file_title")
            content=f"{title}\n\n{body}"
            final_chunks.append({
                "content":content,
                "title":title,
                "parent_title":parent_title,
                "file_title":file_title
            })
        self.logger.info(f"最终切割之后的节点数量是{len(final_chunks)}")
        return final_chunks

    def _back_up(self, final_chunks, state: ImportGraphState):
        """将切分结果备份到 JSON 文件"""
        local_dir = state.get("file_dir", "")
        if not local_dir:
            return
        try:
            os.makedirs(local_dir, exist_ok=True)  # 如果目录存在 不报错
            output_path = os.path.join(local_dir, "chunks.json")

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(final_chunks, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(f"备份失败: {e}")

if __name__ == '__main__':
    document_split_node = DocumentSplitNode()

    md_path = r"D:\pycharm-workspace\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\万用表的使用\hybrid_auto\万用表的使用_new.md"

    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    init_state = {
        "md_content": md_content,
        "file_title": "万用表的使用",
        "file_dir": r"D:\pycharm-workspace\shopkeeper_brain\knowledge\processor\import_processor\temp_dir"
    }
    document_split_node.process(init_state)
    print("输出完成")