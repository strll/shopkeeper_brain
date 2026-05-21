import logging
import os.path
import shutil
import time
import uuid
from datetime import datetime

from Crypto.SelfTest.Cipher.test_CBC import file_name
from fastapi import UploadFile

from knowledge.core.path import get_local_base_dir
from knowledge.processor.import_processor.exceptions import FileProcessingError
from knowledge.processor.import_processor.main_graph import import_app
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.task_util import *

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

class UpLoadService:
    def get_base_dir(self)->str:
       return os.path.join(get_local_base_dir(),datetime.now().strftime("%Y%m%d"))


    """
    文件上传相关逻辑
    """


    def process_upload_file(self,file:UploadFile):

        task_id=str(uuid.uuid4().hex[:8])
        file_dir = self.get_base_dir()
        add_running_task(task_id=task_id,node_name="upload_file")

        import_path=os.path.join(file_dir,file.filename)
        start_time = time.time()
        import_file_path=self.save_upload_file_to_local(file,file_dir)

        self.save_upload_file_to_minio(import_file_path, file.filename)
        add_done_task(task_id=task_id, node_name="upload_file")
        end_time = time.time()


        add_node_duration(task_id, "upload_file", end_time - start_time)
        return task_id,import_file_path,file_dir

    def run_import_graph(self,task_id:str,import_file_path:str,file_dir:str):


        update_task_status(task_id,TASK_STATUS_PROCESSING)

        grap_state = {
            "task_id":task_id,
            "import_file_path": import_file_path,
            "file_dir": file_dir
        }

        try:
            for event in import_app.stream(grap_state):
                final_state = {}
                for event in import_app.stream(grap_state):
                    final_state = {}
                    for key, value in event.items():
                        print(f"当前正在执行的节点：---------->{key}----------")

                        final_state = value
                update_task_status(task_id, TASK_STATUS_FAILED)
        except Exception as e:
            logger.error(f"[{task_id}] .0任务执行失败:{e}")
            update_task_status(task_id, TASK_STATUS_FAILED)
        return final_state




    def save_upload_file_to_local(self, file:UploadFile, file_dir:str):
        os.makedirs(file_dir,exist_ok=True)
        file_path=os.path.join(file_dir,file.filename)
        try:
            with open(file_path, "wb") as buffer:

                shutil.copyfileobj(file.file, buffer)
        except IOError as e:
            logger.info(f"{file.filename}写入临时目录失败 原因是:{str(e)}")
            raise FileProcessingError(f"{file.filename}写入临时目录失败 原因是:{str(e)}")
        return file_path

    def save_upload_file_to_minio(self, import_file_path: str, filename: str):

        try:
            minio_client =StorageClients.get_minio_client()
        except Exception as e:
            logger.error(f"获取minio客户端失败:{e}")
            raise FileProcessingError(f"获取minio客户端失败:{e}")

        bucket_name=os.getenv("MINIO_BUCKET_NAME")
        object_name=f"origin_files/{datetime.now().strftime('%Y%m%d')}/{file_name}"
        try:
            minio_client.fput_object(bucket_name, object_name, import_file_path)
        except Exception as e:
            logger.error(f"上传文件到minio失败:{e}")
            raise FileProcessingError(f"上传文件到minio失败:{e}")