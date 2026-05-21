import os.path

import uvicorn
from fastapi import FastAPI, Depends, BackgroundTasks
from starlette.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

from knowledge.core.deps import get_upload_file_service
from knowledge.core.path import get_front_page_dir
from knowledge.schema.upload_schema import UploadResponse, TaskStatusResponse
from knowledge.services.upload_service import *
from knowledge.utils.task_util import *


def create_app():
    app = FastAPI(description="掌柜智库导入的应用", version="1")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 允许任意的源
        allow_credentials=True,  # 允许cookie中携带任意的自定义参数
        allow_methods=["*"],  # 允许任意的请求方式
        allow_headers=["*"],  # 允许请求头中携带任意的我自定义参数
    )

    front_page_dir=get_front_page_dir()
    if front_page_dir and os.path.exists(front_page_dir):
        app.mount("/front", StaticFiles(directory=front_page_dir))

    register_router(app)

    return app


def register_router(app: FastAPI):
    @app.get("/")
    def Hello_word():
        return {"flag": "sueccess"}

    @app.post("/upload", response_model=UploadResponse)
    def upload_endpoint(file:UploadFile,
                        background_tasks:BackgroundTasks,
                        upload_service:UpLoadService=Depends(get_upload_file_service)):
        print("fileName",file.filename)
        #上传 文件
        task_id,import_file_path,file_dir=upload_service.process_upload_file(file)
        #运行整个导入的图谱 后台运行
        background_tasks.add_task(upload_service.run_import_graph,task_id,import_file_path,file_dir)

        return UploadResponse(message=f"{file.filename}上传成功",task_id=task_id)

    @app.post("/status")
    def get_task_status_endpoint():
        return None

    @app.get("/status/{task_id}")
    def get_task_status_endpoint(task_id:str):

        task_info=get_task_info(task_id)
        return TaskStatusResponse(**task_info)

if __name__ == '__main__':
    uvicorn.run(app=create_app(), host='127.0.0.1', port=8000, log_level='info')
