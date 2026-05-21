from functools import cache

from cachetools.func import lru_cache

from knowledge.services.upload_service import *


@cache
@lru_cache
def get_upload_file_service():
    return UpLoadService()