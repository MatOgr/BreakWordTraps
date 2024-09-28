from fastapi import FastAPI, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from break_word_traps.utils.service_types import ServiceType


class _FastAPIServer:
    _app = FastAPI()

    def __init__(self, service_type: ServiceType):
        self.service_type = service_type
        self.api_prefix = "/api"

        self._app.add_middleware(
            CORSMiddleware,
            allow_origins="http://localhost:5173",
            allow_methods="*",
            allow_headers="*",
        )
        endpoints = [
            ("get", "/health", self.health),
        ]
        if self.service_type == ServiceType.MAIN:
            endpoints.append(("post", "/process-video", self.process_video))
        elif self.service_type == ServiceType.ASR:
            # TODO import and add whisper
            pass
        elif self.service_type == ServiceType.FER:
            # TODO import and add FEr
            pass
        elif self.service_type == ServiceType.LLM:
            # TODO import and add LLM
            pass

        for method, endpoint, func in endpoints:
            register_func = getattr(self._app, method, None)
            if register_func is None:
                raise Exception(f"Method {method} not known")
            register_func(self.api_prefix + endpoint)(func)

    def health(self):
        return {"health": "OK"}

    def process_video(self, request: Request, files: List[UploadFile]):
        # TODO add processing
        return {"result": "OK"}

    @property
    def app(self):
        return self._app


_server_app = None


def server_app(**kwargs):
    global _server_app
    if _server_app is None:
        _server_app = _FastAPIServer(**kwargs)
    return _server_app


def get_app():
    return _FastAPIServer().app
