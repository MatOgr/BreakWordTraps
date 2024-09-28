from typing import Literal
from fastapi import FastAPI, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .tools.fer.analyzer import Emotion, retrieve_emotion


class _FastAPIServer:
    _app = FastAPI()

    def __init__(self):
        self.api_prefix = "/api"

        self._app.add_middleware(
            CORSMiddleware,
            allow_origins="http://localhost:5173",
            allow_methods="*",
            allow_headers="*",
        )
        for method, endpoint, func in (
            ("get", "/health", self.health),
            ("post", "/process-video", self.process_video),
            ("post", "/retrieve-emotion-test", self.process_image),
        ):
            register_func = getattr(self._app, method, None)
            if register_func is None:
                raise Exception(f"Method {method} not known")
            register_func(self.api_prefix + endpoint)(func)

    def health(self):
        return {"health": "OK"}

    def process_video(self, request: Request, files: List[UploadFile]):
        # TODO add processing
        return {"result": "OK"}

    async def process_image(
        self, request: Request, file: UploadFile
    ) -> Emotion | Literal["No emotion detected"]:
        content = await file.read()
        return retrieve_emotion(content)

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
