import os
from typing import Literal

import numpy as np
from fastapi import FastAPI, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .tools.fer.analyzer import FER, Emotion


class _FastAPIServer:
    _app = FastAPI()

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_prefix = "/api"

        # Disable api-key verification
        # self._app.middleware("http")(self.validate_api_key)
        self._app.add_middleware(
            CORSMiddleware,
            allow_origins="http://localhost:5173",
            allow_methods="*",
            allow_headers="*",
        )
        for method, endpoint, func in (
            ("get", "/health", self.health),
            ("post", "/process-video", self.process_video),
            ("post", "/retrieve-emotion", self.process_image),
        ):
            register_func = getattr(self._app, method, None)
            if register_func is None:
                raise Exception(f"Method {method} not known")
            register_func(self.api_prefix + endpoint)(func)

    async def validate_api_key(self, request: Request, call_next):
        return JSONResponse(content={"error": "bad API key"}, status_code=403)

    def health(self):
        return {"health": "OK"}

    def process_video(self, request: Request, file: UploadFile):
        # TODO add processing
        return {"result": "OK"}

    def process_image(
        self, request: Request, file: UploadFile
    ) -> Emotion | Literal["No emotion detected"]:
        return self.retrieve_emotion(file.file)

    def retrieve_emotion(self, image: np.ndarray) -> Emotion | None:
        fer = FER()
        fer.eval()
        emotion = fer.analyse_image(image)
        return emotion

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
    api_key = os.environ.get("API_KEY", None)
    if not api_key:
        raise Exception("API_KEY missing")
    return _FastAPIServer(api_key).app
