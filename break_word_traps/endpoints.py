import os
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import JSONResponse
from typing import List

API_KEY_NAME = "x-api-key"


class _FastAPIServer:
    _app = FastAPI()

    def __init__(self, api_key: str):
        self.api_key = api_key

        self._app.middleware("http")(self.validate_api_key)
        for method, endpoint, func in (
            ("get", "/health", self.health),
            ("post", "/process-video", self.process_video),
        ):
            register_func = getattr(self._app, method, None)
            if register_func is None:
                raise Exception(f"Method {method} not known")
            register_func(endpoint)(func)

    async def validate_api_key(self, request: Request, call_next):
        api_key = request.headers.get(API_KEY_NAME, None)
        if api_key == self.api_key:
            return await call_next(request)
        return JSONResponse(content={"error": "bad API key"}, status_code=403)

    def health(self):
        return {"health": "OK"}

    def process_video(self, request: Request, file: UploadFile):
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
    api_key = os.environ.get("API_KEY", None)
    if not api_key:
        raise Exception("API_KEY missing")
    return _FastAPIServer(api_key).app
