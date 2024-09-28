import aiofile
import asyncio
import shutil
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pathlib import Path
from uuid import uuid4

from break_word_traps.extract_autio import extract


async def save_uploaded_file(file: UploadFile, path: Path, chunk_size=2048):
    async with aiofile.async_open(str(path), "wb") as fd:
        while chunk := await file.read(chunk_size):
            await fd.write(chunk)
    return path


class _FastAPIServer:
    def __init__(self):
        self._app = FastAPI(lifespan=self.lifecycle)
        self.api_prefix = "/api"
        self.resources_path = Path("./resources")

        self._app.add_middleware(
            CORSMiddleware,
            allow_origins="http://localhost:5173",
            allow_methods="*",
            allow_headers="*",
        )
        for method, endpoint, func in (
            ("get", "/health", self.health),
            ("post", "/process-video", self.process_video),
        ):
            register_func = getattr(self._app, method, None)
            if register_func is None:
                raise Exception(f"Method {method} not known")
            register_func(self.api_prefix + endpoint)(func)

    @asynccontextmanager
    async def lifecycle(self, app: FastAPI):
        self.resources_path.mkdir(exist_ok=True, parents=True)
        yield
        shutil.rmtree(self.resources_path)

    def health(self):
        return {"health": "OK"}

    async def process_video(self, request: Request, files: List[UploadFile]):
        saved_files = await asyncio.gather(
            *[
                save_uploaded_file(
                    file,
                    self.resources_path
                    / f"{uuid4()}{'_' + file.filename if hasattr(file, "filename") else ''}",
                )
                for file in files
            ]
        )
        for saved_file in saved_files:
            extract(saved_file, saved_file.with_suffix(".wav"))
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
