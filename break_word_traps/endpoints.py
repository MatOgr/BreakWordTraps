import os
import aiofile
import asyncio
import shutil
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Callable, Optional
from pathlib import Path
from uuid import uuid4

from break_word_traps.extract_autio import extract
from break_word_traps.utils.service_types import ServiceType

API_KEY_NAME = "x-api-key"
LOG = logging.Logger(Path(__file__).name)


async def save_uploaded_file(file: UploadFile, path: Path, chunk_size=2048):
    async with aiofile.async_open(str(path), "wb") as fd:
        while chunk := await file.read(chunk_size):
            await fd.write(chunk)
    return path


class _FastAPIServer:
    def __init__(
        self,
        service_type: ServiceType,
        api_key: str,
        asr_server_address: Optional[str] = None,
        fer_server_address: Optional[str] = None,
        llm_server_address: Optional[str] = None,
    ):
        self._app = FastAPI(lifespan=self.lifecycle)
        self.api_key = api_key
        self.service_type = service_type
        self.api_prefix = "/api"
        self.resources_path = Path(f"./resources_{service_type.value}")
        self.asr_server_address = asr_server_address
        self.fer_server_address = fer_server_address
        self.llm_server_address = llm_server_address
        self._lock = asyncio.Lock()

        self._app.add_middleware(
            CORSMiddleware,
            allow_origins="http://localhost:5173",
            allow_methods="*",
            allow_headers="*",
        )

        endpoints = [
            ("get", "/health", self.health),
        ]
        # Register endpoints based on service type
        if self.service_type == ServiceType.MAIN:
            if not self.asr_server_address or not self.fer_server_address:
                LOG.warn("Addresses for ASR and FER servers should be defined")
            endpoints.append(("post", "/process-video", self.process_video))
        elif self.service_type == ServiceType.ASR:
            if not self.llm_server_address:
                LOG.warn("Address for LLM servers is required")
            # TODO import and add whisper
            endpoints.append(
                ("post", "/process-audio", self.create_process_audio_endpoint())
            )
        elif self.service_type == ServiceType.FER:
            # TODO import and add FEr
            endpoints.append(
                ("post", "/process-images", self.create_process_images_endpoint())
            )
        elif self.service_type == ServiceType.LLM:
            # TODO import and add LLM
            endpoints.append(
                ("post", "/process-text", self.create_process_text_endpoint())
            )
        # Add API_KEY validation for subservices
        if self.service_type != ServiceType.MAIN:
            self._app.middleware("http")(self.validate_api_key)

        for method, endpoint, func in endpoints:
            register_func = getattr(self._app, method, None)
            if register_func is None:
                raise Exception(f"Method {method} not known")
            register_func(self.api_prefix + endpoint)(func)

    @asynccontextmanager
    async def lifecycle(self, app: FastAPI):
        self.resources_path.mkdir(exist_ok=True, parents=True)
        yield
        shutil.rmtree(self.resources_path)

    async def validate_api_key(self, request: Request, call_next):
        api_key = request.headers.get(API_KEY_NAME, None)
        if api_key == self.api_key:
            return await call_next(request)
        return JSONResponse(content={"error": "Wrong API key"}, status_code=403)

    def health(self):
        return {"health": "OK"}

    async def receive_files(self, files: List[UploadFile]):
        return await asyncio.gather(
            *[
                save_uploaded_file(
                    file,
                    self.resources_path
                    / f"{uuid4()}{'_' + file.filename if hasattr(file, "filename") else ''}",
                )
                for file in files
            ]
        )

    async def process_video(self, request: Request, files: List[UploadFile]):
        saved_files = await self.receive_files(files)
        try:
            for saved_file in saved_files:
                extract(saved_file, saved_file.with_suffix(".wav"))
            # TODO add frames extractions, send requests and collect data
            return {"result": "OK"}
        finally:
            for saved_file in saved_files:
                saved_file.unlink()

    async def create_process_audio_endpoint(self, func: Callable):
        async def process_audio(files: List[UploadFile]):
            saved_audio_files = await self.receive_files(files)
            try:
                await self._lock.acquire()
                func(saved_audio_files)
            finally:
                self._lock.release()
                for saved_file in saved_audio_files:
                    saved_file.unlink()

        return process_audio

    async def create_process_images_endpoint(self, func: Callable):
        async def process_images(images: List[UploadFile]):
            images = await asyncio.gather(*[image.read() for image in images])
            try:
                await self._lock.acquire()
                return func(images)
            finally:
                self._lock.release()

        return process_images

    async def create_process_text_endpoint(self, func: Callable):
        async def process_text(texts: List[str]):
            try:
                await self._lock.acquire()
                return func(texts)
            finally:
                self._lock.release()

        return process_text

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
    return _FastAPIServer(service_type=ServiceType.MAIN, api_key=api_key).app
