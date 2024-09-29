import asyncio
import logging
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, List, Literal, Optional
from uuid import uuid4

import aiofile

import cv2
import requests
from fastapi import FastAPI, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from break_word_traps.extract_autio import extract
from break_word_traps.schemas import ResultsDTO
from break_word_traps.utils.service_types import ServiceType

API_KEY_NAME = "x-api-key"
LOG = logging.Logger(Path(__file__).name)


async def save_uploaded_file(file: UploadFile, path: Path, chunk_size=2048):
    async with aiofile.async_open(str(path), "wb") as fd:
        while chunk := await file.read(chunk_size):
            await fd.write(chunk)
    return path


async def read_file(file: UploadFile, chunk_size=2048):
    b = bytearray()
    while chunk := await file.read(chunk_size):
        b.extend(chunk)
    return b


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
        # self.async_client = aiosonic.HTTPClient()
        self.prepare_func = None

        self._app.add_middleware(
            CORSMiddleware,
            allow_origins="http://localhost:5173",
            allow_methods="*",
            allow_headers="*",
        )

        endpoints = [
            ("get", "/health", self.health),
            ("post", "/process-video", self.process_video),
        ]
        # Register endpoints based on service type
        if self.service_type == ServiceType.MAIN:
            if not self.asr_server_address or not self.fer_server_address:
                LOG.warn("Addresses for ASR and FER servers should be defined")
            endpoints.append(("post", "/process-video", self.process_video))
        elif self.service_type == ServiceType.ASR:
            from break_word_traps.tools.whisper.model import (
                prepare_model,
                transcribe_file,
            )

            if not self.llm_server_address:
                LOG.warn("Address for LLM servers is required")
            self.prepare_func = prepare_model
            endpoints.append(
                (
                    "post",
                    "/process-audio",
                    self.create_process_audio_endpoint(transcribe_file),
                )
            )
        elif self.service_type == ServiceType.FER:
            from break_word_traps.tools.fer.analyzer import (
                prepare_model,
                retrieve_emotion,
            )

            self.prepare_func = prepare_model
            endpoints.append(
                (
                    "post",
                    "/process-images",
                    self.create_process_images_endpoint(retrieve_emotion),
                )
            )
        elif self.service_type == ServiceType.LLM:
            from break_word_traps.tools.bielik.model import prepare_model, analyze_text

            self.prepare_func = prepare_model
            endpoints.append(
                (
                    "post",
                    "/process-text",
                    self.create_process_text_endpoint(analyze_text),
                )
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
        if self.service_type != ServiceType.MAIN and self.prepare_func:
            LOG.info("Preparing runtime")
            self.prepare_func()
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
                    / f"{uuid4()}{'_' + file.filename if hasattr(file, 'filename') else ''}",
                )
                for file in files
            ]
        )

    async def process_video(self, request: Request, files: List[UploadFile]):
        saved_files = await self.receive_files(files)
        try:
            audio_files = []
            for saved_file in saved_files:
                audio_files.append(extract(saved_file, saved_file.with_suffix(".wav")))
            buffers = [audio_file.open("rb") for audio_file in audio_files]
            asr_request = asyncio.to_thread(
                requests.post,
                f"http://{self.asr_server_address}/api/process-audio",
                files=[
                    ("files", (file.name, buf))
                    for file, buf in zip(audio_files, buffers)
                ],
                headers={API_KEY_NAME: self.api_key},
            )

            fer_requests = []
            for saved_file in saved_files:
                vidcap = cv2.VideoCapture(str(saved_file))
                success, image = vidcap.read()
                images = [self.resources_path / f"{uuid4()}_frame_0.png"]
                cv2.imwrite(images[-1], image)
                count = 0
                while success:
                    success, image = vidcap.read()
                    if count % 25 == 0:
                        images.append(
                            self.resources_path / f"{uuid4()}_frame_{count}.png"
                        )
                        cv2.imwrite(images[-1], image)
                    count += 1
                img_buff = [image.open("rb") for image in images]

                fer_requests.append(
                    asyncio.to_thread(
                        requests.post,
                        f"http://{self.fer_server_address}/api/process-images",
                        files=[
                            (
                                "images",
                                (
                                    image.name,
                                    buf,
                                ),
                            )
                            for image, buf in zip(images, img_buff)
                        ],
                        headers={API_KEY_NAME: self.api_key},
                    )
                )

            results = await asyncio.gather(asr_request, *fer_requests)
            [b.close() for b in buffers]
            [image.close() for image in img_buff]
            asr_results = results[0].json()
            fer_results = [r.json() for r in results[1:]]
            video_results = []
            for asr, fer in zip(asr_results, fer_results):
                video_results.append(
                    asr
                    | {
                        "ferResults": [
                            {"emotion": f, "timestamp": i} for i, f in enumerate(fer)
                        ],
                        "targetGroup": "",
                        "sentiment": "",
                        "questions": [],
                        "errors": [],
                    }
                )

            # TODO add frames extractions, send requests and collect data
            return {
                "videoResults": video_results,
                "summary": {
                    "overall": {
                        "totalFiles": 0,
                        "totalErrors": 0,
                        "wordsPerMinute": 0,
                    },
                    "statistics": [],
                },
            }
        finally:
            for saved_file in saved_files:
                saved_file.unlink()

    def create_process_audio_endpoint(self, func: Callable):
        async def process_audio(files: List[UploadFile]):
            saved_audio_files = await self.receive_files(files)
            results = None
            try:
                await self._lock.acquire()
                results = func(saved_audio_files)
            finally:
                self._lock.release()
                for saved_file in saved_audio_files:
                    saved_file.unlink()

            if self.llm_server_address:
                response = await asyncio.to_thread(
                    requests.post,
                    f"http://{self.llm_server_address}/api/process-text",
                    data={"text": [r["text"] for r in results]},
                    headers={API_KEY_NAME: self.api_key},
                )
                response = response.json()
                results |= response
            return results

        return process_audio

    def create_process_images_endpoint(self, func: Callable):
        async def process_images(
            images: List[UploadFile],
        ):
            names = [image.filename for image in images]
            saved_images = await asyncio.gather(
                *[
                    save_uploaded_file(
                        image, self.resources_path / f"{uuid4()}_frame.png"
                    )
                    for image in images
                ]
            )
            try:
                await self._lock.acquire()
                results = []
                for name, image in zip(names, saved_images):
                    img = cv2.imread(str(image))
                    emotion = func(img)
                    if emotion:
                        results.append(emotion)
                    else:
                        results.append("No emotion detected")
                return results
            finally:
                self._lock.release()

        return process_images

    def create_process_text_endpoint(self, func: Callable):
        async def process_text(texts: List[str]):
            try:
                await self._lock.acquire()
                return func(texts)
            finally:
                self._lock.release()

        return process_text

    def return_mock_summary(self, files: List[UploadFile]) -> ResultsDTO:
        response = ResultsDTO.model_validate(
            {
                "videosResults": [
                    {
                        "ferResults": [
                            {
                                "emotion": "happy",
                                "timestamp": 0,
                            }
                        ],
                        "transcript": [
                            {
                                "timestamp": 0,
                                "text": "Hello, how are you doing today?",
                            }
                        ],
                        "readability": {
                            "fleschScore": 100.0,
                            "fleschGrade": "A",
                            "gunningFogScore": 100.0,
                            "gunningFogGrade": "A",
                        },
                        "errors": [],
                    }
                    for _ in files
                ],
                "summary": {
                    "overall": {
                        "totalFiles": len(files),
                        "totalErrors": 0,
                        "wordsPerMinute": 100.0,
                    },
                    "statistics": [
                        {
                            "name": "happy",
                            "quantity": len(files),
                        }
                    ],
                },
            },
        )
        print(response)
        return response

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
