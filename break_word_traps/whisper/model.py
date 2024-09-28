from pathlib import Path
from typing import List, Dict

import stable_whisper
from stable_whisper.result import WordTiming


def word_timing_to_json(timings: List[WordTiming]) -> Dict:
    return [
        {"word": word.word, "start": word.start, "end": word.end} for word in timings
    ]


def word_timing_to_text(timings: List[WordTiming]) -> str:
    return ("".join([word.word for word in timings])).strip()


class WhisperModel:
    def __init__(self, model_type: str, device: str = "cuda"):
        self.model_type = model_type
        self.model = None
        self.device = device

    def prepare_model(self):
        self.model = stable_whisper.load_hf_whisper(
            self.model_type,
            device=self.device,
        )

    def transcribe(self, audio: Path) -> List[WordTiming]:
        results = self.model.transcribe(str(audio), word_timestamps=True)
        return results.all_words()
