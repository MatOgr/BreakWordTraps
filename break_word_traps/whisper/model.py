from pathlib import Path
from typing import List, Dict, Optional

import nltk
import stable_whisper
from stable_whisper.result import WordTiming

from break_word_traps.calculate_readability import calculate_scores


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


MODEL: Optional[WhisperModel] = None


def prepare_model():
    # Ensure NLTK corpus is downloaded
    nltk.download("punkt_tab")
    global MODEL
    MODEL = WhisperModel("large")
    MODEL.prepare_model()


def transcribe_file(files: List[Path]):
    results = []
    for file in files:
        word_timings = MODEL.transcribe(file)
        transcribed_text = word_timing_to_text(word_timings)
        readability_scores = calculate_scores(transcribed_text)
        results.append(
            {
                "transcribed_text": transcribed_text,
            }
            | readability_scores
        )
    return results
