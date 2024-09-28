from typing import Literal, NewType

Emotion = NewType(
    "Emotion",
    Literal["Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger"],
)
