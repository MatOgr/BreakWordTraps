from typing import List

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Bounding box schema."""

    top: int
    left: int
    width: int
    height: int


class FERResult(BaseModel):
    """Facial expression recognition results schema."""

    emotion: str
    timestamp: int


class Error(BaseModel):
    timestamp: int
    name: str
    details: str | None


class Readability(BaseModel):
    flesch_score: float = Field(alias="fleschScore")
    flesch_grade: str = Field(alias="fleschGrade")
    gunning_fog_score: float = Field(alias="gunningFogScore")
    gunning_fog_grade: str = Field(alias="gunningFogGrade")

    class Config:
        allow_population_by_field_name = True


class Transcript(BaseModel):
    timestamp: float
    text: str


class VideoResult(BaseModel):
    fer_results: List[FERResult] = Field(alias="ferResults")
    transcript: List[Transcript]
    target_group: str | None = Field(alias="targetGroup")
    sentiment: str | None
    questions: List[str] | None
    readability: Readability
    errors: List[Error]

    class Config:
        allow_population_by_field_name = True


class Overall(BaseModel):
    total_files: int = Field(alias="totalFiles")
    total_errors: int = Field(alias="totalErrors")
    words_per_minute: float = Field(alias="wordsPerMinute")

    class Config:
        allow_population_by_field_name = True


class Statistic(BaseModel):
    name: str
    quantity: int


class Summary(BaseModel):
    overall: Overall
    statistics: List[Statistic]


class ResultsDTO(BaseModel):
    videos_results: List[VideoResult] = Field(alias="videosResults")
    summary: List[Summary]

    class Config:
        allow_population_by_field_name = True
