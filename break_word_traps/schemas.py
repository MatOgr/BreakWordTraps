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
    details: str | None = Field(default=None)


class Readability(BaseModel):
    flesch_score: float = Field(alias="fleschScore")
    flesch_grade: str = Field(alias="fleschGrade")
    gunning_fog_score: float = Field(alias="gunningFogScore")
    gunning_fog_grade: str = Field(alias="gunningFogGrade")

    class Config:
        populate_by_name = True


class Transcript(BaseModel):
    timestamp: float
    text: str


class VideoResult(BaseModel):
    fer_results: List[FERResult] = Field(alias="ferResults")
    transcript: List[Transcript]
    target_group: str | None = Field(default=None, alias="targetGroup")
    sentiment: str | None = Field(default=None)
    questions: List[str] | None = Field(default=None)
    readability: Readability
    errors: List[Error] = Field(default_factory=list)

    class Config:
        populate_by_name = True


class Overall(BaseModel):
    total_files: int = Field(alias="totalFiles")
    total_errors: int = Field(alias="totalErrors")
    words_per_minute: float = Field(alias="wordsPerMinute")

    class Config:
        populate_by_name = True


class Statistic(BaseModel):
    name: str
    quantity: int


class Summary(BaseModel):
    overall: Overall
    statistics: List[Statistic]


class ResultsDTO(BaseModel):
    videos_results: List[VideoResult] = Field(alias="videosResults")
    summary: Summary

    class Config:
        populate_by_name = True
