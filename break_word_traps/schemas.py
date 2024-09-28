from pydantic import BaseModel


class BoundingBox(BaseModel):
    """Bounding box schema."""

    top: int
    left: int
    width: int
    height: int


class FERResults(BaseModel):
    """Facial expression recognition results schema."""

    emotion: str
    confidence: float | None
    bounding_box: BoundingBox | None
    landmarks: list[list[int]]


class AnalysisResults(BaseModel):
    """General analysis results schema."""

    fer_results: FERResults
