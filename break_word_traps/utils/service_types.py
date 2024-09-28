from enum import Enum


class ServiceType(str, Enum):
    MAIN = "main"
    ASR = "asr"
    FER = "fer"
    LLM = "llm"
