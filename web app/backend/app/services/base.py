from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas import DetectResponse


class InferenceService(ABC):
    @abstractmethod
    def startup(self) -> None:
        pass

    @abstractmethod
    def shutdown(self) -> None:
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        pass

    @abstractmethod
    def detect(self, text: str) -> DetectResponse:
        pass
