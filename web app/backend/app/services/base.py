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

    def is_model_ready(self) -> bool:
        """Return whether inference can run without additional model loading."""
        return self.is_ready()

    def model_status(self) -> str:
        return "ready" if self.is_model_ready() else "not_ready"

    @abstractmethod
    def detect(self, text: str) -> DetectResponse:
        pass
