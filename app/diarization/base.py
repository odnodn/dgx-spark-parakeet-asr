"""
Abstract base class for speaker diarization backends.

All diarization backends must implement this interface, allowing
the API to swap between Sortformer, pyannote, or other models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DiarizationSegment:
    """A single speaker segment with timing information."""

    speaker: str
    start: float
    end: float
    text: Optional[str] = None

    def to_dict(self) -> dict:
        d = {
            "speaker": self.speaker,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
        }
        if self.text is not None:
            d["text"] = self.text
        return d


@dataclass
class DiarizationResult:
    """Result of a diarization run."""

    segments: list[DiarizationSegment] = field(default_factory=list)
    num_speakers: int = 0
    duration: float = 0.0
    backend: str = ""

    def to_dict(self) -> dict:
        return {
            "segments": [s.to_dict() for s in self.segments],
            "num_speakers": self.num_speakers,
            "duration": round(self.duration, 3),
            "backend": self.backend,
        }


class DiarizationBackend(ABC):
    """Abstract base class for diarization backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this backend."""
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether the model is loaded and ready for inference."""
        ...

    @abstractmethod
    def load_model(self) -> None:
        """Load the diarization model. Called once at startup."""
        ...

    @abstractmethod
    def diarize(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> DiarizationResult:
        """
        Perform speaker diarization on audio.

        Args:
            audio_bytes: Raw audio file bytes (any format ffmpeg can read)
            filename:    Original filename (for format detection)
            num_speakers: Exact number of speakers (if known)
            min_speakers: Minimum expected speakers
            max_speakers: Maximum expected speakers

        Returns:
            DiarizationResult with speaker segments
        """
        ...
