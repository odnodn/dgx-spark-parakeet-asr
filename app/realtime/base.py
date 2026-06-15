"""
Abstract base class for realtime streaming ASR backends.

All streaming backends must implement this interface, allowing
the API to swap between Riva/NIM, local Parakeet streaming, or other
WebSocket-based ASR services.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional


@dataclass
class StreamingTranscript:
    """A single streaming transcript result."""

    text: str
    is_final: bool = False
    confidence: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    language: str = ""

    def to_dict(self) -> dict:
        d = {
            "text": self.text,
            "is_final": self.is_final,
            "confidence": round(self.confidence, 3),
        }
        if self.start_time or self.end_time:
            d["start_time"] = round(self.start_time, 3)
            d["end_time"] = round(self.end_time, 3)
        if self.language:
            d["language"] = self.language
        return d


@dataclass
class StreamingConfig:
    """Configuration for a streaming session."""

    language: str = "en-US"
    sample_rate: int = 16000
    encoding: str = "LINEAR16"
    channels: int = 1
    interim_results: bool = True
    punctuation: bool = True
    word_timestamps: bool = False
    max_alternatives: int = 1
    chunk_duration_ms: int = 100

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "sample_rate": self.sample_rate,
            "encoding": self.encoding,
            "channels": self.channels,
            "interim_results": self.interim_results,
            "punctuation": self.punctuation,
            "word_timestamps": self.word_timestamps,
            "max_alternatives": self.max_alternatives,
            "chunk_duration_ms": self.chunk_duration_ms,
        }


class StreamingASRBackend(ABC):
    """Abstract base class for realtime streaming ASR backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this backend."""
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Whether the backend is configured and reachable."""
        ...

    @abstractmethod
    async def stream(
        self,
        audio_chunks: AsyncIterator[bytes],
        config: Optional[StreamingConfig] = None,
    ) -> AsyncIterator[StreamingTranscript]:
        """
        Stream audio chunks and yield transcription results.

        Args:
            audio_chunks: Async iterator of raw audio bytes (PCM 16-bit, 16kHz mono)
            config: Streaming configuration. Uses defaults if None.

        Yields:
            StreamingTranscript objects (partial/interim and final results)
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close any open connections or resources."""
        ...
