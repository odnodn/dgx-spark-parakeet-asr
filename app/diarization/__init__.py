"""
Speaker Diarization Module

Modular speaker diarization pipeline supporting multiple backends.
Currently supported:
  - NVIDIA Sortformer (NeMo-based neural diarization)

Future backends can be added by subclassing DiarizationBackend.
"""

from app.diarization.base import DiarizationBackend, DiarizationSegment
from app.diarization.factory import get_diarization_backend, list_backends

__all__ = [
    "DiarizationBackend",
    "DiarizationSegment",
    "get_diarization_backend",
    "list_backends",
]
