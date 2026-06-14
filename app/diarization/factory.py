"""
Diarization backend factory.

Provides a registry of available backends and instantiation logic.
New backends (e.g. pyannote) can be added here.
"""

import os
import logging
from typing import Optional

from app.diarization.base import DiarizationBackend

logger = logging.getLogger(__name__)

# ── Registry of available backends ───────────────────────────────────────────
_BACKENDS: dict[str, type[DiarizationBackend]] = {}
_instances: dict[str, DiarizationBackend] = {}

DEFAULT_BACKEND = os.getenv("DIARIZATION_BACKEND", "sortformer")


def _register_backends():
    """Register all available diarization backends."""
    global _BACKENDS

    # Always register Sortformer
    from app.diarization.sortformer import SortformerBackend

    _BACKENDS["sortformer"] = SortformerBackend

    # Future: register pyannote, whisperX, etc.
    # try:
    #     from app.diarization.pyannote import PyannoteBackend
    #     _BACKENDS["pyannote"] = PyannoteBackend
    # except ImportError:
    #     pass


def list_backends() -> list[str]:
    """Return names of all registered diarization backends."""
    if not _BACKENDS:
        _register_backends()
    return list(_BACKENDS.keys())


def get_diarization_backend(
    name: Optional[str] = None,
) -> DiarizationBackend:
    """
    Get (or create) a diarization backend instance.

    Args:
        name: Backend name. Defaults to DIARIZATION_BACKEND env var or 'sortformer'.

    Returns:
        Singleton instance of the requested backend.

    Raises:
        ValueError: If backend name is not registered.
    """
    if not _BACKENDS:
        _register_backends()

    backend_name = name or DEFAULT_BACKEND

    if backend_name not in _BACKENDS:
        available = ", ".join(_BACKENDS.keys())
        raise ValueError(
            f"Unknown diarization backend: '{backend_name}'. "
            f"Available: {available}"
        )

    # Return cached instance (singleton per backend)
    if backend_name not in _instances:
        logger.info(f"Creating diarization backend: {backend_name}")
        _instances[backend_name] = _BACKENDS[backend_name]()

    return _instances[backend_name]
