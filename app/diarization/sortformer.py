"""
NVIDIA Sortformer — NeMo-based Speaker Diarization Backend

Sortformer is NVIDIA's sort-based end-to-end neural diarization model
that directly predicts speaker activities from audio. It is optimized
for GPU inference and integrates naturally with the NeMo toolkit.

Model: nvidia/diar_sortformer_4spk-v1
"""

import os
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Optional

import torch
import soundfile as sf
import numpy as np

from app.diarization.base import (
    DiarizationBackend,
    DiarizationResult,
    DiarizationSegment,
)

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────
SORTFORMER_MODEL = os.getenv(
    "SORTFORMER_MODEL", "nvidia/diar_sortformer_4spk-v1"
)
DIARIZATION_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SAMPLE_RATE = 16000


class SortformerBackend(DiarizationBackend):
    """NVIDIA Sortformer speaker diarization backend."""

    def __init__(self):
        self._model = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "sortformer"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load_model(self) -> None:
        """Load the Sortformer model onto GPU."""
        if self._loaded:
            return

        logger.info(f"Loading Sortformer model: {SORTFORMER_MODEL}")
        logger.info(f"  Device: {DIARIZATION_DEVICE}")

        import nemo.collections.asr as nemo_asr

        self._model = nemo_asr.models.NeuralDiarizer.from_pretrained(
            SORTFORMER_MODEL
        )
        self._model.eval()

        if DIARIZATION_DEVICE == "cuda":
            self._model = self._model.cuda()

        self._loaded = True
        logger.info("Sortformer model loaded successfully")

    def _convert_to_wav_16k_mono(
        self, audio_bytes: bytes, original_filename: str
    ) -> str:
        """Convert any audio format to WAV 16kHz mono using ffmpeg."""
        suffix = (
            Path(original_filename).suffix.lower()
            if original_filename
            else ".wav"
        )

        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False
        ) as tmp_in:
            tmp_in.write(audio_bytes)
            tmp_in_path = tmp_in.name

        tmp_out_path = tmp_in_path + ".converted.wav"

        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", tmp_in_path,
                "-ac", "1",
                "-ar", "16000",
                "-sample_fmt", "s16",
                "-f", "wav",
                tmp_out_path,
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            if result.returncode != 0:
                stderr = result.stderr.decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"ffmpeg conversion failed: {stderr[:500]}"
                )
            return tmp_out_path
        finally:
            try:
                os.unlink(tmp_in_path)
            except OSError:
                pass

    def diarize(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> DiarizationResult:
        """
        Run Sortformer diarization on audio.

        Returns DiarizationResult with speaker-labeled time segments.
        """
        if not self._loaded:
            self.load_model()

        wav_path = None
        try:
            # Convert to WAV 16kHz mono
            wav_path = self._convert_to_wav_16k_mono(audio_bytes, filename)

            # Get audio duration
            data, sr = sf.read(wav_path)
            duration = len(data) / sr
            logger.info(
                f"Diarization: {filename}, {duration:.1f}s audio"
            )

            # Run diarization inference
            with torch.no_grad():
                annotations = self._model.diarize(
                    audio=[wav_path],
                    batch_size=1,
                    num_speakers=num_speakers,
                    max_num_speakers=max_speakers,
                )

            # Parse annotations into segments
            segments = self._parse_annotations(annotations)

            # Determine number of unique speakers
            speakers = set(s.speaker for s in segments)

            # Free GPU memory
            if DIARIZATION_DEVICE == "cuda":
                torch.cuda.empty_cache()

            return DiarizationResult(
                segments=segments,
                num_speakers=len(speakers),
                duration=duration,
                backend=self.name,
            )

        finally:
            if wav_path:
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass

    def _parse_annotations(
        self, annotations
    ) -> list[DiarizationSegment]:
        """
        Parse NeMo diarization output into DiarizationSegment list.

        Sortformer returns annotations in RTTM-like format or as a list
        of (speaker, start, end) tuples depending on the NeMo version.
        """
        segments = []

        if annotations is None:
            return segments

        # Handle list of annotation objects (NeMo returns list per audio file)
        if isinstance(annotations, list) and len(annotations) > 0:
            annotation = annotations[0]
        else:
            annotation = annotations

        # NeMo NeuralDiarizer returns an Annotation object (pyannote-style)
        # with itertracks() method
        if hasattr(annotation, "itertracks"):
            for turn, _, speaker in annotation.itertracks(yield_label=True):
                segments.append(
                    DiarizationSegment(
                        speaker=str(speaker),
                        start=turn.start,
                        end=turn.end,
                    )
                )
        # Handle dict-style output
        elif isinstance(annotation, dict):
            for item in annotation.get("segments", []):
                segments.append(
                    DiarizationSegment(
                        speaker=item.get("speaker", "unknown"),
                        start=item.get("start", 0.0),
                        end=item.get("end", 0.0),
                    )
                )
        # Handle list of tuples (speaker, start, end)
        elif isinstance(annotation, list):
            for item in annotation:
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    segments.append(
                        DiarizationSegment(
                            speaker=str(item[0]),
                            start=float(item[1]),
                            end=float(item[2]),
                        )
                    )
                elif isinstance(item, dict):
                    segments.append(
                        DiarizationSegment(
                            speaker=item.get("speaker", "unknown"),
                            start=item.get("start", 0.0),
                            end=item.get("end", 0.0),
                        )
                    )

        # Sort by start time
        segments.sort(key=lambda s: s.start)
        return segments
