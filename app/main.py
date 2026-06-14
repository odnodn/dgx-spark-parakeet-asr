"""
Parakeet TDT 0.6b v3 — OpenAI-Compatible ASR API

Endpoints:
  POST /v1/audio/transcriptions   — OpenAI Whisper-compatible transcription
  POST /v1/audio/translations     — (stub) maps to transcription
  POST /v1/audio/diarizations     — Speaker diarization + transcription (Sortformer)
  GET  /health                    — Health check
  GET  /v1/models                 — List available models
  GET  /                          — Service info

Compatible with any client that speaks the OpenAI audio API:
  - Open WebUI
  - Home Assistant Whisper integration
  - Any OpenAI SDK client
"""

import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.transcriber import transcriber, MODEL_NAME, DEVICE
from app.diarization import get_diarization_backend, list_backends

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("parakeet-api")


# ── Diarization configuration ────────────────────────────────────────────────
ENABLE_DIARIZATION = os.getenv("ENABLE_DIARIZATION", "true").lower() in ("1", "true", "yes")


# ── Lifespan: load model at startup ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("="*60)
    logger.info("Parakeet TDT 0.6b v3 — Starting up")
    logger.info(f"  Model:  {MODEL_NAME}")
    logger.info(f"  Device: {DEVICE}")
    logger.info(f"  Diarization: {'enabled' if ENABLE_DIARIZATION else 'disabled'}")
    if torch.cuda.is_available():
        logger.info(f"  GPU:    {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"  VRAM:   {mem:.1f} GB")
    logger.info("="*60)

    # Load ASR model (downloads on first run, ~1.2 GB)
    transcriber.load_model()
    logger.info("ASR model ready")

    # Load diarization model if enabled
    if ENABLE_DIARIZATION:
        try:
            diarizer = get_diarization_backend()
            diarizer.load_model()
            logger.info(f"Diarization model ready (backend: {diarizer.name})")
        except Exception as e:
            logger.warning(f"Diarization model failed to load: {e}")
            logger.warning("Diarization endpoints will return 503 until model is available")

    logger.info("All models ready — accepting requests")

    yield

    logger.info("Shutting down...")


# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Parakeet ASR API",
    description="OpenAI-compatible Speech-to-Text powered by NVIDIA Parakeet TDT 0.6b v3",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for local/homelab use
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── GET / — Service info ────────────────────────────────────────────────────
@app.get("/")
async def root():
    diarization_info = None
    if ENABLE_DIARIZATION:
        try:
            diarizer = get_diarization_backend()
            diarization_info = {
                "enabled": True,
                "backend": diarizer.name,
                "ready": diarizer.is_loaded,
                "available_backends": list_backends(),
            }
        except Exception:
            diarization_info = {"enabled": True, "ready": False}
    else:
        diarization_info = {"enabled": False}

    return {
        "service": "parakeet-asr",
        "model": MODEL_NAME,
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "endpoints": {
            "transcribe": "/v1/audio/transcriptions",
            "diarize": "/v1/audio/diarizations",
            "models": "/v1/models",
            "health": "/health",
        },
        "diarization": diarization_info,
        "supported_languages": [
            "en", "de", "fr", "es", "it", "pt", "nl", "pl", "ru", "uk",
            "cs", "sk", "sl", "hr", "bg", "ro", "hu", "el", "da", "sv",
            "fi", "et", "lt", "lv", "mt",
        ],
    }


# ── GET /health ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    gpu_mem = None
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_mem = {"allocated_gb": round(allocated, 2), "total_gb": round(total, 1)}

    diarization_status = "disabled"
    if ENABLE_DIARIZATION:
        try:
            diarizer = get_diarization_backend()
            diarization_status = "ready" if diarizer.is_loaded else "loading"
        except Exception:
            diarization_status = "error"

    return {
        "status": "ready" if transcriber._loaded else "loading",
        "model": MODEL_NAME,
        "device": DEVICE,
        "gpu_memory": gpu_mem,
        "diarization": diarization_status,
    }


# ── GET /v1/models — OpenAI-compatible model list ───────────────────────────
@app.get("/v1/models")
async def list_models():
    models = [
        {
            "id": "parakeet-tdt-0.6b-v3",
            "object": "model",
            "created": 1723593600,  # Aug 2025
            "owned_by": "nvidia",
            "permission": [],
        },
        {
            "id": "whisper-1",
            "object": "model",
            "created": 1723593600,
            "owned_by": "nvidia",
            "permission": [],
            "_note": "Alias — routes to parakeet-tdt-0.6b-v3 for OpenAI client compat",
        },
    ]

    if ENABLE_DIARIZATION:
        models.append({
            "id": "diar_sortformer_4spk-v1",
            "object": "model",
            "created": 1723593600,
            "owned_by": "nvidia",
            "permission": [],
            "_note": "NVIDIA Sortformer speaker diarization (up to 4 speakers)",
        })

    return {
        "object": "list",
        "data": models,
    }


# ── POST /v1/audio/transcriptions — OpenAI Whisper-compatible endpoint ──────
@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: Optional[str] = Form(default="parakeet-tdt-0.6b-v3"),
    language: Optional[str] = Form(default=None),
    response_format: Optional[str] = Form(default="json"),
    temperature: Optional[float] = Form(default=0.0),
    timestamp_granularities: Optional[str] = Form(default=None),
    prompt: Optional[str] = Form(default=None),
):
    """
    OpenAI-compatible audio transcription endpoint.

    Accepts the same parameters as OpenAI's /v1/audio/transcriptions:
      - file: Audio file (wav, mp3, flac, ogg, webm, m4a, mp4, etc.)
      - model: Model name (ignored — always uses parakeet-tdt-0.6b-v3)
      - language: ISO language code or 'auto' for detection
      - response_format: 'json', 'text', 'verbose_json', 'srt', 'vtt'
      - temperature: (ignored — Parakeet is non-generative)
    """
    if not transcriber._loaded:
        raise HTTPException(status_code=503, detail="Model is still loading. Try again shortly.")

    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Read audio bytes
    try:
        audio_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read audio file: {e}")

    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")

    # Check file size (limit to 200 MB)
    max_size = int(os.getenv("MAX_UPLOAD_MB", "200")) * 1024 * 1024
    if len(audio_bytes) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {max_size // (1024*1024)} MB"
        )

    # Include timestamps?
    want_timestamps = (
        response_format in ("verbose_json", "srt", "vtt")
        or timestamp_granularities is not None
    )

    # Transcribe
    logger.info(
        f"Transcription request: {file.filename} "
        f"({len(audio_bytes) / 1024:.0f} KB, lang={language}, fmt={response_format})"
    )

    start_time = time.time()

    try:
        result = transcriber.transcribe(
            audio_bytes=audio_bytes,
            filename=file.filename,
            language=language,
            timestamps=want_timestamps,
        )
    except Exception as e:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    elapsed = time.time() - start_time
    rtfx = result.get("duration", 0) / elapsed if elapsed > 0 else 0

    logger.info(
        f"Transcription complete: {result.get('duration', 0):.1f}s audio "
        f"in {elapsed:.1f}s ({rtfx:.0f}x realtime)"
    )
    # ── Write response in the Log ────────────────────────────────────────
    logger.info(f"Recognized Text: {result['text']}")

    # ── Format response ──────────────────────────────────────────────────
    if response_format == "text":
        return JSONResponse(
            content=result["text"],
            media_type="text/plain",
        )

    if response_format == "verbose_json":
        return {
            "task": "transcribe",
            "language": language or "auto",
            "duration": result.get("duration", 0),
            "text": result["text"],
            "segments": result.get("segments", []),
        }

    if response_format == "srt":
        srt = _to_srt(result)
        return JSONResponse(content=srt, media_type="text/plain")

    if response_format == "vtt":
        vtt = _to_vtt(result)
        return JSONResponse(content=vtt, media_type="text/plain")

    # Default: json (OpenAI format)
    return {"text": result["text"]}


# ── POST /v1/audio/translations — stub (maps to transcription) ──────────────
@app.post("/v1/audio/translations")
async def translate(
    file: UploadFile = File(...),
    model: Optional[str] = Form(default="parakeet-tdt-0.6b-v3"),
    language: Optional[str] = Form(default=None),
    response_format: Optional[str] = Form(default="json"),
    temperature: Optional[float] = Form(default=0.0),
    prompt: Optional[str] = Form(default=None),
):
    """
    Stub for translation endpoint.
    Parakeet TDT v3 does transcription only (no translation).
    Falls back to transcription.
    """
    logger.warning(
        "Translation endpoint called — Parakeet v3 does not translate. "
        "Falling back to transcription."
    )
    return await transcribe(
        file=file,
        model=model,
        language=language,
        response_format=response_format,
        temperature=temperature,
    )


# ── Default speaker names by count ───────────────────────────────────────────
DEFAULT_SPEAKER_NAMES = {
    2: "Arzt,Patient",
    3: "Arzt,Patient,Begleitung",
}


# ── POST /v1/audio/diarizations — Speaker diarization endpoint ──────────────
@app.post("/v1/audio/diarizations")
async def diarize(
    file: UploadFile = File(...),
    model: Optional[str] = Form(default="diar_sortformer_4spk-v1"),
    num_speakers: Optional[int] = Form(default=None),
    min_speakers: Optional[int] = Form(default=None),
    max_speakers: Optional[int] = Form(default=None),
    speaker_names: Optional[str] = Form(default=None),
    transcribe_audio: Optional[bool] = Form(default=True),
    language: Optional[str] = Form(default=None),
    response_format: Optional[str] = Form(default="verbose_json"),
):
    """
    Speaker diarization endpoint with optional transcription.

    Identifies who spoke when in the audio. Returns speaker-labeled time segments
    with transcribed text for each segment.

    Parameters:
      - file: Audio file (wav, mp3, flac, ogg, webm, m4a, mp4, etc.)
      - model: Diarization model (default: diar_sortformer_4spk-v1)
      - num_speakers: Exact number of speakers (if known)
      - min_speakers: Minimum expected speakers
      - max_speakers: Maximum expected speakers (Sortformer supports up to 4)
      - speaker_names: Comma-separated speaker names (e.g. "Arzt,Patient").
            Defaults: "Arzt,Patient" for 2 speakers, "Arzt,Patient,Begleitung" for 3.
      - transcribe_audio: Whether to transcribe and assign text to segments (default: true)
      - language: Language code for transcription (e.g. 'de', 'en')
      - response_format: 'verbose_json' (default) or 'json'
    """
    if not ENABLE_DIARIZATION:
        raise HTTPException(
            status_code=501,
            detail="Speaker diarization is disabled. Set ENABLE_DIARIZATION=true to enable.",
        )

    try:
        diarizer = get_diarization_backend()
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not diarizer.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Diarization model is still loading. Try again shortly.",
        )

    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Read audio bytes
    try:
        audio_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read audio file: {e}")

    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")

    # Check file size
    max_size = int(os.getenv("MAX_UPLOAD_MB", "200")) * 1024 * 1024
    if len(audio_bytes) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {max_size // (1024*1024)} MB",
        )

    logger.info(
        f"Diarization request: {file.filename} "
        f"({len(audio_bytes) / 1024:.0f} KB, speakers={num_speakers})"
    )

    start_time = time.time()

    try:
        result = diarizer.diarize(
            audio_bytes=audio_bytes,
            filename=file.filename,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
    except Exception as e:
        logger.exception("Diarization failed")
        raise HTTPException(status_code=500, detail=f"Diarization failed: {e}")

    # ── Transcribe and assign text to segments ───────────────────────────
    if transcribe_audio and transcriber._loaded and result.segments:
        try:
            transcript = transcriber.transcribe(
                audio_bytes=audio_bytes,
                filename=file.filename,
                language=language,
                timestamps=True,
            )
            _assign_text_to_segments(result, transcript)
        except Exception as e:
            logger.warning(f"Transcription for diarization failed: {e}")
            # Continue without text — diarization segments still valid

    # ── Apply speaker names ──────────────────────────────────────────────
    _apply_speaker_names(result, speaker_names, num_speakers)

    elapsed = time.time() - start_time

    logger.info(
        f"Diarization complete: {result.duration:.1f}s audio, "
        f"{result.num_speakers} speakers in {elapsed:.1f}s"
    )

    # Format response
    if response_format == "json":
        return {
            "num_speakers": result.num_speakers,
            "segments": [s.to_dict() for s in result.segments],
        }

    # Default: verbose_json
    return {
        "task": "diarize",
        "duration": result.duration,
        "num_speakers": result.num_speakers,
        "backend": result.backend,
        "segments": [s.to_dict() for s in result.segments],
    }


def _apply_speaker_names(
    result,
    speaker_names: Optional[str],
    num_speakers: Optional[int],
) -> None:
    """
    Replace anonymous speaker labels (speaker_0, speaker_1, ...) with
    user-provided names or defaults based on speaker count.
    """
    # Determine the name list
    if speaker_names:
        names = [n.strip() for n in speaker_names.split(",")]
    else:
        # Use defaults based on detected or requested speaker count
        count = num_speakers or result.num_speakers
        names_str = DEFAULT_SPEAKER_NAMES.get(count)
        if names_str:
            names = [n.strip() for n in names_str.split(",")]
        else:
            return  # No default mapping — keep original labels

    # Build mapping from anonymous labels to names
    unique_speakers = sorted(set(s.speaker for s in result.segments))
    name_map = {}
    for i, spk in enumerate(unique_speakers):
        if i < len(names):
            name_map[spk] = names[i]
        # else: keep original label

    # Apply mapping
    for seg in result.segments:
        if seg.speaker in name_map:
            seg.speaker = name_map[seg.speaker]


def _assign_text_to_segments(result, transcript: dict) -> None:
    """
    Assign transcribed text to diarization segments based on time overlap.

    Uses word-level timestamps from the transcription to attribute words
    to the correct speaker segment.
    """
    segments_with_ts = transcript.get("segments", [])

    if segments_with_ts:
        # Word-level timestamps available — assign by midpoint overlap
        for diar_seg in result.segments:
            words = []
            for word_seg in segments_with_ts:
                word_mid = (word_seg["start"] + word_seg["end"]) / 2
                if diar_seg.start <= word_mid <= diar_seg.end:
                    text = word_seg.get("text", "").strip()
                    if text:
                        words.append(text)
            if words:
                diar_seg.text = " ".join(words)
    else:
        # No timestamps — split full text proportionally by segment duration
        full_text = transcript.get("text", "").strip()
        if not full_text or not result.segments:
            return

        total_duration = sum(s.end - s.start for s in result.segments)
        if total_duration == 0:
            return

        words = full_text.split()
        total_words = len(words)
        word_idx = 0

        for seg in result.segments:
            if word_idx >= total_words:
                break
            seg_duration = seg.end - seg.start
            proportion = seg_duration / total_duration
            word_count = round(proportion * total_words)
            end_idx = min(word_idx + word_count, total_words)
            seg_words = words[word_idx:end_idx]
            if seg_words:
                seg.text = " ".join(seg_words)
            word_idx = end_idx


# ── Subtitle formatters ─────────────────────────────────────────────────────

def _format_timestamp_srt(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    """Format seconds as VTT timestamp: HH:MM:SS.mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _to_srt(result: dict) -> str:
    """Convert transcription result to SRT format."""
    segments = result.get("segments", [])
    if not segments:
        # No timestamps — return single block
        return f"1\n00:00:00,000 --> 99:59:59,999\n{result['text']}\n"

    lines = []
    for i, seg in enumerate(segments, 1):
        start = _format_timestamp_srt(seg.get("start", 0))
        end = _format_timestamp_srt(seg.get("end", 0))
        lines.append(f"{i}\n{start} --> {end}\n{seg.get('text', '')}\n")
    return "\n".join(lines)


def _to_vtt(result: dict) -> str:
    """Convert transcription result to WebVTT format."""
    segments = result.get("segments", [])
    header = "WEBVTT\n\n"
    if not segments:
        return header + f"00:00:00.000 --> 99:59:59.999\n{result['text']}\n"

    lines = [header]
    for seg in segments:
        start = _format_timestamp_vtt(seg.get("start", 0))
        end = _format_timestamp_vtt(seg.get("end", 0))
        lines.append(f"{start} --> {end}\n{seg.get('text', '')}\n")
    return "\n".join(lines)
