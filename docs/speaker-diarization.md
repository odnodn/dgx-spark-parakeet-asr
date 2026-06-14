# Speaker Diarization

Speaker diarization identifies **who spoke when** in an audio recording. This module uses NVIDIA's **Sortformer** model to perform end-to-end neural speaker diarization on the DGX Spark.

## Overview

| Feature | Details |
|---------|---------|
| **Model** | NVIDIA Sortformer (`diar_sortformer_4spk-v1`) |
| **Max speakers** | 4 (Sortformer v1 limitation) |
| **Endpoint** | `POST /v1/audio/diarizations` |
| **Frameworks** | NeMo toolkit, PyTorch |
| **Target hardware** | DGX Spark (GB10, aarch64, CUDA 13.0) |

---

## Quick Start

### Diarize an Audio File

```bash
curl -s http://localhost:8010/v1/audio/diarizations \
    -F file="@meeting.wav" \
    -F num_speakers=2 \
    | python3 -m json.tool
```

### With Custom Speaker Names

```bash
curl -s http://localhost:8010/v1/audio/diarizations \
    -F file="@meeting.wav" \
    -F num_speakers=2 \
    -F speaker_names="Arzt,Patient" \
    | python3 -m json.tool
```

### Response Format

```json
{
    "task": "diarize",
    "duration": 120.5,
    "num_speakers": 2,
    "backend": "sortformer",
    "segments": [
        {"speaker": "Arzt", "start": 0.0, "end": 5.2, "text": "Guten Tag, wie kann ich Ihnen helfen?"},
        {"speaker": "Patient", "start": 5.4, "end": 12.1, "text": "Ich habe seit drei Tagen Kopfschmerzen."},
        {"speaker": "Arzt", "start": 12.3, "end": 18.7, "text": "Können Sie mir mehr darüber erzählen?"}
    ]
}
```

---

## API Reference

### `POST /v1/audio/diarizations`

Speaker diarization endpoint. Identifies speakers and their time segments.

**Parameters (multipart form):**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | required | Audio file (wav, mp3, flac, ogg, webm, m4a, mp4) |
| `model` | string | `diar_sortformer_4spk-v1` | Diarization model to use |
| `num_speakers` | int | auto | Exact number of speakers (if known) |
| `min_speakers` | int | null | Minimum expected speakers |
| `max_speakers` | int | 4 | Maximum expected speakers |
| `speaker_names` | string | see below | Comma-separated speaker names (e.g. `"Arzt,Patient"`) |
| `transcribe_audio` | bool | `true` | Transcribe audio and assign text to segments |
| `language` | string | auto | Language code for transcription (e.g. `de`, `en`) |
| `response_format` | string | `verbose_json` | `verbose_json` or `json` |

**Default speaker names:**
- 2 speakers: `"Arzt,Patient"`
- 3 speakers: `"Arzt,Patient,Begleitung"`
- Other counts: anonymous labels (`speaker_0`, `speaker_1`, ...)

**Response formats:**

- `verbose_json` — Full result with metadata (task, duration, backend, segments)
- `json` — Compact format with num_speakers and segments only

---

## Configuration

Environment variables for diarization:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_DIARIZATION` | `true` | Enable/disable diarization endpoints |
| `DIARIZATION_BACKEND` | `sortformer` | Which backend to use |
| `SORTFORMER_MODEL` | `nvidia/diar_sortformer_4spk-v1` | NeMo model identifier |

Set these in your `.env` file or `docker-compose.yml`:

```yaml
environment:
  - ENABLE_DIARIZATION=true
  - DIARIZATION_BACKEND=sortformer
  - SORTFORMER_MODEL=nvidia/diar_sortformer_4spk-v1
```

To disable diarization (reduces memory usage):
```yaml
environment:
  - ENABLE_DIARIZATION=false
```

---

## Architecture

The diarization module is designed to be modular and extensible:

```text
app/diarization/
├── __init__.py          # Public API exports
├── base.py              # Abstract base class (DiarizationBackend)
├── factory.py           # Backend registry and instantiation
└── sortformer.py        # NVIDIA Sortformer implementation
```

### Adding a New Backend

To add a new diarization backend (e.g., pyannote):

1. Create `app/diarization/pyannote.py` implementing `DiarizationBackend`
2. Register it in `app/diarization/factory.py`
3. Set `DIARIZATION_BACKEND=pyannote` in your environment

```python
# app/diarization/pyannote.py
from app.diarization.base import DiarizationBackend, DiarizationResult

class PyannoteBackend(DiarizationBackend):
    @property
    def name(self) -> str:
        return "pyannote"

    # ... implement load_model() and diarize() ...
```

---

## Python Client Example

```python
import requests

# Diarize a meeting recording with named speakers
with open("meeting.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8010/v1/audio/diarizations",
        files={"file": f},
        data={
            "num_speakers": 2,
            "speaker_names": "Arzt,Patient",
            "language": "de",
        },
    )

result = response.json()
print(f"Found {result['num_speakers']} speakers:")
for seg in result["segments"]:
    print(f"  [{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['speaker']}: {seg.get('text', '')}")
```

---

## Performance Notes

- **GPU Memory**: Sortformer uses approximately 1-2 GB additional VRAM on top of the ASR model
- **Speed**: Typically processes audio at 20-50x real-time on DGX Spark
- **Startup**: Model loading takes 30-60 seconds on first start (cached thereafter)
- **Unified Memory**: DGX Spark's 128 GB unified memory allows both ASR and diarization models to coexist comfortably

---

## Limitations

- Sortformer v1 supports a maximum of **4 speakers**
- Speaker naming uses order of appearance — no voice identification/verification
- Best accuracy with clear turn-taking; overlapping speech detection depends on model version
- Text assignment to segments uses time-overlap heuristics (word midpoint within segment boundary)

---

## Health Check

Check diarization readiness:

```bash
curl -s http://localhost:8010/health | python3 -m json.tool
```

Response includes a `diarization` field:
```json
{
    "status": "ready",
    "model": "nvidia/parakeet-tdt-0.6b-v3",
    "device": "cuda",
    "gpu_memory": {"allocated_gb": 5.2, "total_gb": 128.0},
    "diarization": "ready"
}
```

Possible values: `ready`, `loading`, `disabled`, `error`
