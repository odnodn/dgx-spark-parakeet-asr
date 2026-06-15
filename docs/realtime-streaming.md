# Realtime Streaming ASR via Riva/NIM WebSocket

This document describes the realtime streaming speech-to-text feature using NVIDIA Riva or NIM WebSocket ASR, integrated into the Parakeet ASR service.

---

## Overview

The Parakeet ASR service supports two transcription modes:

| Mode | Endpoint | Use Case |
|------|----------|----------|
| **Batch** | `POST /v1/audio/transcriptions` | Pre-recorded audio files |
| **Realtime** | `WS /v1/realtime/transcriptions` | Live audio streaming |

Realtime streaming uses a WebSocket connection to stream audio to an NVIDIA Riva or NIM ASR backend, receiving transcription results (both interim/partial and final) in real time. This is ideal for:

- Live dictation / voice typing
- Real-time captioning
- Voice assistants with low-latency requirements
- Medical dictation (German/English)
- Meeting transcription

---

## Architecture

```
┌─────────────────┐     WebSocket      ┌─────────────────────┐     WebSocket      ┌───────────────┐
│   Client App    │ ◄─────────────────► │   Parakeet ASR API  │ ◄─────────────────► │  Riva / NIM   │
│  (Browser/App)  │  audio + transcripts│   (FastAPI)         │   audio + results   │  ASR Service  │
└─────────────────┘                     └─────────────────────┘                     └───────────────┘
```

### Module Structure

```
app/realtime/
├── __init__.py              # Module exports
├── base.py                  # Abstract base class (StreamingASRBackend)
├── riva_client.py           # Riva/NIM WebSocket client implementation
└── websocket_handler.py     # FastAPI WebSocket endpoint handler
```

The module follows the same modular pattern as the diarization backends — new streaming ASR backends can be added by subclassing `StreamingASRBackend`.

---

## Prerequisites

### Option A: NVIDIA Riva (On-Premise)

Deploy Riva ASR on the same machine or network:

```bash
# Pull and start Riva with Riva Quick Start
# See: https://docs.nvidia.com/deeplearning/riva/user-guide/docs/quick-start-guide.html

# 1. Download Riva Quick Start scripts
ngc registry resource download-version nvidia/riva/riva_quickstart:2.17.0

# 2. Edit config.sh — enable ASR, set language models
cd riva_quickstart_v2.17.0
vi config.sh

# 3. Initialize and start
bash riva_init.sh
bash riva_start.sh
```

Riva ASR WebSocket will be available at: `ws://localhost:50051/asr/ws`

### Option B: NVIDIA NIM (Cloud)

Use NVIDIA NIM cloud-hosted ASR:

1. Get an API key from [NVIDIA Build](https://build.nvidia.com/)
2. Set `RIVA_ASR_URL` to the NIM WebSocket endpoint
3. Set `RIVA_API_KEY` to your API key

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_REALTIME` | `true` | Enable/disable the realtime streaming endpoint |
| `RIVA_ASR_URL` | `ws://localhost:50051/asr/ws` | WebSocket URL of Riva/NIM ASR |
| `RIVA_API_KEY` | *(empty)* | API key for NIM cloud (optional for on-premise) |
| `RIVA_LANGUAGE` | `en-US` | Default language code (BCP-47) |

### Docker Compose

Add these to your `docker-compose.yml` or Portainer stack environment:

```yaml
environment:
  - ENABLE_REALTIME=true
  - RIVA_ASR_URL=ws://riva-speech:50051/asr/ws
  - RIVA_API_KEY=${RIVA_API_KEY:-}
  - RIVA_LANGUAGE=en-US
```

### Running Alongside Riva

If running Riva in the same Docker Compose stack:

```yaml
services:
  parakeet-asr:
    # ... existing config ...
    environment:
      - RIVA_ASR_URL=ws://riva-speech:50051/asr/ws
    depends_on:
      - riva-speech

  riva-speech:
    image: nvcr.io/nvidia/riva/riva-speech:2.17.0
    ports:
      - "50051:50051"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

---

## WebSocket Protocol

### Connection

```
ws://<host>:8010/v1/realtime/transcriptions
```

### Message Flow

```
Client                          Server
  │                               │
  │──── Connect ─────────────────►│
  │◄─── Accept ──────────────────│
  │                               │
  │──── Config (JSON) ──────────►│
  │◄─── Config ACK (JSON) ───────│
  │                               │
  │──── Audio bytes ─────────────►│
  │◄─── Interim transcript ───────│
  │──── Audio bytes ─────────────►│
  │◄─── Interim transcript ───────│
  │──── Audio bytes ─────────────►│
  │◄─── Final transcript ────────│
  │──── Audio bytes ─────────────►│
  │◄─── Interim transcript ───────│
  │                               │
  │──── EOS (JSON) ─────────────►│
  │◄─── Session end ──────────────│
  │                               │
  │──── Close ───────────────────►│
```

### 1. Configuration Message (Client → Server)

Send this as the first message after connecting:

```json
{
  "type": "config",
  "language": "en-US",
  "sample_rate": 16000,
  "encoding": "LINEAR16",
  "channels": 1,
  "interim_results": true,
  "punctuation": true,
  "word_timestamps": false,
  "max_alternatives": 1
}
```

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | string | — | Must be `"config"` |
| `language` | string | `en-US` | BCP-47 language code |
| `sample_rate` | int | `16000` | Audio sample rate in Hz |
| `encoding` | string | `LINEAR16` | Audio encoding (LINEAR16, FLAC, OPUS) |
| `channels` | int | `1` | Number of audio channels |
| `interim_results` | bool | `true` | Send partial results |
| `punctuation` | bool | `true` | Enable auto-punctuation |
| `word_timestamps` | bool | `false` | Include word-level timestamps |
| `max_alternatives` | int | `1` | Max alternative transcriptions |

If no config message is sent within 10 seconds, defaults are used.

### 2. Configuration Acknowledgment (Server → Client)

```json
{
  "type": "config_ack",
  "status": "ok",
  "message": "Configuration accepted. Send audio bytes to begin.",
  "config": { /* echoed config */ }
}
```

### 3. Audio Streaming (Client → Server)

Send raw audio bytes as binary WebSocket frames:

- **Format:** PCM 16-bit signed little-endian (LINEAR16)
- **Sample rate:** 16000 Hz (or as configured)
- **Channels:** 1 (mono)
- **Chunk size:** Recommended 3200 bytes (100ms at 16kHz/16-bit)

### 4. Transcript Messages (Server → Client)

```json
{
  "type": "transcript",
  "text": "Hello world",
  "is_final": false,
  "confidence": 0.95,
  "start_time": 0.0,
  "end_time": 1.5
}
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Always `"transcript"` |
| `text` | string | Transcribed text |
| `is_final` | bool | `true` = final result, `false` = interim/partial |
| `confidence` | float | Confidence score (0.0 – 1.0) |
| `start_time` | float | Start time in seconds (if word_timestamps enabled) |
| `end_time` | float | End time in seconds (if word_timestamps enabled) |

### 5. End of Stream (Client → Server)

Signal the end of audio:

```json
{"type": "eos"}
```

Or simply close the WebSocket connection.

### 6. Session End (Server → Client)

```json
{
  "type": "session_end",
  "message": "Transcription session complete."
}
```

### Error Messages (Server → Client)

```json
{
  "type": "error",
  "message": "Riva/NIM streaming is not available. Check RIVA_ASR_URL configuration."
}
```

---

## Client Examples

### Python (websockets)

```python
import asyncio
import json
import wave
import websockets

async def stream_audio(file_path: str, ws_url: str = "ws://localhost:8010/v1/realtime/transcriptions"):
    async with websockets.connect(ws_url) as ws:
        # Send config
        config = {
            "type": "config",
            "language": "en-US",
            "sample_rate": 16000,
            "encoding": "LINEAR16",
            "interim_results": True,
        }
        await ws.send(json.dumps(config))

        # Wait for config ack
        ack = json.loads(await ws.recv())
        print(f"Config: {ack['status']}")

        # Stream audio file
        with wave.open(file_path, 'rb') as wf:
            chunk_size = 3200  # 100ms at 16kHz/16-bit
            while True:
                data = wf.readframes(chunk_size // 2)  # 2 bytes per sample
                if not data:
                    break
                await ws.send(data)
                await asyncio.sleep(0.1)  # Simulate realtime

                # Check for transcripts
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.01)
                    result = json.loads(msg)
                    prefix = "FINAL" if result.get("is_final") else "partial"
                    print(f"[{prefix}] {result['text']}")
                except asyncio.TimeoutError:
                    pass

        # Signal end of stream
        await ws.send(json.dumps({"type": "eos"}))

        # Receive remaining transcripts
        async for msg in ws:
            result = json.loads(msg)
            if result.get("type") == "session_end":
                break
            if result.get("type") == "transcript":
                prefix = "FINAL" if result.get("is_final") else "partial"
                print(f"[{prefix}] {result['text']}")

asyncio.run(stream_audio("recording.wav"))
```

### JavaScript (Browser)

```javascript
const ws = new WebSocket('ws://localhost:8010/v1/realtime/transcriptions');

ws.onopen = () => {
  // Send configuration
  ws.send(JSON.stringify({
    type: 'config',
    language: 'en-US',
    sample_rate: 16000,
    encoding: 'LINEAR16',
    interim_results: true,
  }));
};

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);

  switch (msg.type) {
    case 'config_ack':
      console.log('Ready to stream audio');
      startMicrophone();
      break;
    case 'transcript':
      if (msg.is_final) {
        document.getElementById('final').textContent += msg.text + ' ';
      } else {
        document.getElementById('interim').textContent = msg.text;
      }
      break;
    case 'session_end':
      console.log('Session complete');
      ws.close();
      break;
    case 'error':
      console.error('Error:', msg.message);
      break;
  }
};

// Stream from microphone
async function startMicrophone() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const audioContext = new AudioContext({ sampleRate: 16000 });
  const source = audioContext.createMediaStreamSource(stream);
  const processor = audioContext.createScriptProcessor(1024, 1, 1);

  source.connect(processor);
  processor.connect(audioContext.destination);

  processor.onaudioprocess = (e) => {
    const float32 = e.inputBuffer.getChannelData(0);
    // Convert Float32 to Int16
    const int16 = new Int16Array(float32.length);
    for (let i = 0; i < float32.length; i++) {
      int16[i] = Math.max(-32768, Math.min(32767, float32[i] * 32768));
    }
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(int16.buffer);
    }
  };
}
```

### cURL (for testing with a file)

WebSocket is not natively supported by cURL, but you can use `websocat`:

```bash
# Install websocat
# https://github.com/vi/websocat

# Stream a WAV file (PCM 16-bit, 16kHz mono)
echo '{"type":"config","language":"en-US","sample_rate":16000}' | \
  cat - <(tail -c +45 recording.wav) | \
  websocat ws://localhost:8010/v1/realtime/transcriptions
```

---

## Supported Languages

Language support depends on the Riva/NIM model deployed. Common language codes:

| Language | Code |
|----------|------|
| English (US) | `en-US` |
| English (UK) | `en-GB` |
| German | `de-DE` |
| French | `fr-FR` |
| Spanish | `es-ES` |
| Italian | `it-IT` |
| Portuguese | `pt-BR` |
| Russian | `ru-RU` |
| Dutch | `nl-NL` |
| Polish | `pl-PL` |

See NVIDIA Riva documentation for the full list of supported language models.

---

## Audio Format Requirements

| Property | Requirement |
|----------|-------------|
| Encoding | PCM 16-bit signed little-endian (LINEAR16) |
| Sample rate | 16000 Hz (recommended) |
| Channels | 1 (mono) |
| Chunk duration | 100ms recommended (3200 bytes at 16kHz) |

### Converting audio for streaming

```bash
# Convert any audio to the required format
ffmpeg -i input.mp3 -ar 16000 -ac 1 -f s16le output.pcm

# From a WAV file, strip the header (first 44 bytes)
tail -c +45 input.wav > output.pcm
```

---

## Extending the Streaming Backend

To add a new streaming ASR backend:

1. Create a new file in `app/realtime/` (e.g., `my_backend.py`)
2. Subclass `StreamingASRBackend` from `app/realtime/base.py`
3. Implement `name`, `is_available`, `stream()`, and `close()`
4. Register it in the WebSocket handler or create a factory pattern

```python
from app.realtime.base import StreamingASRBackend, StreamingConfig, StreamingTranscript
from typing import AsyncIterator, Optional

class MyCustomBackend(StreamingASRBackend):
    @property
    def name(self) -> str:
        return "my-custom-backend"

    @property
    def is_available(self) -> bool:
        return True  # Check your backend availability

    async def stream(
        self,
        audio_chunks: AsyncIterator[bytes],
        config: Optional[StreamingConfig] = None,
    ) -> AsyncIterator[StreamingTranscript]:
        # Implement your streaming logic
        async for chunk in audio_chunks:
            result = await self._process(chunk)
            yield StreamingTranscript(text=result, is_final=False)

    async def close(self) -> None:
        pass
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| WebSocket connects but no transcripts | Check `RIVA_ASR_URL` points to a running Riva/NIM instance |
| "Realtime streaming is disabled" | Set `ENABLE_REALTIME=true` in environment |
| Connection refused | Ensure Riva/NIM service is running and accessible |
| Poor transcription quality | Check audio format (must be 16kHz, 16-bit, mono PCM) |
| High latency | Reduce chunk size, check network between services |

### Checking Service Status

```bash
# Check if the realtime endpoint is available
curl http://localhost:8010/ | jq .realtime

# Health check
curl http://localhost:8010/health
```

### Debug Logging

Set `LOG_LEVEL=DEBUG` for verbose WebSocket and streaming logs:

```yaml
environment:
  - LOG_LEVEL=DEBUG
```

---

## Comparison: Batch vs Realtime

| Feature | Batch (`/v1/audio/transcriptions`) | Realtime (`/v1/realtime/transcriptions`) |
|---------|------|----------|
| Protocol | HTTP POST | WebSocket |
| Input | Complete audio file | Audio stream (chunks) |
| Output | Full transcript (after processing) | Incremental results (interim + final) |
| Latency | Seconds–minutes (depends on length) | Sub-second (per utterance) |
| Backend | Local Parakeet NeMo model | Riva/NIM WebSocket |
| GPU usage | On this machine | Riva service (may be remote) |
| Best for | Pre-recorded audio, files | Live audio, microphones |
| Languages | 25 European (auto-detect) | Depends on Riva model |
