"""
Riva/NIM WebSocket ASR streaming client.

Connects to NVIDIA Riva or NIM ASR service via WebSocket for
realtime speech-to-text streaming.

Environment variables:
  RIVA_ASR_URL      — WebSocket URL (e.g. ws://localhost:50051/asr/ws)
  RIVA_API_KEY      — API key for NIM cloud endpoints (optional)
  RIVA_LANGUAGE     — Default language code (default: en-US)
"""

import os
import json
import logging
import asyncio
from typing import AsyncIterator, Optional

import websockets

from app.realtime.base import StreamingASRBackend, StreamingConfig, StreamingTranscript

logger = logging.getLogger(__name__)

# ── Configuration from environment ───────────────────────────────────────────
RIVA_ASR_URL = os.getenv("RIVA_ASR_URL", "ws://localhost:50051/asr/ws")
RIVA_API_KEY = os.getenv("RIVA_API_KEY", "")
RIVA_LANGUAGE = os.getenv("RIVA_LANGUAGE", "en-US")


class RivaStreamingClient(StreamingASRBackend):
    """
    Riva/NIM WebSocket streaming ASR client.

    Connects to a Riva or NIM ASR WebSocket endpoint and streams audio
    for realtime transcription. Supports both on-premise Riva deployments
    and NVIDIA NIM cloud endpoints.

    Protocol:
      1. Client sends a JSON config message on connection
      2. Client streams raw audio bytes (PCM 16-bit, 16kHz mono)
      3. Server sends back JSON transcript messages (interim + final)
      4. Client sends empty bytes or closes connection to signal end
    """

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        language: Optional[str] = None,
    ):
        self._url = url or RIVA_ASR_URL
        self._api_key = api_key or RIVA_API_KEY
        self._language = language or RIVA_LANGUAGE
        self._ws = None
        self._available = False
        self._check_availability()

    @property
    def name(self) -> str:
        return "riva-nim-websocket"

    @property
    def is_available(self) -> bool:
        return self._available

    def _check_availability(self):
        """Check if the Riva/NIM endpoint URL is configured."""
        self._available = bool(self._url and self._url.startswith(("ws://", "wss://")))
        if not self._available:
            logger.warning(
                f"Riva/NIM streaming not available: invalid URL '{self._url}'. "
                "Set RIVA_ASR_URL to a valid WebSocket endpoint."
            )

    def _build_headers(self) -> dict:
        """Build WebSocket connection headers."""
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _build_config_message(self, config: StreamingConfig) -> str:
        """Build the initial JSON configuration message for Riva/NIM."""
        return json.dumps({
            "type": "config",
            "config": {
                "language_code": config.language,
                "sample_rate_hertz": config.sample_rate,
                "encoding": config.encoding,
                "audio_channel_count": config.channels,
                "enable_automatic_punctuation": config.punctuation,
                "enable_word_time_offsets": config.word_timestamps,
                "max_alternatives": config.max_alternatives,
                "interim_results": config.interim_results,
            },
        })

    async def stream(
        self,
        audio_chunks: AsyncIterator[bytes],
        config: Optional[StreamingConfig] = None,
    ) -> AsyncIterator[StreamingTranscript]:
        """
        Stream audio to Riva/NIM and yield transcription results.

        Opens a WebSocket connection, sends configuration, then streams
        audio chunks while receiving transcript responses.

        Args:
            audio_chunks: Async iterator yielding raw PCM audio bytes
            config: Streaming configuration (language, sample rate, etc.)

        Yields:
            StreamingTranscript with interim and final results
        """
        if not self._available:
            raise RuntimeError(
                "Riva/NIM streaming is not available. "
                "Check RIVA_ASR_URL configuration."
            )

        if config is None:
            config = StreamingConfig(language=self._language)

        headers = self._build_headers()

        try:
            async with websockets.connect(
                self._url,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5,
            ) as ws:
                self._ws = ws

                # Send configuration message
                config_msg = self._build_config_message(config)
                await ws.send(config_msg)
                logger.debug(f"Sent config to Riva/NIM: {config_msg}")

                # Start concurrent send/receive tasks
                send_task = asyncio.create_task(
                    self._send_audio(ws, audio_chunks)
                )

                try:
                    async for transcript in self._receive_transcripts(ws):
                        yield transcript
                finally:
                    send_task.cancel()
                    try:
                        await send_task
                    except asyncio.CancelledError:
                        pass

        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"Riva/NIM WebSocket connection closed: {e}")
        except Exception as e:
            logger.error(f"Riva/NIM streaming error: {e}")
            raise
        finally:
            self._ws = None

    async def _send_audio(
        self,
        ws: websockets.WebSocketClientProtocol,
        audio_chunks: AsyncIterator[bytes],
    ) -> None:
        """Send audio chunks over the WebSocket connection."""
        try:
            async for chunk in audio_chunks:
                if chunk:
                    await ws.send(chunk)
            # Signal end of audio stream
            await ws.send(b"")
            logger.debug("Audio stream complete — sent EOS signal")
        except asyncio.CancelledError:
            pass
        except websockets.exceptions.ConnectionClosed:
            pass

    async def _receive_transcripts(
        self,
        ws: websockets.WebSocketClientProtocol,
    ) -> AsyncIterator[StreamingTranscript]:
        """Receive and parse transcript messages from the WebSocket."""
        try:
            async for message in ws:
                if isinstance(message, bytes):
                    # Binary message — skip
                    continue

                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from Riva/NIM: {message[:100]}")
                    continue

                transcript = self._parse_transcript(data)
                if transcript:
                    yield transcript

        except websockets.exceptions.ConnectionClosed:
            logger.debug("Riva/NIM WebSocket closed during receive")

    def _parse_transcript(self, data: dict) -> Optional[StreamingTranscript]:
        """Parse a Riva/NIM transcript message into a StreamingTranscript."""
        msg_type = data.get("type", "")

        if msg_type == "transcript" or "alternatives" in data:
            # Riva format: {"type": "transcript", "alternatives": [...], "is_final": bool}
            alternatives = data.get("alternatives", [])
            if not alternatives:
                return None

            best = alternatives[0]
            text = best.get("transcript", best.get("text", ""))
            confidence = best.get("confidence", 0.0)

            # Word timestamps if available
            words = best.get("words", [])
            start_time = 0.0
            end_time = 0.0
            if words:
                start_time = words[0].get("start_time", 0.0)
                end_time = words[-1].get("end_time", 0.0)

            return StreamingTranscript(
                text=text,
                is_final=data.get("is_final", False),
                confidence=confidence,
                start_time=start_time,
                end_time=end_time,
                language=data.get("language", ""),
            )

        elif msg_type == "result":
            # NIM format: {"type": "result", "text": "...", "final": bool}
            text = data.get("text", "")
            if not text:
                return None

            return StreamingTranscript(
                text=text,
                is_final=data.get("final", data.get("is_final", False)),
                confidence=data.get("confidence", 0.0),
                start_time=data.get("start_time", 0.0),
                end_time=data.get("end_time", 0.0),
                language=data.get("language", ""),
            )

        elif msg_type == "error":
            error_msg = data.get("message", data.get("error", "Unknown error"))
            logger.error(f"Riva/NIM error: {error_msg}")
            return None

        # Unknown message type — try to extract text anyway
        if "text" in data:
            return StreamingTranscript(
                text=data["text"],
                is_final=data.get("is_final", data.get("final", False)),
                confidence=data.get("confidence", 0.0),
            )

        return None

    async def close(self) -> None:
        """Close the WebSocket connection if open."""
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
