"""
FastAPI WebSocket handler for realtime ASR streaming.

Manages WebSocket connections from clients and proxies audio to/from
the configured streaming ASR backend (Riva/NIM).

Protocol (client ↔ server):
  1. Client connects to ws://<host>/v1/realtime/transcriptions
  2. Client sends JSON config: {"type": "config", "language": "en-US", ...}
  3. Client streams raw audio bytes (PCM 16-bit, 16kHz mono)
  4. Server sends JSON transcripts: {"type": "transcript", "text": "...", "is_final": bool}
  5. Client sends {"type": "eos"} or closes connection to end the session
"""

import json
import logging
import asyncio
from typing import AsyncIterator, Optional

from fastapi import WebSocket, WebSocketDisconnect

from app.realtime.base import StreamingConfig, StreamingTranscript
from app.realtime.riva_client import RivaStreamingClient

logger = logging.getLogger(__name__)


class RealtimeWebSocketHandler:
    """
    Handles a single WebSocket session for realtime ASR.

    Bridges between the FastAPI WebSocket and the Riva/NIM
    streaming backend.
    """

    def __init__(self, backend: Optional[RivaStreamingClient] = None):
        self._backend = backend or RivaStreamingClient()
        self._audio_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        self._active = False

    async def handle(self, websocket: WebSocket) -> None:
        """
        Main handler for a WebSocket connection.

        Accepts the connection, waits for config, then streams audio
        to the backend and sends transcripts back to the client.
        """
        await websocket.accept()
        self._active = True

        logger.info(f"Realtime ASR session started: {websocket.client}")

        config = StreamingConfig()

        try:
            # Phase 1: Wait for config message (with timeout)
            config = await self._receive_config(websocket)
            logger.info(
                f"Session config: language={config.language}, "
                f"sample_rate={config.sample_rate}, "
                f"interim_results={config.interim_results}"
            )

            # Send acknowledgment
            await websocket.send_json({
                "type": "config_ack",
                "status": "ok",
                "message": "Configuration accepted. Send audio bytes to begin.",
                "config": config.to_dict(),
            })

            # Phase 2: Stream audio and receive transcripts
            receive_task = asyncio.create_task(
                self._receive_audio(websocket)
            )

            try:
                async for transcript in self._backend.stream(
                    self._audio_iterator(),
                    config=config,
                ):
                    await self._send_transcript(websocket, transcript)
            finally:
                receive_task.cancel()
                try:
                    await receive_task
                except asyncio.CancelledError:
                    pass

            # Send session end message
            await websocket.send_json({
                "type": "session_end",
                "message": "Transcription session complete.",
            })

        except WebSocketDisconnect:
            logger.info(f"Client disconnected: {websocket.client}")
        except RuntimeError as e:
            logger.error(f"Realtime ASR error: {e}")
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e),
                })
            except Exception:
                pass
        except Exception as e:
            logger.exception(f"Unexpected error in realtime session: {e}")
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Internal error: {type(e).__name__}",
                })
            except Exception:
                pass
        finally:
            self._active = False
            await self._backend.close()
            logger.info(f"Realtime ASR session ended: {websocket.client}")

    async def _receive_config(self, websocket: WebSocket) -> StreamingConfig:
        """
        Wait for a config message from the client.

        If the first message is audio bytes (not JSON), use defaults.
        Times out after 10 seconds.
        """
        try:
            message = await asyncio.wait_for(
                websocket.receive(), timeout=10.0
            )
        except asyncio.TimeoutError:
            logger.info("No config received within timeout — using defaults")
            return StreamingConfig()

        # Check if it's a text (JSON) message
        if "text" in message:
            try:
                data = json.loads(message["text"])
                return self._parse_config(data)
            except (json.JSONDecodeError, KeyError):
                logger.warning("Invalid config message — using defaults")
                return StreamingConfig()

        # First message is bytes — push to audio queue and use defaults
        if "bytes" in message and message["bytes"]:
            await self._audio_queue.put(message["bytes"])

        return StreamingConfig()

    def _parse_config(self, data: dict) -> StreamingConfig:
        """Parse a JSON config message into StreamingConfig."""
        return StreamingConfig(
            language=data.get("language", data.get("language_code", "en-US")),
            sample_rate=data.get("sample_rate", data.get("sample_rate_hertz", 16000)),
            encoding=data.get("encoding", "LINEAR16"),
            channels=data.get("channels", data.get("audio_channel_count", 1)),
            interim_results=data.get("interim_results", True),
            punctuation=data.get("punctuation", data.get("enable_automatic_punctuation", True)),
            word_timestamps=data.get("word_timestamps", data.get("enable_word_time_offsets", False)),
            max_alternatives=data.get("max_alternatives", 1),
            chunk_duration_ms=data.get("chunk_duration_ms", 100),
        )

    async def _receive_audio(self, websocket: WebSocket) -> None:
        """Receive audio bytes from the WebSocket and enqueue them."""
        try:
            while self._active:
                message = await websocket.receive()

                if "bytes" in message and message["bytes"]:
                    await self._audio_queue.put(message["bytes"])

                elif "text" in message:
                    try:
                        data = json.loads(message["text"])
                        if data.get("type") == "eos":
                            logger.debug("Received EOS from client")
                            break
                    except json.JSONDecodeError:
                        pass

        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.debug(f"Audio receive ended: {e}")
        finally:
            # Signal end of stream
            await self._audio_queue.put(None)

    async def _audio_iterator(self) -> AsyncIterator[bytes]:
        """Async iterator that yields audio chunks from the queue."""
        while True:
            chunk = await self._audio_queue.get()
            if chunk is None:
                break
            yield chunk

    async def _send_transcript(
        self, websocket: WebSocket, transcript: StreamingTranscript
    ) -> None:
        """Send a transcript message to the WebSocket client."""
        message = {
            "type": "transcript",
            **transcript.to_dict(),
        }
        try:
            await websocket.send_json(message)
        except Exception:
            self._active = False
