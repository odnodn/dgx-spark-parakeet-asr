"""
Realtime Streaming ASR Module

Modular realtime speech-to-text streaming pipeline supporting multiple backends.
Currently supported:
  - NVIDIA Riva/NIM WebSocket ASR

Future backends can be added by subclassing StreamingASRBackend.
"""

from app.realtime.base import StreamingASRBackend, StreamingConfig, StreamingTranscript
from app.realtime.riva_client import RivaStreamingClient
from app.realtime.websocket_handler import RealtimeWebSocketHandler

__all__ = [
    "StreamingASRBackend",
    "StreamingConfig",
    "StreamingTranscript",
    "RivaStreamingClient",
    "RealtimeWebSocketHandler",
]
