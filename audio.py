import asyncio
import os
import traceback
from typing import Any, Awaitable, Callable, Optional

import numpy as np
from google import genai
from google.genai import types

AudioHandler = Callable[[bytes], Awaitable[None]]

MODEL = "models/gemini-2.5-flash-native-audio-preview-09-2025"

CONFIG = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    media_resolution="MEDIA_RESOLUTION_MEDIUM",
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Zephyr")
        )
    ),
    context_window_compression=types.ContextWindowCompressionConfig(
        trigger_tokens=25600,
        sliding_window=types.SlidingWindow(target_tokens=12800),
    ),
)


class AudioLoop:
    def __init__(
        self,
        *,
        client: Optional[genai.Client] = None,
        model: str = MODEL,
        config: types.LiveConnectConfig = CONFIG,
        audio_handler: Optional[AudioHandler] = None,
    ):
        """
        audio_handler: async callable taking a single bytes argument (PCM data).
        If None, the default handler drops audio.
        """
        self.client = client or genai.Client(
            http_options={"api_version": "v1beta"},
            api_key=os.environ.get("GEMINI_API_KEY"),
        )
        self.model = model
        self.config = config

        self.session: Optional[Any] = None
        self.out_queue: Optional[asyncio.Queue[dict[str, bytes]]] = None
        self.audio_handler: AudioHandler = audio_handler or self._default_audio_handler
        self._stop_event = asyncio.Event()

    async def _default_audio_handler(self, _: bytes) -> None:
        """Default audio handler: drop output."""
        return None

    async def append_chunk(self, chunk: np.ndarray) -> None:
        """
        Append an incoming audio chunk (np.int16 PCM) to the send queue.
        Chunks should match the format emitted by telegram.py.
        """
        if self.out_queue is None:
            raise RuntimeError("AudioLoop is not running; call run() before appending audio.")

        if not isinstance(chunk, np.ndarray):
            chunk = np.asarray(chunk)
        if chunk.dtype != np.int16:
            chunk = chunk.astype(np.int16)

        await self.out_queue.put({"data": chunk.tobytes(), "mime_type": "audio/pcm"})

    async def _send_audio(self) -> None:
        assert self.session is not None
        assert self.out_queue is not None
        try:
            while True:
                msg = await self.out_queue.get()
                await self.session.send_realtime_input(audio=msg)
        except asyncio.CancelledError:
            raise

    async def _receive_audio(self) -> None:
        assert self.session is not None
        try:
            while True:
                turn = self.session.receive()
                async for response in turn:
                    if data := response.data:
                        await self.audio_handler(data)
                    if text := response.text:
                        print(text, end="")
        except asyncio.CancelledError:
            raise

    async def stop(self) -> None:
        """Signal the loop to shut down."""
        self._stop_event.set()

    async def run(self) -> None:
        """Open the live connection and pump audio until stopped."""
        self._stop_event.clear()

        try:
            async with (
                self.client.aio.live.connect(model=self.model, config=self.config)
                as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                self.out_queue = asyncio.Queue(maxsize=5)

                tg.create_task(self._send_audio())
                tg.create_task(self._receive_audio())

                await self._stop_event.wait()
        except asyncio.CancelledError:
            pass
        except ExceptionGroup as eg:
            traceback.print_exception(eg)
        finally:
            self.session = None
            self.out_queue = None
            self._stop_event.set()
