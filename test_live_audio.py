import os
import asyncio
import traceback

import pyaudio

from google import genai
from google.genai import types

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-2.5-flash-native-audio-preview-09-2025"

client = genai.Client(
    http_options={"api_version": "v1beta"},
    api_key=os.environ.get("GEMINI_API_KEY"),
)

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

pya = pyaudio.PyAudio()


class AudioLoop:
    def __init__(self, audio_handler=None):
        """
        audio_handler: async callable taking a single bytes argument (PCM data).
        If None, default handler plays audio back via PyAudio.
        """
        self.session = None
        self.mic_stream = None
        self.output_stream = None

        self.out_queue = None  # mic → model

        self.audio_handler = audio_handler or self._default_audio_handler

    async def _default_audio_handler(self, data: bytes):
        """Default audio handler: play audio back using PyAudio."""
        if self.output_stream is None:
            self.output_stream = await asyncio.to_thread(
                pya.open,
                format=FORMAT,
                channels=CHANNELS,
                rate=RECEIVE_SAMPLE_RATE,
                output=True,
            )
        await asyncio.to_thread(self.output_stream.write, data)

    async def send_text(self):
        """Send text messages using send_client_content."""
        while True:
            text = await asyncio.to_thread(input, "message > ")
            if text.lower() == "q":
                break

            user_text = text or "."

            # Turn-based text → model
            await self.session.send_client_content(
                turns=types.Content(
                    role="user",
                    parts=[types.Part(text=user_text)],
                ),
                turn_complete=True,
            )

    async def listen_audio(self):
        """Read from microphone and enqueue audio chunks to be sent to the model."""
        mic_info = pya.get_default_input_device_info()
        self.mic_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )

        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}

        while True:
            data = await asyncio.to_thread(self.mic_stream.read, CHUNK_SIZE, **kwargs)
            # print(type(data))
            # print(np.frombuffer(data, dtype=np.int16))
            # Queue raw PCM for realtime input
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def send_audio(self):
        """Send queued audio to the model using send_realtime_input."""
        while True:
            msg = await self.out_queue.get()
            # msg is a dict: {"data": bytes, "mime_type": "audio/pcm"}
            await self.session.send_realtime_input(audio=msg)

    async def receive_audio(self):
        """
        Read from the websocket and pass PCM chunks to the audio handler.
        response.data aggregates audio bytes from inline_data parts.
        """
        assert self.session is not None
        while True:
            turn = self.session.receive()
            async for response in turn:
                if data := response.data:
                    # Audio (24kHz PCM) from model
                    await self.audio_handler(data)
                if text := response.text:
                    # Any text the model emits (system messages, etc.)
                    print(text, end="")

    async def run(self):
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                self.out_queue = asyncio.Queue(maxsize=5)

                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.listen_audio())
                tg.create_task(self.send_audio())
                tg.create_task(self.receive_audio())

                await send_text_task
                raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            pass
        except ExceptionGroup as EG:
            traceback.print_exception(EG)
        finally:
            try:
                if self.mic_stream is not None:
                    self.mic_stream.close()
            except Exception:
                pass
            try:
                if self.output_stream is not None:
                    self.output_stream.close()
            except Exception:
                pass


if __name__ == "__main__":
    # You can override audio handling by passing your own async handler:
    # async def my_handler(data: bytes): ...
    # main = AudioLoop(audio_handler=my_handler)
    main = AudioLoop()
    asyncio.run(main.run())
