import os
from pathlib import Path
from typing import Awaitable, Callable, Optional

import numpy as np
from dotenv import load_dotenv
from pytgcalls import PyTgCalls, filters, idle
from pytgcalls import filters as fl
from pytgcalls.types import (
    ChatUpdate,
    Device,
    Direction,
    ExternalMedia,
    GroupCallConfig,
    MediaStream,
    RecordStream,
    StreamEnded,
    StreamFrames,
    Update,
)
from pytgcalls.types.raw import AudioParameters
from telethon import TelegramClient as TelethonClient, events

load_dotenv()

SAMPLE_MEDIA = str(Path.home() / "Downloads" / "sample-15s.wav")
SAMPLE_IMAGE = str(Path.home() / "Downloads" / "newplot.png")

AUDIO_PARAMETERS = AudioParameters(bitrate=48000, channels=2)

AudioCallback = Callable[[int, np.ndarray], Awaitable[None]]
MessageCallback = Callable[[int, str], Awaitable[None]]


class TelegramClient:
    def __init__(
        self,
        audio_callback: AudioCallback,
        message_callback: MessageCallback,
    ):
        self.app = TelethonClient(
            "py-tgcalls",
            api_id=int(os.environ["TELEGRAM_API_ID"]),
            api_hash=os.environ["TELEGRAM_API_HASH"],
        )
        self.call_py = PyTgCalls(self.app)

        self.audio_callback: AudioCallback = audio_callback
        self.message_callback: MessageCallback = message_callback

        self._register_handlers()

    async def _start_recording(
        self,
        chat_id: int,
        *,
        enable_video: bool,
        invite_hash: Optional[str] = None,
    ) -> None:
        stream = RecordStream(
            audio=True,
            audio_parameters=AUDIO_PARAMETERS,
            camera=enable_video,
            screen=enable_video,
        )

        config = (
            GroupCallConfig(invite_hash=invite_hash, auto_start=False)
            if enable_video
            else None
        )

        await self.call_py.record(chat_id, stream, config=config)

    # Public async API:

    async def play_audio(self, chat_id: int, media_url: Optional[str] = None) -> None:
        """Play audio into the call."""
        await self.call_py.play(
            chat_id,
            MediaStream(
                media_url or SAMPLE_MEDIA, video_flags=MediaStream.Flags.IGNORE
            ),
        )

    async def show_image(self, chat_id: int, image_url: str) -> None:
        """Display a static image in the call."""
        is_group = chat_id < 0
        config = GroupCallConfig(auto_start=True) if is_group else None
        await self.call_py.play(
            chat_id,
            MediaStream(
                image_url,
                audio_flags=MediaStream.Flags.IGNORE,
            ),
            config=config,
        )

    async def send_chat_message(self, chat_id: int, message: str) -> None:
        """Send a text message to a chat."""
        await self.app.send_message(chat_id, message)

    def start(self) -> None:
        """Start py-tgcalls (and underlying Telethon client) and block."""
        self.call_py.start()  # type: ignore
        print("[*] Running. Call this account from another Telegram client to test.")
        idle()

    def _register_handlers(self) -> None:
        # === Telethon message handlers ===

        @self.app.on(events.NewMessage())
        async def generic_message_handler(event):
            if self.message_callback is not None:
                await self.message_callback(event.chat_id, event.raw_text)

        # === PyTgCalls update handlers ===

        @self.call_py.on_update(filters.chat_update(ChatUpdate.Status.INCOMING_CALL))
        async def incoming_call_handler(_: PyTgCalls, update: Update):
            chat_id = update.chat_id
            print(f"[+] Incoming call from {chat_id}, answering...")

            await self.call_py.play(
                chat_id,
                MediaStream(ExternalMedia.AUDIO, AUDIO_PARAMETERS),
            )
            await self._start_recording(chat_id, enable_video=False)

            print(f"[+] Call answered and recording started for chat {chat_id}")

        @self.call_py.on_update(
            filters.stream_frame(Direction.INCOMING, Device.MICROPHONE)
        )
        async def on_audio_chunk(_: PyTgCalls, update: StreamFrames):
            if not update.frames:
                return

            frame_bytes = update.frames[0].frame
            samples = np.frombuffer(frame_bytes, dtype=np.int16)
            if sum(samples) == 0:
                return

            # Call user-provided audio callback
            await self.audio_callback(update.chat_id, samples)

        @self.call_py.on_update(
            filters.stream_frame(Direction.INCOMING, Device.CAMERA | Device.SCREEN)
        )
        async def on_video_frame(_: PyTgCalls, update: StreamFrames):
            if not update.frames:
                return

            frame = update.frames[0]
            print(f"[video] Chat {update.chat_id} frame size={len(frame.frame)} bytes")

        @self.call_py.on_update(fl.stream_end())
        async def stream_end_handler(_: PyTgCalls, update: StreamEnded):
            print(f"Stream ended in {update.chat_id}", update)


if __name__ == "__main__":

    async def default_audio_callback(chat_id: int, samples: np.ndarray) -> None:
        print(samples[:5])

    async def default_message_callback(chat_id: int, message: str) -> None:
        print(f"[message] Chat {chat_id}: {message}")

    client = TelegramClient(
        audio_callback=default_audio_callback,
        message_callback=default_message_callback,
    )
    client.start()
