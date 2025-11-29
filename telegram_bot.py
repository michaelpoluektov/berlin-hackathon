import asyncio
from typing import Awaitable, Callable, Optional

import numpy as np

from audio import AudioLoop
from telegram import TelegramClient

OutgoingAudioCallback = Callable[[int, bytes], Awaitable[None]]
MessageCallback = Callable[[int, str], Awaitable[None]]


class TelegramBot:
    def __init__(
        self,
        *,
        outgoing_audio_callback: OutgoingAudioCallback,
        message_callback: Optional[MessageCallback] = None,
    ):
        self._outgoing_audio_callback = outgoing_audio_callback
        self._audio_loop = AudioLoop(audio_handler=self._on_model_audio)

        self._active_chat_id: Optional[int] = None

        self._telegram_client = TelegramClient(
            audio_callback=self._on_incoming_audio,
            message_callback=message_callback or self._default_message_callback,
            agent_called=self._on_agent_called,
        )

        self._audio_task: Optional[asyncio.Task] = None
        self._telegram_task: Optional[asyncio.Task] = None

    async def _default_message_callback(self, chat_id: int, message: str) -> None:
        print(f"[telegram message] Chat {chat_id}: {message}")

    async def _on_agent_called(self, chat_id: int) -> None:
        # Track which chat we should stream model audio back to.
        self._active_chat_id = chat_id

    async def _on_incoming_audio(self, samples: np.ndarray) -> None:
        # Forward Telegram audio chunks into the model loop.
        # print(samples[:5])
        await self._audio_loop.append_chunk(samples)

    async def _on_model_audio(self, data: bytes) -> None:
        # Send model audio back to Telegram using the provided callback.
        print("bot")
        if self._active_chat_id is None:
            return
        await self._outgoing_audio_callback(self._active_chat_id, data)

    async def start(self) -> None:
        """
        Start both AudioLoop and TelegramClient concurrently.
        This method blocks until either side exits or is cancelled.
        """
        async with asyncio.TaskGroup() as tg:
            self._audio_task = tg.create_task(self._audio_loop.run())
            self._telegram_task = tg.create_task(self._telegram_client.start())

    async def stop(self) -> None:
        """Stop the audio loop; Telegram client is stopped by cancellation/shutdown."""
        await self._audio_loop.stop()


async def main() -> None:
    async def outgoing_audio(chat_id: int, data: bytes) -> None:
        # TODO: Implement using PyTgCalls/Telethon to stream audio back into the call.
        print(f"[outgoing audio] chat={chat_id} bytes={len(data)}")

    bot = TelegramBot(outgoing_audio_callback=outgoing_audio)
    await bot.start()


if __name__ == "__main__":
    asyncio.run(main())
