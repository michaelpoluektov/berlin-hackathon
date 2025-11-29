import numpy as np
import asyncio
import os

from dotenv import load_dotenv
from google import genai
from playwright.async_api import async_playwright

from agent import ComputerUseAgent
from telegram import TelegramClient  # if you actually use it
from test_live_audio import AudioLoop


SCREEN_WIDTH = 1440
SCREEN_HEIGHT = 900


async def main_async() -> None:
    # Load environment
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in the environment")

    # GenAI client
    client = genai.Client(api_key=api_key)

    # Browser setup
    print("Initializing browser...")
    playwright = await async_playwright().start()
    browser = await playwright.firefox.launch(headless=False)
    context = await browser.new_context(
        viewport={"width": SCREEN_WIDTH, "height": SCREEN_HEIGHT}
    )
    page = await context.new_page()
    await page.goto("https://google.com")

    # Callbacks for the agent
    async def screenshot_callback(screenshot_bytes: bytes) -> None:
        with open("screenshot.png", "wb") as f:
            f.write(screenshot_bytes)
        print("Saved screenshot to screenshot.png")

    def text_callback(text: str) -> None:
        print(f"AI says: {text}")

    # Agent that controls the browser
    agent = ComputerUseAgent(
        client=client,
        page=page,
        screenshot_callback=screenshot_callback,
        text_callback=text_callback,
        screen_width=SCREEN_WIDTH,
        screen_height=SCREEN_HEIGHT,
        turn_limit=5,
    )

    async def default_audio_callback(samples: np.ndarray) -> None:
        pass

    async def default_message_callback(chat_id: int, message: str) -> None:
        print(f"[message] Chat {chat_id}: {message}")

    # Audio loop that will call the agent when needed
    audio_loop = AudioLoop(execute_agent_callback=agent.execute_task)
    chat_idx = 0

    async def agent_called(chat_id: int) -> None:
        nonlocal chat_idx
        chat_idx = chat_id

    # Instantiate client with callbacks
    client = TelegramClient(
        audio_callback=default_audio_callback,
        message_callback=default_message_callback,
        agent_called=agent_called,
    )

    # OPTIONAL: set up Telegram if you need it
    # telegram_client = TelegramClient(...)
    # await telegram_client.connect()

    from pathlib import Path

    def bytes_to_url(screenshot_bytes: bytes) -> str:
        path = Path("screenshot.png")  # or any other path you like
        path.write_bytes(screenshot_bytes)  # save the bytes

        return str(path.resolve())

    # Run long-lived things as background tasks
    audio_task = asyncio.create_task(audio_loop.run(), name="audio_loop")
    client_task = asyncio.create_task(client.start(), name="telegram client")

    async def cb(s):
        await client.show_image(chat_idx, bytes_to_url(s))

    agent.set_screenshot_callback(cb)

    # If you have more long-running coroutines, create tasks for them too:
    # telegram_task = asyncio.create_task(telegram_client.run(), name="telegram")

    try:
        # Keep the process alive while background tasks run.
        # If you have several tasks, you can use asyncio.gather([...]) or TaskGroup.
        await asyncio.gather(audio_task, client_task)

        # await asyncio.gather(audio_task, telegram_task)
    finally:
        # Make sure we clean up properly
        await browser.close()
        await playwright.stop()


def main() -> None:
    # Single entry point that owns the event loop
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
