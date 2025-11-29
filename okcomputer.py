import asyncio
import os

from dotenv import load_dotenv
from google import genai
from playwright.async_api import async_playwright

from agent import ComputerUseAgent


async def main_async():
    load_dotenv()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    SCREEN_WIDTH = 1440
    SCREEN_HEIGHT = 900

    print("Initializing browser...")
    playwright = await async_playwright().start()
    browser = await playwright.firefox.launch(headless=False)
    context = await browser.new_context(
        viewport={"width": SCREEN_WIDTH, "height": SCREEN_HEIGHT}
    )
    page = await context.new_page()
    await page.goto("https://google.com")

    def screenshot_callback(screenshot_bytes: bytes) -> None:
        with open("screenshot.png", "wb") as f:
            f.write(screenshot_bytes)
        print("Saved screenshot to screenshot.png")

    def text_callback(text: str) -> None:
        print(f"AI says: {text}")

    agent = ComputerUseAgent(
        client=client,
        page=page,
        screenshot_callback=screenshot_callback,
        text_callback=text_callback,
        screen_width=SCREEN_WIDTH,
        screen_height=SCREEN_HEIGHT,
        turn_limit=5,
    )

    await agent.execute_task("Search for cat photos")

    await browser.close()
    await playwright.stop()


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
