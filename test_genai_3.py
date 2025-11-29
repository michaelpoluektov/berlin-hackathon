import asyncio
import os
from itertools import count
from typing import Callable, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.types import Content, Part
from playwright.async_api import Page, async_playwright


class ComputerUseAgent:
    def __init__(
        self,
        client: genai.Client,
        page: Page,
        screenshot_callback: Optional[Callable[[bytes], None]] = None,
        text_callback: Optional[Callable[[str], None]] = None,
        screen_width: int = 1440,
        screen_height: int = 900,
        turn_limit: int = 5,
    ) -> None:
        self.client = client
        self.page = page
        self.screenshot_callback = screenshot_callback
        self.text_callback = text_callback
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.turn_limit = turn_limit

        self.config = types.GenerateContentConfig(
            tools=[
                types.Tool(
                    computer_use=types.ComputerUse(
                        environment=types.Environment.ENVIRONMENT_BROWSER
                    )
                )
            ],
            thinking_config=types.ThinkingConfig(include_thoughts=True),
        )

    @staticmethod
    def denormalize_x(x: int, screen_width: int) -> int:
        """Convert normalized x coordinate (0-1000) to actual pixel coordinate."""
        return int(x / 1000 * screen_width)

    @staticmethod
    def denormalize_y(y: int, screen_height: int) -> int:
        """Convert normalized y coordinate (0-1000) to actual pixel coordinate."""
        return int(y / 1000 * screen_height)

    async def _take_screenshot(self) -> bytes:
        """Take a screenshot and invoke the screenshot callback."""
        screenshot_bytes = await self.page.screenshot(type="png")
        if self.screenshot_callback is not None:
            self.screenshot_callback(screenshot_bytes)
        return screenshot_bytes

    async def _execute_function_calls(
        self,
        candidate,
    ):
        results = []
        function_calls = [
            part.function_call
            for part in candidate.content.parts
            if getattr(part, "function_call", None)
        ]

        for function_call in function_calls:
            action_result = {}
            fname = function_call.name
            args = function_call.args
            print(f"  -> Executing: {fname}")

            try:
                if fname == "open_web_browser":
                    # Already open
                    pass
                elif fname == "click_at":
                    actual_x = self.denormalize_x(args["x"], self.screen_width)
                    actual_y = self.denormalize_y(args["y"], self.screen_height)
                    await self.page.mouse.click(actual_x, actual_y)
                elif fname == "type_text_at":
                    actual_x = self.denormalize_x(args["x"], self.screen_width)
                    actual_y = self.denormalize_y(args["y"], self.screen_height)
                    text = args["text"]
                    press_enter = args.get("press_enter", False)

                    await self.page.mouse.click(actual_x, actual_y)
                    # Simple clear (Command+A, Backspace for Mac)
                    await self.page.keyboard.press("Meta+A")
                    await self.page.keyboard.press("Backspace")
                    await self.page.keyboard.type(text)
                    if press_enter:
                        await self.page.keyboard.press("Enter")
                else:
                    print(f"Warning: Unimplemented or custom function {fname}")

                # Wait for potential navigations/renders
                await self.page.wait_for_load_state(timeout=5000)

            except Exception as e:
                print(f"Error executing {fname}: {e}")
                action_result = {"error": str(e)}

            results.append((fname, action_result))

        return results

    async def _get_function_responses(self, results) -> list[types.FunctionResponse]:
        screenshot_bytes = await self._take_screenshot()
        current_url = self.page.url
        function_responses: list[types.FunctionResponse] = []

        for name, result in results:
            response_data = {"url": current_url, "safety_acknowledgement": "true"}
            response_data.update(result)
            function_responses.append(
                types.FunctionResponse(
                    name=name,
                    response=response_data,
                    parts=[
                        types.FunctionResponsePart(
                            inline_data=types.FunctionResponseBlob(
                                mime_type="image/png", data=screenshot_bytes
                            )
                        )
                    ],
                )
            )
        return function_responses

    def _handle_text_from_model(self, candidate) -> None:
        """Extract all textual parts from the model response and call the callback."""
        if candidate.content is None or candidate.content.parts is None:
            return

        for part in candidate.content.parts:
            text = getattr(part, "text", None)
            if text:
                if self.text_callback is not None:
                    self.text_callback(text)

    async def execute_task(self, user_prompt: str) -> None:
        """Run the agent loop for the given user task."""
        print(f"Goal: {user_prompt}")

        # Initial screenshot (callback invoked inside)
        initial_screenshot = await self._take_screenshot()

        contents = [
            Content(
                role="user",
                parts=[
                    Part(text=user_prompt),
                    Part.from_bytes(data=initial_screenshot, mime_type="image/png"),
                ],
            )
        ]

        for i in range(self.turn_limit):
            print(f"\n--- Turn {i + 1} ---")
            print("Thinking...")
            response = await self.client.aio.models.generate_content(
                model="gemini-2.5-computer-use-preview-10-2025",
                contents=contents,  # type: ignore
                config=self.config,
            )

            assert response.candidates is not None
            candidate = response.candidates[0]
            assert candidate.content is not None
            assert candidate.content.parts is not None

            # Handle all text emitted by the model
            self._handle_text_from_model(candidate)

            contents.append(candidate.content)

            has_function_calls = any(
                getattr(part, "function_call", None) for part in candidate.content.parts
            )
            if not has_function_calls:
                # No more actions; model is done
                print("Agent finished.")
                break

            print("Executing actions...")
            results = await self._execute_function_calls(candidate)

            print("Capturing state...")
            function_responses = await self._get_function_responses(results)

            contents.append(
                Content(
                    role="user",
                    parts=[Part(function_response=fr) for fr in function_responses],
                )
            )


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

    # Callbacks
    screenshot_dir = "screenshots"
    os.makedirs(screenshot_dir, exist_ok=True)
    screenshot_counter = count(1)

    def screenshot_callback(screenshot_bytes: bytes) -> None:
        idx = next(screenshot_counter)
        filename = os.path.join(screenshot_dir, f"screenshot_{idx:03d}.png")
        with open(filename, "wb") as f:
            f.write(screenshot_bytes)
        print(f"Saved screenshot to {filename}")

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
