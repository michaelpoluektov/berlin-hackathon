import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.types import Content, Part
from playwright.sync_api import sync_playwright

load_dotenv()
print(os.environ["GEMINI_API_KEY"])
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
# Constants for screen dimensions
SCREEN_WIDTH = 1440
SCREEN_HEIGHT = 900

# Setup Playwright
print("Initializing browser...")
playwright = sync_playwright().start()
browser = playwright.firefox.launch(headless=False)
context = browser.new_context(viewport={"width": SCREEN_WIDTH, "height": SCREEN_HEIGHT})
page = context.new_page()


def denormalize_x(x: int, screen_width: int) -> int:
    """Convert normalized x coordinate (0-1000) to actual pixel coordinate."""
    return int(x / 1000 * screen_width)


def denormalize_y(y: int, screen_height: int) -> int:
    """Convert normalized y coordinate (0-1000) to actual pixel coordinate."""
    return int(y / 1000 * screen_height)


def execute_function_calls(candidate, page, screen_width, screen_height):
    results = []
    function_calls = []
    for part in candidate.content.parts:
        if part.function_call:
            function_calls.append(part.function_call)

    for function_call in function_calls:
        action_result = {}
        fname = function_call.name
        args = function_call.args
        print(f"  -> Executing: {fname}")

        try:
            if fname == "open_web_browser":
                pass  # Already open
            elif fname == "click_at":
                actual_x = denormalize_x(args["x"], screen_width)
                actual_y = denormalize_y(args["y"], screen_height)
                page.mouse.click(actual_x, actual_y)
            elif fname == "type_text_at":
                actual_x = denormalize_x(args["x"], screen_width)
                actual_y = denormalize_y(args["y"], screen_height)
                text = args["text"]
                press_enter = args.get("press_enter", False)

                page.mouse.click(actual_x, actual_y)
                # Simple clear (Command+A, Backspace for Mac)
                page.keyboard.press("Meta+A")
                page.keyboard.press("Backspace")
                page.keyboard.type(text)
                if press_enter:
                    page.keyboard.press("Enter")
            else:
                print(f"Warning: Unimplemented or custom function {fname}")

            # Wait for potential navigations/renders
            page.wait_for_load_state(timeout=5000)
            time.sleep(1)

        except Exception as e:
            print(f"Error executing {fname}: {e}")
            action_result = {"error": str(e)}

        results.append((fname, action_result))

    return results


def get_function_responses(page, results):
    screenshot_bytes = page.screenshot(type="png")
    current_url = page.url
    function_responses = []
    for name, result in results:
        response_data = {"url": current_url}
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


try:
    # Go to initial page
    page.goto("https://google.com")

    # Configure the model (From Step 1)
    config = types.GenerateContentConfig(
        tools=[
            types.Tool(
                computer_use=types.ComputerUse(
                    environment=types.Environment.ENVIRONMENT_BROWSER
                )
            )
        ],
        thinking_config=types.ThinkingConfig(include_thoughts=True),
    )

    # Initialize history
    initial_screenshot = page.screenshot(type="png")
    USER_PROMPT = "Search for cat photos"
    print(f"Goal: {USER_PROMPT}")

    contents = [
        Content(
            role="user",
            parts=[
                Part(text=USER_PROMPT),
                Part.from_bytes(data=initial_screenshot, mime_type="image/png"),
            ],
        )
    ]

    # Agent Loop
    turn_limit = 5
    for i in range(turn_limit):
        print(f"\n--- Turn {i + 1} ---")
        print("Thinking...")
        response = client.models.generate_content(
            model="gemini-2.5-computer-use-preview-10-2025",
            contents=contents,  # type: ignore
            config=config,
        )

        assert response.candidates is not None
        candidate = response.candidates[0]
        assert candidate.content is not None
        assert candidate.content.parts is not None
        contents.append(candidate.content)

        has_function_calls = any(part.function_call for part in candidate.content.parts)
        if not has_function_calls:
            text_response = " ".join(
                [part.text for part in candidate.content.parts if part.text]
            )
            print("Agent finished:", text_response)
            break

        print("Executing actions...")
        results = execute_function_calls(candidate, page, SCREEN_WIDTH, SCREEN_HEIGHT)

        print("Capturing state...")
        function_responses = get_function_responses(page, results)

        contents.append(
            Content(
                role="user",
                parts=[Part(function_response=fr) for fr in function_responses],
            )
        )

finally:
    print("\nClosing browser...")
    browser.close()
    playwright.stop()
