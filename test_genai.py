import os

from google import genai
from google.genai import types
from google.genai.types import Content, Part
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

generate_content_config = genai.types.GenerateContentConfig(
    tools=[
        types.Tool(
            computer_use=types.ComputerUse(
                environment=types.Environment.ENVIRONMENT_BROWSER,
            )
        ),
        # 2. Optional: Custom user-defined functions
        # types.Tool(function_declarations=custom_functions)
    ],
)

PROMPT = "Search for highly rated smart fridges with touchscreen, 2 doors, around 25 cu ft, priced below 4000 dollars on Google Shopping. Create a bulleted list of the 3 cheapest options in the format of name, description, price in an easy-to-read layout."

contents = [Content(role="user", parts=[Part(text=PROMPT)])]

# Generate content with the configured settings
response = client.models.generate_content(
    model="gemini-2.5-computer-use-preview-10-2025",
    contents=contents,  # type: ignore
    config=generate_content_config,
)

# Print the response output
print(response)
