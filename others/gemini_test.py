from google import genai
import os

client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
)

response_1 = client.models.generate_content(
    model='gemini-2.0-flash',
    contents='Hello',
)

print(response_1.text)