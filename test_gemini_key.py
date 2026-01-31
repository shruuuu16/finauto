import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("ERROR: GEMINI_API_KEY not found in environment!")
    exit(1)

print(f"Testing Gemini with API Key: {api_key[:10]}...")

try:
    gemini = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    print(f"Using model: {model}")
    
    response = gemini.models.generate_content(
        model=model,
        contents="Say 'Hello from Gemini API!'"
    )
    print("SUCCESS!")
    print(response.text)
except Exception as e:
    print(f"FAILED: {e}")
