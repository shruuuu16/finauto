import os
from google import genai

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\keys\finmate-service-90fb14600045.json"

# Test different locations and models
test_configs = [
    # Gemini models (Google)
    ("us-central1", "gemini-1.5-flash-002"),
    ("us-central1", "gemini-1.5-flash"),
    ("us-central1", "gemini-1.5-pro"),
    ("us-central1", "gemini-2.0-flash-exp"),
    ("us-central1", "gemini-1.0-pro"),
    
    # Claude models (Anthropic)
    ("us-east5", "claude-3-5-sonnet@20240620"),
    ("us-east5", "claude-3-opus@20240229"),
    ("us-east5", "claude-3-sonnet@20240229"),
    ("us-east5", "claude-3-haiku@20240307"),
    
    # Llama models (Meta)
    ("us-central1", "llama-3.1-405b-instruct-maas"),
    ("us-central1", "llama-3.2-90b-vision-instruct-maas"),
]

print("Testing available models across different locations...\n")

for location, model_name in test_configs:
    try:
        client = genai.Client(vertexai=True, project="finmate-service", location=location)
        response = client.models.generate_content(
            model=model_name,
            contents="Say 'OK'"
        )
        print(f"✅ {model_name} ({location}) - AVAILABLE")
    except Exception as e:
        if "404" in str(e):
            print(f"❌ {model_name} ({location}) - NOT AVAILABLE")
        elif "403" in str(e):
            print(f"🔒 {model_name} ({location}) - NO ACCESS")
        else:
            print(f"⚠️  {model_name} ({location}) - ERROR: {str(e)[:60]}")

print("\n✓ Test complete!")

