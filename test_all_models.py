import os
from google import genai

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\keys\finmate-service-90fb14600045.json"

# All possible models to test
models_to_test = {
    "Gemini": [
        "gemini-1.5-flash-002",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-2.0-flash-exp",
        "gemini-1.0-pro",
    ],
    "Claude": [
        "claude-3-5-sonnet@20240620",
        "claude-3-opus@20240229",
        "claude-3-sonnet@20240229",
        "claude-3-haiku@20240307",
    ],
    "Llama": [
        "llama-3.2-90b-vision-instruct-maas",
        "llama-3.1-405b-instruct-maas",
        "llama-3.1-70b-instruct-maas",
    ],
}

locations_to_test = ["us-central1", "us-east5"]

print("Testing ALL available models across regions...\n")

for category, models in models_to_test.items():
    print(f"\n{'='*50}")
    print(f"{category} Models:")
    print(f"{'='*50}")
    
    for model_name in models:
        found = False
        for location in locations_to_test:
            try:
                client = genai.Client(vertexai=True, project="finmate-service", location=location)
                response = client.models.generate_content(
                    model=model_name,
                    contents="Say 'OK'"
                )
                print(f"✅ {model_name} ({location}) - AVAILABLE")
                found = True
                break
            except Exception as e:
                continue
        
        if not found:
            print(f"❌ {model_name} - NOT AVAILABLE in any region")

print("\n\nTest complete!")
