from google.cloud import aiplatform_v1
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value

PROJECT_ID = "finmate-service"
LOCATION = "global"
MODEL_ID = "gemini-2.5-flash"

client = aiplatform_v1.PredictionServiceClient()

endpoint = (
    f"projects/{PROJECT_ID}/locations/{LOCATION}"
    f"/publishers/google/models/{MODEL_ID}"
)

instance = {
    "contents": [
        {
            "role": "user",
            "parts": [{"text": "Reply with exactly: Gemini is LIVE"}]
        }
    ]
}

instances = [json_format.ParseDict(instance, Value())]

response = client.predict(endpoint=endpoint, instances=instances)

print(response.predictions)
