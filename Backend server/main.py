from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import requests

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TF Serving REST API endpoint
TF_SERVING_ENDPOINT = "http://localhost:8501/v1/models/potatoes_model:predict"

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((256, 256))
    image = np.array(image)
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Preprocess image
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0) 

    # Prepare JSON for TF Serving
    json_data = {"instances": img_batch.tolist()}

    try:
        # Send request to TF Serving
        response = requests.post(TF_SERVING_ENDPOINT, json=json_data)
        response.raise_for_status()
        prediction = np.array(response.json()["predictions"][0])
    except Exception as e:
        return {"error": str(e)}

    predicted_index = int(np.argmax(prediction))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(np.max(prediction))

    return {
        "class": predicted_class,
        "class_index": predicted_index,
        "confidence": confidence
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
