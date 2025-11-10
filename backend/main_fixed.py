import os
import deep_translator
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import requests
from deep_translator import GoogleTranslator
from deep_translator.exceptions import NotValidPayload, RequestError
from typing import List
import html
import sys
import time
import asyncio

# Import chatbot module (now in same directory)
try:
    from chatbot import router as chatbot_router
    CHATBOT_AVAILABLE = True
except ImportError:
    CHATBOT_AVAILABLE = False

app = FastAPI(title="Potato Disease Detection API", version="2.0.0")

# Include chatbot routes if available
if CHATBOT_AVAILABLE:
    app.include_router(chatbot_router)

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:5500",
    "http://localhost:5501",
    "http://127.0.0.1:5500",
    "http://127.0.0.1:5501"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TF_SERVING_ENDPOINT = "http://localhost:8501/v1/models/potatoes_model:predict"

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

class TranslationRequest(BaseModel):
    texts: List[str]
    target: str


def unescape_translations(translations, originals):
    return [html.unescape(item) if isinstance(item, str) else original for item, original in zip(translations, originals)]

@app.post("/translate")
async def translate_text(payload: TranslationRequest):
    if not payload.texts:
        return {"translations": []}
    target_lang = payload.target.lower() if payload.target else "en"

    # If English requested, return original strings
    if target_lang == "en":
        return {"translations": payload.texts}
    
    texts = [text or "" for text in payload.texts]

    try:
        # Try Google Translator with retry logic
        translator = GoogleTranslator(source="auto", target=target_lang)
        translations = []
        
        # Translate in smaller batches to avoid rate limiting
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                batch_translations = translator.translate_batch(batch)
                translations.extend(batch_translations)
                # Add delay between batches to avoid rate limiting
                if i + batch_size < len(texts):
                    await asyncio.sleep(0.2)
            except Exception as batch_error:
                # If batch fails, translate individually
                for text in batch:
                    try:
                        translation = translator.translate(text) if text else text
                        translations.append(translation)
                        await asyncio.sleep(0.1)  # Small delay between individual requests
                    except:
                        translations.append(text)  # Fallback to original
        
        translations = unescape_translations(translations, payload.texts)
        return {"translations": translations}
        
    except Exception as e:
        # If all translation fails, return original texts with error message
        print(f"Translation error: {str(e)}")
        return {
            "translations": payload.texts,
            "error": "Translation service temporarily unavailable. Showing original text.",
            "success": False
        }

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    """
    CRITICAL FIX: Properly preprocess image to match model training pipeline
    The model was trained with Rescaling(1./255) layer, so we need [0, 1] range
    """
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((256, 256))
    
    # Convert to numpy array
    img_array = np.array(image, dtype=np.float32)
    
    # *** CRITICAL FIX: Normalize to [0, 1] range ***
    img_array = img_array / 255.0
    
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0) 

    
    json_data = {"instances": img_batch.tolist()}

    try:
    
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
