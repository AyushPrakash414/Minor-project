"""
AI Chatbot Backend - Separate module for Ollama integration
"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
import requests

router = APIRouter(prefix="/chat", tags=["chatbot"])

# Store conversation context
conversation_context = {}

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    disease_context: Optional[dict] = None
    language: str = "en"

class ClearChatRequest(BaseModel):
    session_id: str = "default"


@router.post("")
async def chat(payload: ChatRequest):
    """
    Chat endpoint that integrates with Ollama Llama 3.2
    """
    OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
    
    # Build context-aware system prompt
    system_prompt = """You are an expert agricultural AI assistant specializing in potato plant diseases. 
You provide practical, actionable advice to farmers in a friendly and supportive manner.
Keep responses concise (2-3 sentences for simple questions, longer for complex topics).
Always prioritize farmer safety and crop health."""
    
    # Add disease context if available
    if payload.disease_context:
        disease = payload.disease_context.get("class", "Unknown")
        confidence = payload.disease_context.get("confidence", 0)
        system_prompt += f"\n\nCurrent Diagnosis: {disease} with {confidence:.1%} confidence."
        system_prompt += "\nProvide advice specific to this diagnosis when relevant."
    
    # Language instruction
    if payload.language == "hi":
        system_prompt += "\n\nIMPORTANT: Respond in Hindi (Devanagari script)."
    
    # Get conversation history
    session_id = payload.session_id
    if session_id not in conversation_context:
        conversation_context[session_id] = []
    
    # Build conversation history
    history = conversation_context[session_id]
    
    # Construct full prompt with history
    full_prompt = system_prompt + "\n\n"
    for msg in history[-6:]:  # Last 3 exchanges (6 messages)
        full_prompt += f"{msg['role']}: {msg['content']}\n"
    full_prompt += f"User: {payload.message}\nAssistant:"
    
    try:
        # Call Ollama API
        ollama_response = requests.post(
            OLLAMA_ENDPOINT,
            json={
                "model": "llama3.2",
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 300
                }
            },
            timeout=30
        )
        
        if ollama_response.status_code == 200:
            response_data = ollama_response.json()
            assistant_message = response_data.get("response", "").strip()
            
            # Update conversation history
            conversation_context[session_id].append({"role": "User", "content": payload.message})
            conversation_context[session_id].append({"role": "Assistant", "content": assistant_message})
            
            # Keep only last 10 messages (5 exchanges)
            if len(conversation_context[session_id]) > 10:
                conversation_context[session_id] = conversation_context[session_id][-10:]
            
            return {
                "response": assistant_message,
                "session_id": session_id,
                "success": True
            }
        else:
            return {
                "response": "I'm having trouble connecting to my knowledge base. Please make sure Ollama is running.",
                "error": f"Ollama returned status {ollama_response.status_code}",
                "success": False
            }
    
    except requests.exceptions.ConnectionError:
        return {
            "response": "❌ Cannot connect to Ollama. Please install and start Ollama first.\n\nQuick Start:\n1. Download from https://ollama.ai\n2. Run: ollama pull llama3.2\n3. Ollama runs automatically on port 11434",
            "error": "Connection to Ollama failed",
            "success": False
        }
    except Exception as e:
        return {
            "response": "I encountered an error. Please try again.",
            "error": str(e),
            "success": False
        }


@router.post("/clear")
async def clear_chat(payload: ClearChatRequest):
    """Clear conversation history for a session"""
    if payload.session_id in conversation_context:
        conversation_context[payload.session_id] = []
    return {"message": "Conversation cleared", "session_id": payload.session_id}
