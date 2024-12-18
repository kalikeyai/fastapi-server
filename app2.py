import os
import logging
import requests
from typing import Dict
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
bubble_api_url = os.getenv("BUBBLE_API_URL")  # Bubble Data API URL
bubble_api_key = os.getenv("BUBBLE_API_KEY")  # Bubble API Key

if not openrouter_api_key or not bubble_api_url or not bubble_api_key:
    raise ValueError("OPENROUTER_API_KEY, BUBBLE_API_URL, and BUBBLE_API_KEY must be set as environment variables.")

port = int(os.getenv('PORT', '8000'))

# FastAPI instance
app = FastAPI()

# Pydantic Models
class CharacterProfile(BaseModel):
    name: str
    description: str
    personality: str

class QuestionRequest(BaseModel):
    question: str
    user_id: str
    character_name: str

# Character profiles stored in memory
character_profiles: Dict[str, Dict] = {}

# LLM Configuration
def create_llm():
    return ChatOpenAI(
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=openrouter_api_key,
        model_name="openai/gpt-3.5-turbo",
        temperature=0.7
    )

# Bubble API Functions
def fetch_user_context(user_id: str, character_name: str) -> str:
    """Fetch user context from Bubble API."""
    endpoint = f"{bubble_api_url}/obj/user_contexts"  # Bubble API endpoint
    headers = {"Authorization": f"Bearer {bubble_api_key}"}
    params = {"constraints": f"[{{'key':'user_id','constraint_type':'equals','value':'{user_id}'}},"
                             f"{{'key':'character_name','constraint_type':'equals','value':'{character_name}'}}]"}
    response = requests.get(endpoint, headers=headers, params={"constraints": params})
    
    if response.status_code != 200:
        logger.error(f"Error fetching user context: {response.text}")
        raise HTTPException(status_code=500, detail="Failed to fetch user context.")

    results = response.json().get("response", {}).get("results", [])
    if results:
        return results[0].get("context", "")
    return ""

def save_user_context(user_id: str, character_name: str, context: str):
    """Save or update user context to Bubble API."""
    endpoint = f"{bubble_api_url}/obj/user_contexts"
    headers = {"Authorization": f"Bearer {bubble_api_key}", "Content-Type": "application/json"}
    
    data = {
        "user_id": user_id,
        "character_name": character_name,
        "context": context
    }

    # Try saving or updating the context
    response = requests.post(endpoint, headers=headers, json=data)
    if response.status_code not in [200, 201]:
        logger.error(f"Error saving user context: {response.text}")
        raise HTTPException(status_code=500, detail="Failed to save user context.")

# Context Management
def load_user_context(user_id: str, character_name: str) -> ConversationBufferWindowMemory:
    """Load context for a user and character from Bubble API."""
    memory = ConversationBufferWindowMemory(
        k=10, memory_key="chat_history", return_messages=True
    )
    context = fetch_user_context(user_id, character_name)
    if context:
        memory.load_memory_variables({"chat_history": context})
    return memory

# Endpoints
@app.post("/add-character-profile")
async def add_character_profile(profile: CharacterProfile):
    """Add a digital character profile."""
    if profile.name in character_profiles:
        raise HTTPException(status_code=400, detail="Character already exists.")
    character_profiles[profile.name] = {
        "description": profile.description,
        "personality": profile.personality
    }
    logger.info(f"Character '{profile.name}' added.")
    return {"message": f"Character '{profile.name}' added successfully."}

@app.post("/query-character")
async def query_character(request: QuestionRequest):
    """Chat with a character while maintaining user-specific memory."""
    if request.character_name not in character_profiles:
        raise HTTPException(status_code=404, detail="Character not found.")
    
    # Retrieve character and user-specific memory
    character = character_profiles[request.character_name]
    memory = load_user_context(request.user_id, request.character_name)

    # LLM with character personality prompt
    llm = create_llm()
    system_message = (
        f"You are {character['name']}. {character['description']}. "
        f"Your personality: {character['personality']}."
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=None,
        memory=memory
    )

    # Invoke chain
    response = chain.invoke({
        "question": request.question,
        "system_message": system_message
    })

    # Save updated context to Bubble API
    updated_context = memory.load_memory_variables({})["chat_history"]
    save_user_context(request.user_id, request.character_name, updated_context)
    return {"answer": response['answer']}

@app.get("/list-characters")
async def list_characters():
    """List all available characters."""
    return {"characters": list(character_profiles.keys())}

@app.post("/reset-user-context")
async def reset_user_context(user_id: str, character_name: str):
    """Reset chat context for a specific user and character."""
    save_user_context(user_id, character_name, "")  # Clear context in Bubble API
    logger.info(f"Context reset for user '{user_id}' and character '{character_name}'.")
    return {"message": "Chat context reset successfully."}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting the server...")
    uvicorn.run(app, host="0.0.0.0", port=port)
