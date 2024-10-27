import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from backend.qweli import process_query,get_conversation_history
from backend.initial_questions import initial_questions  # Import the questions
import uvicorn
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
import uuid
import random
import json


# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI()

# CORS configuration
origins = [
    "http://localhost:5173",  # Your frontend URL
    "http://localhost:3000",  # Add this if you're also using port 3000
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatInput(BaseModel):
    user_id: str
    user_input: str

class ChatResponse(BaseModel):
    qweli_response: str
    suggested_question: str

@app.post("/start-session")
async def start_session():
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}

@app.post("/api/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        user_input = data.get("user_input", "")
        session_id = data.get("session_id", "")
        user_id = data.get("user_id", "default_user")
        conversation_id = data.get("conversation_id", f"conv_{user_id}")

        # Process the query
        final_output = process_query(user_input, session_id, user_id, conversation_id)

        logger.info(f"Full final output: {json.dumps(final_output, indent=2)}")

        qweli_response = "Sorry, I couldn't generate a response."
        suggested_question = ""

        if isinstance(final_output, dict) and 'qweli_agent_RAG' in final_output:
            qweli_agent_output = final_output['qweli_agent_RAG']
            qweli_response = qweli_agent_output.get('qweli_response', qweli_response)
            suggested_question = qweli_agent_output.get('suggested_question', "")

        # If suggested_question is missing or empty, pick a random question from initial_questions
        if not suggested_question:
            suggested_question = random.choice(initial_questions)

        logger.info(f"Extracted qweli_response: {qweli_response[:100]}...")
        logger.info(f"Extracted suggested_question: {suggested_question}")

        return JSONResponse(content={
            "qweli_response": qweli_response,
            "suggested_question": suggested_question
        })

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return JSONResponse(content={"error": "An error occurred while processing your request."}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
