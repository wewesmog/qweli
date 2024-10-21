import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from backend.qweli import create_graph, get_postgres_connection
import uvicorn
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
import uuid

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

# Create the workflow
rag_workflow = create_graph()
app_workflow = rag_workflow.compile()

class ChatInput(BaseModel):
    user_id: str
    user_input: str

class ChatResponse(BaseModel):
    qweli_response: str

def get_postgres_connection():
    try:
        return psycopg2.connect(
            dbname=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            host=os.getenv("POSTGRES_HOST"),
            port=os.getenv("POSTGRES_PORT")
        )
    except psycopg2.Error as e:
        logger.error(f"Unable to connect to the database: {e}")
        raise

def get_conversation_history(user_id: str, limit: int = 5):
    try:
        conn = get_postgres_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT role, content
                FROM conversation_history
                WHERE user_id = %s
                ORDER BY timestamp DESC
                LIMIT %s
            """, (user_id, limit))
            history = cur.fetchall()
        conn.close()
        return [{"role": entry["role"], "content": entry["content"]} for entry in history]
    except Exception as e:
        logger.error(f"Error fetching conversation history: {e}")
        return []

conversation_histories = {}

@app.post("/start-session")
async def start_session():
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}

@app.post("/api/chat")
async def chat(request: Request):
    data = await request.json()
    user_id = data.get("user_id", "default_user")
    session_id = data.get("session_id")
    user_input = data["user_input"]

    if not session_id:
        return JSONResponse(status_code=400, content={"error": "Session ID is required"})

    conversation_history = conversation_histories.get(session_id, [])
    conversation_history.append({"role": "user", "content": user_input})

    # Use session_id to maintain conversation context
    conversation_history = get_conversation_history(user_id)

    # Prepare input for the workflow
    workflow_input = {
        "user_id": user_id,
        "session_id": session_id,
        "conversation_id": f"conv_{user_id}",
        "user_input": user_input,
        "conversation_history": conversation_history
    }

    # Run the workflow
    final_output = None
    logger.info("Starting workflow stream")
    for output in app_workflow.stream(workflow_input):
        logger.info(f"Intermediate output: {output}")
        final_output = output
    logger.info("Workflow stream completed")

    # Extract Qweli's response
    qweli_response = None
    if final_output and isinstance(final_output, dict):
        # Check all nodes for qweli_response
        for node_output in final_output.values():
            if isinstance(node_output, dict) and 'qweli_response' in node_output:
                qweli_response = node_output['qweli_response']
                break
            
            # If not found in node outputs, check top level
            if qweli_response is None and 'qweli_response' in final_output:
                qweli_response = final_output['qweli_response']

    if qweli_response is None:
        logger.error(f"No qweli_response found in final output. Final output: {final_output}")
        qweli_response = "Sorry, I couldn't generate a response."

    logger.info(f"Final Qweli response: {qweli_response}")

    # Return the response
    return JSONResponse(content={"qweli_response": qweli_response})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
