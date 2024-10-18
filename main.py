import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from qweli import create_graph, get_postgres_connection
import uvicorn
import logging
import psycopg2
from psycopg2.extras import RealDictCursor

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI()

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

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_input: ChatInput):
    logger.info(f"Received chat request for user: {chat_input.user_id}")
    
    try:
        # Fetch conversation history
        conversation_history = get_conversation_history(chat_input.user_id)

        # Prepare input for the workflow
        workflow_input = {
            "user_id": chat_input.user_id,
            "session_id": f"session_{chat_input.user_id}",
            "conversation_id": f"conv_{chat_input.user_id}",
            "user_input": chat_input.user_input,
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

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
