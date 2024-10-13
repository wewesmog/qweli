import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from rag_application import create_graph, get_conversation_history
import uvicorn
import logging

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

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_input: ChatInput):
    try:
        # Fetch conversation history
        conversation_history = get_conversation_history(chat_input.user_id)

        # Prepare input for the workflow
        workflow_input = {
            "user_id": chat_input.user_id,
            "session_id": f"session_{chat_input.user_id}",  # You might want to manage sessions differently
            "conversation_id": f"conv_{chat_input.user_id}",  # Same for conversation IDs
            "user_input": chat_input.user_input,
            "conversation_history": conversation_history
        }

        # Run the workflow
        for output in app_workflow.stream(workflow_input):
            logger.info(f"Intermediate output: {output}")

        # Get the final result (last item from the stream)
        final_output = list(app_workflow.stream(workflow_input))[-1]

        # Extract Qweli's response
        qweli_response = final_output.get("qweli_response", "Sorry, I couldn't generate a response.")

        return ChatResponse(qweli_response=qweli_response)

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
