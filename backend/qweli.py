import os
import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime, timezone
import json
from typing import List, Dict, Any, Optional, TypedDict, Union
from dotenv import load_dotenv
import google.generativeai as genai
import psycopg2
import requests
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolExecutor
from psycopg2.extras import Json
from google.generativeai import GenerativeModel
import numpy as np
from openai import OpenAI


# Load environment variables
load_dotenv()

# Initialize clients
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY is not set in the environment variables")

genai.configure(api_key=google_api_key)


# Setup logger

def setup_logger(log_level=logging.INFO):
    main_project_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    log_folder = os.path.join(main_project_directory, "logs")
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    log_file_path = os.path.join(log_folder, f"query_state_log_{datetime.now().strftime('%Y-%m-%d')}.log")
    logger = logging.getLogger("QueryStateLogger")
    if not logger.handlers:
        logger.setLevel(log_level)
        file_handler = TimedRotatingFileHandler(log_file_path, when="midnight", interval=1, backupCount=30)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger

logger = setup_logger()

logger = setup_logger()

# Define the main state structure
class MainState(TypedDict):
    user_id: str
    session_id: str
    conversation_id: str
    user_input: str
    selected_tools: List[Dict[str, str]]
    conversation_history: List[Dict[str, str]]
    documents: List[Dict[str, Any]]
    tavily_results: List[Dict[str, Any]]
    qweli_response: str
    comprehensive_query: str
    similar_questions: List[str]

# Helper functions
def call_llm_api1(messages: List[Dict[str, str]]) -> str:
    try:
        # Create the model
        model = genai.GenerativeModel('gemini-pro')
        
        # Prepare the chat messages
        chat = model.start_chat(history=[])
        
        # Add each message to the chat
        for message in messages:
            if message['role'] == 'user':
                chat.send_message(message['content'])
            elif message['role'] == 'system':
                # For system messages, we'll prepend it to the first user message
                next_user_message = next((m for m in messages if m['role'] == 'user'), None)
                if next_user_message:
                    next_user_message['content'] = f"{message['content']}\n\n{next_user_message['content']}"
        
        # Get the last response
        last_response = chat.last
        return last_response.text
    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}")
        return ""

def call_llm_api1(messages: List[Dict[str, str]]) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json"
    }
    data = {
        #"model": "openai/gpt-3.5-turbo-0613",  # You can change this to another model if needed
        "model": "meta-llama/llama-3.1-8b-instruct",
        "messages": messages
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        logger.error(f"Error calling OpenRouter API: {e}")
        return ""
# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#  gpt-3.5-turbo #gpt-4o-2024-08-06 #GPT-4o mini
def call_embedding_api(text: str, model: str = "text-embedding-ada-002") -> List[float]:
    """
    Make a call to the OpenAI API for text embeddings.
    
    :param text: The text to embed
    :param model: The model to use for the embedding
    :return: The embedding as a list of floats
    """
    try:
        response = client.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        # Log the error and re-raise it to be handled by the calling function
        print(f"Error in OpenAI Embedding API call: {e}")

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#  gpt-3.5-turbo #gpt-4o-2024-08-06 #GPT-4o mini
def call_llm_api(messages: List[Dict[str, str]], model: str = "gpt-4o-mini", max_tokens: int = 2000, temperature: float = 0.3) -> Any:
    """
    Make a call to the OpenAI API for chat completions.
    
    :param messages: List of message dictionaries to send to the API
    :param model: The model to use for the API call
    :param max_tokens: Maximum number of tokens in the response
    :param temperature: Controls randomness in the response
    :return: The API response
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        # Log the error and re-raise it to be handled by the calling function
        print(f"Error in OpenAI API call: {e}")
        raise
        raise

import re
import json

def call_llm_api1(messages: List[Dict[str, str]]) -> str:
    url = "http://localhost:11434/api/chat"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": "phi3:mini-4k", #"llama3.2:1b", #"mistral:latest",
        "messages": messages,
        "stream": False  # Set this to False to get a complete response
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        response_json = response.json()
        print(f"Ollama API response: {response_json}")
        if 'message' in response_json and 'content' in response_json['message']:
            content = response_json['message']['content']
            # Try to extract JSON-like content
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    extracted_json = json.loads(json_match.group())
                    return json.dumps(extracted_json)
                except json.JSONDecodeError:
                    print(f"Failed to parse extracted JSON: {json_match.group()}")
                    pass
            # If JSON extraction fails, return the raw content
            print(f"Returning raw content: {content}")
            return content
        else:
            print(f"Unexpected response structure from Ollama API: {response_json}")
            return ""
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from Ollama API: {e}")
        print(f"Raw response: {response.text}")
        return ""
    except requests.RequestException as e:
        print(f"Error calling Ollama API: {e}")
        return ""

def get_postgres_connection(table_name: str):
    """
    Establish and return a connection to the PostgreSQL database.
    
    :param table_name: Name of the table to interact with
    :return: Connection object
    """
    db_host = os.getenv("DB_HOST", "").strip()
    db_user = os.getenv("DB_USER", "").strip()
    db_password = 'wes@1234' #os.getenv("DB_PASSWORD", "").strip()
    db_port = os.getenv("DB_PORT", "5432").strip()
    db_name = "postgres" #os.getenv("DB_NAME", "postgres").strip()

    try:
        conn = psycopg2.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            port=db_port,
            dbname=db_name
        )
        logger.info(f"Successfully connected to database: {db_name}")
        return conn
    except psycopg2.OperationalError as e:
        logger.error(f"Unable to connect to the database. Error: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise


#Main Functions
def get_conversation_history(session_id: str, limit: int = 500) -> List[Dict[str, str]]:
    logger.info(f"Fetching conversation history for user: {session_id}")
    conn = get_postgres_connection("respond_to_human")  # Changed this line
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT state->>'qweli_response' as assistant, state->>'user_input' as human
                FROM respond_to_human
                WHERE state->>'session_id' = %s
                ORDER BY id DESC
                LIMIT %s
            """, (session_id, limit))
            
            rows = cur.fetchall()
            
            history = []
            for row in rows:
                if row[1]:  # human input
                    history.append(f"human: {row[1]}")
                if row[0]:  # assistant response
                    history.append(f"AI: {row[0]}")
            
            logger.info(f"Retrieved {len(history)} conversation entries for session: {session_id}")
            return list(reversed(history))  # Reverse to get chronological order
    except Exception as e:
        logger.error(f"Error fetching conversation history: {e}")
        return []
    finally:
        conn.close()


def handle_user_input(state: MainState) -> MainState:
    logger.info(f"Handling user input: {state['user_input']}")
    conversation_history = state.get("conversation_history", [])
    user_query = state.get("user_input", "")
    clarified_query = state.get("understood_query", "")

    prompt = f"""
    Given the user input and conversation history, follow these steps:

    User input: {user_query}
    Conversation history: {conversation_history}
    Clarified query: {clarified_query}

    Tools available:
    1. **RAG**: Use this tool to answer specific queries related to information, services, or products.
    2. **Chitchat**: Use this tool for casual or social interactions, such as greetings, small talk, or personal questions.

    1. Carefully analyze the user input and conversation history to select the appropriate tool. You can only select one tool.
    2. Determine if the input is a **chitchat** (e.g., greetings, personal questions like "What is your name?", "How are you?", or other social interactions). For this, select the **chitchat** tool.
    3. If the input is related to specific services, products, or information (including details about loans, banking services, or financial products), select the **RAG** tool.
    4. Chitchat examples are strictly limited to: "Hello", "How are you?", "What is your name?", "What's the weather like?", "Tell me a joke".
    5. If it is a chitchat query, respond appropriately and update the parameters field with the response.
        Examples of chitchat responses include: "Hello! How can I assist you today?", "I'm Qweli, your virtual assistant. How may I help you?", "Nice to meet you! How can I assist you today?", "I'm here to help you with any questions you have. What's on your mind?"
    6. Use **chitchat** to ask for clarifications only if the user's query is completely unrelated to any banking or financial topics.
    7. When in doubt, always select RAG. Only select chitchat for the most obvious and basic social interactions.
    8. Ensure that you provide the required parameters & justification for the tool selected.

    Expected format if the tool selected is RAG:
    {{
        "response_type": "tool_selection",
        "selected_tool": {{
            "name": "RAG",
            "reason": "User query is about specific information or products",
            "parameters": {{
                "user_query": "{user_query}"
            }}
        }}
    }}


    Expected format if the tool selected is chitchat:
    {{
        "response_type": "tool_selection",
        "selected_tool": {{
            "name": "chitchat",
            "reason": "User query is a chitchat",
            "parameters": {{
                "response": The reponse to the user chitchat.
            }}
        }}
    }}
    """

    messages = [
        {"role": "system", "content": "You are a helpful assistant that handles user queries and selects appropriate tools."},
        {"role": "user", "content": prompt}
    ]

    try:
        llm_response = call_llm_api(messages)
        logger.debug(f"LLM response: {llm_response}")
        
        llm_response = json.loads(llm_response)
        
        print(f"LLM response JSON: {llm_response}")
        # Ensure only one tool is selected
        selected_tool = llm_response.get("selected_tool", {})
        if not selected_tool:
            raise ValueError("No tool selected from the LLM response.")
        
        # Store the selected tool in the state
        state["selected_tool"] = {
            "name": selected_tool["name"],
            "reason": selected_tool["reason"],
            "parameters": selected_tool.get("parameters", {})
        }

        
        # Update conversation history with user input
        state["conversation_history"].append({"role": "user", "content": user_query})

        # Check if the selected tool is chitchat
        if selected_tool["name"].lower() == "chitchat":
            # Explicitly update qweli_response for chitchat
            state["qweli_response"] = selected_tool["parameters"]["response"]
            # Update conversation history with the chitchat response
            state["conversation_history"].append({"role": "assistant", "content": state["qweli_response"]})
            
            # Establish PostgreSQL connection
            table_name = "public.respond_to_human"
            conn = get_postgres_connection(table_name)
            
            # Insert the state into the database
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO respond_to_human (state)
                        VALUES (%s)
                    """, (Json(state),))
                
                conn.commit()
                logger.info("Successfully inserted state into respond_to_human table")
            except Exception as e:
                logger.error(f"Error inserting state into database: {e}")
                conn.rollback()
            finally:
                conn.close()

        # Log the selected tool and handle RAG appropriately
        logger.info(f"Selected tool: {state['selected_tool']['name']}")
        if selected_tool["name"].lower() == "rag":
            logger.info(f"Tool RAG selected. No qweli_response update needed.")

    except Exception as e:
        logger.error(f"Error in handle_user_input: {e}")
        state["error"] = str(e)
        # Only update qweli_response for non-RAG errors
        if not state.get('selected_tool') or state['selected_tool']['name'].lower() != 'rag':
            state["qweli_response"] = "I'm sorry, but I encountered an error while processing your request."

    return state


def refine_query_for_RAG(state: MainState) -> MainState:
    logger.info("Refining query for RAG")
    conversation_history = state.get("conversation_history", [])
    user_query = state.get("user_input", "")

    prompt = f"""
    Given the conversation history and user input, follow these steps:

    User input: {user_query}
    Conversation history: {conversation_history}

    1. Review the user input and the context from the conversation history.
    2. If the user query is vague or incomplete, use the conversation history to complete the query.
    3. After determining the comprehensive query, generate 3 semantically similar questions to help with wider recall.

    Expected format:
    {{
        "comprehensive_query": "The refined or original query here",
        "similar_questions": [
            "Question 1",
            "Question 2",
            "Question 3"
        ]
    }}
    """

    messages = [
        {"role": "system", "content": "You are a helpful assistant in Swiftcash bank that refines user queries based on conversation history and generates similar questions for wider recall."},
        {"role": "user", "content": prompt}
    ]

    try:
        llm_response = call_llm_api(messages)
        llm_response = json.loads(llm_response)
        
        state["comprehensive_query"] = llm_response.get("comprehensive_query", user_query)
        state["similar_questions"] = llm_response.get("similar_questions", [])

        logger.info(f"Refined query: {state['comprehensive_query']}")
        logger.info(f"Similar questions: {state['similar_questions']}")

    except Exception as e:
        logger.error(f"Error in refine_query_for_RAG: {e}")
        state["error"] = str(e)

    return state

def retrieval(state: MainState) -> MainState:
    logger.info("Starting document retrieval")
    comprehensive_query = state.get("comprehensive_query", ""); 
    similar_questions = state.get("similar_questions", [])

    logger.info(f"Comprehensive query: {comprehensive_query}")
    logger.info(f"Similar questions: {similar_questions}")

    all_documents = []
    table_name = "public.world_bank_report"
    
    try:
        conn = get_postgres_connection(table_name)
        logger.info("Successfully connected to the database")

        queries = [comprehensive_query] + similar_questions
        
        for query in queries:
            try:
                embedding = call_embedding_api(query)  # Use your existing embedding function
                
                if not embedding:
                    logger.warning(f"Failed to generate embedding for query: {query}")
                    continue

                embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT id, 
                               json_build_object(
                                   'title', 'Swiftcash Bank Documentation',
                                   'page', "metadata"
                               )::jsonb AS metadata,
                               content,
                               1 - (embeddings <=> %s::vector) AS similarity
                        FROM public.swiftcash
                        WHERE 1 - (embeddings <=> %s::vector) >= %s
                        ORDER BY similarity DESC
                        LIMIT %s
                    """, (embedding_str, embedding_str, 0.7, 3))
                    
                    
                    rows = cur.fetchall()
                    
                    for row in rows:
                        document = {
                            "id": row[0],
                            "metadata": row[1],
                            "content": row[2],
                            "similarity": float(row[3])
                        }
                        all_documents.append(document)

            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")
                logger.error(f"Full error details: {str(e)}")
                # Continue with the next query

    except Exception as e:
        logger.error(f"Error during document retrieval: {e}")
        logger.error(f"Full error details: {str(e)}")
        # If there's any error, we'll just return an empty list for documents
    
    finally:
        if 'conn' in locals() and conn:
            conn.close()

    # Deduplicate documents based on id
    seen_ids = set()
    unique_documents = []
    for doc in all_documents:
        if doc['id'] not in seen_ids:
            seen_ids.add(doc['id'])
            unique_documents.append(doc)

    logger.info(f"Retrieved {len(unique_documents)} unique documents")
    state['documents'] = unique_documents

    return state

def check_document_relevance(state: MainState) -> MainState:
    logger.info("Checking document relevance")
    comprehensive_query = state.get("comprehensive_query", "")
    documents = state.get("documents", [])

    for document in documents:
        doc_content = document["content"]
        
        prompt = f"""
        Given the comprehensive query:
        Query: "{comprehensive_query}"
        
        And the following document content:
        Document: "{doc_content}"

        Analyze whether this document provides useful information related to the comprehensive query. Consider the following:
        1. Does the document contribute to answering at least 20% of the comprehensive query?
        2. Does the document address any main points or subtopics of the query?
        3. Is the content relevant to the query topic?
        4. Does it provide any insights or details that could be helpful in formulating a response?

        Even if the document doesn't fully answer the query, respond with "Yes" if it covers at least 20% of the query or provides significant relevant information. Otherwise, respond with "No".

        After your analysis, respond with only "Yes" or "No".
        """

        messages = [
            {"role": "system", "content": "You are a helpful assistant in Swiftcash bank tasked with analyzing document relevance."},
            {"role": "user", "content": prompt}
        ]

        try:
            llm_response = call_llm_api(messages)
            document["answers_query"] = "Yes" if "yes" in llm_response.strip().lower() else "No"
        except Exception as e:
            logger.error(f"Error analyzing document ID {document.get('id', 'unknown')}: {e}")
            document["answers_query"] = "Yes"  # Default to 'Yes' on error

    logger.info(f"Relevant documents after check: {len([doc for doc in state['documents'] if doc['answers_query'] == 'Yes'])}")
    return state
    



def qweli_agent_RAG(state: MainState) -> MainState:
    logger.info("Qweli agent processing RAG results")
    relevant_documents = state.get("documents", [])
    relevant_documents = [doc for doc in relevant_documents if doc.get("answers_query") == "Yes"]
    
    if not relevant_documents:
        state["qweli_response"] = "I apologize, but after searching our documents, I couldn't find a relevant answer to your query. Could you please rephrase your question or provide more details?"
        state["suggested_question"] = ""
        return state

    prompt = f"""
    User Query: "{state['user_input']}"

    Conversation History:
    {state['conversation_history']}

    Relevant Documents:
    {json.dumps(relevant_documents, indent=2)}

    Instructions:
    1. Using the information from the relevant documents and considering the conversation history, 
       provide a professional & comprehensive answer to the user's query as well as a suggested next question. 
    2. Cite your sources based on the metadata provided. Use the file name or URL as the source.
    3. If multiple sources are used, cite each one separately, however check and ensure that there is no repetition of sources.
    4. Do not start your response with "Based on the information provided", or "Answer*" : Just give the answer.
    5. Do not make up any information.
    6. If the documents do not contain sufficient information to answer the query, state that clearly.
    7. Present your answer in a well-formatted markdown style.
    8. After providing the answer, suggest a logical follow-up question that a customer might ask based on the information provided.
    9. Format your response strictly as a JSON object with two fields: "answer" and "suggested_question".

    Example format:
    {{
        "answer": "Your comprehensive answer here, with proper citations and markdown formatting.",
        "suggested_question": "A logical follow-up question based on the provided answer."
    }}
    """

    messages = [
        {"role": "system", "content": "You are Qweli, a professional assistant in Swiftcash bank, a single source of truth for all staff. You are tasked with answering user queries based on provided documents and conversation history & provide a suggested next question. Your responses should be accurate, concise, well-formatted in markdown style, and include source citations."},
        {"role": "user", "content": prompt}
    ]

    try:
        llm_response = call_llm_api(messages)
        print(f"LLM response: {llm_response}")
        response_data = json.loads(llm_response.strip())
        
        state["qweli_response"] = response_data["answer"]
        state["suggested_question"] = response_data["suggested_question"]
        
        logger.info(f"Qweli response generated: {state['qweli_response'][:100]}...")
        logger.info(f"Suggested question generated: {state['suggested_question']}")
    except Exception as e:
        logger.error(f"Error in Qweli agent: {e}")
        state["qweli_response"] = "I apologize, but I encountered an error while processing your query. Please try again later."
        state["suggested_question"] = ""
    
    # Update conversation history
    state["conversation_history"].append({"role": "assistant", "content": state["qweli_response"]})

    print(f"State after Qweli agent: {state}")
    # Establish PostgreSQL connection
    table_name = "public.respond_to_human"
    conn = get_postgres_connection(table_name)  # Changed this line
    
    # Insert the state into the database
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO respond_to_human (state)
                VALUES (%s)
            """, (Json(state),))
        
        conn.commit()
        logger.info("Successfully inserted state into respond_to_human table")
    except Exception as e:
        logger.error(f"Error inserting state into database: {e}")
        conn.rollback()
    finally:
        conn.close()
    return state


def process_query(user_input: str, session_id: str, user_id: str, conversation_id: str) -> Dict[str, Any]:
    # Initialize the state
    state = {
        "user_id": user_id,
        "session_id": session_id,
        "conversation_id": conversation_id,
        "user_input": user_input,
        "conversation_history": [],
        "documents": [],
        "qweli_response": "",
        "suggested_question": "",
    }

    # Handle user input
    state = handle_user_input(state)

    # Check if it's chitchat
    if state.get("selected_tool", {}).get("name") == "chitchat":
        # For chitchat, we already have the response in the state
        return {
            "qweli_agent_RAG": {
                "qweli_response": state.get("qweli_response", ""),
                "suggested_question": ""
            }
        }

    # If it's not chitchat, proceed with RAG
    if state.get("selected_tool", {}).get("name") == "RAG":
        # Refine query for RAG
        state = refine_query_for_RAG(state)

        # Retrieval
        state = retrieval(state)

        # Check document relevance
        state = check_document_relevance(state)

        # Qweli agent RAG
        state = qweli_agent_RAG(state)

    # Prepare the final output
    final_output = {
        "qweli_agent_RAG": {
            "qweli_response": state.get("qweli_response", ""),
            "suggested_question": state.get("suggested_question", "What else would you like to know?")
        }
    }

    return final_output