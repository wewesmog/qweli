import psycopg2
from psycopg2.extras import Json, execute_batch
import os
from dotenv import load_dotenv
import requests
import numpy as np
from openai import OpenAI
from typing import List, Dict, Any, Optional, TypedDict, Union

# Load environment variables
load_dotenv()



def get_postgres_connection():
    """
    Establish and return a connection to the PostgreSQL database.
    """
    db_host = os.getenv("DB_HOST", "").strip()
    db_user = os.getenv("DB_USER", "").strip()
    db_password = os.getenv("DB_PASSWORD", "").strip()
    db_port = os.getenv("DB_PORT", "5432").strip()
    db_name = "postgres" #os.getenv("DB_NAME", "postgres").strip()

    conn = psycopg2.connect(
        host=db_host,
        user=db_user,
        password=db_password,
        port=db_port,
        dbname=db_name
    )
    
    return conn
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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

def update_embeddings(conn, table_name: str):
    """
    Update the embeddings for all rows in the specified table.
    
    Args:
    conn: PostgreSQL connection object.
    table_name (str): Name of the table to update embeddings.
    """
    # Drop the existing embeddings column
    conn = get_postgres_connection()
    table_name = "public.world_bank_report"
    with conn.cursor() as cur:
        # Drop the existing embeddings column
        cur.execute(f"ALTER TABLE {table_name} DROP COLUMN IF EXISTS embeddings")
        
        # Recreate the embeddings column with the correct vector size
        cur.execute(f"ALTER TABLE {table_name} ADD COLUMN embeddings vector(1536)")
        
        # Fetch all rows
        cur.execute(f"SELECT id, content FROM {table_name}")
        rows = cur.fetchall()
        
        for row in rows:
            id, content = row
            embedding = call_embedding_api(content)
            
            if embedding:  # Only update if embedding is not empty
                # Format the embedding as a PostgreSQL array
                embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                cur.execute(f"""
                    UPDATE {table_name} 
                    SET embeddings = %s::vector 
                    WHERE id = %s
                """, (embedding_str, id))
                print(f"Updated embeddings for ID: {id}")

        conn.commit()  # Commit all changes to the database

def update_embeddings_for_world_bank_report():
    conn = get_postgres_connection()
    try:
        with conn.cursor() as cur:
            # Fetch all rows that need embeddings
            cur.execute("""
                SELECT id, content
                FROM public.world_bank_report
                WHERE embeddings IS NULL
            """)
            rows = cur.fetchall()

            print(f"Found {len(rows)} rows to update.")

            # Prepare data for batch update
            update_data = []
            for row in rows:
                id, content = row
                embedding = generate_nomic_embedding(content)
                if embedding:
                    update_data.append((embedding, id))
                else:
                    print(f"Failed to generate embedding for id: {id}")

            # Perform batch update
            execute_batch(cur, """
                UPDATE public.world_bank_report
                SET embeddings = %s::vector
                WHERE id = %s
            """, update_data)

            conn.commit()
            print(f"Successfully updated {len(update_data)} rows with embeddings.")

    except Exception as e:
        print(f"An error occurred: {e}")
        conn.rollback()
    finally:
        conn.close()

def main():
    table_name = "public.faq"  # Adjust this if your table name is different

    conn = get_postgres_connection()
    
    try:
        update_embeddings(conn, table_name)
        print("Embeddings update completed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
