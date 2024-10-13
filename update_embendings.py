import psycopg2
import requests


import google.generativeai as genai
GOOGLE_API_KEY = "AIzaSyAl2S13gS3AK5set7lu4ppeQ-jUSyYnGWE"
genai.configure(api_key=GOOGLE_API_KEY)

def get_postgres_connection(dbname: str, table_name: str):
    """
    Establish and return a connection to the PostgreSQL database.
    
    :param dbname: Name of the database to connect to
    :param table_name: Name of the table to interact with
    :return: Connection object
    """
    db_host = "www.sema-ai.com"
    db_user = "postgres"
    db_password = "wes@1234"

    conn = psycopg2.connect(
        host=db_host,
        user=db_user,
        password=db_password,
        dbname=dbname
    )
    
    return conn

def get_gemini_embedding(text: str) -> list:
    """
    Generate an embedding for the given text using the Gemini API.
    
    Args:
    text (str): The input text to embed.
    
    Returns:
    list: The embedding vector as a list of floats.
    """
    model = 'models/embedding-001'
    try:
        embedding = genai.embed_content(model=model,
                                        content=text,
                                        task_type="retrieval_document")
        return embedding['embedding']  # This is already a list
    except Exception as e:
        print(f"Error generating Gemini embedding: {e}")
        return []

def update_embeddings(conn, table_name: str):
    """
    Update the embeddings for all rows in the specified table.
    
    Args:
    conn: PostgreSQL connection object.
    table_name (str): Name of the table to update embeddings.
    """
    with conn.cursor() as cur:
        cur.execute(f"SELECT id, content FROM {table_name}")
        rows = cur.fetchall()
        
        for row in rows:
            id, content = row
            embedding = get_gemini_embedding(content)
            
            if embedding:  # Only update if embedding is not empty
                # Format the embedding as a PostgreSQL array
                embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                cur.execute(f"""
                    UPDATE {table_name} 
                    SET embeddings = %s 
                    WHERE id = %s
                """, (embedding_str, id))
                print(f"Updated embeddings for ID: {id}")

        conn.commit()  # Commit all changes to the database

def main():
    dbname = "postgres"
    table_name = "public.faq"

    conn = get_postgres_connection(dbname, table_name)
    
    try:
        update_embeddings(conn, table_name)
    finally:
        conn.close()

if __name__ == "__main__":
    main()
