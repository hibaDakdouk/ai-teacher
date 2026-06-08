import uuid

import chromadb
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """
    Splits the input text into chunks of specified size with optional overlap.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The size of each chunk in characters. Default is 500.
        overlap (int): The number of characters to overlap between chunks. Default is 50.

    Returns:
        list: A list of text chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if overlap < 0:
        raise ValueError("overlap must be a non-negative integer.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size.")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks

def embed_chunks(chunks: list) -> list:
    """
    Placeholder function for embedding text chunks.

    Args:
        chunks (list): A list of text chunks to be embedded.
    Returns:
        list: A list of embedded vectors corresponding to the input chunks. 
    """

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunks  # list of strings
    )

    vectors = [item.embedding for item in response.data]
    return vectors


def store_chunks(chunks: list, embeddings: list) -> None:
    """
    Stores the chunks and their corresponding embeddings in a vector database.

    Args:
        chunks (list): A list of text chunks.
        embeddings (list): A list of embeddings corresponding to each chunk.

    Returns:
        None
    """
    # Create a persistent client — saves to disk so data survives restarts
    client = chromadb.PersistentClient(path="./chroma_db")

    # delete old collections
    try:
        client.delete_collection("documents")
    except:
        pass  # collection didn't exist yet, that's fine
    
    # Get or create a collection — like a table in a regular database
    collection = client.get_or_create_collection(name="documents")

    # generate ids for each chunk
    ids = [str(uuid.uuid4()) for i in range(len(chunks))]

    # Add the chunks and their embeddings to the collection
    collection.add(ids=ids, documents=chunks, embeddings=embeddings)    

def index_document(text:str) -> int:
    """
    Indexes a document by chunking the text, generating embeddings, and storing them in a vector database.

    Args:
        text (str): The input text to be indexed.   
    Returns:
        None    
    """

    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    store_chunks(chunks, embeddings)

    return len(chunks)


def search(query: str, n_results: int = 3) -> list:
    # embed the query (single string, not a list)
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    )
    query_embedding = response.data[0].embedding

    # connect to ChromaDB and get the collection
    chromadb_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chromadb_client.get_collection(name="documents")    
    
    # query the collection
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)

    # return the list of relevant chunks
    return results['documents'][0]  # Since we only send one query at a time, we take the first (and only) result set