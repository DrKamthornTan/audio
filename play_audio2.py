import os
import openai
import streamlit as st
import chromadb
from pydub import AudioSegment

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configure ChromaDB client
CHROMA_PATH = "chroma"
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="my_collection")

# Function to perform semantic search
def semantic_search_query(query, size=5):
    results = collection.query(
        query_texts=[query],
        n_results=size
    )
    return results

# Function to generate human-like responses using OpenAI API
def generate_human_like_response(query, results):
    response_texts = []
    for i in range(len(results['documents'])):
        doc_text = results['documents'][i]
        prompt = f"Based on the following information:\n\n{doc_text}\n\nQ: {query}\nA:"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300  # Increase the max_tokens to ensure more complete responses
        )
        response_texts.append(response['choices'][0]['message']['content'].strip())
    return response_texts

# Streamlit application
st.set_page_config(page_title="ChromaDB Semantic Search H", layout="wide")
st.title("AI Suggestions for Your Stress Management")
st.write("Click play song to calm down....")

# Auto-play the audio1.mp3 file when the browser is opened
audio_path = "audio1.mp3"
if os.path.exists(audio_path):
    st.audio(audio_path, format="audio/mp3", start_time=0)
else:
    st.error("audio1.mp3 file not found. Please ensure the file exists in the directory.")

# User input for query
query = "suggest stress management for high stress level"

# Perform semantic search
model_name = "sentence-transformers/all-mpnet-base-v2"
results = semantic_search_query(query)

# Display search results
if results['documents']:
    human_like_responses = generate_human_like_response(query, results)
    for i in range(len(results['documents'])):
        doc_id = results['ids'][i]
        doc_text = results['documents'][i]
        metadata = results['metadatas'][i]
        filename = metadata[0]['filename'] if isinstance(metadata, list) and metadata else 'Unknown'
        st.write(f"**Human-like Response:**")
        st.write(human_like_responses[i])
        st.write("---")
else:
    st.write("No documents found.")
