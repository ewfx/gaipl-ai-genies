import os
import pandas as pd
import numpy as np
import streamlit as st
import storage_db as sd
import invoke_llm as llm
from load_data import load_data
from storage_db import store_data_in_chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from streamlit_chat import message
import uvloop
import asyncio
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Load SentenceTransformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load environment variables
load_dotenv()
st.set_page_config(
    page_title="* Welcome to ECM AI Model *",
    page_icon="🤖",
    layout="centered"
)

if 'user_input' not in st.session_state:
    st.session_state['user_input'] = []
if 'openai_response' not in st.session_state:
    st.session_state['openai_response'] = []

st.markdown("""
<style>
.chat-message {
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 10px;
    max-width: 80%;
}
.user-message {
    background-color: #e6f3ff;
    align-self: flex-end;
    margin-left: auto;
}
.ai-message {
    background-color: #f0f0f0;
    align-self: flex-start;
}
</style>
""", unsafe_allow_html=True)


with st.spinner('Loading dataset...'):
    df = load_data('data/incidents_dataset.csv')
    
#     # Store data in ChromaDB
with st.spinner('Generating embeddings and storing local...'):
    collection = store_data_in_chromadb(df, embedding_model)

def main():
    st.title("🤖 Welcome to ECM AI Chatbot")
    st.write("Chat with an intelligent AI assistant powered by Google's Generative AI")
    
    user_input = st.chat_input("Type your message...")
    if user_input:
        st.session_state['user_input'].append(user_input)        
        # Generate AI Response
        with st.spinner('Generating response...'):
            ai_response = llm.generate_resolution_steps(user_input, embedding_model, collection)

        # Add AI response to chat history
        st.session_state['openai_response'].append(ai_response)

    # Sidebar Options
    st.sidebar.header("🛠️ Chat Settings")
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state['user_input'] = []
        st.session_state['openai_response'] = []

    if st.session_state['user_input']:
        for i in range(len(st.session_state['user_input']) - 1, -1, -1):
            message(st.session_state['user_input'][i], key=str(i),avatar_style="personas", is_user=True)
            message(st.session_state['openai_response'][i], avatar_style="adventurer", is_user=False, key=str(i)+"data_by_user")

if __name__ == "__main__":
    main()