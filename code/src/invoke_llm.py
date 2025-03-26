import google.generativeai as genai
import os
import storage_db as sd
from dotenv import load_dotenv
import json
# from google import genai
# Load environment variables
load_dotenv()
MODEL_CONFIG = {
    'model': 'gemini-2.0-flash',
    'temperature': 0.7,
    'max_output_tokens': 100
}


# Set API Key
# genai.configure(api_key='')need to add Key
genai.configure(api_key='')



def generate_resolution_steps(query_text, embedding_model, collection, top_n=2):
    """Finds similar incidents & generates resolution steps using Gemini."""
    try:
        # 1️⃣ **Embed query**
        query_embedding = embedding_model.encode(query_text).tolist()
        
        # 2️⃣ **Retrieve top matching incidents**
        results = collection.query(query_embeddings=[query_embedding], n_results=top_n)
        print(results)
        # Extract incidents & resolution notes
        top_matches = results["documents"][0]
        resolutions = [json.loads(doc).get("Resolution Notes", "No resolution found") for doc in top_matches]
        
        # 3️⃣ **Prepare prompt for Gemini**
        prompt = f"""You are an IT support AI assistant. 
        A user has reported the following issue: **{query_text}**
        
        Based on past incidents, here are some relevant resolution notes:
        {resolutions}
        
        Please generate a structured resolution guide to resolve this issue.
        Provide clear step-by-step instructions.
        """

        # 4️⃣ **Call Gemini API**
        # response = genai.generate_text(prompt)
        model = genai.GenerativeModel(MODEL_CONFIG['model'])
        chat = model.start_chat(history=[])
        response = chat.send_message(query_text)
        
        # 5️⃣ **Display generated resolution steps**
        # print("🔧 Suggested Resolution Steps:")
        # print(response.text)
        return response.text
    except Exception as e:
        return "I'm having trouble generating a response right now."

# Example Query
# generate_resolution_steps("Email not syncing with Outlook 365")
