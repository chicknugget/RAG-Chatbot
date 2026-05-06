import chromadb
from sentence_transformers import SentenceTransformer

from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="./chroma_db")
msg_collection = client.get_collection("messages")

def retrieve(query, top_k=5):
    query_embedding = model.encode(query).tolist()
    
    results = msg_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    retrieved = []
    for i in range(len(results['documents'][0])):
        retrieved.append({
            'text': results['documents'][0][i],
            'metadata': results['metadatas'][0][i],
            'distance': results['distances'][0][i]
        })
    
    return retrieved


groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def answer_question(query, top_k=5):
    retrieved = retrieve(query, top_k=top_k)
    
    context = "\n".join([f"- {r['text']}" for r in retrieved])
    
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "user",
            "content": f"""Answer the question based on the conversation excerpts below.
            
            Context:
            {context}

            Question: {query}

            Answer concisely based only on the context provided."""
        }]
    )
    
    return {
        "answer": response.choices[0].message.content,
        "sources": retrieved
    }

if __name__ == "__main__":
    result = answer_question("What do people say about Portland?")
    print("Answer:", result['answer'])
    print("\nSources used:")
    for s in result['sources']:
        print(f"  - {s['text']}")
