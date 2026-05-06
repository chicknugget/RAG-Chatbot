import pickle
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
import json
from dotenv import load_dotenv
import os

load_dotenv()

model = SentenceTransformer('all-MiniLM-L6-v2')
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./chroma_db")
msg_collection = chroma_client.get_collection("messages")


with open('breakpoints.pkl', 'rb') as f:
    breakpoints = pickle.load(f)


try:
    with open('summaries.pkl', 'rb') as f:
        summaries = pickle.load(f)
except:
    summaries = []


try:
    with open('personas.pkl', 'rb') as f:
        personas = pickle.load(f)
except:
    personas = []


def retrieve_chunks(query, top_k=5):
    query_embedding = model.encode(query).tolist()
    results = msg_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return [
        {'text': results['documents'][0][i], 'metadata': results['metadatas'][0][i]}
        for i in range(len(results['documents'][0]))
    ]


def retrieve_summaries(query, top_k=3):
    if not summaries:
        return []
    query_lower = query.lower()
    scored = []
    for i, s in enumerate(summaries):
        words = set(query_lower.split())
        summary_words = set(s['summary'].lower().split())
        score = len(words & summary_words)
        if score > 0:
            scored.append((score, i, s)) 
    scored.sort(reverse=True)
    return [s for _, _, s in scored[:top_k]] 


def get_persona_context():
    if not personas:
        return "No persona data available."
    
    all_habits = []
    all_traits = []
    all_facts = []
    
    for p in personas:
        persona = p['persona']
        all_habits.extend(persona.get('habits', []))
        all_traits.extend(persona.get('personality_traits', []))
        all_facts.extend(persona.get('personal_facts', []))
    
    return {
        "habits": list(set(all_habits)),
        "personality_traits": list(set(all_traits)),
        "personal_facts": list(set(all_facts))
    }


def answer(query):
    chunks = retrieve_chunks(query)
    relevant_summaries = retrieve_summaries(query)
    persona = get_persona_context()

    chunk_context = "\n".join([f"- {c['text']}" for c in chunks])
    summary_context = "\n".join([f"- {s['summary']}" for s in relevant_summaries]) or "None available"
    persona_context = json.dumps(persona, indent=2)

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "user",
            "content": f"""You are a helpful assistant that answers questions about users based on their conversations.

Persona data:
{persona_context}

Relevant topic summaries:
{summary_context}

Relevant conversation excerpts:
{chunk_context}

Question: {query}

Answer based only on the data provided above."""
        }]
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print("Chatbot ready! Ask anything about the users.")
    print("Type 'quit' to exit\n")

    while True:
        query = input("You: ").strip()
        if query.lower() == 'quit':
            break
        if not query:
            continue
        print(f"\nBot: {answer(query)}\n")