import pickle
from classifier import IntentClassifier
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

chroma_client = PersistentClient(path="chroma_db")
msg_collection = chroma_client.get_collection("messages")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
clf = IntentClassifier()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

EMOTIONAL_KEYWORDS = [
    "love", "hate", "sad", "happy", "angry", "scared", "lonely",
    "excited", "anxious", "frustrated", "proud", "ashamed", "miss",
    "hurt", "afraid", "cry", "laugh", "worried", "stressed", "joy"
]

def emotional_weight(text):
    text_lower = text.lower()
    score = sum(1 for word in EMOTIONAL_KEYWORDS if word in text_lower)
    return score

def recency_score(metadata, total_messages):
    idx = metadata.get('index', 0)
    return idx / total_messages if total_messages > 0 else 0

def score_chunks(chunks, total_messages, recency_weight=0.6, emotion_weight=0.4):
    scored = []
    for chunk in chunks:
        r_score = recency_score(chunk['metadata'], total_messages)
        e_score = emotional_weight(chunk['text'])
        e_score_norm = min(e_score / 5, 1.0)
        final_score = (recency_weight * r_score) + (emotion_weight * e_score_norm)
        scored.append({**chunk, 'score': round(final_score, 3)})
    
    return sorted(scored, key=lambda x: x['score'], reverse=True)

def detect_contradictions(chunks):

    positive = {"close", "love", "happy", "together", "great", "good"}
    negative = {"apart", "hate", "sad", "distant", "bad", "not", "never", "haven't"}
    
    contradictions = []
    for i in range(len(chunks)):
        for j in range(i+1, len(chunks)):
            words_i = set(chunks[i]['text'].lower().split())
            words_j = set(chunks[j]['text'].lower().split())
            
            i_positive = bool(words_i & positive)
            i_negative = bool(words_i & negative)
            j_positive = bool(words_j & positive)
            j_negative = bool(words_j & negative)
            
            if (i_positive and j_negative) or (i_negative and j_positive):
                contradictions.append((i, j))
    
    return contradictions

def resolve(query, top_k=6):
    query_embedding = embed_model.encode(query).tolist()
    results = msg_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    chunks = [
        {
            'text': results['documents'][0][i],
            'metadata': results['metadatas'][0][i]
        }
        for i in range(len(results['documents'][0]))
    ]
    
    total_messages = msg_collection.count()
    
    scored_chunks = score_chunks(chunks, total_messages)
    
    contradictions = detect_contradictions(scored_chunks)
    
    context = "\n".join([
        f"[Score: {c['score']}] {c['text']}"
        for c in scored_chunks
    ])
    
    contradiction_note = ""
    if contradictions:
        contradiction_note = f"\nNote: {len(contradictions)} potential contradiction(s) detected between chunks."
    
    # step 5: generate merged answer
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "user",
            "content": f"""You are analyzing conversation history to answer a question.
Chunks are ranked by recency and emotional weight (higher score = more recent/emotional).
{contradiction_note}

Conversation chunks:
{context}

Question: {query}

Instructions:
- Prioritize higher scored chunks
- If contradictions exist, mention that the user's views changed over time
- Give a merged coherent answer

Answer:"""
        }]
    )
    
    return {
        "answer": response.choices[0].message.content,
        "chunks_used": scored_chunks,
        "contradictions_found": len(contradictions),
        "contradiction_pairs": contradictions
    }


if __name__ == "__main__":
    result = resolve("Did I mention anything about my sister?")
    print("=== RESOLVED ANSWER ===")
    print(result['answer'])
    print(f"\nContradictions found: {result['contradictions_found']}")
    print("\nChunks ranked by score:")
    for c in result['chunks_used']:
        print(f"  [{c['score']}] {c['text'][:60]}")