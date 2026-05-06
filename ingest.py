import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle


def parse_conversation(raw_text, start_index = 0):
    messages = []
    parts = re.split(r'(User \d+:)', raw_text)
    parts = [part.strip() for part in parts if part.strip()]

    msg_index = start_index
    for i in range(0, len(parts)-1, 2):
        speaker = parts[i].rstrip(':')
        text = parts[i+1]
        messages.append({
            'speaker': speaker,
            'text': text,
            'index': msg_index
        })
        msg_index += 1
    return messages, msg_index



model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_messages(all_messages):
    
    texts = [m['text'] for m in all_messages]
    embeddings = model.encode(texts, show_progress_bar=True)
    for i, msg in enumerate(all_messages):
        msg['embedding'] = embeddings[i]
    return all_messages


def detect_topic_changes(all_messages, threshold=0.1):
   
    breakpoints = [0] 

    for i in range(1, len(all_messages)):
        sim = cosine_similarity(
            all_messages[i]['embedding'].reshape(1, -1),
            all_messages[i-1]['embedding'].reshape(1, -1)
        )
        if sim < threshold:
            breakpoints.append(i)
    
    return breakpoints

if __name__ == "__main__":
    df = pd.read_csv('conversations.csv')
    # df = df.head(200)
    all_messages = []
    global_index = 0

    for _, row in df.iterrows():
        conversation_text = row.iloc[0]
        messages, global_index = parse_conversation(conversation_text, start_index=global_index)
        all_messages.extend(messages)

    all_messages = embed_messages(all_messages)
    breakpoints = detect_topic_changes(all_messages)
    with open('messages.pkl', 'wb') as f:
        pickle.dump(all_messages, f)

    with open('breakpoints.pkl', 'wb') as f:
        pickle.dump(breakpoints, f)

    print("Saved!")

    