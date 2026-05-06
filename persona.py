import pandas as pd
import json
import pickle
import os
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def extract_persona(conversation_text):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "user",
            "content": f"""Analyze this conversation and extract persona information.
Return ONLY a JSON object with no extra text, no markdown, no backticks.
Every array must contain only plain strings, no nested objects.

{{
  "habits": ["habit1", "habit2"],
  "personal_facts": ["fact1", "fact2"],
  "personality_traits": ["trait1", "trait2"],
  "communication_style": {{
    "tone": "friendly/neutral/formal",
    "message_length": "short/medium/long",
    "emoji_usage": "none/occasional/frequent",
    "other": ""
  }}
}}

Rules:
- Arrays must contain plain strings only, no nested objects
- Only include what is explicitly stated in the conversation
- If a field has no evidence, use an empty array

Conversation:
{conversation_text}"""
        }]
    )
    
    raw = response.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw, "parse_error": True}


def build_personas(csv_path, checkpoint_every=50):
    
    if os.path.exists('personas.pkl'):
        with open('personas.pkl', 'rb') as f:
            personas = pickle.load(f)
        start_from = len(personas)
        print(f"Resuming from conversation {start_from}")
    else:
        personas = []
        start_from = 0

    df = pd.read_csv(csv_path)
    df = df.head(200)

    for i in range(start_from, len(df)):
        conversation_text = df.iloc[i, 0]
        persona = extract_persona(conversation_text)
        personas.append({
            "conversation_id": i,
            "persona": persona
        })

        if i % checkpoint_every == 0:
            with open('personas.pkl', 'wb') as f:
                pickle.dump(personas, f)
            print(f"Progress: {i}/{len(df)} conversations done")

    with open('personas.pkl', 'wb') as f:
        pickle.dump(personas, f)

    return personas


if __name__ == "__main__":
    personas = build_personas('conversations.csv')