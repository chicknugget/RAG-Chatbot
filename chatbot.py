import pickle
import json
import os
from classifier import IntentClassifier
from resolver import resolve
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
clf = IntentClassifier()

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

try:
    with open('drift_timeline.pkl', 'rb') as f:
        drift_timeline = pickle.load(f)
except:
    drift_timeline = []


def get_persona_context():
    if not personas:
        return "No persona data available."
    all_habits, all_traits, all_facts = [], [], []
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


def get_drift_context():
    if not drift_timeline:
        return "No drift data available."
    drifts = [
        f"Day {d['day']}: {', '.join(d['traits'])} (trigger: {d['trigger']})"
        for d in drift_timeline if d['drift_detected']
    ]
    return "\n".join(drifts) if drifts else "No significant drift detected."


def answer(query):
    intent = clf.classify(query)

    if intent['intent'] == 'emotional-support':
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{
                "role": "user",
                "content": f"The user seems to need emotional support. Respond warmly and empathetically to: {query}"
            }]
        )
        return f"[Intent: emotional-support]\n{response.choices[0].message.content}"

    elif intent['intent'] in ['reminder', 'action-item']:
        return f"[Intent: {intent['intent']}] Got it! I've noted: '{query}'"

    else:
        persona_context = json.dumps(get_persona_context(), indent=2)
        drift_context = get_drift_context()

        resolved = resolve(query)

        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{
                "role": "user",
                "content": f"""You are a helpful assistant that answers questions about users based on their conversations.

Persona data:
{persona_context}

Persona drift timeline (days where mood/tone changed):
{drift_context}

Retrieved conversation context:
{resolved['answer']}

Question: {query}

Answer based only on the data provided. If contradictions exist in the data, mention that the user's views changed over time."""
            }]
        )

        contradiction_note = ""
        if resolved['contradictions_found'] > 0:
            contradiction_note = f"\n[Note: {resolved['contradictions_found']} contradiction(s) detected in retrieved chunks]"

        return f"[Intent: {intent['intent']}]{contradiction_note}\n{response.choices[0].message.content}"


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