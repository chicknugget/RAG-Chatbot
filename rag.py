from dotenv import load_dotenv
import os
import pandas as pd
import pickle
import time

from groq import Groq

load_dotenv()

with open('messages.pkl', 'rb') as f:
    all_messages = pickle.load(f)

with open('breakpoints.pkl', 'rb') as f:
    breakpoints = pickle.load(f)

client = Groq(api_key=os.getenv('GROQ_API_KEY'))

def summarize_segment(messages_slice, retries=3):
    conversation = "\n".join([f"{m['speaker']}: {m['text']}" for m in messages_slice])
    
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                timeout=30, 
                messages=[{
                    "role": "user",
                    "content": f"Summarize the topic of this conversation segment in 2-3 sentences:\n\n{conversation}"
                }]
            )
            return response.choices[0].message.content
        
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt) 
    
    return "Summary unavailable" 


def generate_topic_summaries(all_messages, breakpoints, checkpoints_every=100):
    if os.path.exists('summaries.pkl'):
        with open('summaries.pkl', 'rb') as f:
            summaries = pickle.load(f)
        start_from = len(summaries)
        print(f"Resuming from topic {start_from}")
    else:
        summaries = []
        start_from = 0
    
    for i in range(start_from,len(breakpoints)):
        start = breakpoints[i]
        end = breakpoints[i+1] if i+1 < len(breakpoints) else len(all_messages)
        
        messages_slice = all_messages[start:end]
        
        summary = summarize_segment(messages_slice)
        summaries.append({
            "topic_id": i,
            "start": start,
            "end": end,
            "summary": summary
        })
        if i % checkpoints_every == 0:
            with open('summaries.pkl', 'wb') as f:
                pickle.dump(summaries, f)
            print(f"Progress: {i}/{len(breakpoints)} topics done")
        time.sleep(0.1)
    
    with open('summaries.pkl', 'wb') as f:
        pickle.dump(summaries, f)
    
    return summaries

if __name__ == "__main__":
    test_summaries = generate_topic_summaries(all_messages, breakpoints)
    print(test_summaries)