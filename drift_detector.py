import pickle
import json
from classifier import IntentClassifier
import pandas as pd

with open('4/personas.pkl', 'rb') as f:
    personas = pickle.load(f)

clf = IntentClassifier()

def extract_tone(persona):
    cs = persona.get('communication_style', {})
    if isinstance(cs, dict):
        return cs.get('tone', 'unknown')
    elif isinstance(cs, list) and len(cs) > 0:
        return cs[0] 
    return 'unknown'

def extract_traits(persona):
    traits = persona.get('personality_traits', [])
    if isinstance(traits, list):
        return set(traits)
    return set()

def detect_drift(personas):
    timeline = []
    
    for i, p in enumerate(personas):
        day = p['conversation_id'] + 1
        persona = p['persona']
        
        traits = extract_traits(persona)
        tone = extract_tone(persona)
        facts = persona.get('personal_facts', [])
        
        drift = False
        lost_traits = set()
        gained_traits = set()
        trigger = None
        
        if i > 0:
            prev_traits = extract_traits(personas[i-1]['persona'])
            lost_traits = prev_traits - traits
            gained_traits = traits - prev_traits
            
            if len(gained_traits) > 1 or len(lost_traits) > 1:
                drift = True
                trigger = facts[0] if facts else "unknown"
        
        timeline.append({
            "day": day,
            "tone": tone,
            "traits": list(traits),
            "drift_detected": drift,
            "gained_traits": list(gained_traits),
            "lost_traits": list(lost_traits),
            "trigger": trigger
        })
    
    return timeline

if __name__ == "__main__":
    timeline = detect_drift(personas)
    
    print("=== PERSONA DRIFT TIMELINE ===\n")
    for entry in timeline:
        drift_marker = "⚠ DRIFT" if entry['drift_detected'] else "stable"
        print(f"Day {entry['day']:>3} [{drift_marker}]")
        print(f"         Tone: {entry['tone']}")
        print(f"         Traits: {', '.join(entry['traits']) or 'none'}")
        if entry['drift_detected']:
            print(f"         Gained: {', '.join(entry['gained_traits'])}")
            print(f"         Lost:   {', '.join(entry['lost_traits'])}")
            print(f"         Trigger: {entry['trigger']}")
        print()
    
    with open('drift_timeline.pkl', 'wb') as f:
        pickle.dump(timeline, f)
    print("Saved to drift_timeline.pkl")