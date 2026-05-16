import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class IntentClassifier:
    def __init__(self, model_path="./intent_model/final"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def classify(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=64
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)[0]
        pred = torch.argmax(probs).item()
        return {
            "intent": self.id2label[pred],
            "confidence": round(probs[pred].item(), 3)
        }


if __name__ == "__main__":
    clf = IntentClassifier()
    tests = [
        "remind me to call mom tomorrow",
        "i'm feeling really sad and lonely today",
        "can you book a flight to Delhi for me",
        "haha yeah that's so funny",
        "asdfgh"
    ]
    for t in tests:
        print(clf.classify(t))