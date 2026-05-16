import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import time

with open('training_data.json', 'r') as f:
    data = json.load(f)

label2id = {"reminder": 0, "emotional-support": 1, "action-item": 2, "small-talk": 3, "unknown": 4}
id2label = {v: k for k, v in label2id.items()}

class IntentDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.encodings = tokenizer(
            [d['text'] for d in data],
            truncation=True,
            padding=True,
            max_length=64,
            return_tensors='pt'
        )
        self.labels = torch.tensor([label2id[d['label']] for d in data])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }

model_name = "prajjwal1/bert-tiny"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=5,
    id2label=id2label,
    label2id=label2id
)

dataset = IntentDataset(data, tokenizer)

training_args = TrainingArguments(
    output_dir="./intent_model",
    num_train_epochs=50, 
    per_device_train_batch_size=8,
    learning_rate=5e-5,
    logging_steps=10,
    save_strategy="epoch",
    no_cuda=True           
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

model.save_pretrained("./intent_model/final")
tokenizer.save_pretrained("./intent_model/final")
print("Model saved to ./intent_model/final")

def classify_timed(text):
    start = time.time()
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)[0]
    pred = torch.argmax(probs).item()
    elapsed = (time.time() - start) * 1000 
    return id2label[pred], probs[pred].item(), elapsed


tests = [
    "remind me to call mom tomorrow",
    "i'm feeling really sad and lonely today",
    "can you book a flight to Delhi for me",
    "haha yeah that's so funny",
    "asdfgh"
]

print("\nTest results:")
for t in tests:
    label, score, ms = classify_timed(t)
    print(f"{t[:45]:<45} → {label} ({score:.0%}) [{ms:.1f}ms]")