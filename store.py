import pickle
import chromadb

with open('messages.pkl', 'rb') as f:
    all_messages = pickle.load(f)

with open('breakpoints.pkl', 'rb') as f:
    breakpoints = pickle.load(f)

client = chromadb.PersistentClient(path="./chroma_db")

msg_collection = client.get_or_create_collection("messages")

def store_messages(all_messages):
    ids = [str(m['index']) for m in all_messages]
    embeddings = [m['embedding'].tolist() for m in all_messages]
    documents = [m['text'] for m in all_messages]
    metadatas = [{'speaker': m['speaker'], 'index': m['index']} for m in all_messages]

    batch_size = 1000
    for i in range(0, len(all_messages), batch_size):
        msg_collection.add(
            ids=ids[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size]
        )
        print(f"Stored {min(i+batch_size, len(all_messages))}/{len(all_messages)} messages")

if __name__ == "__main__":
    store_messages(all_messages)
    print("Done! Messages stored in ChromaDB")