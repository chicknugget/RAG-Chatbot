import gradio as gr
import os
from chatbot import answer

def chat(message, history):
    return answer(message)

demo = gr.ChatInterface(fn=chat, title="Conversation RAG Chatbot")
demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))