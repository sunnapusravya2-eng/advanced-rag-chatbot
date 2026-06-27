import os
from typing import Optional
import google.generativeai as genai

def create_chat_model(api_key: Optional[str] = None):
    key = api_key or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError("GOOGLE_API_KEY not provided.")
    genai.configure(api_key=key)
    model_name = os.getenv("CHAT_MODEL", "gemini-2.5-flash")
    temperature = float(os.getenv("CHAT_TEMPERATURE", "0.2"))
    return {"client": genai, "model": model_name, "temperature": temperature}

def generate_answer(
    question: str,
    retrieved_chunks: list,
    chat_model: dict,
    history: list,
) -> str:
    context = "\n\n".join(
        f"[Chunk {c['index']+1}]: {c['content']}" for c in retrieved_chunks
    )
    system_prompt = (
        "You are a helpful assistant that answers questions using only the PDF content below. "
        "Always cite the chunk numbers you used.\n\n"
        f"CONTEXT:\n{context}"
    )
    # Build conversation history
    chat_history = []
    for msg in history[-6:]:
        if msg["role"] == "user":
            chat_history.append({"role": "user", "parts": [msg["content"]]})
        elif msg["role"] == "assistant":
            chat_history.append({"role": "model", "parts": [msg["content"]]})

    model = chat_model["client"].GenerativeModel(
        model_name=chat_model["model"],
        system_instruction=system_prompt,
        generation_config={"temperature": chat_model["temperature"], "max_output_tokens": 1024},
    )
    chat = model.start_chat(history=chat_history)
    response = chat.send_message(question)
    return response.text