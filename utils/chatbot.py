from typing import List
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage, SystemMessage


def create_chat_model(api_key: str | None = None, model: str = "gemini-2.5-flash") -> ChatGoogleGenerativeAI:
    """Create a Gemini 2.5 Flash chat model instance."""
    kwargs = {
        "model": model,
        "temperature": 0.2,
        "max_tokens": 1024,
    }
    if api_key:
        kwargs["api_key"] = api_key
    return ChatGoogleGenerativeAI(**kwargs)


def build_prompt(query: str, retrieved_chunks: list[dict], history: list[dict]) -> list[list[SystemMessage | HumanMessage]]:
    """Build a chat prompt that includes retrieved PDF context and prior messages."""
    system_instruction = (
        "You are a helpful assistant. Answer using only the provided PDF context. "
        "If the answer is not contained in the context, reply honestly that the information is unavailable. "
        "Keep responses concise and factual."
    )

    context_blocks = []
    for item in retrieved_chunks:
        context_blocks.append(f"Chunk {item['index'] + 1}: {item['content']}")

    context_text = "\n\n".join(context_blocks)
    history_text = "\n".join(
        f"{entry['role'].capitalize()}: {entry['content']}" for entry in history[-4:]
    )

    prompt = (
        f"PDF context:\n{context_text}\n\n"
        f"Conversation history:\n{history_text}\n\n"
        f"Question:\n{query}\n\n"
        "Answer using only the PDF content above."
    )

    return [[
        SystemMessage(content=system_instruction),
        HumanMessage(content=prompt),
    ]]


def generate_answer(query: str, retrieved_chunks: list[dict], chat_model: ChatGoogleGenerativeAI, history: list[dict]) -> str:
    """Generate an answer from Gemini 2.5 Flash using retrieved PDF context."""
    if not retrieved_chunks:
        return "I could not find relevant content in the uploaded PDF to answer that question."

    messages = build_prompt(query, retrieved_chunks, history)
    result = chat_model.generate(messages)
    generation = result.generations[0][0]
    return generation.text.strip()
