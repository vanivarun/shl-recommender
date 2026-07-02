from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.agent import get_agent_response

app = FastAPI(title="SHL Assessment Recommender")

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str

class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty")

    # Enforce max 8 turns
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    if len(messages) > 8:
        messages = messages[-8:]

    result = get_agent_response(messages)
    return ChatResponse(
        reply=result["reply"],
        recommendations=[Recommendation(**r) for r in result["recommendations"]],
        end_of_conversation=result["end_of_conversation"]
    )

@app.on_event("startup")
async def startup_event():
    # Pre-load index on startup
    from agent.retriever import load_index
    load_index()
    print("SHL index loaded successfully")