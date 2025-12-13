from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
from qa import answer_question

app = FastAPI(title="Lightweight RAG Q&A")

class QARequest(BaseModel):
    question: str

@app.post("/qa")
def qa(req: QARequest):
    return answer_question(Path("policy.txt"), req.question)
