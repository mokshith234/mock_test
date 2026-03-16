"""
PrepAI — FastAPI Backend
Handles: secure Groq API calls, session saving, user history via Supabase
"""

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import httpx
import json
import os
import uuid
from datetime import datetime
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────
#  INIT
# ─────────────────────────────────────────
app = FastAPI(title="PrepAI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # lock this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL    = "llama-3.3-70b-versatile"

SUPABASE_URL  = os.getenv("SUPABASE_URL")
SUPABASE_KEY  = os.getenv("SUPABASE_SERVICE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ─────────────────────────────────────────
#  MODELS
# ─────────────────────────────────────────
class GenerateQuestionRequest(BaseModel):
    topics: list[str]
    difficulty: str         # easy | medium | hard | mixed
    question_number: int
    total_questions: int
    asked_questions: list[str] = []

class EvaluateAnswerRequest(BaseModel):
    question: str
    topic: str
    user_answer: str
    session_id: str

class SaveSessionRequest(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    topics: list[str]
    difficulty: str
    total_questions: int
    results: list[dict]
    score_pct: int
    duration_seconds: Optional[int] = None

class GetHistoryRequest(BaseModel):
    user_id: str
    limit: int = 10

# ─────────────────────────────────────────
#  HEALTH
# ─────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "PrepAI backend running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

# ─────────────────────────────────────────
#  GROQ HELPER
# ─────────────────────────────────────────
async def call_groq(messages: list, max_tokens: int = 600) -> str:
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured on server")

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            GROQ_BASE_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": GROQ_MODEL,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.7
            }
        )

    if response.status_code != 200:
        err = response.json().get("error", {})
        raise HTTPException(status_code=502, detail=f"Groq error: {err.get('message','Unknown error')}")

    return response.json()["choices"][0]["message"]["content"]

def parse_json_response(raw: str) -> dict:
    clean = raw.strip().replace("```json","").replace("```","").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        # try to extract JSON object from response
        start = clean.find("{")
        end   = clean.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(clean[start:end])
        raise HTTPException(status_code=502, detail="Invalid JSON from AI model")

# ─────────────────────────────────────────
#  GENERATE QUESTION
# ─────────────────────────────────────────
@app.post("/api/question/generate")
async def generate_question(req: GenerateQuestionRequest):
    topics_str = ", ".join(req.topics)
    asked_str  = "\n- ".join(req.asked_questions) if req.asked_questions else "None"

    # auto-escalate difficulty for mixed mode
    if req.difficulty == "mixed":
        third = req.total_questions / 3
        diff  = "easy" if req.question_number <= third else "medium" if req.question_number <= 2*third else "hard"
    else:
        diff = req.difficulty

    prompt = f"""You are a senior technical interviewer at a top tech company like Google or Microsoft.

Generate interview question #{req.question_number} of {req.total_questions}.
Topics: {topics_str}
Difficulty: {diff}

Already asked (DO NOT repeat or ask something similar):
- {asked_str}

Rules:
- One clear, specific question — no multi-part
- Mix theory and real-world application
- Professional tone used in actual FAANG interviews
- Vary question style: definition, explain-why, compare, scenario, code-concept

Respond ONLY with valid JSON — no markdown, no extra text:
{{"question":"...","topic":"exact topic name from the list","difficulty":"{diff}","type":"conceptual|practical|scenario|comparison"}}"""

    raw  = await call_groq([{"role": "user", "content": prompt}], max_tokens=400)
    data = parse_json_response(raw)
    data["session_question_id"] = str(uuid.uuid4())
    return data

# ─────────────────────────────────────────
#  EVALUATE ANSWER
# ─────────────────────────────────────────
@app.post("/api/answer/evaluate")
async def evaluate_answer(req: EvaluateAnswerRequest):
    skipped = not req.user_answer or req.user_answer.strip() == ""

    prompt = f"""You are a strict but encouraging senior technical interviewer.

Question: {req.question}
Topic: {req.topic}
Candidate's answer: {req.user_answer if not skipped else "(skipped — no answer provided)"}

Evaluate thoroughly. Respond ONLY with valid JSON — no markdown, no extra text:
{{
  "score": "correct" | "partial" | "wrong",
  "points": <integer 0-100>,
  "feedback": "<2-3 sentences — what was right, what was wrong, be specific>",
  "ideal_answer": "<clear complete answer in 3-5 sentences a senior engineer would give>",
  "tip": "<one specific actionable improvement tip for interviews>"
}}

Scoring guide:
- correct  = covers all key concepts accurately (80-100 pts)
- partial  = right idea but incomplete or minor errors (40-79 pts)  
- wrong    = incorrect, missing key concept, or skipped (0-39 pts)"""

    raw        = await call_groq([{"role": "user", "content": prompt}], max_tokens=700)
    evaluation = parse_json_response(raw)
    evaluation["session_id"] = req.session_id
    return evaluation

# ─────────────────────────────────────────
#  SAVE SESSION
# ─────────────────────────────────────────
@app.post("/api/session/save")
async def save_session(req: SaveSessionRequest):
    try:
        session_data = {
            "session_id":        req.session_id,
            "user_id":           req.user_id,
            "topics":            req.topics,
            "difficulty":        req.difficulty,
            "total_questions":   req.total_questions,
            "correct":           sum(1 for r in req.results if r.get("score") == "correct"),
            "partial":           sum(1 for r in req.results if r.get("score") == "partial"),
            "wrong":             sum(1 for r in req.results if r.get("score") == "wrong"),
            "score_pct":         req.score_pct,
            "duration_seconds":  req.duration_seconds,
            "results":           req.results,
            "created_at":        datetime.utcnow().isoformat(),
        }
        supabase.table("sessions").insert(session_data).execute()
        return {"status": "saved", "session_id": req.session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save session: {str(e)}")

# ─────────────────────────────────────────
#  GET USER HISTORY
# ─────────────────────────────────────────
@app.get("/api/history/{user_id}")
async def get_history(user_id: str, limit: int = 10):
    try:
        resp = (
            supabase.table("sessions")
            .select("session_id,topics,difficulty,total_questions,correct,partial,wrong,score_pct,duration_seconds,created_at")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return {"sessions": resp.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─────────────────────────────────────────
#  GET SESSION DETAIL
# ─────────────────────────────────────────
@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    try:
        resp = (
            supabase.table("sessions")
            .select("*")
            .eq("session_id", session_id)
            .single()
            .execute()
        )
        return resp.data
    except Exception as e:
        raise HTTPException(status_code=404, detail="Session not found")

# ─────────────────────────────────────────
#  LEADERBOARD
# ─────────────────────────────────────────
@app.get("/api/leaderboard")
async def get_leaderboard(topic: Optional[str] = None, limit: int = 10):
    try:
        query = (
            supabase.table("sessions")
            .select("user_id,score_pct,topics,total_questions,created_at")
            .order("score_pct", desc=True)
            .limit(limit)
        )
        if topic:
            query = query.contains("topics", [topic])
        resp = query.execute()
        return {"leaderboard": resp.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
