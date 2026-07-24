"""
PrepAI — FastAPI Backend
Handles: secure Groq API calls, session saving, user history via Supabase
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import httpx
import json
import os
import uuid
from datetime import datetime
from supabase import create_client, Client
from dotenv import load_dotenv
import logging
import dns.resolver
import smtplib
import socket
import secrets
import hashlib

load_dotenv()

# ─────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
#  INIT & CONFIG
# ─────────────────────────────────────────
app = FastAPI(title="PrepAI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL    = "llama-3.3-70b-versatile"

SUPABASE_URL  = os.getenv("SUPABASE_URL")
SUPABASE_KEY  = (
    os.getenv("SUPABASE_SERVICE_ROLE_KEY") or 
    os.getenv("SUPABASE_SERVICE_KEY") or 
    os.getenv("SUPABASE_KEY") or 
    os.getenv("SUPABASE_ANON_KEY")
)
RESEND_API_KEY = os.getenv("RESEND_API_KEY")

supabase: Optional[Client] = None

if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info(f"Supabase successfully connected to: {SUPABASE_URL[:30]}...")
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
else:
    logger.warning("SUPABASE_URL or SUPABASE_KEY missing from environment variables.")

if RESEND_API_KEY:
    logger.info("Resend API key configured for email delivery.")
else:
    logger.warning("RESEND_API_KEY missing. Email verification will not work.")

# ─────────────────────────────────────────
#  MODELS
# ─────────────────────────────────────────
class GenerateQuestionRequest(BaseModel):
    topics: List[str]
    difficulty: str         # easy | medium | hard | mixed
    question_number: int
    total_questions: int
    asked_questions: List[str] = []

class EvaluateAnswerRequest(BaseModel):
    question: str
    topic: str
    user_answer: str
    session_id: str

class SaveUserRequest(BaseModel):
    email: str
    name: str

class SendOtpRequest(BaseModel):
    email: str
    name: Optional[str] = None

class VerifyOtpRequest(BaseModel):
    email: str
    name: str
    otp: str

class VerifyTokenRequest(BaseModel):
    token: str
    name: str

class SaveSessionRequest(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    topics: List[str]
    difficulty: str
    total_questions: int
    results: List[Dict[str, Any]]
    score_pct: int
    duration_seconds: Optional[int] = None

class GetHistoryRequest(BaseModel):
    user_id: str
    limit: int = 10

# ─────────────────────────────────────────
#  HEALTH ENDPOINTS
# ─────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "PrepAI backend running", "version": "1.0.0"}

@app.api_route("/health", methods=["GET", "HEAD"])
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
    clean = raw.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
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

    if req.difficulty == "mixed":
        third = req.total_questions / 3
        diff  = "easy" if req.question_number <= third else "medium" if req.question_number <= 2 * third else "hard"
    else:
        diff = req.difficulty

    prompt = f"""You are a senior technical interviewer at a top tech company.

Generate interview question #{req.question_number} of {req.total_questions}.
Topics: {topics_str}
Difficulty: {diff}

Already asked (DO NOT repeat):
- {asked_str}

Respond ONLY with valid JSON — no markdown, no extra text:
{{"question":"...","topic":"exact topic name from list","difficulty":"{diff}","type":"conceptual|practical|scenario|comparison"}}"""

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

    prompt = f"""You are a senior technical interviewer.

Question: {req.question}
Topic: {req.topic}
Candidate's answer: {req.user_answer if not skipped else "(skipped — no answer provided)"}

Evaluate thoroughly. Respond ONLY with valid JSON:
{{
  "score": "correct" | "partial" | "wrong",
  "points": <integer 0-100>,
  "feedback": "<2-3 sentences feedback>",
  "ideal_answer": "<3-5 sentences ideal answer>",
  "tip": "<1 actionable tip>"
}}"""

    raw        = await call_groq([{"role": "user", "content": prompt}], max_tokens=700)
    evaluation = parse_json_response(raw)
    evaluation["session_id"] = req.session_id
    return evaluation

# ─────────────────────────────────────────
#  SAVE USER (IMMEDIATE ON START)
# ─────────────────────────────────────────
def verify_email_exists(email: str) -> bool:
    """
    Checks if the email's domain has MX records and does a basic SMTP ping.
    Note: Some providers (like Gmail/Yahoo) might still accept or block the ping, 
    so this isn't 100% foolproof, but it catches most fake emails.
    """
    try:
        domain = email.split('@')[1]
        
        # 1. Check MX records
        try:
            records = dns.resolver.resolve(domain, 'MX')
        except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN, dns.exception.Timeout):
            return False
            
        mx_record = str(records[0].exchange)
        
        # 2. SMTP Ping
        try:
            server = smtplib.SMTP(timeout=3)
            server.set_debuglevel(0)
            server.connect(mx_record)
            server.helo(server.local_hostname or socket.gethostname())
            server.mail('verify@example.com')
            code, message = server.rcpt(email)
            server.quit()
            
            # SMTP code 250 means OK (user exists)
            # Code 550 means user unknown
            # Other codes (like 451) usually mean greylisting, which means domain is real
            if code == 550:
                return False
                
            return True
        except Exception:
            # If we fail to connect but MX exists, we assume it's valid to avoid false negatives
            return True
            
    except Exception:
        return False

@app.post("/api/user/send-otp")
async def send_otp(req: SendOtpRequest, request: Request):
    if not supabase:
        return {"status": "error", "message": "Database not configured"}
    if not RESEND_API_KEY:
        return {"status": "error", "message": "Email service not configured (RESEND_API_KEY missing)"}
        
    clean_email = req.email.strip().lower()
    clean_name = req.name.strip() if req.name else None

    if not verify_email_exists(clean_email):
        return {"status": "error", "code": "INVALID_EMAIL", "message": "The email address does not exist or is unreachable."}

    # Check if user already exists in our database — skip verification for returning users
    try:
        existing_user = supabase.table("users").select("name, email").eq("email", clean_email).execute()
        if existing_user.data:
            stored_name = existing_user.data[0]["name"]
            # Update last_active timestamp
            supabase.table("users").update({"last_active": datetime.utcnow().isoformat()}).eq("email", clean_email).execute()
            logger.info(f"Returning user detected: {clean_email} ({stored_name})")
            return {"status": "existing_user", "name": stored_name, "email": clean_email}
    except Exception as e:
        logger.error(f"Error checking existing user: {str(e)}", exc_info=True)
        
    # If name is provided, verify it's unique
    if clean_name:
        try:
            existing_user_resp = supabase.table("users").select("email").ilike("name", clean_name).execute()
            if existing_user_resp.data:
                for user in existing_user_resp.data:
                    if user["email"] != clean_email:
                        return {"status": "error", "code": "NAME_EXISTS", "message": "Username already exists. Please choose a different name."}
        except Exception as e:
            logger.error(f"ERROR verifying name uniqueness: {str(e)}", exc_info=True)
            return {"status": "error", "detail": f"Failed to check username: {str(e)}"}

    # Generate a secure token
    raw_token = secrets.token_urlsafe(48)
    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    expires_at = datetime.utcnow().isoformat()

    # Store token hash in DB
    try:
        # Clean up any old tokens for this email
        supabase.table("verification_tokens").delete().eq("email", clean_email).execute()
        supabase.table("verification_tokens").insert({
            "email": clean_email,
            "token_hash": token_hash,
            "created_at": datetime.utcnow().isoformat(),
        }).execute()
    except Exception as e:
        logger.error(f"ERROR storing verification token: {str(e)}", exc_info=True)
        return {"status": "error", "message": "Failed to create verification token."}

    # Build the verification link
    origin = request.headers.get("origin") or request.headers.get("referer") or ""
    if origin.endswith('/'):
        origin = origin[:-1]
    verify_link = f"{origin}#token={raw_token}"

    # Send email via Resend
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.resend.com/emails",
                headers={
                    "Authorization": f"Bearer {RESEND_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "from": "PrepAI <onboarding@resend.dev>",
                    "to": [clean_email],
                    "subject": "PrepAI — Verify your email to start your interview",
                    "html": f"""
                    <div style="font-family:sans-serif;max-width:480px;margin:0 auto;padding:40px 20px;">
                        <h2 style="color:#333;">PrepAI Interview Verification</h2>
                        <p style="color:#555;font-size:16px;line-height:1.6;">Click the button below to verify your email and start your mock interview session.</p>
                        <a href="{verify_link}" style="display:inline-block;padding:14px 28px;background:#635BFF;color:white;text-decoration:none;border-radius:8px;font-size:16px;font-weight:600;margin:20px 0;">Start Interview →</a>
                        <p style="color:#999;font-size:13px;">This link expires in 30 minutes. If you didn't request this, you can ignore this email.</p>
                    </div>
                    """,
                },
            )
            if resp.status_code >= 400:
                logger.error(f"Resend API error: {resp.status_code} {resp.text}")
                return {"status": "error", "message": f"Failed to send email: {resp.text}"}
        
        logger.info(f"Verification email sent to {clean_email} via Resend")
        return {"status": "success", "message": "Verification email sent! Check your inbox."}
    except Exception as e:
        logger.error(f"ERROR sending email via Resend: {str(e)}", exc_info=True)
        return {"status": "error", "message": f"Failed to send email: {str(e)}"}

@app.post("/api/user/verify-token")
async def verify_token_and_save(req: VerifyTokenRequest):
    if not supabase:
        logger.warning("Supabase client not initialized. User not saved to database.")
        return {"status": "skipped", "message": "Database not configured"}

    clean_name = req.name.strip()
    token_hash = hashlib.sha256(req.token.encode()).hexdigest()
    
    # 1. Look up the token in our DB
    try:
        token_resp = supabase.table("verification_tokens").select("*").eq("token_hash", token_hash).execute()
        if not token_resp.data:
            return {"status": "error", "code": "INVALID_TOKEN", "message": "Invalid or expired verification link."}
        
        token_record = token_resp.data[0]
        clean_email = token_record["email"]
        
        # Check if token is older than 30 minutes
        created_at = datetime.fromisoformat(token_record["created_at"].replace("Z", "+00:00").replace("+00:00", ""))
        age_minutes = (datetime.utcnow() - created_at).total_seconds() / 60
        if age_minutes > 30:
            supabase.table("verification_tokens").delete().eq("token_hash", token_hash).execute()
            return {"status": "error", "code": "INVALID_TOKEN", "message": "Verification link has expired. Please request a new one."}
        
        # Delete the token (single use)
        supabase.table("verification_tokens").delete().eq("token_hash", token_hash).execute()
    except Exception as e:
        logger.error(f"Token lookup failed: {str(e)}")
        return {"status": "error", "code": "INVALID_TOKEN", "message": "Failed to verify token."}

    # 2. Save user
    try:
        existing_user_resp = supabase.table("users").select("email").ilike("name", clean_name).execute()
        if existing_user_resp.data:
            for user in existing_user_resp.data:
                if user["email"] != clean_email:
                    return {"status": "error", "code": "NAME_EXISTS", "message": "Username already exists. Please choose a different name."}

        user_data = {
            "email": clean_email,
            "name": clean_name,
            "last_active": datetime.utcnow().isoformat()
        }
        response = supabase.table("users").upsert(user_data, on_conflict="email").execute()
        logger.info(f"Supabase user save success: {response}")
        return {"status": "saved", "email": clean_email, "name": clean_name}
    except Exception as e:
        logger.error(f"ERROR saving user to Supabase: {str(e)}", exc_info=True)
        return {"status": "error", "detail": str(e)}

# ─────────────────────────────────────────
#  SAVE SESSION (FIXED USER_ID & SUPABASE INSERT)
# ─────────────────────────────────────────
@app.post("/api/session/save")
async def save_session(req: SaveSessionRequest):
    if not supabase:
        logger.warning("Supabase client not initialized. Session not saved to database.")
        return {"status": "skipped", "message": "Database not configured"}

    # Sanitize user_id (lowercase email or string)
    clean_user_id = req.user_id.strip().lower() if req.user_id and req.user_id.strip() else "anonymous"
    logger.info(f"Saving session {req.session_id} for user_id: '{clean_user_id}'")

    try:
        session_data = {
            "session_id":        req.session_id,
            "user_id":           clean_user_id,
            "user_name":         req.user_name.strip() if req.user_name else None,
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

        try:
            response = supabase.table("sessions").insert(session_data).execute()
        except Exception as insert_err:
            logger.warning(f"Initial session insert failed (possibly missing user_name column), retrying fallback: {insert_err}")
            session_data_fallback = {k: v for k, v in session_data.items() if k != "user_name"}
            response = supabase.table("sessions").insert(session_data_fallback).execute()

        logger.info(f"Supabase insert success: {response}")
        return {"status": "saved", "session_id": req.session_id, "user_id": clean_user_id}

    except Exception as e:
        logger.error(f"ERROR saving session to Supabase: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to save session: {str(e)}")

# ─────────────────────────────────────────
#  GET USER HISTORY
# ─────────────────────────────────────────
@app.get("/api/history/{user_id}")
async def get_history(user_id: str, limit: int = 10):
    if not supabase:
        return {"sessions": []}

    clean_user_id = user_id.strip().lower()
    try:
        logger.info(f"Fetching history for user_id: '{clean_user_id}'")
        resp = (
            supabase.table("sessions")
            .select("session_id,topics,difficulty,total_questions,correct,partial,wrong,score_pct,duration_seconds,created_at")
            .eq("user_id", clean_user_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return {"sessions": resp.data or []}
    except Exception as e:
        logger.error(f"ERROR fetching history: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ─────────────────────────────────────────
#  GET SESSION DETAIL
# ─────────────────────────────────────────
@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    if not supabase:
        raise HTTPException(status_code=503, detail="Database connection uninitialized")

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
        logger.error(f"ERROR fetching session: {str(e)}", exc_info=True)
        raise HTTPException(status_code=404, detail="Session not found")

# ─────────────────────────────────────────
#  LEADERBOARD
# ─────────────────────────────────────────

@app.get("/api/leaderboard")
async def get_leaderboard(topic: Optional[str] = None, limit: int = 10):
    if not supabase:
        return {"leaderboard": []}

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
        return {"leaderboard": resp.data or []}
    except Exception as e:
        logger.error(f"ERROR fetching leaderboard: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
