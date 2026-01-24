import os
import io
import wave
import base64
import time
import traceback
from collections import OrderedDict
from typing import Literal
import json
import re
import requests
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import threading
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import List, Optional

import sqlalchemy
import databases
from passlib.hash import pbkdf2_sha256


# ===== Load .env =====
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# =====================
# ENV
# =====================
GOOGLE_TTS_API_KEY = os.getenv("GOOGLE_TTS_API_KEY")

# STT: if not provided, fallback to TTS key
GOOGLE_STT_API_KEY = os.getenv("GOOGLE_STT_API_KEY", GOOGLE_TTS_API_KEY)
GOOGLE_STT_LANG = os.getenv("GOOGLE_STT_LANG", "en-US")

# =====================
# Modal LLM
# =====================

MODAL_LLM_URL = os.getenv(
    "MODAL_LLM_URL",
    "https://essamahmoud04--qwen-gemini-like-api-web-app.modal.run/v1beta/models/anything:generateContent",
)
MODAL_LLM_HEALTH_URL = os.getenv(
    "MODAL_LLM_HEALTH_URL",
    "https://essamahmoud04--qwen-gemini-like-api-web-app.modal.run/health",
)


GOOGLE_TTS_LANG = os.getenv("GOOGLE_TTS_LANG", "en-US")
GOOGLE_TTS_VOICE = os.getenv("GOOGLE_TTS_VOICE", "en-US-Standard-C")
GOOGLE_TTS_SR = int(os.getenv("GOOGLE_TTS_SR", "24000"))

CACHE_TTL_SEC = 600          # 10 minutes
MIN_INTERVAL_SEC = 3.0       # throttle

# UX line at the end of every answer
CTA_LINE = "\n\n(Press 'A' to ask me anything else!)"


# =====================
# App
# =====================



# --- DATABASE CONFIG ---
DATABASE_URL = "postgresql://medilearn_user:xO64NTMEOUI9T2zZKzIkMo2Ul53bWWRg@dpg-d3c2oremcj7s73d7d3h0-a.frankfurt-postgres.render.com/medilearn?ssl=require"

database = databases.Database(DATABASE_URL)
metadata = sqlalchemy.MetaData()







app = FastAPI(title="Z-Anatomy AI Backend")





@app.on_event("startup")
async def startup():
    await database.connect()
    print("--- DB Connected ---")

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def create_session() -> requests.Session:
    session = requests.Session()
    retry_strategy = Retry(
        total=6,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        respect_retry_after_header=True,
    )

    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=20,
        pool_maxsize=50,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"Connection": "keep-alive"})
    return session

_llm_session = create_session()

def _modal_keep_alive_worker():
    while True:
        try:
            time.sleep(300)  # 5 minutes
            _llm_session.get(MODAL_LLM_HEALTH_URL, timeout=10)
        except Exception:
            pass

threading.Thread(target=_modal_keep_alive_worker, daemon=True).start()

# =====================
# In-memory cache
# =====================
_text_cache = OrderedDict()     # key -> (ts, text)
_audio_cache = OrderedDict()    # key -> (ts, wav)
_last_request_time = 0.0


def _cache_get(cache, key):
    now = time.time()
    if key in cache:
        ts, val = cache[key]
        if now - ts < CACHE_TTL_SEC:
            cache.move_to_end(key)
            return val
        del cache[key]
    return None


def _cache_set(cache, key, val, max_items=200):
    cache[key] = (time.time(), val)
    cache.move_to_end(key)
    while len(cache) > max_items:
        cache.popitem(last=False)


# =====================
# Helpers & Prompts (IMPROVED PERSONA)
# =====================
def clean_bone_name(name: str) -> str:
    if not name:
        return "unknown bone"
    return (
        name.replace("(R)", "")
            .replace("(L)", "")
            .replace("+", " ")
            .strip()
    )


def build_prompt(bone: str, mode: Literal["short", "more"]) -> str:
    # FIXED: Changed persona to be friendly and enthusiastic instead of boring
    if mode == "more":
        return (
            "You are a friendly, enthusiastic VR anatomy guide. Not a boring textbook.\n"
            f"Explain details about the {bone} like you are talking to a curious student.\n"
            "Cover location, function, landmarks, and a cool clinical fact.\n"
            "Keep sentences punchy and clear. Speak for about 60 seconds.\n"
            "No markdown.\n"
        )

    return (
        "You are a friendly, enthusiastic VR anatomy guide.\n"
        f"Give me 4 quick, fascinating facts about the {bone}.\n"
        "Keep it light and easy to understand.\n"
        "No markdown.\n"
    )

def build_voice_chat_prompt(bone: str, user_text: str) -> str:
    bone_part = f"Current bone: {bone}." if bone else "No specific bone."
    return (
        "You are a helpful and energetic VR medical assistant. You love anatomy!\n"
        f"{bone_part}\n"
        "Answer the student's question directly and conversationally.\n"
        "Avoid super long monologues unless asked. Be encouraging.\n"
        "No markdown.\n\n"
        f"Student question: {user_text}\n"
    )

def build_quiz_prompt(bone: str) -> str:
    return (
        f"Act as a medical examination expert. Generate a 2-question multiple-choice quiz on the anatomy of the {bone}.\n"
        "Requirements:\n"
        "Each question must have exactly 4 options labeled A through D.\n"
        "The output must be strictly raw JSON format.\n"
        "Do not include markdown code blocks (like ```json), introductory text, or concluding remarks.\n"
        "Your response should start with [ and end with ].\n"
        "Each object in the JSON array must include:\n"
        "question: The text of the question.\n"
        "options: An object containing keys A, B, C, D.\n"
        "correct_answer: Just the letter of the correct option (e.g., 'A').\n"
        "explanation: A concise 1-2 sentence medical explanation justifying the correct answer.\n\n"
        "Use this exact JSON structure:\n"
        "[\n"
        "  {\n"
        '    "question": "Question text here?",\n'
        '    "options": {\n'
        '      "A": "Option one",\n'
        '      "B": "Option two",\n'
        '      "C": "Option three",\n'
        '      "D": "Option four"\n'
        '    },\n'
        '    "correct_answer": "C",\n'
        '    "explanation": "This is the explanation."\n'
        "  }\n"
        "]"
    )


def _append_cta(text: str) -> str:
    if not text:
        return text
    if "Press 'A'" in text:
        return text
    return text.strip() + CTA_LINE






# ====================
# log in 
# ====================
# Define User Table
# Matches your screenshot: id, email, password (NO name column visible)
users = sqlalchemy.Table(
    "users", 
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("email", sqlalchemy.String, unique=True),
    sqlalchemy.Column("password", sqlalchemy.String), # <--- CHANGED to "password"
)

# --- ADD THIS CLASS ---
class LoginRequest(BaseModel):
    email: str
    password: str



@app.post("/login")
async def login(req: LoginRequest):
    email = (req.email or "").strip().lower()
    password = (req.password or "")

    print(f"Login Attempt: {email}")
    print(f"DEBUG: pw_chars={len(password)} pw_bytes={len(password.encode('utf-8', 'ignore'))}")

    # 1) fetch user
    query = users.select().where(users.c.email == email)
    user = await database.fetch_one(query)

    # 2) auto-register if not found
    if not user:
        print("User not found -> Creating new account...")

        try:
            hashed_pw = pbkdf2_sha256.hash(password)
        except Exception as e:
            print("Hash error:", repr(e))
            raise HTTPException(status_code=500, detail="Password hashing failed")

        insert_query = users.insert().values(
            email=email,
            password=hashed_pw,
        )
        await database.execute(insert_query)

        return {"ok": True, "msg": "Account created & logged in", "email": email}

    # 3) verify password
    stored_pw = user["password"]

    try:
        # ✅ Support BOTH: new pbkdf2 hashes + old bcrypt hashes (optional, helps migration)
        if stored_pw.startswith("$pbkdf2-sha256$"):
            ok = pbkdf2_sha256.verify(password, stored_pw)
        else:
            # old/unknown hash -> refuse or treat as wrong
            # (لو عايز تعمل migration من bcrypt لازم نرجّع bcrypt بشكل سليم، لكن حالياً Render بيكسره)
            ok = False

        if ok:
            return {"ok": True, "msg": "Login successful", "email": user["email"]}

    except Exception as e:
        print("Password verify error:", repr(e))
        raise HTTPException(status_code=500, detail="Password verification failed")

    print("Wrong password")
    raise HTTPException(status_code=401, detail="Incorrect password")


# =====================
# Qwen
# =====================
def gemini_generate_text(
    prompt: str,
    cache_key: str,
    *,
    temperature: float = 0.7,
    max_output_tokens: int = 8192,
    use_cache: bool = True,
) -> str:
    """
    Modal-only LLM call.
    Kept the name gemini_generate_text() so the rest of the code does not change.
    """
    global _last_request_time

    # -------- cache --------
    if use_cache:
        cached = _cache_get(_text_cache, cache_key)
        if cached:
            return cached

    # -------- throttle (global) --------
    now = time.time()
    dt = now - _last_request_time
    if dt < MIN_INTERVAL_SEC:
        time.sleep(MIN_INTERVAL_SEC - dt)
    _last_request_time = time.time()

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": float(temperature),
            "maxOutputTokens": int(max_output_tokens),
        },
    }

    # -------- retry/backoff --------
    fallback_text = "I'm a bit busy right now (rate limit). Please try again in a few seconds."
    max_attempts = 5
    base_sleep = 1.5  # seconds

    last_err = None

    for attempt in range(1, max_attempts + 1):
        try:
            r = _llm_session.post(MODAL_LLM_URL, json=payload, timeout=(10, 180))

            # 429 = rate limit -> wait + retry
            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                sleep_s = float(retry_after) if retry_after and retry_after.isdigit() else (base_sleep * attempt)
                time.sleep(sleep_s)
                continue

            # any non-200 -> retry a bit then fallback
            if r.status_code != 200:
                last_err = RuntimeError(f"LLM HTTP {r.status_code}: {r.text[:800]}")
                time.sleep(base_sleep * attempt)
                continue

            data = r.json()

            try:
                text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            except Exception:
                last_err = RuntimeError(f"Bad LLM response shape: {str(data)[:800]}")
                time.sleep(base_sleep * attempt)
                continue

            if not text:
                last_err = RuntimeError("LLM returned empty text")
                time.sleep(base_sleep * attempt)
                continue

            if use_cache:
                _cache_set(_text_cache, cache_key, text)

            return text

        except Exception as e:
            last_err = e
            time.sleep(base_sleep * attempt)
            continue

    # -------- final fallback (do NOT crash VR endpoints) --------
    # Optional: print last error for debugging
    try:
        print("LLM failed after retries:", repr(last_err))
    except Exception:
        pass

    return fallback_text

# =====================
# Google TTS (FIXED)
# =====================
def google_tts_wav(text: str, cache_key: str) -> bytes:
    cached = _cache_get(_audio_cache, cache_key)
    if cached:
        return cached

    if not GOOGLE_TTS_API_KEY:
        raise RuntimeError("Missing GOOGLE_TTS_API_KEY")

    # --- THE FIX IS IN THIS LINE BELOW (No brackets, no markdown) ---
    url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={GOOGLE_TTS_API_KEY}"
    
    payload = {
        "input": {"text": text},
        "voice": {"languageCode": GOOGLE_TTS_LANG, "name": GOOGLE_TTS_VOICE},
        "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": GOOGLE_TTS_SR}
    }

    r = requests.post(url, json=payload, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Google TTS HTTP {r.status_code}: {r.text}")

    audio_b64 = r.json().get("audioContent")
    if not audio_b64:
        raise RuntimeError("Google TTS missing audioContent")

    pcm = base64.b64decode(audio_b64)

    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(GOOGLE_TTS_SR)
        wf.writeframes(pcm)

    wav = wav_io.getvalue()
    _cache_set(_audio_cache, cache_key, wav)
    return wav

# =====================
# Google STT (FIXED)
# =====================
def google_stt_text(wav_bytes: bytes, language_code: str = None) -> str:
    if not GOOGLE_STT_API_KEY:
        raise RuntimeError("Missing GOOGLE_STT_API_KEY")

    language_code = language_code or GOOGLE_STT_LANG

    # read sample rate from WAV header (dynamic for VR mic compatibility)
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            sr = wf.getframerate()
    except Exception:
        # Fallback if header is weird/missing
        sr = 48000 

    # --- THE FIX IS IN THIS LINE BELOW (No brackets, no markdown) ---
    url = f"https://speech.googleapis.com/v1/speech:recognize?key={GOOGLE_STT_API_KEY}"

    audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
    payload = {
        "config": {
            "encoding": "LINEAR16",
            "sampleRateHertz": sr,
            "languageCode": language_code,
        },
        "audio": {"content": audio_b64}
    }

    r = requests.post(url, json=payload, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Google STT HTTP {r.status_code}: {r.text}")

    data = r.json()
    results = data.get("results", [])
    if not results:
        return ""

    parts = []
    for res in results:
        alts = res.get("alternatives", [])
        if alts and alts[0].get("transcript"):
            parts.append(alts[0]["transcript"])

    return " ".join([p.strip() for p in parts if p.strip()]).strip()


# =====================
# Endpoints
# =====================
@app.get("/")
def root():
    return {
        "ok": True,
        "llm_mode": "modal",
        "modal_llm_url": MODAL_LLM_URL,
        "tts_key_loaded": bool(GOOGLE_TTS_API_KEY),
        "stt_key_loaded": bool(GOOGLE_STT_API_KEY),
    }


@app.get("/bone_talk_text")
def bone_talk_text(bone_name: str, mode: str = "short"):
    try:
        mode = mode.lower()
        bone = clean_bone_name(bone_name)
        prompt = build_prompt(bone, mode)
        key = f"text::{mode}::{bone.lower()}"

        answer = gemini_generate_text(prompt, key)
        answer = _append_cta(answer)

        print("\nPROMPT >>>", prompt)
        print("ANSWER >>>", answer)

        return JSONResponse({
            "bone": bone,
            "mode": mode,
            "answer": answer
        })
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.get("/bone_talk_audio")
def bone_talk_audio(bone_name: str, mode: str = "short"):
    try:
        mode = mode.lower()
        bone = clean_bone_name(bone_name)
        prompt = build_prompt(bone, mode)

        text_key = f"text::{mode}::{bone.lower()}"
        audio_key = f"audio::{mode}::{bone.lower()}"

        answer = gemini_generate_text(prompt, text_key)
        answer = _append_cta(answer)

        wav = google_tts_wav(answer, audio_key)

        print("\nPROMPT >>>", prompt)
        print("ANSWER >>>", answer[:200], "...")
        print("AUDIO BYTES >>>", len(wav))

        return Response(content=wav, media_type="audio/wav")

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.post("/voice_chat_audio")
async def voice_chat_audio(
    audio: UploadFile = File(...),
    bone_name: str = Form(""),
    language: str = Form("en-US"),
):
    """
    Upload WAV audio (PCM16 preferred) -> STT -> Gemini -> TTS WAV response
    """
    try:
        bone = clean_bone_name(bone_name)
        wav_bytes = await audio.read()
        if not wav_bytes:
            raise RuntimeError("Empty audio upload")

        # 1) STT
        user_text = google_stt_text(wav_bytes, language_code=language)
        if not user_text:
            fallback = "Sorry, I didn't catch that. Please press A and try again."
            wav = google_tts_wav(fallback, "audio::fallback::stt")
            return Response(content=wav, media_type="audio/wav")

        # 2) Gemini
        prompt = build_voice_chat_prompt(bone, user_text)
        text_key = f"voice_text::{bone.lower()}::{user_text.lower()[:80]}"
        audio_key = f"voice_audio::{bone.lower()}::{user_text.lower()[:80]}"

        answer = gemini_generate_text(prompt, text_key)
        answer = _append_cta(answer)

        # 3) TTS
        wav = google_tts_wav(answer, audio_key)

        print("\nVOICE PROMPT >>>", prompt)
        print("USER STT >>>", user_text)
        print("ANSWER >>>", answer[:200], "...")
        print("AUDIO BYTES >>>", len(wav))

        # Optional transcript header for captions in Unity
        return Response(
            content=wav,
            media_type="audio/wav",
            headers={"X-Transcript": user_text}
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))

def _extract_json_array(text: str) -> str:
    """
    Extract the first JSON array [...] from a messy LLM response.
    Also strips ```json fences if present.
    """
    if not text:
        raise ValueError("Empty LLM response")

    s = text.strip()
    s = s.replace("```json", "").replace("```", "").strip()

    match = re.search(r"\[[\s\S]*\]", s)
    if not match:
        raise ValueError(f"No JSON array found in response. Head: {s[:300]}")

    return match.group(0).strip()


def _repair_json_loose(s: str) -> str:
    """
    Best-effort repair for common LLM JSON mistakes:
    - unquoted keys: question: -> "question":
    - trailing commas: }, ] etc
    - single-quoted strings: 'text' -> "text"
    - comments (rare): // ... or /* ... */
    """
    if not s:
        return s

    # Remove JS-style comments if any
    s = re.sub(r"//.*?$", "", s, flags=re.MULTILINE)
    s = re.sub(r"/\*[\s\S]*?\*/", "", s)

    # Quote unquoted keys AFTER { or ,  e.g. { question: "..." } or , correct_answer: "A"
    s = re.sub(r'(?m)([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', s)

    # Remove trailing commas before } or ]
    s = re.sub(r",\s*([}\]])", r"\1", s)

    # Convert single-quoted strings to double-quoted strings (best-effort)
    # Example: 'A' -> "A", 'Option one' -> "Option one"
    s = re.sub(
        r"(?<!\\)'([^'\\]*(?:\\.[^'\\]*)*)'",
        lambda m: '"' + m.group(1).replace('"', '\\"') + '"',
        s
    )

    return s.strip()


def _loads_json_with_repair(text_response: str):
    """
    1) Extract array
    2) Try strict json.loads
    3) If fail, repair then loads again
    """
    raw_array = _extract_json_array(text_response)

    try:
        return json.loads(raw_array)
    except json.JSONDecodeError:
        repaired = _repair_json_loose(raw_array)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError as e2:
            # Print useful debug (first ~1500 chars)
            print("=== LLM RAW ARRAY (truncated) ===")
            print(raw_array[:1500])
            print("=== LLM REPAIRED ARRAY (truncated) ===")
            print(repaired[:1500])
            raise e2
    
@app.get("/quiz_generate")
def quiz_generate(bone_name: str):
    try:
        bone = clean_bone_name(bone_name)
        prompt = build_quiz_prompt(bone)
        key = f"quiz_v2::{bone.lower()}" 

        # 1. Ask Gemini
        text_response = gemini_generate_text(
            prompt,
            key,
            temperature=0.2,          # lower randomness -> more valid JSON
            max_output_tokens=1200,   # enough space for 5 Qs + explanations
            use_cache=False,          # avoids “stuck” quiz outputs during testing
        )

        # 2-3) Extract + parse JSON robustly (with repair if needed)
        quiz_data = _loads_json_with_repair(text_response)

        # 4. TRANSFORMATION FOR UNITY
        for q in quiz_data:
            if isinstance(q.get("options"), dict):
                opts = q["options"]
                q["options"] = [
                    f"A: {opts.get('A', '-')}",
                    f"B: {opts.get('B', '-')}",
                    f"C: {opts.get('C', '-')}",
                    f"D: {opts.get('D', '-')}"
                ]

        print(f"\nQUIZ GENERATED for {bone}")
        return JSONResponse({"data": quiz_data})

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

class TTSRequest(BaseModel):
    text: str

@app.post("/tts")
def tts_endpoint(req: TTSRequest):
    try:
        text = req.text
        if not text:
            raise HTTPException(400, "Empty text")
            
        key = f"tts_quiz::{text[:50].lower()}"
        wav = google_tts_wav(text, key)
        
        return Response(content=wav, media_type="audio/wav")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))
    

# Assessment of the quiz
class AssessmentAnswer(BaseModel):
    question: str
    chosen_letter: str
    chosen_text: str
    correct_letter: str
    correct_text: str
    is_correct: bool
    explanation: Optional[str] = ""

class QuizAssessmentRequest(BaseModel):
    bone: Optional[str] = ""
    total: int
    correct: int
    answers: List[AssessmentAnswer]


def build_assessment_prompt(bone: str, total: int, correct: int, answers: list[AssessmentAnswer]) -> str:
    bone = bone or "general anatomy"
    total = max(int(total), 0)
    correct = max(int(correct), 0)
    wrong = max(total - correct, 0)
    percent = (correct / total * 100.0) if total > 0 else 0.0

    incorrect = [a for a in answers if not a.is_correct]
    incorrect = incorrect[:5]  # limit to keep prompt compact

    missed_block_lines = []
    for i, a in enumerate(incorrect, start=1):
        missed_block_lines.append(
            f"{i}) Q: {a.question}\n"
            f"   You answered: {a.chosen_letter} - {a.chosen_text}\n"
            f"   Correct answer: {a.correct_letter} - {a.correct_text}\n"
            f"   Explanation: {a.explanation or ''}"
        )
    missed_block = "\n".join(missed_block_lines).strip()

    return (
        "You are an experienced anatomy tutor.\n"
        "Write a short performance-based assessment addressed directly to the learner using 'you' language.\n"
        "Rules:\n"
        "- Plain text only. No markdown.\n"
        "- 20 to 30 words.\n"
        "- Address the learner as 'You' (e.g., 'You did well on...'). Do not say 'the student'.\n"
        "- Be specific, supportive, and actionable.\n"
        "- Include: score interpretation, 2 strengths, 2 weaknesses, and a 3-step study plan.\n"
        "- If there are incorrect answers, refer to them as themes (do not repeat every full question).\n\n"
        f"Topic/Bone: {bone}\n"
        f"Score: {correct}/{total} ({percent:.0f}%)\n"
        f"Incorrect count: {wrong}\n\n"
        f"Missed examples (for pattern detection):\n{missed_block if missed_block else 'None'}\n"
    )

@app.post("/quiz_assessment")
def quiz_assessment(req: QuizAssessmentRequest):
    try:
        # Basic validation/sanity
        if req.total <= 0:
            raise HTTPException(400, "total must be > 0")
        if req.correct < 0 or req.correct > req.total:
            raise HTTPException(400, "correct must be between 0 and total")
        if not req.answers:
            raise HTTPException(400, "answers list is required")

        bone = clean_bone_name(req.bone or "")
        prompt = build_assessment_prompt(bone, req.total, req.correct, req.answers)

        # Do NOT cache assessments while testing
        key = f"assessment::{bone.lower()}::{req.correct}/{req.total}"
        assessment = gemini_generate_text(
            prompt,
            key,
            temperature=0.4,
            max_output_tokens=100,
            use_cache=False,
        ).strip()

        return JSONResponse({"assessment": assessment})
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)
    
    
    
