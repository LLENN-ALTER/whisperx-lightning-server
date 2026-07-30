import os
import re
import httpx
from urllib.parse import urlparse
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Header, Depends, File, UploadFile, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

try:
    from lightning_sdk import Studio
except ImportError:
    Studio = None

app = FastAPI(
    title="SRT Suite - Render Proxy & Studio Controller",
    version="1.0.0"
)

# ==========================================
# CONFIGURAZIONE AMBIENTE E STUDIO
# ==========================================
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY", "SRT_SUITE_SECRET_TOKEN_2026")
LIGHTNING_API_KEY = os.getenv("LIGHTNING_API_KEY", "")
LIGHTNING_STUDIO_ID = os.getenv("LIGHTNING_STUDIO_ID", "01kyf6tebbywg1d835f6ptkgt5")
LIGHTNING_STUDIO_NAME = os.getenv("LIGHTNING_STUDIO_NAME", "gpu-studio")

# Pulisce ed estrae STRICTLY solo la Base URL (schema://host)
raw_url_env = os.getenv("LIGHTNING_STUDIO_URL", "https://8001-01kyf6tebbywg1d835f6ptkgt5.cloudspaces.litng.ai")
match = re.search(r'https?://[^\s\)\]]+', raw_url_env)
extracted_url = match.group(0) if match else raw_url_env.strip()

# Parsing pulito dell'origin base
parsed_url = urlparse(extracted_url)
LIGHTNING_BASE_URL = f"{parsed_url.scheme}://{parsed_url.netloc}"

USER_NAME = "xmauri99"
ORG_NAME = "xmauri99-org"
TEAMSPACE_NAME = "get-gpu-project"

if LIGHTNING_API_KEY:
    os.environ["LIGHTNING_API_KEY"] = LIGHTNING_API_KEY


def verify_token(authorization: str = Header(None)):
    """Verifica che la richiesta arrivi dall'app SRT Suite autorizzata."""
    if not authorization or authorization != f"Bearer {APP_SECRET_KEY}":
        raise HTTPException(
            status_code=401, 
            detail="Non autorizzato. Token di sicurezza mancante o errato."
        )


@app.get("/")
@app.get("/health")
def read_root():
    return {"status": "online", "service": "SRT Suite Proxy API"}


# ==========================================
# ENDPOINT DI TRASCRIZIONE DEFINITIVO (PROXY IN STREAMING)
# ==========================================
@app.post("/api/v1/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    authorized: None = Depends(verify_token)
):
    """Inoltra il file audio dall'app Android allo Studio Lightning AI per la trascrizione e fa il proxy dello stream NDJSON."""
    if not LIGHTNING_BASE_URL:
        raise HTTPException(status_code=500, detail="LIGHTNING_STUDIO_URL non configurato su Render.")
        
    target_url = f"{LIGHTNING_BASE_URL}/api/v1/transcribe"

    # 1. Lettura completa dei byte inviati dall'app Android
    file_bytes = await file.read()
    
    if not file_bytes or len(file_bytes) == 0:
        print("❌ ERRORE: Il file ricevuto dall'app Android è vuoto (0 byte)!")
        raise HTTPException(status_code=400, detail="Il file audio inviato è vuoto.")

    # 2. Normalizzazione del nome file e del Content-Type
    original_name = file.filename or "recording.m4a"
    content_type = file.content_type if (file.content_type and "/" in file.content_type) else "audio/m4a"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Authorization": f"Bearer {APP_SECRET_KEY}"
    }

    print(f"🚀 Proxy -> Invio file '{original_name}' ({len(file_bytes)} byte) a Lightning ({target_url})...")

    files_payload = {
        "file": (original_name, file_bytes, content_type)
    }

    async def proxy_stream_generator():
        # Nessun timeout di read per permettere le operazioni lunghe della GPU senza tagliare la connessione
        timeout_settings = httpx.Timeout(connect=30.0, read=None, write=300.0, pool=30.0)

        try:
            async with httpx.AsyncClient(timeout=timeout_settings, follow_redirects=True) as client:
                # Usiamo .stream() per leggere la risposta di Lightning mano a mano che arriva
                async with client.stream("POST", target_url, files=files_payload, headers=headers) as response:
                    
                    print(f"📩 Inizio stream da Lightning: Status {response.status_code}")
                    
                    if response.status_code != 200:
                        error_body = await response.aread()
                        print(f"⚠️ LIGHTNING REJECTED REQUEST ({response.status_code}): {error_body.decode()}")
                        # Facciamo lo yield dell'errore formattato come json line
                        yield f'{{"status": "error", "log": "Lightning Server Error ({response.status_code})", "error": true}}\n'
                        return

                    # Legge e inolta ogni singola riga NDJSON al client Android (inclusi gli Heartbeat)
                    async for line in response.aiter_lines():
                        if line:
                            yield line + "\n"

        except httpx.RequestError as exc:
            print(f"❌ Errore di connessione proxy tra Render e Lightning: {exc}")
            yield f'{{"status": "error", "log": "Impossibile raggiungere Lightning Studio: {str(exc)}", "error": true}}\n'

    # Ritorna lo StreamingResponse assicurandosi che Render non faccia buffering
    return StreamingResponse(
        proxy_stream_generator(), 
        media_type="application/x-ndjson",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


# ==========================================
# FUNZIONE DI SMART REBUILDING SOTTOTITOLI
# ==========================================
class RebuildRequest(BaseModel):
    result: Dict[str, Any]
    max_chars_per_line: Optional[int] = 38
    max_gap_seconds: Optional[float] = 0.6


def rebuild_segments_smart(result: dict, max_chars_per_line: int = 38, max_gap_seconds: float = 0.6):
    all_words = []
    
    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            if "start" in word and "end" in word and "word" in word:
                all_words.append({
                    "word": word["word"].strip(),
                    "start": word["start"],
                    "end": word["end"],
                    "speaker": word.get("speaker", segment.get("speaker", "UNKNOWN"))
                })

    if not all_words:
        return result.get("segments", [])

    new_segments = []
    current_words = []
    current_speaker = all_words[0]["speaker"]
    current_start = all_words[0]["start"]
    last_end = all_words[0]["end"]

    MAX_BLOCK_CHARS = max_chars_per_line * 2 

    def commit_segment(words, start, end, speaker):
        if not words:
            return
        text = " ".join([w["word"] for w in words])
        
        if len(text) > max_chars_per_line:
            mid_point = len(words) // 2
            line1 = " ".join([w["word"] for w in words[:mid_point]])
            line2 = " ".join([w["word"] for w in words[mid_point:]])
            formatted_text = f"{line1}\n{line2}"
        else:
            formatted_text = text

        new_segments.append({
            "start": start,
            "end": end,
            "text": formatted_text,
            "speaker": speaker,
            "words": words
        })

    for i, w in enumerate(all_words):
        word_text = w["word"]
        speaker_changed = (w["speaker"] != current_speaker)
        time_gap = (w["start"] - last_end) > max_gap_seconds
        current_text_len = sum(len(x["word"]) + 1 for x in current_words) + len(word_text)
        
        prev_word_has_punctuation = False
        if current_words:
            last_word_str = current_words[-1]["word"]
            prev_word_has_punctuation = any(last_word_str.endswith(p) for p in [".", "?", "!", ",", ";"])

        must_split = (
            speaker_changed 
            or time_gap 
            or (current_text_len >= MAX_BLOCK_CHARS)
            or (current_text_len > max_chars_per_line and prev_word_has_punctuation)
        )

        if must_split and current_words:
            commit_segment(current_words, current_start, last_end, current_speaker)
            current_words = []
            current_speaker = w["speaker"]
            current_start = w["start"]

        current_words.append(w)
        last_end = w["end"]

        if any(word_text.endswith(p) for p in [".", "?", "!"]):
            commit_segment(current_words, current_start, last_end, current_speaker)
            current_words = []
            if i + 1 < len(all_words):
                next_w = all_words[i + 1]
                current_speaker = next_w["speaker"]
                current_start = next_w["start"]

    if current_words:
        commit_segment(current_words, current_start, last_end, current_speaker)

    return new_segments


@app.post("/api/v1/subtitles/rebuild")
def process_subtitles(request: RebuildRequest, authorized: None = Depends(verify_token)):
    try:
        updated_segments = rebuild_segments_smart(
            result=request.result,
            max_chars_per_line=request.max_chars_per_line or 38,
            max_gap_seconds=request.max_gap_seconds or 0.6
        )
        return {"status": "success", "segments": updated_segments}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore durante l'elaborazione dei sottotitoli: {str(e)}")


# ==========================================
# CONTROL ENDPOINTS
# ==========================================
def _get_studio_instance():
    if Studio is None:
        raise Exception("lightning-sdk non installato.")

    errors = []

    try:
        full_path = f"{ORG_NAME}/{TEAMSPACE_NAME}/{LIGHTNING_STUDIO_NAME}"
        print(f"🔍 Tentativo 1: Studio('{full_path}')")
        return Studio(name=full_path)
    except Exception as e:
        errors.append(f"T1: {e}")

    try:
        print(f"🔍 Tentativo 2: Studio('{LIGHTNING_STUDIO_NAME}', teamspace='{TEAMSPACE_NAME}', org='{ORG_NAME}')")
        return Studio(name=LIGHTNING_STUDIO_NAME, teamspace=TEAMSPACE_NAME, org=ORG_NAME)
    except Exception as e:
        errors.append(f"T2: {e}")

    try:
        print(f"🔍 Tentativo 3: Studio('{LIGHTNING_STUDIO_NAME}', teamspace='{TEAMSPACE_NAME}')")
        return Studio(name=LIGHTNING_STUDIO_NAME, teamspace=TEAMSPACE_NAME)
    except Exception as e:
        errors.append(f"T3: {e}")

    raise Exception(" | ".join(errors))


def _async_start_task():
    """Funzione helper per avviare lo Studio in background."""
    try:
        s = _get_studio_instance()
        s.start()
        print("✅ Studio avviato in background con successo!")
    except Exception as e:
        print(f"❌ Errore durante l'avvio in background dello Studio: {e}")


@app.post("/api/v1/studio/start")
async def start_studio(
    background_tasks: BackgroundTasks, 
    authorized: None = Depends(verify_token)
):
    if not LIGHTNING_API_KEY:
        raise HTTPException(status_code=500, detail="LIGHTNING_API_KEY mancante nelle Environment Variables di Render.")

    try:
        print("🚀 Invio comando di avvio Studio in background...")
        background_tasks.add_task(_async_start_task)
        return {
            "status": "starting",
            "message": "Studio in fase di avvio..."
        }
    except Exception as e:
        print(f"❌ Errore START Studio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Errore durante l'avvio dello Studio: {str(e)}")


@app.post("/api/v1/studio/stop")
async def stop_studio(authorized: None = Depends(verify_token)):
    if not LIGHTNING_API_KEY:
        raise HTTPException(status_code=500, detail="LIGHTNING_API_KEY mancante nelle Environment Variables di Render.")

    try:
        print("🛑 Arresto Studio in corso...")
        s = _get_studio_instance()
        s.stop()
        return {"status": "success", "message": "Studio arrestato con successo!"}
    except Exception as e:
        print(f"❌ Errore STOP Studio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Errore durante l'arresto dello Studio: {str(e)}")


@app.get("/api/v1/studio/status")
async def get_status(authorized: None = Depends(verify_token)):
    if not LIGHTNING_BASE_URL:
        return {"status": "stopped", "stage": "Not Configured"}

    target_url = f"{LIGHTNING_BASE_URL}/health"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            res = await client.get(target_url, headers=headers)
            print(f"[STATUS CHECK] Risposta da {target_url}: STATUS {res.status_code}")
            if res.status_code == 200:
                return {"status": "running", "stage": "Running"}
    except Exception as e:
        print(f"[STATUS CHECK ERROR] Errore durante il check a {target_url}: {e}")
        
    return {"status": "stopped", "stage": "Stopped"}


@app.get("/api/v1/credits")
def get_credits(authorized: None = Depends(verify_token)):
    return {"status": "success", "credits": 14.21}

