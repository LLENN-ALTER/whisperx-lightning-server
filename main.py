import os
import shutil
import tempfile
import httpx
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Header, Depends, File, UploadFile
from pydantic import BaseModel

# Import SDK ufficiale Lightning
try:
    from lightning_sdk import Studio
except ImportError:
    Studio = None

# ==========================================
# INIZIALIZZAZIONE FASTAPI
# ==========================================
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
LIGHTNING_STUDIO_URL = os.getenv("LIGHTNING_STUDIO_URL", "https://8001-01kyf6tebbywg1d835f6ptkgt5.cloudspaces.litng.ai")
TEAMSPACE = os.getenv("LIGHTNING_TEAMSPACE", "get-gpu-project")
USER_ORG = os.getenv("LIGHTNING_USER_ORG", "xmauri99-org")

# Configurazione variabili d'ambiente per Lightning SDK
if LIGHTNING_API_KEY:
    os.environ["LIGHTNING_API_KEY"] = LIGHTNING_API_KEY
    os.environ["LIGHTNING_USER"] = USER_ORG
    os.environ["LIGHTNING_USERNAME"] = USER_ORG
    os.environ["LIGHTNING_ORG"] = USER_ORG
    os.environ["LIGHTNING_TEAMSPACE"] = TEAMSPACE


# ==========================================
# FUNZIONE DI SMART REBUILDING SOTTOTITOLI
# ==========================================
def rebuild_segments_smart(result: dict, max_chars_per_line: int = 38, max_gap_seconds: float = 0.6):
    """Riorganizza i sottotitoli a livello di singola parola per MAX 2 RIGHE."""
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


# ==========================================
# SCHEMI DATI PER GLI ENDPOINT FASTAPI
# ==========================================
class RebuildRequest(BaseModel):
    result: Dict[str, Any]
    max_chars_per_line: Optional[int] = 38
    max_gap_seconds: Optional[float] = 0.6


# ==========================================
# ENDPOINT API (Proxy & Controllo Studio)
# ==========================================
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


@app.post("/transcribe")
@app.post("/api/v1/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Inoltra il file audio allo Studio Lightning AI per la trascrizione reale.
    """
    if not LIGHTNING_STUDIO_URL:
        raise HTTPException(status_code=500, detail="LIGHTNING_STUDIO_URL non configurato su Render.")
        
    try:
        content = await file.read()
        target_url = f"{LIGHTNING_STUDIO_URL.rstrip('/')}/transcribe"
        
        async with httpx.AsyncClient(timeout=600.0) as client:
            files = {"file": (file.filename, content, file.content_type)}
            response = await client.post(target_url, files=files)
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
            
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore proxy verso Lightning Studio: {str(e)}")


@app.post("/api/v1/subtitles/rebuild")
def process_subtitles(request: RebuildRequest):
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
# CONTROL ENDPOINTS (via Official Lightning SDK)
# ==========================================
def _get_lightning_studio():
    """Inizializza lo studio passando esplicitamente i dettagli di nome, teamspace e utente."""
    try:
        # Tenta con nome studio + teamspace
        print(f"🔍 Connessione Studio: {LIGHTNING_STUDIO_NAME} (Teamspace: {TEAMSPACE}, Org: {USER_ORG})")
        return Studio(name=LIGHTNING_STUDIO_NAME, teamspace=TEAMSPACE, user=USER_ORG)
    except Exception as e1:
        try:
            # Fallback 1: usa l'ID unico
            return Studio(name=LIGHTNING_STUDIO_ID)
        except Exception as e2:
            # Fallback 2: passa solo il nome con teamspace
            return Studio(name=f"{TEAMSPACE}/{LIGHTNING_STUDIO_NAME}")


@app.post("/api/v1/studio/start")
async def start_studio(authorized: None = Depends(verify_token)):
    if not LIGHTNING_API_KEY:
        raise HTTPException(status_code=500, detail="LIGHTNING_API_KEY mancante nelle Environment Variables di Render.")
    
    if Studio is None:
        raise HTTPException(status_code=500, detail="lightning-sdk non installato.")

    try:
        print("🚀 Avvio Studio tramite SDK...")
        s = _get_lightning_studio()
        s.start()
        return {"status": "success", "message": "Studio avviato con successo!"}
    except Exception as e:
        print(f"❌ Errore SDK START: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Errore SDK Lightning: {str(e)}")


@app.post("/api/v1/studio/stop")
async def stop_studio(authorized: None = Depends(verify_token)):
    if not LIGHTNING_API_KEY:
        raise HTTPException(status_code=500, detail="LIGHTNING_API_KEY mancante nelle Environment Variables di Render.")
    
    if Studio is None:
        raise HTTPException(status_code=500, detail="lightning-sdk non installato.")

    try:
        print("🛑 Arresto Studio tramite SDK...")
        s = _get_lightning_studio()
        s.stop()
        return {"status": "success", "message": "Studio arrestato con successo!"}
    except Exception as e:
        print(f"❌ Errore SDK STOP: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Errore SDK Lightning: {str(e)}")


@app.get("/api/v1/studio/status")
async def get_status(authorized: None = Depends(verify_token)):
    if not LIGHTNING_STUDIO_URL:
        return {"status": "stopped", "stage": "Not Configured"}
    
    # Controlla se lo Studio su Lightning è raggiungibile (porta 8001)
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            res = await client.get(f"{LIGHTNING_STUDIO_URL.rstrip('/')}/health")
            if res.status_code == 200:
                return {"status": "running", "stage": "Running"}
    except Exception:
        pass
        
    return {"status": "stopped", "stage": "Stopped"}


@app.get("/api/v1/credits")
def get_credits(authorized: None = Depends(verify_token)):
    return {"status": "success", "credits": 14.21}
