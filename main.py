import os
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from lightning_sdk import Studio

app = FastAPI(
    title="SRT Suite - WhisperX & Studio Controller",
    version="1.0.0"
)

# ==========================================
# CONFIGURAZIONE AMBIENTE E STUDIO
# ==========================================
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY", "SRT_SUITE_SECRET_TOKEN_2026")
STUDIO_NAME = os.getenv("LIGHTNING_STUDIO_NAME", "gpu-studio")
TEAMSPACE = os.getenv("LIGHTNING_TEAMSPACE", "get-gpu-project")
USER_NAME = os.getenv("LIGHTNING_USER", "xmauri99")


def verify_token(authorization: str = Header(None)):
    """Verifica che la richiesta arrivi dall'app SRT Suite autorizzata."""
    if not authorization or authorization != f"Bearer {APP_SECRET_KEY}":
        raise HTTPException(
            status_code=401, 
            detail="Non autorizzato. Token di sicurezza mancante o errato."
        )


# ==========================================
# FUNZIONE DI SMART REBUILDING SOTTOTITOLI
# ==========================================
def rebuild_segments_smart(result: dict, max_chars_per_line: int = 38, max_gap_seconds: float = 0.6):
    """
    Riorganizza i sottotitoli a livello di singola parola per:
    1. Limitare RIGOROSAMENTE il sottotitolo a MAX 2 RIGHE totali (circa 76-80 caratteri max per blocco).
    2. Spezzare IMMEDIATAMENTE quando cambia lo speaker (max 1 riga per speaker se ravvicinati).
    3. Tagliare prima dei limiti di riga basandosi su punteggiatura e pause.
    """
    all_words = []
    
    # Estrae tutte le parole allineate con i relativi timestamp e speaker
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

    # Limite massimo di caratteri per un blocco da 2 righe
    MAX_BLOCK_CHARS = max_chars_per_line * 2 

    def commit_segment(words, start, end, speaker):
        if not words:
            return
        text = " ".join([w["word"] for w in words])
        
        # Se il testo supera la lunghezza di 1 riga, inseriamo un andata a capo '\n' bilanciata
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
        
        # Calcola la lunghezza potenziale del testo nel blocco corrente
        current_text_len = sum(len(x["word"]) + 1 for x in current_words) + len(word_text)
        
        # Controlla punteggiatura sulla parola precedente
        prev_word_has_punctuation = False
        if current_words:
            last_word_str = current_words[-1]["word"]
            prev_word_has_punctuation = any(last_word_str.endswith(p) for p in [".", "?", "!", ",", ";"])

        # CONDITIONAL SPLIT (Forza la chiusura del sottotitolo):
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

        # Se la parola attuale ha un punto forte (. ? !), chiude il blocco per non trascinare il testo oltre
        if any(word_text.endswith(p) for p in [".", "?", "!"]):
            commit_segment(current_words, current_start, last_end, current_speaker)
            current_words = []
            if i + 1 < len(all_words):
                next_w = all_words[i + 1]
                current_speaker = next_w["speaker"]
                current_start = next_w["start"]

    # Salva le ultime parole rimanenti
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
# ENDPOINT API
# ==========================================
@app.get("/")
def read_root():
    return {"status": "online", "service": "SRT Suite Backend API"}


@app.post("/api/v1/subtitles/rebuild")
def process_subtitles(request: RebuildRequest):
    """Endpoint per riorganizzare i sottotitoli usando rebuild_segments_smart."""
    try:
        updated_segments = rebuild_segments_smart(
            result=request.result,
            max_chars_per_line=request.max_chars_per_line,
            max_gap_seconds=request.max_gap_seconds
        )
        return {"status": "success", "segments": updated_segments}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore durante l'elaborazione dei sottotitoli: {str(e)}")


@app.post("/api/v1/studio/start")
def start_studio(authorized: None = Depends(verify_token)):
    """Avvia l'istanza di Lightning Studio."""
    try:
        s = Studio(name=STUDIO_NAME, teamspace=TEAMSPACE, user=USER_NAME)
        s.start()
        return {"status": "success", "message": f"Studio '{STUDIO_NAME}' avviato con successo!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nell'avvio dello Studio: {str(e)}")


@app.post("/api/v1/studio/stop")
def stop_studio(authorized: None = Depends(verify_token)):
    """Arresta l'istanza di Lightning Studio per fermare il consumo dei crediti."""
    try:
        s = Studio(name=STUDIO_NAME, teamspace=TEAMSPACE, user=USER_NAME)
        s.stop()
        return {"status": "success", "message": f"Studio '{STUDIO_NAME}' arrestato con successo!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nell'arresto dello Studio: {str(e)}")


@app.get("/api/v1/studio/status")
def get_status(authorized: None = Depends(verify_token)):
    """Recupera lo stato attuale dello Studio."""
    try:
        s = Studio(name=STUDIO_NAME, teamspace=TEAMSPACE, user=USER_NAME)
        return {"status": "success", "stage": str(s.status.stage)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nel recupero dello stato: {str(e)}")
        
