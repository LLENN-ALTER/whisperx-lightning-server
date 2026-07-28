import os
import torch
import uvicorn
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import whisperx
import whisperx.diarize as whisperx_diarize

# Variabili globali per i modelli
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

whisper_model = None
diarize_model = None
is_loading = False

async def load_models_in_background():
    """Carica i modelli in sottofondo SENZA bloccare il bind della porta 8000."""
    global whisper_model, diarize_model, is_loading
    is_loading = True
    print(f"⚡ Inizializzazione WhisperX Engine in background su {device}...")

    try:
        # Caricamento modello WhisperX (operazione bloccante eseguita in thread pool)
        whisper_model = await asyncio.to_thread(
            whisperx.load_model, "large-v2", device, compute_type=compute_type
        )
        print("✅ Modello WhisperX caricato con successo!")

        HF_TOKEN = os.getenv("HF_TOKEN", "")
        if HF_TOKEN:
            try:
                diarize_model = await asyncio.to_thread(
                    whisperx_diarize.DiarizationPipeline, token=HF_TOKEN, device=device
                )
                print("✅ Modello Diarizzazione caricato!")
            except Exception as e:
                print(f"⚠️ Errore caricamento Diarizzazione: {e}")
        else:
            print("⚠️ HF_TOKEN non trovato. La diarizzazione sarà disabilitata.")

    except Exception as e:
        print(f"❌ Errore critico durante il caricamento dei modelli: {e}")
    finally:
        is_loading = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Avvia la task di caricamento dei modelli SENZA attendere il suo completamento
    asyncio.create_task(load_models_in_background())
    yield
    # Pulizia alla chiusura dell'app
    global whisper_model, diarize_model
    whisper_model = None
    diarize_model = None

# Inizializzazione FastAPI
app = FastAPI(title="WhisperX Engine Server", lifespan=lifespan)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": device,
        "whisper_loaded": whisper_model is not None,
        "diarize_loaded": diarize_model is not None,
        "is_loading": is_loading
    }

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form("it"),
    diarize: bool = Form(True)
):
    if whisper_model is None:
        raise HTTPException(
            status_code=503, 
            detail="I modelli AI sono ancora in fase di caricamento in memoria. Riprova tra pochi secondi."
        )

    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # Elaborazione audio
        audio = whisperx.load_audio(temp_path)
        
        # Trascrizione
        result = await asyncio.to_thread(
            whisper_model.transcribe, audio, batch_size=16, language=language
        )

        # Allineamento temporale
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = await asyncio.to_thread(
            whisperx.align, result["segments"], model_a, metadata, audio, device, False
        )

        # Diarizzazione (se richiesta ed abilitata)
        if diarize and diarize_model:
            diarize_segments = await asyncio.to_thread(diarize_model, audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)

        return {"segments": result["segments"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
