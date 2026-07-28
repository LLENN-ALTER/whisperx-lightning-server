import os
import shutil
import tempfile
import torch
import whisperx
import uvicorn
from fastapi import FastAPI, HTTPException, File, UploadFile

app = FastAPI(title="SRT Suite - WhisperX GPU Engine")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"
HF_TOKEN = os.getenv("HF_TOKEN", "")

print(f"⚡ Inizializzazione WhisperX Engine su {DEVICE}...")
whisper_model = whisperx.load_model("large-v2", DEVICE, compute_type=COMPUTE_TYPE, language="it")

diarize_model = None
if HF_TOKEN:
    try:
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
        print("✅ Modello Diarizzazione caricato!")
    except Exception as e:
        print(f"⚠️ Errore caricamento Diarizzazione: {e}")

@app.get("/")
@app.get("/health")
def health():
    return {"status": "online", "engine": "WhisperX GPU"}

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name

        # 1. Trascrizione
        audio = whisperx.load_audio(temp_path)
        result = whisper_model.transcribe(audio, batch_size=BATCH_SIZE)

        # 2. Alignment
        language_code = result.get("language", "it")
        model_a, metadata = whisperx.load_align_model(language_code=language_code, device=DEVICE)
        result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE, return_char_alignments=False)

        # 3. Diarizzazione
        if diarize_model is not None:
            try:
                diarize_segments = diarize_model(audio)
                result = whisperx.assign_word_speakers(diarize_segments, result)
            except Exception as d_err:
                print(f"Errore Diarizzazione: {d_err}")

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
