from fastapi import FastAPI, UploadFile, File
import whisperx
from whisperx.diarize import DiarizationPipeline
import torch
import shutil
import os

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16
compute_type = "float16" if device == "cuda" else "int8"

# Prende il token dall'ambiente di sistema
HF_TOKEN = os.getenv("HF_TOKEN")

# 1. Carica modello WhisperX
model = whisperx.load_model("large-v2", device, compute_type=compute_type)

# 2. Carica modello Diarizzazione
try:
    if HF_TOKEN:
        diarize_model = DiarizationPipeline(token=HF_TOKEN, device=device)
        print("✅ Diarizzazione configurata con successo!")
    else:
        print("⚠️ HF_TOKEN non trovato! La diarizzazione non sarà attiva.")
        diarize_model = None
except Exception as e:
    print(f"⚠️ Errore caricamento diarizzazione: {e}")
    diarize_model = None

@app.get("/")
def home():
    return {"status": "ok", "device": device}

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    audio = whisperx.load_audio(temp_path)
    result = model.transcribe(audio, batch_size=batch_size)
    
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    
    if diarize_model:
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)
    
    os.remove(temp_path)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
