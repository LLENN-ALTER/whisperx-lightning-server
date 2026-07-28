import os
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import whisperx
import whisperx.diarize as whisperx_diarize

app = FastAPI(title="WhisperX Engine Server")

device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

print(f"⚡ Inizializzazione WhisperX Engine su {device}...")

whisper_model = whisperx.load_model("large-v2", device, compute_type=compute_type)

HF_TOKEN = os.getenv("HF_TOKEN", "")
diarize_model = None

if HF_TOKEN:
    try:
        # Nota: si usa token= invece del vecchio use_auth_token=
        diarize_model = whisperx_diarize.DiarizationPipeline(token=HF_TOKEN, device=device)
        print("✅ Modello Diarizzazione caricato!")
    except Exception as e:
        print(f"⚠️ Errore caricamento Diarizzazione: {e}")
else:
    print("⚠️ HF_TOKEN non trovato. La diarizzazione sarà disabilitata.")

@app.get("/health")
def health():
    return {"status": "ok", "device": device}

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form("it"),
    diarize: bool = Form(True)
):
    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        audio = whisperx.load_audio(temp_path)
        result = whisper_model.transcribe(audio, batch_size=16, language=language)

        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

        if diarize and diarize_model:
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)

        return {"segments": result["segments"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
