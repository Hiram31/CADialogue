import os
import sounddevice as sd
import soundfile as sf
import tempfile
import whisper
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn


# Environment Setup
# Inject ffmpeg path explicitly (for Mac Homebrew)
os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin"

# Project root = folder containing this file
project_root = os.path.dirname(os.path.abspath(__file__))

# Logs directory inside project
log_dir = os.path.join(project_root, "logs")
os.makedirs(log_dir, exist_ok=True)

# Full path for transcription output
output_txt = os.path.join(log_dir, "whisper_prompt.txt")

# Ensure default microphone input device
devices = sd.query_devices()
try:
    if not sd.default.device or sd.default.device[0] is None:
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                sd.default.device = (i, None)
                break
except Exception:
    pass  # Use fallback default if necessary

# Load Whisper Model Once
print("Loading Whisper model...")
model = whisper.load_model("base.en")  # Use your preferred model
print("Model loaded!")

# CAD initial prompt
cad_prompt = (
    "This is about computer-aided design using FreeCAD. "
    "Common terms include create, built, add, edit, delete, remove, move, help, selected, cube, box, cylinder, cone, sphere, torus, prism, polyline, helix, revolve, loft, sweep, and boolean operations like union, cut, and intersection. "
    "Sketching terms include line, arc, circle, constraint, coincident, vertical, horizontal, symmetric, and fully constrained sketch. "
    "3D operations include pad, pocket, fillet, chamfer, revolve, mirror, pattern, extrude, and groove. "
    "Other keywords include datum plane, body, part, face, edge, vertex, solid, wire, shell, compound, comp solid, section view, parametric, macro, and placement."
)

# FastAPI Setup
app = FastAPI()

# Define request model
class RecordRequest(BaseModel):
    duration : int = 10  # seconds
    samplerate : int = 16000
    channels : int = 1
    save_to_file : bool = False  # Optional: whether to save transcription to a text file

@app.post("/record_and_transcribe")
def record_and_transcribe(req: RecordRequest):
    filename = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    try:
        print(f"🎙 Recording {req.duration} seconds...")
        recording = sd.rec(int(req.duration * req.samplerate), samplerate=req.samplerate, channels=req.channels, dtype='int16')
        sd.wait()
        sf.write(filename, recording, req.samplerate)

        print("🧠 Transcribing...")
        result = model.transcribe(filename, initial_prompt=cad_prompt)
        text = result.get("text", "").strip()

        # Optional: Save transcription to file
        if req.save_to_file:
            os.makedirs(os.path.dirname(output_txt), exist_ok=True)
            with open(output_txt, "w") as f:
                f.write(text)
            print(f"💾 Transcription saved at {output_txt}")

        return {"text": text}

    except Exception as e:
        return {"error": str(e)}

    finally:
        # Always clean up temp audio
        if os.path.exists(filename):
            os.remove(filename)

if __name__ == "__main__":
    # Warm up the microphone to reduce first call delay
    try:
        print("🔧 Warming up microphone...")
        with sd.InputStream(samplerate=16000, channels=1, dtype='int16'):
            pass
        print("🎤 Microphone ready.")
    except Exception as e:
        print(f"⚠️ Failed to warm up microphone: {e}")

    uvicorn.run(app, host="127.0.0.1", port=5005)
