import os
import sounddevice as sd
import soundfile as sf
import tempfile
import whisper


# Inject ffmpeg path explicitly (adjust this path if needed)
os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin"

# Project root = folder containing this file
project_root = os.path.dirname(os.path.abspath(__file__))

# Logs directory inside project
log_dir = os.path.join(project_root, "logs")
os.makedirs(log_dir, exist_ok=True)

# Full path to whisper_prompt.txt
output_txt = os.path.join(log_dir, "whisper_prompt.txt")

# Settings
duration = 5  # Seconds
samplerate = 16000
channels = 1

# Ensure default input device is valid
devices = sd.query_devices()
try:
    if not sd.default.device or sd.default.device[0] is None:
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                sd.default.device = (i, None)
                break
except Exception:
    pass  # Fallback to default if setting fails

# Record audio
print("Recording...")
filename = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype='int16')
sd.wait()
sf.write(filename, audio, samplerate)

# Transcribe with Whisper
print("Transcribing with Whisper...")
model = whisper.load_model("base.en")

cad_prompt = (
    "This is about computer-aided design using FreeCAD. "
    "Common terms include create, built, add, edit, delete, remove, move, help, selected, cube, box, cylinder, cone, sphere, torus, prism, polyline, helix, revolve, loft, sweep, and boolean operations like union, cut, and intersection. "
    "Sketching terms include line, arc, circle, constraint, coincident, vertical, horizontal, symmetric, and fully constrained sketch. "
    "3D operations include pad, pocket, fillet, chamfer, revolve, mirror, pattern, extrude, and groove. "
    "Other keywords include datum plane, body, part, face, edge, vertex, solid, wire, shell, compound, comp solid, section view, parametric, macro, and placement."
)
result = model.transcribe(filename, initial_prompt=cad_prompt)
text = result.get("text", "").strip()

print(f"Transcription: {text}")

# Save text result
with open(output_txt, "w") as f:
    f.write(text)

# Clean up temp audio file
print(f"Temporary file saved at {filename}")
os.remove(filename)
