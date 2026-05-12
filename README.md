# Meeting Summarizer (100% Local AI Pipeline)

A robust, fully on-premise AI pipeline that transcribes audio, identifies distinct speakers (diarization), and generates concise summaries and actionable items using entirely local models. No cloud APIs, no data leaving your machine!

## Core Architecture

The application is built on a 4-stage pipeline that ensures maximum privacy and capability:

1. **Audio Pre-processing (`services/audio_converter.py`)**
   - Automatically detects formats (WAV, MP3, M4A, WEBM, OGG, FLAC).
   - Uses `ffmpeg` to standardize everything to `16kHz Mono WAV` format, which is strictly required by the downstream AI models.

2. **Transcription (`core/audio_processor.py`)**
   - **Model:** Local OpenAI Whisper (`medium` model).
   - Converts raw audio waveforms into text with word-level timestamps and confidence scores.

3. **Speaker Diarization (`core/audio_processor.py`)**
   - **Model:** Pyannote Audio (`pyannote/speaker-diarization-3.1`).
   - Analyzes acoustic voice fingerprints (pitch, resonance) to detect *who* is speaking *when*. It groups the voice segments and labels them (e.g., Speaker 0, Speaker 1) without needing facial recognition.
   - The timelines from Whisper and Pyannote are merged to create a perfectly aligned transcript.

4. **Summarization & Action Extraction (`core/lg.py`)**
   - **Model:** Mistral (via local Ollama).
   - **Orchestration:** LangGraph state machine.
   - The LLM reads the beautifully formatted transcript (`[00:01:23] Speaker 0: Let's launch on Friday.`) and generates a structured JSON payload containing the overall summary, key decisions, and assigned action items.

---

##  Handling Long Recordings (>15 Minutes)

LLMs have restricted "context windows" and Whisper can suffer from memory exhaustion or hallucination loops on extremely long audio files. To solve this, the app uses a **Hierarchical Map-Reduce** strategy (`core/long_recording_processor.py`):

1. **Audio Slicing:** The `pydub` library physically cuts the massive audio file into manageable **10-minute chunks**.
2. **Context Overlaps:** It adds a **15-second overlap** between chunks so that if someone is mid-sentence at the 10-minute mark, their word isn't severed in half.
3. **Sequential Processing:** It runs Whisper and Pyannote on each 10-minute chunk individually, stitching the final text and timestamps back together.
4. **Hierarchical Summarization:** The text is passed back in blocks. Mistral writes a "mini-summary" for each 10-minute block. Finally, all the mini-summaries are concatenated, and Mistral reads that document to write the "Master Summary" and final Action List.

---

## 🌍 Multilingual Support

The Whisper `medium` model is highly capable of understanding and transcribing non-English languages. 

The UI explicitly supports selecting the following languages to optimize accuracy (by providing Whisper with a strict language hint):
- **English** (`en`)
- **Hindi** (`hi`)
- **Spanish** (`es`)
- **French** (`fr`)
- **German** (`de`)
- **Chinese** (`zh`)
- **Japanese** (`ja`)
- **Russian** (`ru`)
- **Arabic** (`ar`)

**Auto-Detect:** If you select "Auto-detect", Whisper analyzes the first 30 seconds of audio to detect the language automatically. It supports over 90 languages out-of-the-box.

---

## 🚀 Setup & Installation

### Prerequisites
<<<<<<< HEAD

- Python 3.9+
- An OpenAI API key (for GPT-4o access)

### Download speaker diarization models
```
# run this ONCE with your HF token
from huggingface_hub import snapshot_download

HF_TOKEN = "your_hf_token_here"  # Needed for one-time download

print("Downloading speaker-diarization-3.1...")
snapshot_download(
    "pyannote/speaker-diarization-3.1",
    token=HF_TOKEN,
    local_dir="./models/speaker-diarization-3.1"
)

print("Downloading segmentation-3.0...")
snapshot_download(
    "pyannote/segmentation-3.0",
    token=HF_TOKEN,
    local_dir="./models/segmentation-3.0"
)

print("All models downloaded.")
```
### Installation

1. Clone this repository
2. Install dependencies:

1. **Ollama:** Must be installed and running locally.
   ```bash
   ollama pull mistral
   ollama serve
   ```
2. **FFmpeg:** Must be installed on your system for audio conversion.
   ```bash
   # macOS
   brew install ffmpeg
   ```
3. **HuggingFace Token:** Pyannote requires a free HuggingFace access token.
   ```bash
   export HUGGINGFACE_TOKEN="your_token_here"
   ```

### Running the App
```bash
pip install -r requirements.txt
streamlit run app.py
```

Navigate to `http://localhost:8502` to start uploading your meeting recordings!


