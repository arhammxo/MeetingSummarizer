# Codebase Architecture & Walkthrough

If you want to understand how the Meeting Summarizer goes from a raw audio file to a clean, structured JSON summary, you need to follow the data as it flows through the application. 

Here is the exact journey of a meeting through your codebase.

---

## Stage 1: The User Interface (`app.py`)

This is the front door of your application, built using Streamlit.

1. **Input Handling:** 
   The user can either upload an audio file, upload a text transcript, or paste a transcript directly. If they upload an audio file, it triggers the heavy lifting.
2. **Context & Participants:**
   Before hitting "Generate", the app collects the `participants` list and the optional `meeting_context` (which we just built!).
3. **Execution Routing:**
   When the user clicks "Generate Summary", `app.py` looks at the length of the meeting. 
   - If it's a short meeting, it sends the text directly to `summarize_meeting` in `core/lg.py`.
   - If it's flagged as a long recording, it routes the text to `summarize_long_meeting` in `core/summarize_long_transcripts.py`.

---

## Stage 2: Audio Processing (`core/audio_processor.py`)

If the user uploaded an audio file (like an `.mp3` or `.webm`), the raw audio must be converted into text before the AI can summarize it.

1. **Format Standardization:** 
   The code checks if the audio is a 16kHz Mono WAV file (the only format AI models accept natively). If it's not, it uses `ffmpeg` to forcefully convert it.
2. **Transcription (Whisper):** 
   The audio is fed into the local **OpenAI Whisper** model. Whisper returns the spoken text along with timestamps (e.g., *[00:15 - 00:20] "Let's update the database"*).
3. **Diarization (Pyannote):** 
   Simultaneously, the audio is fed to **Pyannote**. Pyannote listens to the pitch and tone of the voices and groups them together (e.g., *Voice A spoke from 00:15 - 00:20*).
4. **Merging the Timelines:**
   The code loops through Whisper's text and overlaps it with Pyannote's voice tags. This results in the beautiful transcript you see in the app: `[00:15] Speaker 0: Let's update the database.`

---

## Stage 3: The AI Brain (LangGraph) (`core/lg.py`)

Once we have a text transcript, we need to extract intelligence from it. We use **LangGraph** to build a "State Machine" that forces the Mistral LLM to think step-by-step, rather than trying to do everything at once.

1. **The State (`AgentState`):** 
   Think of this as the AI's short-term memory backpack. It carries the `transcript`, `participants`, and your `context` instructions.
2. **Node 1: `analyze_node`**
   The AI reads the transcript and generates a high-level JSON map of the meeting: the emotional tone, participation levels, and main topics discussed.
3. **Node 2: `summarize_node`**
   The AI looks at the transcript *and* the analysis from Node 1, and writes a 3-sentence summary and the key decisions.
4. **Node 3: `extract_actions_node`**
   The AI makes a third pass over the transcript, strictly looking for promises, tasks, and deadlines. It outputs these as a strict JSON array.
5. **Node 4: `format_output_node`**
   The final step merges all the JSON data into one clean payload and hands it back to `app.py` to be displayed on the screen.

---

## Stage 4: Long Meetings (`core/summarize_long_transcripts.py`)

If a meeting is 2 hours long, passing the entire transcript to Mistral at once will cause it to run out of memory or hallucinate. We solve this using a **Hierarchical Map-Reduce** strategy.

1. **The Slicer (`chunk_transcript_by_time`):**
   The code physically cuts the massive transcript into 10-minute chunks.
2. **The Map Phase (`summarize_transcript_chunk`):**
   The code loops through the chunks, asking Mistral to summarize Chunk 1, then Chunk 2, then Chunk 3 entirely independently. (Your specific `context` instruction is injected into every single one of these chunks so Mistral stays on track).
3. **The Reduce Phase (`hierarchical_summarize`):**
   The code takes the mini-summaries from all the chunks, stitches them together into one document, and asks Mistral to read *that* document to write the Final Master Summary.

---

### In Summary:
The app converts audio to text (`audio_processor`), decides if it needs to slice it up into chunks (`summarize_long_transcripts.py`), and uses a step-by-step AI workflow (`lg.py`) to extract highly accurate, contextual summaries before displaying them on your screen (`app.py`).
