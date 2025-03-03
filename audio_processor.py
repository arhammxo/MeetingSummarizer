import whisper
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import torch
import os
import librosa
import time
from datetime import datetime

# Enable for better performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def transcribe_audio(audio_file):
    """
    Transcribe audio file using Whisper model
    Returns segments with start time, end time, and text
    """
    model = whisper.load_model("base")
    
    # Add explicit path to ffmpeg if needed
    # Uncomment and modify if ffmpeg is not in your PATH
    # os.environ["PATH"] += os.pathsep + 'C:\\ffmpeg\\bin'
    
    # Load audio with librosa
    audio = librosa.load(audio_file, sr=16000)[0]  # Whisper requires 16kHz sample rate
    result = model.transcribe(audio)
    return result["segments"]  # Contains 'start', 'end', 'text'

def diarize_audio(audio_file):
    """
    Perform speaker diarization on audio file
    Returns segments with speaker identifications
    """
    # Replace with your own HuggingFace token or use environment variable
    hf_token = os.environ.get("HUGGINGFACE_TOKEN", "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")
    
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token)
    
    # Use GPU if available
    diarization_pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    with ProgressHook() as hook:
        diarization_result = diarization_pipeline(audio_file, hook=hook)
    
    return diarization_result

def format_conversation(diarization_result, transcription_segments):
    """
    Align transcription with speaker segments
    Returns conversation data with speaker labels
    """
    conversation_data = []  # List to hold speaker segments
    used_segments = set()
    
    # Sort both diarization results and transcription segments by start time
    diarization_turns = sorted(diarization_result.itertracks(yield_label=True), key=lambda x: x[0].start)
    transcription_segments = sorted(transcription_segments, key=lambda x: x["start"])
    
    for turn, _, speaker in diarization_turns:
        turn_start = turn.start
        turn_end = turn.end
        segment_text = []
        
        for seg_idx, segment in enumerate(transcription_segments):
            if seg_idx in used_segments:
                continue  # Skip already used segments
                
            seg_start = segment["start"]
            seg_end = segment["end"]
            
            # Calculate overlap duration
            overlap_start = max(turn_start, seg_start)
            overlap_end = min(turn_end, seg_end)
            overlap_duration = max(0, overlap_end - overlap_start)
            
            # Require at least 50% overlap with the speaker turn
            segment_duration = seg_end - seg_start
            if overlap_duration / segment_duration >= 0.5:
                segment_text.append(segment["text"].strip())
                used_segments.add(seg_idx)  # Mark segment as used
                
        if segment_text:
            conversation_data.append({
                "speaker": speaker,
                "text": ' '.join(segment_text).strip(),
                "start_time": turn_start,
                "end_time": turn_end
            })
    
    return conversation_data

def format_time(seconds):
    """Format seconds into HH:MM:SS format"""
    return datetime.utcfromtimestamp(seconds).strftime('%H:%M:%S')

def process_audio_file(audio_file_path):
    """
    Process audio file to extract transcript with speaker identification
    
    Args:
        audio_file_path: Path to the audio file
        
    Returns:
        Dictionary containing the transcript data and processing metrics
    """
    metrics = {
        'total_time': 0,
        'step_times': {},
        'transcript': [],
        'formatted_transcript': []
    }
    
    try:
        start_total = time.time()
        
        # Transcribe with timing
        step_start = time.time()
        transcription_segments = transcribe_audio(audio_file_path)
        metrics['step_times']['transcription'] = time.time() - step_start
        
        # Diarize with timing
        step_start = time.time()
        diarization_result = diarize_audio(audio_file_path)
        metrics['step_times']['diarization'] = time.time() - step_start
        
        # Format with timing
        step_start = time.time()
        conversation_data = format_conversation(diarization_result, transcription_segments)
        metrics['step_times']['formatting'] = time.time() - step_start
        
        # Create enhanced transcript with timestamps
        formatted_transcript = []
        for seg in conversation_data:
            time_str = format_time(seg['start_time'])
            formatted_line = f"[{time_str}] Speaker {seg['speaker']}: {seg['text']}"
            formatted_transcript.append(formatted_line)
        
        # Final metrics
        metrics['total_time'] = time.time() - start_total
        metrics['formatted_transcript'] = formatted_transcript
        metrics['transcript'] = [{
            'speaker': seg['speaker'],
            'text': seg['text'],
            'start_time': seg['start_time'],
            'end_time': seg['end_time'],
            'start_time_formatted': format_time(seg['start_time']),
            'end_time_formatted': format_time(seg['end_time'])
        } for seg in conversation_data]
        
        return metrics
    
    except Exception as e:
        print(f"Processing failed: {str(e)}")
        raise