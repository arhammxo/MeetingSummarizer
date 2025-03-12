import whisper
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import torch
import os
import numpy as np
import librosa
import time
from datetime import datetime
import tempfile
import logging
from typing import Dict, List, Any, Optional, Callable
from pydub import AudioSegment
import soundfile as sf

# Configure logging
logger = logging.getLogger("audio-service")

# Enable for better performance if using CUDA
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def format_time(seconds: float) -> str:
    """Format seconds into HH:MM:SS format"""
    return datetime.utcfromtimestamp(seconds).strftime('%H:%M:%S')

def transcribe_audio(audio_file: str, language: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Transcribe audio file using Whisper model
    Returns segments with start time, end time, and text
    
    Args:
        audio_file: Path to the audio file
        language: Optional language code (e.g., 'hi' for Hindi, None for auto-detection)
    """
    logger.info(f"Transcribing audio file {audio_file} with language={language}")
    
    # Load a larger model for better multilingual support
    model = whisper.load_model("medium")
    
    # Load audio with librosa
    audio = librosa.load(audio_file, sr=16000)[0]  # Whisper requires 16kHz sample rate
    
    # Transcribe with language specification if provided
    if language and language != "auto":
        result = model.transcribe(audio, language=language)
        logger.info(f"Transcribed with specified language: {language}")
    else:
        try:
            # Convert audio to torch tensor
            audio_tensor = torch.tensor(audio)
            result = model.transcribe(audio)
            logger.info(f"Transcribed with auto-detected language: {result.get('language', 'unknown')}")
        except Exception as e:
            logger.error(f"Error in language detection: {str(e)}")
            # Fallback to English if detection fails
            result = model.transcribe(audio, language="en")
            logger.info("Falling back to English transcription")
    
    return result["segments"]  # Contains 'start', 'end', 'text'

def diarize_audio(audio_file: str) -> Any:
    """
    Perform speaker diarization on audio file
    Returns segments with speaker identifications
    """
    logger.info(f"Diarizing audio file {audio_file}")
    
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

def format_conversation(diarization_result: Any, transcription_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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

def process_audio_file(audio_file_path: str, language: Optional[str] = None) -> Dict[str, Any]:
    """
    Process audio file to extract transcript with speaker identification
    
    Args:
        audio_file_path: Path to the audio file
        language: Optional language code (e.g., 'hi' for Hindi, None for auto-detection)
        
    Returns:
        Dictionary containing the transcript data and processing metrics
    """
    metrics = {
        'total_time': 0,
        'step_times': {},
        'transcript': [],
        'formatted_transcript': [],
        'language': language or 'auto-detect'
    }
    
    try:
        start_total = time.time()
        
        # Transcribe with timing
        step_start = time.time()
        transcription_segments = transcribe_audio(audio_file_path, language)
        metrics['step_times']['transcription'] = time.time() - step_start
        
        # If language was auto-detected, get the detected language
        if not language and transcription_segments:
            # The language is available in the first segment's language attribute
            try:
                metrics['language'] = transcription_segments[0].get('language', 'auto-detected')
            except:
                # If we can't get it from the segment, at least we tried
                pass
        
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
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
        raise

def split_audio(audio_path: str, chunk_duration: int = 600, overlap: int = 15) -> List[Dict[str, Any]]:
    """
    Split long audio files into smaller chunks with overlap
    
    Args:
        audio_path: Path to the audio file
        chunk_duration: Duration of each chunk in seconds (default: 600s = 10 minutes)
        overlap: Overlap between chunks in seconds (default: 15s)
        
    Returns:
        List of paths to the temporary chunk files
    """
    logger.info(f"Splitting audio file {audio_path} into chunks of {chunk_duration}s with {overlap}s overlap")
    
    # Load audio file with pydub (handles more formats)
    try:
        audio = AudioSegment.from_file(audio_path)
    except Exception as e:
        logger.error(f"Error loading audio with pydub: {e}")
        # Fallback to librosa
        audio_array, sr = librosa.load(audio_path, sr=None)
        total_duration = len(audio_array) / sr
        logger.info(f"Loaded with librosa. Duration: {total_duration}s, Sample rate: {sr}Hz")
        
        # Create chunks with librosa
        chunk_files = []
        for start_time in range(0, int(total_duration), chunk_duration - overlap):
            end_time = min(start_time + chunk_duration, total_duration)
            if end_time - start_time < 30:  # Skip very short segments
                continue
                
            # Extract chunk
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            chunk = audio_array[start_sample:end_sample]
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                sf.write(temp_file.name, chunk, sr)
                chunk_files.append({
                    "path": temp_file.name,
                    "start_time": start_time,
                    "end_time": end_time
                })
                
        return chunk_files
    
    # Process with pydub if successful
    audio_duration = len(audio) / 1000  # pydub uses milliseconds
    logger.info(f"Loaded with pydub. Duration: {audio_duration}s")
    
    chunk_files = []
    for start_time in range(0, int(audio_duration), chunk_duration - overlap):
        end_time = min(start_time + chunk_duration, audio_duration)
        if end_time - start_time < 30:  # Skip very short segments
            continue
            
        # Extract chunk (pydub uses milliseconds)
        chunk = audio[start_time * 1000:(end_time * 1000)]
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            chunk.export(temp_file.name, format="wav")
            chunk_files.append({
                "path": temp_file.name,
                "start_time": start_time,
                "end_time": end_time
            })
    
    logger.info(f"Split audio into {len(chunk_files)} chunks")
    return chunk_files

def process_long_audio(
    audio_file_path: str, 
    language: Optional[str] = None, 
    chunk_duration: int = 600,
    progress_callback: Optional[Callable[[int, str], None]] = None
) -> Dict[str, Any]:
    """
    Process a long audio file by splitting it into chunks
    
    Args:
        audio_file_path: Path to the audio file
        language: Optional language code
        chunk_duration: Duration of each chunk in seconds
        progress_callback: Optional callback function for progress updates
        
    Returns:
        Dictionary containing the transcript data and processing metrics
    """
    # Implementation based on long_recording_processor.py
    # This would include chunk processing, merging results, etc.
    # For brevity, this is omitted here but would follow the same pattern as in the original file
    
    # For this example, we'll provide a simplified implementation:
    start_total = time.time()
    
    metrics = {
        'total_time': 0,
        'step_times': {},
        'transcript': [],
        'formatted_transcript': [],
        'language': language or 'auto-detect',
        'chunks_processed': 0,
        'total_chunks': 0
    }
    
    try:
        # Split audio into chunks
        if progress_callback:
            progress_callback(5, "Splitting audio into chunks")
            
        chunks = split_audio(audio_file_path, chunk_duration=chunk_duration)
        metrics['total_chunks'] = len(chunks)
        
        if progress_callback:
            progress_callback(10, f"Split audio into {len(chunks)} chunks")
        
        # Process each chunk individually and merge results
        all_transcriptions = []
        all_conversation_data = []
        
        for i, chunk in enumerate(chunks):
            if progress_callback:
                progress = 10 + (i / len(chunks) * 80)
                progress_callback(int(progress), f"Processing chunk {i+1}/{len(chunks)}")
                
            # Process this chunk
            chunk_result = process_audio_file(chunk["path"], language)
            
            # Adjust timestamps
            for segment in chunk_result["transcript"]:
                segment["start_time"] += chunk["start_time"]
                segment["end_time"] += chunk["start_time"]
                segment["start_time_formatted"] = format_time(segment["start_time"])
                segment["end_time_formatted"] = format_time(segment["end_time"])
                all_conversation_data.append(segment)
            
            # Clean up temporary file
            try:
                os.unlink(chunk["path"])
            except Exception as e:
                logger.error(f"Error removing temp file: {str(e)}")
            
            metrics['chunks_processed'] += 1
            
        # Sort by start time
        all_conversation_data.sort(key=lambda x: x["start_time"])
        
        # Create formatted transcript
        formatted_transcript = []
        for seg in all_conversation_data:
            time_str = format_time(seg['start_time'])
            formatted_line = f"[{time_str}] Speaker {seg['speaker']}: {seg['text']}"
            formatted_transcript.append(formatted_line)
        
        # Final metrics
        metrics['total_time'] = time.time() - start_total
        metrics['formatted_transcript'] = formatted_transcript
        metrics['transcript'] = all_conversation_data
        
        if progress_callback:
            progress_callback(100, "Processing complete")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Long audio processing failed: {str(e)}", exc_info=True)
        raise