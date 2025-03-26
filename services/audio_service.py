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
from core.audio_processor import format_time, process_audio_file

# Configure logging
logger = logging.getLogger("audio-service")

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
        
        # Create formatted transcript with confidence indicators
        formatted_transcript = []
        for seg in all_conversation_data:
            time_str = format_time(seg['start_time'])
            
            # Add confidence indicator based on confidence level if available
            if 'confidence_level' in seg:
                confidence_indicator = ""
                if seg['confidence_level'] == "high":
                    confidence_indicator = "✓ "  # Check mark for high confidence
                elif seg['confidence_level'] == "medium":
                    confidence_indicator = "~ "  # Tilde for medium confidence
                else:
                    confidence_indicator = "? "  # Question mark for low confidence
                
                formatted_line = f"[{time_str}] {confidence_indicator}Speaker {seg['speaker']}: {seg['text']}"
            else:
                formatted_line = f"[{time_str}] Speaker {seg['speaker']}: {seg['text']}"
                
            formatted_transcript.append(formatted_line)
        
        # Calculate confidence metrics from all segments with confidence scores
        if all_conversation_data:
            confidences = [seg.get('confidence', 0) for seg in all_conversation_data if 'confidence' in seg]
            if confidences:
                metrics['confidence_metrics'] = {
                    "average": round(sum(confidences) / len(confidences), 2),
                    "min": round(min(confidences), 2),
                    "max": round(max(confidences), 2)
                }
                
                # Calculate low confidence segments stats
                low_confidence_segments = [s for s in all_conversation_data if s.get('confidence_level') == 'low']
                metrics['confidence_metrics']["low_confidence_count"] = len(low_confidence_segments)
                metrics['confidence_metrics']["low_confidence_percentage"] = round(
                    100 * len(low_confidence_segments) / len(all_conversation_data), 2
                )
        
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