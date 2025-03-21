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
from pathlib import Path

# Configure logging
logger = logging.getLogger("audio-service")

# Enable for better performance if using CUDA
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def format_time(seconds: float) -> str:
    """Format seconds into HH:MM:SS format"""
    return datetime.utcfromtimestamp(seconds).strftime('%H:%M:%S')

def preprocess_audio(audio_file: str) -> str:
    """
    Preprocess audio file to ensure compatibility with diarization
    
    Args:
        audio_file: Path to the audio file
        
    Returns:
        Path to preprocessed audio file
    """
    logger.info(f"Preprocessing audio file: {audio_file}")
    
    try:
        # Create a temporary file for the processed audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            processed_file = tmp_file.name
            
        # Load the audio file with pydub (handles many formats)
        audio = AudioSegment.from_file(audio_file)
        
        # Convert to standard format: WAV, mono, 16kHz, 16-bit
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        
        # Export to the temporary file
        audio.export(processed_file, format="wav")
        
        logger.info(f"Preprocessed audio saved to: {processed_file}")
        return processed_file
        
    except Exception as e:
        logger.error(f"Error preprocessing audio: {str(e)}")
        # If preprocessing fails, return the original file
        return audio_file

def transcribe_audio(audio_file: str, language: Optional[str] = None) -> Dict[str, Any]:
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
    detected_language = None
    if language and language != "auto":
        result = model.transcribe(audio, language=language)
        logger.info(f"Transcribed with specified language: {language}")
    else:
        try:
            # Convert audio to torch tensor
            audio_tensor = torch.tensor(audio)
            result = model.transcribe(audio)
            # Important: Store the detected language
            detected_language = result.get('language')
            logger.info(f"Transcribed with auto-detected language: {detected_language}")
        except Exception as e:
            logger.error(f"Error in language detection: {str(e)}")
            # Fallback to English if detection fails
            result = model.transcribe(audio, language="en")
            logger.info("Falling back to English transcription")
    
    # Add the detected language to the result if it wasn't provided
    if not language and detected_language:
        result['detected_language'] = detected_language
    
    return result

def diarize_audio(audio_file: str) -> Any:
    """
    Perform speaker diarization on audio file
    Returns segments with speaker identifications
    
    Args:
        audio_file: Path to the audio file
        
    Returns:
        Diarization result
    """
    from services.audio_converter import is_mp3_file, convert_audio_to_wav
    
    logger.info(f"Starting diarization for file: {audio_file}")
    
    # Track if we create a temporary file that needs cleanup
    temp_file = None
    
    try:
        # Convert MP3 to WAV if needed
        if is_mp3_file(audio_file):
            logger.info(f"Converting MP3 file to WAV before diarization: {audio_file}")
            temp_file = convert_audio_to_wav(audio_file)
            logger.info(f"Using converted file for diarization: {temp_file}")
            audio_file_to_process = temp_file
        else:
            audio_file_to_process = audio_file
        
        # Replace with your own HuggingFace token or use environment variable
        hf_token = os.environ.get("HUGGINGFACE_TOKEN", "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")
        
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token)
        
        # Use GPU if available
        diarization_pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Process with progress hook
        with ProgressHook() as hook:
            diarization_result = diarization_pipeline(audio_file_to_process, hook=hook)
            logger.info("Diarization completed successfully")
            return diarization_result
            
    except Exception as e:
        logger.error(f"Diarization error: {str(e)}")
        
        # Fall back to simpler diarization approach
        from pyannote.core import Segment, Annotation
        
        logger.warning("Creating fallback diarization result")
        annotation = Annotation()
        
        try:
            # Try to get file duration
            y, sr = librosa.load(audio_file, sr=16000, mono=True, duration=10)
            duration = librosa.get_duration(y=y, sr=sr)
        except:
            # Default duration if we can't determine it
            duration = 300  # 5 minutes
            
        # Create a single segment with one speaker
        segment = Segment(0, duration)
        annotation[segment] = "SPEAKER_00"
        
        return annotation
        
    finally:
        # Clean up any temporary file we created
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
                logger.info(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Could not delete temporary file {temp_file}: {str(e)}")

def create_fallback_diarization(audio_file: str) -> Any:
    """
    Create a fallback diarization result when pyannote fails
    
    Args:
        audio_file: Path to the audio file
        
    Returns:
        A mock diarization result
    """
    logger.warning(f"Creating fallback diarization for {audio_file}")
    
    try:
        # Import pyannote core for annotation objects
        from pyannote.core import Segment, Annotation
        
        # Create a simple annotation with just one speaker if we can't process properly
        annotation = Annotation()
        
        # Try to get file duration
        try:
            y, sr = librosa.load(audio_file, sr=16000, mono=True, duration=10)
            duration = librosa.get_duration(y=y, sr=sr)
        except:
            # Default duration if we can't determine it
            duration = 300  # 5 minutes
            
        # Create a single segment with one speaker
        segment = Segment(0, duration)
        annotation[segment] = "SPEAKER_00"
        
        logger.info(f"Created fallback diarization with duration {duration}s")
        return annotation
        
    except Exception as e:
        logger.error(f"Fallback diarization failed: {str(e)}")
        
        # If all else fails, create an extremely minimal annotation
        from pyannote.core import Segment, Annotation
        annotation = Annotation()
        segment = Segment(0, 300)  # Assume 5 minutes
        annotation[segment] = "SPEAKER_00"
        
        return annotation

def fallback_diarization(audio_file: str) -> Any:
    """
    Provide a fallback diarization when pyannote fails
    This creates a very simple speaker separation based on silence detection
    
    Args:
        audio_file: Path to the audio file
        
    Returns:
        Mock diarization result that can be used by the formatter
    """
    logger.warning("Using fallback diarization method")
    
    try:
        # Load audio
        y, sr = librosa.load(audio_file, sr=16000, mono=True)
        
        # Detect speech segments based on energy levels
        non_silent_intervals = librosa.effects.split(y, top_db=30)
        
        # Create a mock diarization result
        # This is a simplified version that mimics pyannote.audio's format
        from pyannote.core import Segment, Timeline, Annotation
        
        # Create an annotation to hold our segments
        annotation = Annotation()
        
        # Create some segments based on silence detection
        current_speaker = 0
        for i, (start, end) in enumerate(non_silent_intervals):
            # Convert sample indices to seconds
            start_sec = start / sr
            end_sec = end / sr
            
            # Change speaker when we detect a longer pause (more than 1 second)
            if i > 0:
                prev_end = non_silent_intervals[i-1][1] / sr
                if start_sec - prev_end > 1.0:
                    current_speaker = (current_speaker + 1) % 3  # Rotate between 3 speakers
            
            # Add segment to the annotation
            segment = Segment(start_sec, end_sec)
            annotation[segment] = str(current_speaker)
        
        logger.info(f"Fallback diarization created {len(annotation)} segments with {len(set(annotation.labels()))} speakers")
        return annotation
        
    except Exception as e:
        logger.error(f"Fallback diarization failed: {str(e)}")
        
        # Create an extremely simple mock result with just one speaker
        from pyannote.core import Segment, Annotation
        
        annotation = Annotation()
        
        # Just create one segment for the entire audio
        try:
            # Try to get audio duration
            y, sr = librosa.load(audio_file, sr=16000, mono=True, duration=10)  # Just load a bit to get info
            duration = librosa.get_duration(y=y, sr=sr)
            segment = Segment(0, duration)
        except:
            # If all else fails, assume a 5-minute audio
            segment = Segment(0, 300)
            
        annotation[segment] = "0"
        
        logger.warning("Created emergency single-speaker diarization")
        return annotation

def format_conversation(diarization_result: Any, transcription_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Align transcription with speaker segments
    Returns conversation data with speaker labels
    """
    conversation_data = []  # List to hold speaker segments
    used_segments = set()
    
    try:
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
                if segment_duration > 0 and overlap_duration / segment_duration >= 0.5:
                    segment_text.append(segment["text"].strip())
                    used_segments.add(seg_idx)  # Mark segment as used
                    
            if segment_text:
                conversation_data.append({
                    "speaker": speaker,
                    "text": ' '.join(segment_text).strip(),
                    "start_time": turn_start,
                    "end_time": turn_end
                })
    except Exception as e:
        logger.error(f"Error formatting conversation: {str(e)}")
        
        # Fallback: assign all unassigned segments to speakers in sequence
        if not conversation_data:
            logger.warning("Conversation formatting failed, using emergency fallback")
            
            # Reset used segments
            used_segments = set()
            
            # Just assign segments to speakers sequentially
            current_speaker = 0
            for seg_idx, segment in enumerate(transcription_segments):
                if seg_idx in used_segments:
                    continue
                    
                conversation_data.append({
                    "speaker": str(current_speaker),
                    "text": segment["text"].strip(),
                    "start_time": segment["start"],
                    "end_time": segment["end"]
                })
                
                used_segments.add(seg_idx)
                current_speaker = (current_speaker + 1) % 3  # Rotate between 3 speakers
    
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
        
        # Check if we're dealing with an MP3 file
        file_ext = Path(audio_file_path).suffix.lower()
        if file_ext == '.mp3':
            logger.info(f"Processing MP3 file: {audio_file_path}")
            metrics['file_type'] = 'mp3'
        else:
            metrics['file_type'] = file_ext.lstrip('.')
        
        # Step 1: Transcribe with timing
        step_start = time.time()
        transcription_result = transcribe_audio(audio_file_path, language)
        transcription_segments = transcription_result["segments"]
        metrics['step_times']['transcription'] = time.time() - step_start
        
        # Step 2: Extract language if auto-detecting
        if not language or language == "auto":
            # Check if whisper detected a language
            detected_language = None
            
            # Try different possible locations for the language info
            if 'detected_language' in transcription_result:
                detected_language = transcription_result['detected_language']
            elif 'language' in transcription_result:
                detected_language = transcription_result['language']
            elif transcription_segments and 'language' in transcription_segments[0]:
                detected_language = transcription_segments[0]['language']
            
            if detected_language:
                metrics['language'] = detected_language
                logger.info(f"Detected language: {detected_language}")
            else:
                logger.warning("Could not detect specific language")
        
        # Step 3: Diarize with timing - with special handling for MP3
        step_start = time.time()
        diarization_result = diarize_audio(audio_file_path)
        metrics['step_times']['diarization'] = time.time() - step_start
        
        # Step 4: Format conversation
        step_start = time.time()
        conversation_data = format_conversation(diarization_result, transcription_segments)
        metrics['step_times']['formatting'] = time.time() - step_start
        
        # Step 5: Create formatted transcript
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
            'end_time_formatted': format_time(seg['end_time']),
            'language': metrics['language']
        } for seg in conversation_data]
        
        logger.info(f"Successfully processed {metrics['file_type']} file in {metrics['total_time']:.2f}s")
        return metrics
    
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
        raise