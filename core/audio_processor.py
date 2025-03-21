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
logger = logging.getLogger("audio-processor")

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
    Returns segments with start time, end time, text, and confidence scores
    
    Args:
        audio_file: Path to the audio file
        language: Optional language code (e.g., 'hi' for Hindi, None for auto-detection)
    """
    logger.info(f"Transcribing audio file {audio_file} with language={language}")
    
    # Load a larger model for better multilingual support
    model = whisper.load_model("medium")
    
    # Load audio with librosa
    audio = librosa.load(audio_file, sr=16000)[0]  # Whisper requires 16kHz sample rate
    
    # Configure transcription to include word-level timestamps and confidence scores
    # Note: We need to enable word timestamps to get per-segment confidence
    transcribe_options = {
        "word_timestamps": True,  # Enable word-level details
        "suppress_tokens": [-1],  # Don't suppress any tokens
        "without_timestamps": False,  # Keep timestamps
        "max_initial_timestamp": None,
        "fp16": torch.cuda.is_available()  # Use fp16 if GPU is available
    }
    
    # Add language if specified
    if language and language != "auto":
        transcribe_options["language"] = language
    
    # Transcribe with options
    detected_language = None
    try:
        # Convert audio to torch tensor if not already
        if not isinstance(audio, torch.Tensor):
            audio_tensor = torch.tensor(audio)
        else:
            audio_tensor = audio
            
        # Run transcription with our options
        result = model.transcribe(audio_tensor, **transcribe_options)
        
        # Extract detected language
        detected_language = result.get('language')
        
        if language:
            logger.info(f"Transcribed with specified language: {language}")
        else:
            logger.info(f"Transcribed with auto-detected language: {detected_language}")
    except Exception as e:
        logger.error(f"Error in transcription: {str(e)}")
        # Fallback to English if transcription fails
        transcribe_options["language"] = "en"
        try:
            result = model.transcribe(audio, **transcribe_options)
            logger.info("Falling back to English transcription")
        except Exception as e2:
            logger.error(f"Even fallback transcription failed: {str(e2)}")
            raise RuntimeError(f"Failed to transcribe audio: {str(e2)}")
    
    # Add the detected language to the result if it wasn't provided
    if not language and detected_language:
        result['detected_language'] = detected_language
    
    # Process segments to add confidence scores
    enhanced_segments = []
    for segment in result["segments"]:
        # Extract basic segment info
        segment_info = {
            "id": segment.get("id", 0),
            "start": segment.get("start", 0),
            "end": segment.get("end", 0),
            "text": segment.get("text", "").strip(),
        }
        
        # Extract confidence scores
        # Method 1: Average token probabilities if available
        if "avg_logprob" in segment:
            # Convert log probability to confidence percentage (0-100)
            # logprob is negative, closer to 0 means higher confidence
            # Typical values range from -1 (high confidence) to -5 (low confidence)
            logprob = segment["avg_logprob"]
            # Scale to 0-100 range, with -1 or better mapping to ~90-100%
            confidence = min(100, max(0, 100 + 20 * logprob))
            segment_info["confidence"] = round(confidence, 2)
            segment_info["avg_logprob"] = logprob
        else:
            # If avg_logprob not available, try a different approach
            segment_info["confidence"] = None
            
        # Method 2: Try to get token-level probabilities if available
        if "tokens" in segment and "token_probs" in segment:
            tokens = segment.get("tokens", [])
            token_probs = segment.get("token_probs", [])
            
            # Only process if we have valid data
            if tokens and token_probs and len(tokens) == len(token_probs):
                # Calculate average probability for non-None values
                valid_probs = [p for p in token_probs if p is not None]
                if valid_probs:
                    avg_token_prob = sum(valid_probs) / len(valid_probs)
                    # Convert to percentage
                    token_confidence = 100 * avg_token_prob
                    segment_info["token_confidence"] = round(token_confidence, 2)
                    
                    # If we didn't have avg_logprob, use this as the main confidence
                    if segment_info["confidence"] is None:
                        segment_info["confidence"] = segment_info["token_confidence"]
                        
                # Include token-level details if detailed logging is needed
                token_details = []
                for i, (token, prob) in enumerate(zip(tokens, token_probs)):
                    if prob is not None:
                        token_details.append({
                            "token": token,
                            "probability": prob
                        })
                segment_info["tokens"] = token_details
        
        # Method 3: If no probability data available, estimate from no_speech_prob
        if segment_info["confidence"] is None and "no_speech_prob" in segment:
            # Lower no_speech_prob means higher speech confidence
            speech_conf = 100 * (1 - segment.get("no_speech_prob", 0))
            segment_info["confidence"] = round(speech_conf, 2)
            segment_info["no_speech_prob"] = segment.get("no_speech_prob", 0)
        
        # Final fallback: If we still don't have confidence, set a default
        if segment_info["confidence"] is None:
            segment_info["confidence"] = 50.0  # Default mid-range confidence
            segment_info["confidence_source"] = "default"
        
        # Add word-level timestamps and confidence if available
        if "words" in segment:
            words_with_confidence = []
            for word in segment["words"]:
                word_info = {
                    "word": word.get("word", ""),
                    "start": word.get("start", 0),
                    "end": word.get("end", 0)
                }
                
                # Add probability if available
                if "probability" in word:
                    word_info["confidence"] = round(100 * word.get("probability", 0), 2)
                
                words_with_confidence.append(word_info)
            
            segment_info["words"] = words_with_confidence
        
        # Add segment confidence categorization
        if segment_info["confidence"] >= 90:
            segment_info["confidence_level"] = "high"
        elif segment_info["confidence"] >= 70:
            segment_info["confidence_level"] = "medium"
        else:
            segment_info["confidence_level"] = "low"
            
        enhanced_segments.append(segment_info)
    
    # Replace the original segments with our enhanced ones
    result["segments"] = enhanced_segments
    
    # Add overall confidence metrics
    if enhanced_segments:
        confidences = [seg["confidence"] for seg in enhanced_segments if "confidence" in seg]
        if confidences:
            result["overall_confidence"] = {
                "average": round(sum(confidences) / len(confidences), 2),
                "min": round(min(confidences), 2),
                "max": round(max(confidences), 2)
            }
            
            # Flag if there are any low confidence segments
            low_confidence_segments = [s for s in enhanced_segments if s.get("confidence_level") == "low"]
            result["low_confidence_count"] = len(low_confidence_segments)
            result["low_confidence_percentage"] = round(100 * len(low_confidence_segments) / len(enhanced_segments), 2)
    
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

def format_conversation(diarization_result, transcription_segments):
    """
    Align transcription with speaker segments
    Returns conversation data with speaker labels and confidence scores
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
        segment_confidence = []
        segment_details = []
        
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
            # Add safety check for zero-length segments
            if segment_duration > 0 and overlap_duration / segment_duration >= 0.5:
                segment_text.append(segment["text"].strip())
                
                # Add confidence score if available
                if "confidence" in segment:
                    segment_confidence.append(segment["confidence"])
                
                # Store full segment details for later reference
                segment_details.append({
                    "text": segment["text"].strip(),
                    "start": segment["start"],
                    "end": segment["end"],
                    "confidence": segment.get("confidence", None),
                    "confidence_level": segment.get("confidence_level", None),
                    "words": segment.get("words", [])
                })
                
                used_segments.add(seg_idx)  # Mark segment as used
                
        if segment_text:
            # Prepare the conversation segment
            conversation_segment = {
                "speaker": speaker,
                "text": ' '.join(segment_text).strip(),
                "start_time": turn_start,
                "end_time": turn_end
            }
            
            # Add confidence metrics if available
            if segment_confidence:
                conversation_segment["confidence"] = round(sum(segment_confidence) / len(segment_confidence), 2)
                
                # Categorize the overall segment confidence
                if conversation_segment["confidence"] >= 90:
                    conversation_segment["confidence_level"] = "high"
                elif conversation_segment["confidence"] >= 70:
                    conversation_segment["confidence_level"] = "medium"
                else:
                    conversation_segment["confidence_level"] = "low"
            
            # Add detailed segment information
            if segment_details:
                conversation_segment["segments"] = segment_details
            
            conversation_data.append(conversation_segment)
    
    return conversation_data

def process_audio_file(audio_file_path: str, language: Optional[str] = None) -> Dict[str, Any]:
    """
    Process audio file to extract transcript with speaker identification and confidence scores
    
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
        'language': language or 'auto-detect',
        'confidence_metrics': {}
    }
    
    try:
        start_total = time.time()
        
        # Transcribe with timing
        step_start = time.time()
        transcription_result = transcribe_audio(audio_file_path, language)
        transcription_segments = transcription_result["segments"]  # Contains 'start', 'end', 'text', 'confidence'
        metrics['step_times']['transcription'] = time.time() - step_start
        
        # Include overall confidence metrics if available
        if "overall_confidence" in transcription_result:
            metrics['confidence_metrics'] = transcription_result["overall_confidence"]
            
            # Add counts of low confidence segments
            if "low_confidence_count" in transcription_result:
                metrics['confidence_metrics']["low_confidence_count"] = transcription_result["low_confidence_count"]
                metrics['confidence_metrics']["low_confidence_percentage"] = transcription_result["low_confidence_percentage"]
        
        # If language was auto-detected, get the detected language
        if not language or language == "auto":
            # First try to get the language from the detected_language field we added
            if 'detected_language' in transcription_result:
                detected_language = transcription_result['detected_language']
                metrics['language'] = detected_language  # Store the actual language code
                logger.info(f"Using explicit detected_language field: {detected_language}")
            # Fall back to looking in the language field
            elif 'language' in transcription_result:
                detected_language = transcription_result['language']
                metrics['language'] = detected_language  # Store the actual language code
                logger.info(f"Using language field from result: {detected_language}")
            # Fall back to looking in the segments if needed
            elif transcription_segments and len(transcription_segments) > 0:
                first_segment = transcription_segments[0]
                if 'language' in first_segment:
                    detected_language = first_segment['language']
                    metrics['language'] = detected_language  # Store the actual language code
                    logger.info(f"Extracted language from first segment: {detected_language}")
                else:
                    logger.warning(f"No language field in first segment. Keys: {list(first_segment.keys())}")
            else:
                logger.warning("Could not determine specific language from transcription")
                # Keep auto-detect in this case
        
        # Diarize with timing
        logger.info(f"Diarizing audio file {audio_file_path}")
        step_start = time.time()
        diarization_result = diarize_audio(audio_file_path)
        metrics['step_times']['diarization'] = time.time() - step_start
        
        # Format with timing
        step_start = time.time()
        conversation_data = format_conversation(diarization_result, transcription_segments)
        metrics['step_times']['formatting'] = time.time() - step_start
        
        # Create enhanced transcript with timestamps and confidence indicators
        formatted_transcript = []
        for seg in conversation_data:
            time_str = format_time(seg['start_time'])
            
            # Add confidence indicator to formatted text if available
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
        
        # Final metrics
        metrics['total_time'] = time.time() - start_total
        metrics['formatted_transcript'] = formatted_transcript
        
        # Enhanced transcript with confidence scores
        metrics['transcript'] = [{
            'speaker': seg['speaker'],
            'text': seg['text'],
            'start_time': seg['start_time'],
            'end_time': seg['end_time'],
            'start_time_formatted': format_time(seg['start_time']),
            'end_time_formatted': format_time(seg['end_time']),
            'confidence': seg.get('confidence', None),  # Include confidence if available
            'confidence_level': seg.get('confidence_level', None),  # Include confidence level
            'segments': seg.get('segments', []),  # Include detailed segment info
            'language': metrics['language']
        } for seg in conversation_data]
        
        # Final check to confirm the language is being properly returned
        logger.info(f"Final language being returned: {metrics['language']}")
        logger.info(f"Overall confidence: {metrics.get('confidence_metrics', {}).get('average', 'N/A')}%")
        
        return metrics
    
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
        raise