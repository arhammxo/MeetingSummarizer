import whisper
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import torch
import os
import librosa
import time
import numpy as np
from datetime import datetime
import tempfile
import soundfile as sf
from pydub import AudioSegment
import logging
from services.audio_service import process_audio_file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("long_audio_processor")

# Enable for better performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def split_audio(audio_path, chunk_duration=600, overlap=15):
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
        
        # If total duration is very short, just process the whole file
        if total_duration <= chunk_duration:
            logger.info(f"Audio is shorter than chunk size ({total_duration}s < {chunk_duration}s), processing as single chunk")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                sf.write(temp_file.name, audio_array, sr)
                chunk_files.append({
                    "path": temp_file.name,
                    "start_time": 0,
                    "end_time": total_duration
                })
            return chunk_files
        
        # For longer files, proceed with chunking
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
    
    # If audio is shorter than chunk size, just use the whole file
    if audio_duration <= chunk_duration:
        logger.info(f"Audio is shorter than chunk size ({audio_duration}s < {chunk_duration}s), processing as single chunk")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            audio.export(temp_file.name, format="wav")
            chunk_files.append({
                "path": temp_file.name,
                "start_time": 0,
                "end_time": audio_duration
            })
        return chunk_files
    
    # For longer files, create chunks
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

def transcribe_audio_chunk(chunk_path, offset_seconds=0, language=None):
    """
    Transcribe a single audio chunk with time offset adjustment
    
    Args:
        chunk_path: Path to the audio chunk
        offset_seconds: Time offset in seconds to adjust timestamps
        language: Optional language code
        
    Returns:
        List of transcription segments with adjusted timestamps and confidence scores
    """
    # Load a larger model for better multilingual support
    model = whisper.load_model("medium")
    
    # Load audio
    audio = librosa.load(chunk_path, sr=16000)[0]
    
    # Transcribe with language specification if provided
    if language:
        result = model.transcribe(audio, language=language)
    else:
        # First, detect the language from a sample
        audio_sample = audio[:48000]  # Use a short sample for detection
        detection_result = model.detect_language(audio_sample)
        detected_language = detection_result[0]
        
        # Then transcribe with the detected language
        result = model.transcribe(audio, language=detected_language)
        logger.info(f"Detected language: {detected_language}")
    
    # Adjust timestamps and add confidence information
    for segment in result["segments"]:
        segment["start"] += offset_seconds
        segment["end"] += offset_seconds
        
        # Extract or estimate confidence
        if "avg_logprob" in segment:
            logprob = segment["avg_logprob"]
            confidence = min(100, max(0, 100 + 20 * logprob))
            segment["confidence"] = round(confidence, 2)
            
            # Add confidence level categorization
            if confidence >= 90:
                segment["confidence_level"] = "high"
            elif confidence >= 70:
                segment["confidence_level"] = "medium"
            else:
                segment["confidence_level"] = "low"
    
    return result["segments"]

def diarize_audio_chunk(chunk_path, offset_seconds=0):
    """
    Perform speaker diarization on an audio chunk with time offset adjustment
    
    Args:
        chunk_path: Path to the audio chunk
        offset_seconds: Time offset in seconds to adjust timestamps
        
    Returns:
        Diarization result with adjusted timestamps
    """
    # Replace with your own HuggingFace token or use environment variable
    hf_token = os.environ.get("HUGGINGFACE_TOKEN", "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")
    
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token)
    
    # Use GPU if available
    diarization_pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    with ProgressHook() as hook:
        diarization_result = diarization_pipeline(chunk_path, hook=hook)
    
    # We need to adjust the timestamps in the diarization result
    adjusted_result = diarization_result.shift(offset_seconds)
    
    return adjusted_result

def merge_diarization_results(diarization_results):
    """
    Merge multiple diarization results into a single result
    with consistent speaker IDs across chunks
    
    Args:
        diarization_results: List of (diarization_result, chunk_info) tuples
        
    Returns:
        Merged diarization result
    """
    # This is complex and would require pyannote internals
    # For simplicity, we'll concatenate the results and map speakers
    # A production version would need speaker embedding clustering to identify same speakers across chunks
    
    # Map speaker IDs across chunks (simple approach)
    speaker_map = {}  # Maps original_chunk_id + speaker_id to global_speaker_id
    next_global_id = 0
    
    merged_turns = []
    
    for idx, (diarization, _) in enumerate(diarization_results):
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            chunk_speaker_key = f"{idx}_{speaker}"
            
            # Assign a global speaker ID if we haven't seen this chunk_speaker before
            if chunk_speaker_key not in speaker_map:
                speaker_map[chunk_speaker_key] = next_global_id
                next_global_id += 1
                
            global_speaker = speaker_map[chunk_speaker_key]
            merged_turns.append((turn, global_speaker))
    
    # Sort turns by start time
    merged_turns.sort(key=lambda x: x[0].start)
    
    return merged_turns

def format_conversation(diarization_turns, transcription_segments):
    """
    Align transcription with speaker segments
    Returns conversation data with speaker labels and confidence scores
    """
    conversation_data = []
    used_segments = set()
    
    # Sort transcription segments by start time
    transcription_segments = sorted(transcription_segments, key=lambda x: x["start"])
    
    for turn, speaker in diarization_turns:
        turn_start = turn.start
        turn_end = turn.end
        segment_text = []
        segment_confidence = []
        
        for seg_idx, segment in enumerate(transcription_segments):
            if seg_idx in used_segments:
                continue
                
            seg_start = segment["start"]
            seg_end = segment["end"]
            
            # Calculate overlap
            overlap_start = max(turn_start, seg_start)
            overlap_end = min(turn_end, seg_end)
            overlap_duration = max(0, overlap_end - overlap_start)
            
            segment_duration = seg_end - seg_start
            if segment_duration > 0 and overlap_duration / segment_duration >= 0.5:
                segment_text.append(segment["text"].strip())
                
                # Add confidence score collection
                if "confidence" in segment:
                    segment_confidence.append(segment["confidence"])
                
                used_segments.add(seg_idx)
                
        if segment_text:
            # Create conversation segment with confidence information
            conversation_segment = {
                "speaker": speaker,
                "text": ' '.join(segment_text).strip(),
                "start_time": turn_start,
                "end_time": turn_end
            }
            
            # Add confidence metrics if available
            if segment_confidence:
                avg_confidence = sum(segment_confidence) / len(segment_confidence)
                conversation_segment["confidence"] = round(avg_confidence, 2)
                
                # Add confidence level categorization
                if avg_confidence >= 90:
                    conversation_segment["confidence_level"] = "high"
                elif avg_confidence >= 70:
                    conversation_segment["confidence_level"] = "medium"
                else:
                    conversation_segment["confidence_level"] = "low"
            
            conversation_data.append(conversation_segment)
    
    return conversation_data

def format_time(seconds):
    """Format seconds into HH:MM:SS format"""
    return datetime.utcfromtimestamp(seconds).strftime('%H:%M:%S')

def process_long_audio(audio_file_path, language=None, chunk_duration=600, progress_callback=None):
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
        # Check if file is MP3 and convert if needed
        from pathlib import Path
        from services.audio_converter import is_mp3_file, convert_audio_to_wav
        
        original_file = audio_file_path
        if is_mp3_file(audio_file_path):
            if progress_callback:
                progress_callback(5, f"Converting MP3 to WAV format")
            try:
                audio_file_path = convert_audio_to_wav(audio_file_path)
                logger.info(f"Converted MP3 to WAV: {audio_file_path}")
            except Exception as e:
                logger.error(f"Error converting MP3: {str(e)}")
                # Continue with original file
        
        # Split audio into chunks
        step_start = time.time()
        if progress_callback:
            progress_callback(10, f"Splitting audio into chunks")
            
        chunks = split_audio(audio_file_path, chunk_duration=chunk_duration)
        metrics['total_chunks'] = len(chunks)
        metrics['step_times']['splitting'] = time.time() - step_start
        
        if progress_callback:
            progress_callback(15, f"Split audio into {len(chunks)} chunks")
        
        # Handle case where no chunks were created (should not happen with our updated split_audio)
        if len(chunks) == 0:
            logger.warning("No chunks were created. This should not happen with the updated code.")
            logger.info("Processing the entire file as a single chunk")
            
            # Process the entire file directly using standard processing
            if progress_callback:
                progress_callback(20, "Processing entire file as single chunk")
                
            result = process_audio_file(audio_file_path, language=language)
            
            # Copy relevant data to our metrics structure
            metrics['transcript'] = result['transcript']
            metrics['formatted_transcript'] = result['formatted_transcript']
            metrics['language'] = result['language']
            metrics['total_time'] = time.time() - start_total
            
            if progress_callback:
                progress_callback(100, "Processing complete")
                
            return metrics
        
        # Process each chunk
        all_transcription_segments = []
        all_diarization_results = []
        
        for i, chunk in enumerate(chunks):
            if progress_callback:
                progress_percentage = 20 + (i / len(chunks) * 60)  # 20-80% for processing chunks
                progress_callback(progress_percentage, f"Processing chunk {i+1}/{len(chunks)}")
            
            chunk_path = chunk["path"]
            offset = chunk["start_time"]
            
            # Transcribe chunk
            logger.info(f"Transcribing chunk {i+1}/{len(chunks)}")
            chunk_transcription = transcribe_audio_chunk(chunk_path, offset, language)
            all_transcription_segments.extend(chunk_transcription)
            
            # If first chunk, store detected language
            if i == 0 and not language and chunk_transcription:
                try:
                    metrics['language'] = chunk_transcription[0].get('language', 'auto-detected')
                except:
                    pass
            
            # Diarize chunk
            logger.info(f"Diarizing chunk {i+1}/{len(chunks)}")
            chunk_diarization = diarize_audio_chunk(chunk_path, offset)
            all_diarization_results.append((chunk_diarization, chunk))
            
            # Clean up temporary file
            try:
                os.unlink(chunk_path)
            except:
                pass
            
            metrics['chunks_processed'] += 1
            
        if progress_callback:
            progress_callback(80, "Merging results")
        
        # Merge diarization results
        step_start = time.time()
        merged_diarization_turns = merge_diarization_results(all_diarization_results)
        metrics['step_times']['merging'] = time.time() - step_start
        
        if progress_callback:
            progress_callback(90, "Formatting conversation")
        
        # Format into conversation
        step_start = time.time()
        conversation_data = format_conversation(merged_diarization_turns, all_transcription_segments)
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
        metrics['transcript'] = [{
            'speaker': seg['speaker'],
            'text': seg['text'],
            'start_time': seg['start_time'],
            'end_time': seg['end_time'],
            'start_time_formatted': format_time(seg['start_time']),
            'end_time_formatted': format_time(seg['end_time'])
        } for seg in conversation_data]
        
        # Calculate overall confidence statistics
        if len(metrics['transcript']) > 0:
            # Calculate confidence metrics from all segments that have confidence scores
            confidences = [seg.get('confidence', 0) for seg in metrics['transcript'] if 'confidence' in seg]
            if confidences:
                metrics['confidence_metrics'] = {
                    "average": round(sum(confidences) / len(confidences), 2),
                    "min": round(min(confidences), 2),
                    "max": round(max(confidences), 2)
                }
                
                # Calculate number and percentage of low confidence segments
                low_confidence_segments = [s for s in metrics['transcript'] if s.get('confidence_level') == 'low']
                metrics['confidence_metrics']["low_confidence_count"] = len(low_confidence_segments)
                metrics['confidence_metrics']["low_confidence_percentage"] = round(
                    100 * len(low_confidence_segments) / len(metrics['transcript']), 2
                )
        
        # Calculate speaker-specific confidence metrics
        speaker_confidences = {}
        
        # Group segments by speaker
        for segment in conversation_data:
            speaker = segment['speaker']
            if speaker not in speaker_confidences:
                speaker_confidences[speaker] = []
                
            if 'confidence' in segment:
                speaker_confidences[speaker].append(segment['confidence'])
        
        # Calculate average confidence for each speaker
        speaker_metrics = {}
        for speaker, confidences in speaker_confidences.items():
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                speaker_metrics[speaker] = {
                    "average_confidence": round(avg_confidence, 2),
                    "min_confidence": round(min(confidences), 2) if confidences else None,
                    "max_confidence": round(max(confidences), 2) if confidences else None,
                    "confidence_level": "high" if avg_confidence >= 90 else 
                                      ("medium" if avg_confidence >= 70 else "low"),
                    "segment_count": len(confidences)
                }
        
        # Add to metrics
        if speaker_metrics:
            metrics['speaker_confidence_metrics'] = speaker_metrics
        
        if progress_callback:
            progress_callback(100, "Processing complete")
            
        # Clean up the converted file if we created one
        if original_file != audio_file_path and os.path.exists(audio_file_path):
            try:
                os.unlink(audio_file_path)
            except:
                pass
                
        return metrics
    
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
        raise