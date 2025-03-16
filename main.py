from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Form, HTTPException, Depends, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
import uuid
import os
import tempfile
import json
import logging
from datetime import datetime

# Import configuration
from config import settings, get_settings

# Import processing modules
from services.audio_service import process_audio_file, process_long_audio
from services.text_service import extract_participants
from services.summarization_service import summarize_meeting, summarize_long_meeting, generate_speaker_summaries
from services.job_service import JobStatus, get_job_status, update_job_status, save_job_result
from services.utils import format_time

# Configure logging
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL.upper()),
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("meeting-summarizer-api")

# Configure more detailed logging for language detection
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('language_detection.log')
    ]
)

# Create a specific logger for language detection
lang_logger = logging.getLogger('language-detection')
# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Initialize FastAPI app
app = FastAPI(
    title="Meeting Summarizer API",
    description="API for summarizing meeting recordings and transcripts",
    version="1.0.0",
    max_upload_size=100 * 1024 * 1024  # 100MB limit
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models for request/response
class TextRequest(BaseModel):
    transcript: str
    participants: Optional[List[str]] = None
    language: Optional[str] = None

class ProcessRequest(BaseModel):
    transcript: str
    participants: List[str]
    language: Optional[str] = None
    is_long_recording: bool = False

class JobResponse(BaseModel):
    job_id: str
    status: str

class LanguageOption(BaseModel):
    code: str
    name: str

# Serve the main application page
@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """Serve the main application page"""
    return templates.TemplateResponse("index.html", {"request": request})

# Endpoint to get supported languages
@app.get("/api/languages", response_model=List[LanguageOption])
async def get_languages():
    """Get a list of supported languages for transcription and summarization"""
    languages = [
        {"code": "auto", "name": "Auto-detect"},
        {"code": "en", "name": "English"},
        {"code": "hi", "name": "Hindi"},
        {"code": "es", "name": "Spanish"},
        {"code": "fr", "name": "French"},
        {"code": "de", "name": "German"},
        {"code": "zh", "name": "Chinese"},
        {"code": "ja", "name": "Japanese"},
        {"code": "ru", "name": "Russian"},
        {"code": "ar", "name": "Arabic"}
    ]
    return languages

# Endpoint to upload and process audio
@app.post("/api/upload-audio", response_model=JobResponse)
async def upload_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    is_long_recording: bool = Form(False)
):
    """
    Upload an audio file for processing.
    
    - The file will be processed in the background
    - Speaker diarization will be performed automatically
    - Returns a job ID that can be used to check progress
    """
    # Generate a unique job ID
    job_id = str(uuid.uuid4())
    
    # Save the uploaded file to a temporary location
    audio_file_path = None
    try:
        # Create a temporary file with the correct extension
        suffix = f".{file.filename.split('.')[-1]}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            # Write the uploaded file content
            content = await file.read()
            tmp.write(content)
            audio_file_path = tmp.name
        
        # Initialize job status
        update_job_status(job_id, JobStatus.PENDING, "Audio file received, processing will start soon")
        
        # Process the audio file in the background
        background_tasks.add_task(
            process_audio_background,
            job_id,
            audio_file_path,
            is_long_recording,
            language
        )
        
        return {"job_id": job_id, "status": "pending"}
    except Exception as e:
        logger.error(f"Error processing audio upload: {str(e)}")
        # Clean up temporary file if it exists
        if audio_file_path and os.path.exists(audio_file_path):
            os.unlink(audio_file_path)
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

# Background task to process audio
async def process_audio_background(job_id: str, audio_path: str, is_long_recording: bool, language: Optional[str] = None):
    """Process audio file in the background and update job status"""
    try:
        update_job_status(job_id, JobStatus.PROCESSING, "Processing audio file")
        
        # Process audio based on length
        if is_long_recording:
            # Define a progress callback function
            def progress_callback(progress, status):
                update_job_status(
                    job_id, 
                    JobStatus.PROCESSING, 
                    status,
                    progress=progress
                )
            
            # Process long audio with progress updates
            result = process_long_audio(audio_path, language=language, progress_callback=progress_callback)
        else:
            # Process standard audio
            result = process_audio_file(audio_path, language=language)
            # Simulate progress updates
            update_job_status(job_id, JobStatus.PROCESSING, "Transcribing audio", progress=33)
            update_job_status(job_id, JobStatus.PROCESSING, "Identifying speakers", progress=66)
            update_job_status(job_id, JobStatus.PROCESSING, "Finalizing transcript", progress=90)
        
        # Check result structure - debug log to see what's coming from audio processing
        logger.debug(f"Audio processing result structure keys: {list(result.keys())}")
        
        # Important: Get the actual detected language from the result
        detected_language = result.get('language', 'auto')
        logger.info(f"Detected language for audio: {detected_language}")
        
        # If language detection was problematic (returning auto-detection terms instead of a specific language)
        if detected_language in ['auto', 'auto-detect', 'auto-detected']:
            # CRITICAL FIX: Try to extract actual language from transcription segments
            # This is where the real detection from Whisper should be found
            try:
                # Check the first segment which might contain language information
                if 'transcript' in result and result['transcript'] and len(result['transcript']) > 0:
                    # Look at first segment metadata
                    first_segment = result['transcript'][0]
                    logger.debug(f"First transcript segment: {first_segment}")
                    
                    # Check if there's any language metadata saved in the segment
                    # We need to access whatever field contains the language info
                    # This will vary based on how Whisper's result was processed
                    if hasattr(first_segment, 'get') and first_segment.get('language'):
                        detected_language = first_segment.get('language')
                        logger.info(f"Extracted language from transcript segment: {detected_language}")
                        
                        # Update the result with the correctly detected language
                        result['language'] = detected_language
            except Exception as e:
                logger.error(f"Error extracting language from transcript: {str(e)}")
        
        # Note for auto-detection scenario - helpful message if we failed to extract a specific language
        if language is None or language.lower() in ['auto', 'auto-detect', 'auto-detected']:
            logger.info(f"Auto-detection was used, setting language to detected language: {detected_language}")
            
            # If we still have 'auto' at this point, a true language wasn't extracted
            if detected_language in ['auto', 'auto-detect', 'auto-detected']:
                logger.warning("Failed to extract specific language code from audio! Will default to English.")
                # Optional: set a default language for summarization
                # result['language'] = 'en'  # Uncomment if you want to force a default
        
        # Save the result - this will include the corrected language value
        save_job_result(job_id, result)
        update_job_status(job_id, JobStatus.COMPLETED, "Audio processing complete", progress=100)
        
        # Clean up the temporary file
        if os.path.exists(audio_path):
            os.unlink(audio_path)
            
    except Exception as e:
        logger.error(f"Error in background processing: {str(e)}")
        update_job_status(job_id, JobStatus.FAILED, f"Error processing audio: {str(e)}")
        
        # Clean up the temporary file
        if os.path.exists(audio_path):
            os.unlink(audio_path)

# Endpoint to get job status
@app.get("/api/job/{job_id}", response_model=Dict[str, Any])
async def check_job_status(job_id: str):
    """Check the status of a background job"""
    job_status = get_job_status(job_id)
    if not job_status:
        raise HTTPException(status_code=404, detail=f"Job ID {job_id} not found")
    return job_status

# Endpoint to upload text file
@app.post("/api/upload-text", response_model=Dict[str, Any])
async def upload_text(file: UploadFile = File(...)):
    """Upload a transcript text file"""
    try:
        content = await file.read()
        transcript = content.decode("utf-8")
        
        # Extract participants from the transcript
        participants = extract_participants(transcript)
        
        return {
            "success": True,
            "transcript": transcript,
            "participants": participants,
            "character_count": len(transcript)
        }
    except Exception as e:
        logger.error(f"Error processing text upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")

# Endpoint to extract participants from text
@app.post("/api/extract-participants", response_model=List[str])
async def get_participants(request: TextRequest):
    """Extract participant names from a transcript"""
    try:
        participants = extract_participants(request.transcript)
        return participants
    except Exception as e:
        logger.error(f"Error extracting participants: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error extracting participants: {str(e)}")

# Endpoint to generate meeting summary
@app.post("/api/summarize", response_model=JobResponse)
async def summarize(background_tasks: BackgroundTasks, request: ProcessRequest):
    """
    Generate a meeting summary from a transcript
    
    - Takes a transcript and list of participants
    - Generates a summary with key points, decisions, and action items
    - Returns a job ID that can be used to check progress
    """
    # Generate a unique job ID
    job_id = str(uuid.uuid4())
    
    # Initialize job status
    update_job_status(job_id, JobStatus.PENDING, "Summarization request received")
    
    # Process the summary in the background
    background_tasks.add_task(
        summarize_background,
        job_id,
        request.transcript,
        request.participants,
        request.language,
        request.is_long_recording
    )
    
    return {"job_id": job_id, "status": "pending"}

# Background task to generate summary
async def summarize_background(
    job_id: str, 
    transcript: str, 
    participants: List[str], 
    language: Optional[str],
    is_long_recording: bool
):
    """Generate meeting summary in the background and update job status"""
    try:
        update_job_status(job_id, JobStatus.PROCESSING, "Analyzing transcript", progress=10)
        
        # CRITICAL: Fix language handling
        # Check if we're getting a placeholder instead of a real language code
        if language in ["auto", "auto-detect", "auto-detected"]:
            # This is problematic - log a warning
            logger.warning(f"Received '{language}' as language for summarization, which is not a specific language code")
            logger.warning("Attempting to detect language from transcript content")
            
            # Optionally try to detect language from transcript content
            # This is a fallback if the audio processing didn't provide a proper language code
            try:
                from langdetect import detect
                detected_lang = detect(transcript[:1000])  # Use first 1000 chars
                logger.info(f"Detected language from transcript text: {detected_lang}")
                language = detected_lang
            except Exception as e:
                logger.error(f"Error detecting language from transcript: {str(e)}")
                # Default to English if all else fails
                logger.warning("Defaulting to English for summarization")
                language = "en"
        
        # Log the final language being used
        logger.info(f"Generating summary with language: {language}")
        
        # Process based on length
        if is_long_recording:
            # Parse transcript into required format for long meeting
            # This would need to be implemented based on how the transcript is structured
            transcript_data = parse_transcript_to_segments(transcript)
            
            # Define progress callback
            def progress_callback(progress, status):
                update_job_status(job_id, JobStatus.PROCESSING, status, progress=progress)
            
            # Summarize long meeting, explicitly passing the language
            result = summarize_long_meeting(
                transcript_data,
                language=language,
                progress_callback=progress_callback
            )
        else:
            # Regular summarization, explicitly passing the language
            update_job_status(job_id, JobStatus.PROCESSING, "Generating meeting summary", progress=30)
            result = summarize_meeting(transcript, participants, language=language)
            update_job_status(job_id, JobStatus.PROCESSING, "Extracting action items", progress=60)
        
        # Generate speaker summaries, passing the same language parameter
        update_job_status(job_id, JobStatus.PROCESSING, "Creating speaker summaries", progress=80)
        speaker_summaries = generate_speaker_summaries(transcript, participants, language=language)
        
        # Get the language name for display
        from services.utils import get_language_name
        language_name = get_language_name(language)
        
        # Combine results
        final_result = {
            "meeting_summary": result["meeting_summary"],
            "action_items": result["action_items"],
            "speaker_summaries": speaker_summaries,
            "metadata": {
                "language": language,
                "language_name": language_name,
                "participant_count": len(participants),
                "is_long_recording": is_long_recording,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Save the result
        save_job_result(job_id, final_result)
        update_job_status(job_id, JobStatus.COMPLETED, "Summarization complete", progress=100)
        
    except Exception as e:
        logger.error(f"Error in summarization: {str(e)}")
        update_job_status(job_id, JobStatus.FAILED, f"Error generating summary: {str(e)}")

# Helper function to parse transcript text into segment format
def parse_transcript_to_segments(transcript_text):
    """
    Parse plain text transcript into the segment format needed for long meeting summarization
    This is a simplified implementation - would need enhancement based on transcript format
    """
    lines = transcript_text.split('\n')
    segments = []
    current_time = 0
    
    for line in lines:
        if not line.strip():
            continue
            
        # Try to extract speaker and text
        parts = line.split(':', 1)
        if len(parts) == 2:
            speaker_part = parts[0].strip()
            text = parts[1].strip()
            
            # Check if there's a speaker number in the speaker part
            speaker_match = None
            if "Speaker" in speaker_part:
                import re
                speaker_match = re.search(r'Speaker\s+(\d+)', speaker_part)
            
            speaker = speaker_match.group(1) if speaker_match else "1"
            
            # Create segment
            segment = {
                'speaker': speaker,
                'text': text,
                'start_time': current_time,
                'end_time': current_time + 30,  # Estimate 30s per utterance
                'start_time_formatted': format_time(current_time),
                'end_time_formatted': format_time(current_time + 30)
            }
            segments.append(segment)
            current_time += 30
    
    return segments

# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Simple health check endpoint"""
    return {
        "status": "healthy", 
        "api_version": "1.0.0",
        "environment": os.environ.get("ENVIRONMENT", "development"),
        "timestamp": datetime.now().isoformat()
    }

# Main entry point
if __name__ == "__main__":
    import uvicorn
    # Check for OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        logger.warning("⚠️ OpenAI API key not found! Set it with 'export OPENAI_API_KEY=your-key' before running")
    
    uvicorn.run(
        "main:app", 
        host=settings.HOST, 
        port=settings.PORT, 
        reload=True,
        workers=settings.WORKERS
    )