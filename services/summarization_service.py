from typing import Dict, List, Optional, Any, Callable
import os
import sys
import json
import logging
import time
from pathlib import Path

# Configure logging
logger = logging.getLogger("summarization-service")

# Add core directory to path if needed
core_dir = Path(__file__).parent.parent / "core"
if str(core_dir) not in sys.path:
    sys.path.append(str(core_dir))

# Import the real implementation functions with exception handling
try:
    from core.lg import summarize_meeting as lg_summarize_meeting
    from core.speaker_summarizer import generate_speaker_summaries as ss_generate_speaker_summaries
    from core.summarize_long_transcripts import summarize_long_meeting as slt_summarize_long_meeting
    logger.info("Successfully imported core summarization modules")
except ImportError as e:
    logger.error(f"Error importing core modules: {str(e)}")
    logger.warning("Using mock implementations as fallback")
    
    # Mock implementations for fallback
    def lg_summarize_meeting(transcript, participants, language=None):
        """Mock implementation for documentation"""
        logger.warning("Using mock implementation of summarize_meeting!")
        return {
            "meeting_summary": {
                "summary": "This is a mock meeting summary",
                "key_points": ["Mock key point 1", "Mock key point 2"],
                "decisions": ["Mock decision 1"]
            },
            "action_items": [
                {"action": "Mock action", "assignee": "Mock Person", "due_date": "Tomorrow", "priority": "high"}
            ]
        }
    
    def ss_generate_speaker_summaries(transcript, participants, language=None):
        """Mock implementation for documentation"""
        logger.warning("Using mock implementation of generate_speaker_summaries!")
        return {
            participant: {
                "brief_summary": f"Mock summary for {participant}",
                "key_contributions": [f"Mock contribution 1 for {participant}", f"Mock contribution 2 for {participant}"],
                "action_items": [f"Mock action for {participant}"],
                "questions_raised": [f"Mock question from {participant}"]
            } for participant in participants
        }
    
    def slt_summarize_long_meeting(transcript_data, language=None, progress_callback=None):
        """Mock implementation for documentation"""
        logger.warning("Using mock implementation of summarize_long_meeting!")
        if progress_callback:
            progress_callback(50, "Simulating long meeting summarization")
        return {
            "meeting_summary": {
                "summary": "This is a mock long meeting summary",
                "key_points": ["Mock key point 1", "Mock key point 2"],
                "decisions": ["Mock decision 1"]
            },
            "action_items": [
                {"action": "Mock action", "assignee": "Mock Person", "due_date": "Tomorrow", "priority": "high"}
            ],
            "metadata": {
                "total_duration_minutes": 60,
                "chunks_analyzed": 6,
                "language": language or "en"
            }
        }

def summarize_meeting(
    transcript: str, 
    participants: List[str], 
    language: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run the meeting summarizer on a transcript and return the summary and action items.
    
    Args:
        transcript: Meeting transcript
        participants: List of participants
        language: Optional language code
        
    Returns:
        Dictionary with meeting summary and action items
    """
    logger.info(f"Summarizing meeting with {len(participants)} participants, language={language}")
    
    try:
        start_time = time.time()
        
        # Validate inputs
        if not transcript or not transcript.strip():
            raise ValueError("Meeting transcript cannot be empty")
        
        if not participants or len(participants) == 0:
            raise ValueError("Participants list cannot be empty")
        
        # Call the core summarization function
        result = lg_summarize_meeting(transcript, participants, language)
        
        logger.info(f"Meeting summarized in {time.time() - start_time:.2f} seconds")
        return result
    except Exception as e:
        logger.error(f"Error summarizing meeting: {str(e)}", exc_info=True)
        raise

def generate_speaker_summaries(
    transcript: str, 
    participants: List[str], 
    language: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Generate individual summaries for each speaker in the meeting.
    
    Args:
        transcript: Meeting transcript
        participants: List of participants
        language: Optional language code
        
    Returns:
        Dictionary mapping each speaker to their summary
    """
    logger.info(f"Generating speaker summaries for {len(participants)} participants, language={language}")
    
    try:
        start_time = time.time()
        
        # Validate inputs
        if not transcript or not transcript.strip():
            raise ValueError("Meeting transcript cannot be empty")
        
        if not participants or len(participants) == 0:
            raise ValueError("Participants list cannot be empty")
            
        # Call the core speaker summarization function
        result = ss_generate_speaker_summaries(transcript, participants, language)
        
        logger.info(f"Speaker summaries generated in {time.time() - start_time:.2f} seconds")
        return result
    except Exception as e:
        logger.error(f"Error generating speaker summaries: {str(e)}", exc_info=True)
        raise

def summarize_long_meeting(
    transcript_data: List[Dict[str, Any]],
    language: Optional[str] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None
) -> Dict[str, Any]:
    """
    Generate a summary for a long meeting using hierarchical summarization.
    
    Args:
        transcript_data: List of transcript segments with speaker IDs and text
        language: Optional language code
        progress_callback: Optional callback function for progress updates
        
    Returns:
        Dictionary with summary and action items
    """
    logger.info(f"Summarizing long meeting with {len(transcript_data)} segments, language={language}")
    
    try:
        start_time = time.time()
        
        # Validate inputs
        if not transcript_data or len(transcript_data) == 0:
            raise ValueError("Transcript data cannot be empty")
        
        # Call the core long meeting summarization function
        result = slt_summarize_long_meeting(transcript_data, language, progress_callback)
        
        logger.info(f"Long meeting summarized in {time.time() - start_time:.2f} seconds")
        return result
    except Exception as e:
        logger.error(f"Error summarizing long meeting: {str(e)}", exc_info=True)
        raise

# Function to verify if the real implementations are available
def check_summarizer_availability():
    """Check if the real summarizer implementations are available"""
    try:
        # Try importing one of the core functions directly
        from core.lg import summarize_meeting
        # If it works, return True
        return True
    except ImportError:
        # If importing fails, return False
        return False

# Report availability status
REAL_SUMMARIZERS_AVAILABLE = check_summarizer_availability()
if REAL_SUMMARIZERS_AVAILABLE:
    logger.info("Real meeting summarizer implementations are available")
else:
    logger.warning("Using mock implementations - real summarizers not available")