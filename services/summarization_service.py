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
    
    # Import the multilingual summarizer module
    try:
        from core.multilingual_summarizer import summarize_meeting_multilingual, get_language_name
        logger.info("Successfully imported multilingual summarizer module")
        HAS_MULTILINGUAL = True
    except ImportError as e:
        logger.warning(f"Multilingual summarizer not available: {str(e)}")
        HAS_MULTILINGUAL = False
        
    # Create a simple fallback get_language_name if not imported
    if not HAS_MULTILINGUAL:
        def get_language_name(language_code):
            language_map = {
                "hi": "Hindi", "en": "English", "es": "Spanish", "fr": "French", 
                "de": "German", "zh": "Chinese", "ja": "Japanese", "ru": "Russian", 
                "ar": "Arabic", "auto": "Auto-detected", None: "English"
            }
            return language_map.get(language_code, "English")
        
except ImportError as e:
    logger.error(f"Error importing core modules: {str(e)}")
    logger.warning("Using mock implementations as fallback")
    HAS_MULTILINGUAL = False
    
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
    
    def get_language_name(language_code):
        """Simple language name getter"""
        language_map = {
            "hi": "Hindi", "en": "English", "es": "Spanish", "fr": "French", 
            "de": "German", "zh": "Chinese", "ja": "Japanese", "ru": "Russian", 
            "ar": "Arabic", "auto": "Auto-detected", None: "English"
        }
        return language_map.get(language_code, "English")
    
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

# Custom implementation of multilingual speaker summaries
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

def generate_speaker_summaries_multilingual(transcript, participants, language):
    """
    Generate speaker summaries in the specified language
    
    Args:
        transcript (str): Meeting transcript
        participants (list): List of participant names
        language (str): Language code or name
    
    Returns:
        dict: Speaker summaries in the specified language
    """
    # If no specific language, English, or no LangChain, use the original function
    if not language or language.lower() == "en" or language.lower() == "english":
        return ss_generate_speaker_summaries(transcript, participants)
    
    try:
        # Get proper language name for instructions
        language_name = get_language_name(language)
        
        # Use the LLM service factory instead of directly creating a ChatOpenAI instance
        # This respects the configured LLM provider (Ollama or OpenAI)
        from services.llm_service import get_llm, create_chat_prompt_template, create_output_parser, get_ollama_llm
        from config import settings
        
        # Initialize the LLM with increased temperature for better multilingual generation
        llm = get_llm(temperature=0.2, purpose="multilingual")
        
        # Create a prompt template for generating multilingual speaker summaries
        system_message = f"""You are an expert meeting analyst working with {language_name} content.
        Your task is to create a concise summary of what a specific participant contributed to a meeting.
        ALL YOUR OUTPUT MUST BE IN {language_name} ONLY.
        
        Focus on:
        1. Main points they raised
        2. Questions they asked
        3. Action items they took on or assigned
        4. Key decisions they influenced
        5. Their primary concerns or interests
        
        Format your response as a JSON object with these keys:
        - "key_contributions": List of 2-4 main points they contributed (IN {language_name})
        - "action_items": List of any tasks they agreed to do or assigned (IN {language_name})
        - "questions_raised": List of important questions they asked, if any (IN {language_name})
        - "brief_summary": A 1-2 sentence summary of their overall participation (IN {language_name})
        
        Keep your response focused only on this speaker's contributions.
        If the transcript contains English or any other language, translate your response to {language_name}.
        """
        
        user_message = """Speaker: {speaker}
        
        Their contributions:
        {contributions}
        
        Please summarize this speaker's participation in the meeting in {language_name} language."""
        
        # Create the prompt using our factory function
        prompt = create_chat_prompt_template(system_message, user_message)
        
        # Create the output parser
        json_parser = create_output_parser()
        
        # Group transcript segments by speaker
        speaker_contributions = {}
        
        # Process the transcript to group text by speaker
        for participant in participants:
            # Create a pattern to match this speaker's lines
            speaker_pattern = f"{participant}:"
            
            # Find all lines from this speaker
            speaker_lines = []
            for line in transcript.split('\n'):
                if speaker_pattern in line:
                    # Extract just the text (remove speaker prefix)
                    text = line.split(speaker_pattern, 1)[1].strip()
                    speaker_lines.append(text)
            
            # Store all this speaker's contributions
            if speaker_lines:
                speaker_contributions[participant] = '\n'.join(speaker_lines)
        
        # Add robust error handling and fallbacks
        from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
        
        # Create a retry wrapper for API call failures
        @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), 
              retry=retry_if_exception_type((ConnectionError, TimeoutError)))
        def generate_summary_with_retry(speaker, contributions, language_name):
            try:
                # For Ollama with structured output support
                if settings.LLM_PROVIDER == "ollama" and settings.OLLAMA_USE_STRUCTURED_OUTPUT:
                    schema = {
                        "type": "object",
                        "properties": {
                            "key_contributions": {"type": "array", "items": {"type": "string"}},
                            "action_items": {"type": "array", "items": {"type": "string"}},
                            "questions_raised": {"type": "array", "items": {"type": "string"}},
                            "brief_summary": {"type": "string"}
                        },
                        "required": ["key_contributions", "action_items", "questions_raised", "brief_summary"]
                    }
                    
                    # Get a fresh LLM instance with format schema
                    structured_llm = get_ollama_llm(
                        temperature=0.2,
                        purpose="multilingual",
                        format_schema=schema
                    )
                    
                    # Create chain without JSON parser for structured output
                    structured_chain = prompt | structured_llm
                    result = structured_chain.invoke({
                        "speaker": speaker,
                        "contributions": contributions,
                        "language_name": language_name
                    })
                    
                    # Parse the structured response
                    if hasattr(result, 'content'):
                        return json.loads(result.content)
                    else:
                        return json.loads(str(result))
                else:
                    # Regular chain with JSON parser
                    chain = prompt | llm | json_parser
                    return chain.invoke({
                        "speaker": speaker,
                        "contributions": contributions,
                        "language_name": language_name
                    })
            except Exception as e:
                logger.error(f"Error in API call for {speaker}: {str(e)}")
                raise
        
        # Generate summaries for each speaker
        speaker_summaries = {}
        
        for speaker, contributions in speaker_contributions.items():
            if contributions.strip():  # Only process if they have actual contributions
                try:
                    # Try using our retry-enabled function
                    summary = generate_summary_with_retry(speaker, contributions, language_name)
                    speaker_summaries[speaker] = summary
                except Exception as e:
                    logger.error(f"Error generating multilingual summary for {speaker}: {str(e)}")
                    
                    # First fallback: Try with a simpler prompt that might be easier to process
                    try:
                        # Create a simpler prompt
                        simple_system = f"Summarize the contributions of {speaker} in {language_name} language."
                        simple_user = f"Speaker contributions:\n{contributions[:2000]}"  # Limit length
                        
                        simple_prompt = create_chat_prompt_template(simple_system, simple_user)
                        simple_chain = simple_prompt | llm
                        
                        # Get a text summary instead of JSON
                        simple_result = simple_chain.invoke({})
                        
                        # Format into our expected structure
                        speaker_summaries[speaker] = {
                            "key_contributions": ["See brief summary"],
                            "action_items": [],
                            "questions_raised": [],
                            "brief_summary": str(simple_result)
                        }
                    except Exception as fallback_error:
                        logger.error(f"Fallback summary generation failed: {str(fallback_error)}")
                        
                        # Final fallback: Use the original function to get at least some summary
                        try:
                            logger.warning(f"Falling back to standard speaker summarizer for {speaker}")
                            speaker_text = {speaker: contributions}
                            orig_summary = ss_generate_speaker_summaries('\n'.join([f"{s}: {t}" for s, t in speaker_text.items()]), [speaker])
                            speaker_summaries[speaker] = orig_summary.get(speaker, {
                                "key_contributions": ["Error processing contributions"],
                                "action_items": [],
                                "questions_raised": [],
                                "brief_summary": f"Error generating summary for {speaker}: {str(e)}"
                            })
                        except Exception as final_error:
                            # Ultimate fallback
                            speaker_summaries[speaker] = {
                                "key_contributions": ["Error processing contributions"],
                                "action_items": [],
                                "questions_raised": [],
                                "brief_summary": f"Error generating summary for {speaker}: {str(e)}"
                            }
        
        return speaker_summaries
    except Exception as e:
        logger.error(f"Error in multilingual speaker summarization: {str(e)}")
        # Fall back to the original function
        logger.warning("Falling back to standard speaker summarizer for all speakers")
        return ss_generate_speaker_summaries(transcript, participants)

def summarize_meeting(
    transcript: str, 
    participants: List[str], 
    language: Optional[str] = None,
    additional_context: Optional[str] = None  # Add parameter
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
        
        # Critical change: Don't treat "auto-detected" as a special case
        # Instead, if language is auto-detected, we should have the actual 
        # detected language code from the audio processing
        
        # Don't check for "auto-detected" here - if the language is already detected,
        # it will be a specific language code, not "auto-detected"
        
        # Use multilingual summarizer if a language is specified (other than English)
        if HAS_MULTILINGUAL and language and language.lower() not in ["en", "english", "auto", "auto-detected"]:
            logger.info(f"Using multilingual summarizer for language: {language}")
            # Update this line - Pass only 3 arguments if the function only accepts 3
            if additional_context:
                # Check if the multilingual function has been updated to accept context
                if hasattr(summarize_meeting_multilingual, "__code__") and summarize_meeting_multilingual.__code__.co_argcount >= 4:
                    result = summarize_meeting_multilingual(transcript, participants, language, additional_context)
                else:
                    # Log that we're ignoring additional context due to function signature
                    logger.warning(f"Additional context provided but summarize_meeting_multilingual can't accept it")
                    result = summarize_meeting_multilingual(transcript, participants, language)
            else:
                # No context provided, call with 3 args
                result = summarize_meeting_multilingual(transcript, participants, language)
        else:
            # Call the core summarization function with the detected language
            logger.info(f"Using standard summarizer with language: {language if language else 'default'}")
            # Similar pattern here - check if function can accept context
            if additional_context:
                if hasattr(lg_summarize_meeting, "__code__") and lg_summarize_meeting.__code__.co_argcount >= 4:
                    result = lg_summarize_meeting(transcript, participants, language, additional_context)
                else:
                    logger.warning(f"Additional context provided but lg_summarize_meeting can't accept it")
                    result = lg_summarize_meeting(transcript, participants, language)
            else:
                result = lg_summarize_meeting(transcript, participants, language)
        
        logger.info(f"Meeting summarized in {time.time() - start_time:.2f} seconds")
        
        # Add language info to result
        if "metadata" not in result:
            result["metadata"] = {}
        result["metadata"]["language"] = language or "en"
        result["metadata"]["language_name"] = get_language_name(language)
        
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
        
        # Use multilingual speaker summarizer if a language is specified (other than English)
        if language and language.lower() not in ["en", "english"]:
            logger.info(f"Using multilingual speaker summarizer for language: {language}")
            result = generate_speaker_summaries_multilingual(transcript, participants, language)
        else:
            # Call the core speaker summarization function
            logger.info(f"Using standard speaker summarizer with language: {language if language else 'default'}")
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
        # Note: The core function already supports multilingual summaries internally
        result = slt_summarize_long_meeting(transcript_data, language, progress_callback)
        
        # Add language info to result
        if "metadata" not in result:
            result["metadata"] = {}
        result["metadata"]["language"] = language or "en"
        result["metadata"]["language_name"] = get_language_name(language)
        
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