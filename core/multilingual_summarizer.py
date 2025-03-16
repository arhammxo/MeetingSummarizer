from lg import summarize_meeting as original_summarize_meeting
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from services.llm_service import get_llm, create_chat_prompt_template, create_output_parser

def summarize_meeting_multilingual(transcript, participants, language=None):
    """
    Creates meeting summaries in the specified language
    
    Args:
        transcript (str): Meeting transcript
        participants (list): List of participant names
        language (str, optional): Language code (e.g., 'hi') or language name
        
    Returns:
        dict: Meeting summary and action items in the specified language
    """
    # If no specific language is provided, use the original function
    if not language or language.lower() == "en" or language.lower() == "english":
        return original_summarize_meeting(transcript, participants)
    
    # Get proper language name for instructions
    language_name = get_language_name(language)
    
    # Initialize the LLM with increased temperature for better multilingual generation
    llm = get_llm(temperature=0.2, purpose="multilingual")
    
    # First, get the meeting summary
    system_message = f"""You are an expert meeting summarizer working with {language_name} content.
    Your task is to create a meeting summary ENTIRELY IN {language_name}.

    Based on the meeting transcript, create a concise summary that captures what was discussed and decided.

    Follow this structure:
    1. Summary: A 2-3 sentence overview of the meeting IN {language_name}
    2. Key Points: Bullet points of important topics discussed IN {language_name}
    3. Decisions: Bullet points of decisions made during the meeting IN {language_name}

    Your response must be a JSON object with the fields 'summary', 'key_points', and 'decisions'.
    ENSURE ALL TEXT IS IN {language_name} ONLY. DO NOT MIX LANGUAGES.

    If the transcript contains English, translate all summary content to {language_name}.
    """

    user_message = """Meeting Transcript: {transcript}

    Participants: {participants}

    Please summarize this meeting COMPLETELY in {language_name}, ensuring all output is in {language_name} only."""

    summary_prompt = create_chat_prompt_template(system_message, user_message)
        
    json_parser = create_output_parser()
    summary_chain = summary_prompt | llm | json_parser    
    # Then, extract action items
    action_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are an expert at identifying action items from meeting transcripts.
        
        IMPORTANT: YOU MUST GENERATE ALL OUTPUT ENTIRELY IN {language_name}.
        
        For each action item, identify:
        - The specific action to be taken
        - Who is responsible for the action
        - Any mentioned deadline or due date
        - The priority level (high, medium, low) based on context
        
        Return your results as a JSON array of action items. If no action items are mentioned, return an empty array.
        Each action item should have fields: 'action', 'assignee', 'due_date', and 'priority'.
        
        ALL TEXT MUST BE IN {language_name} ONLY.
        
        IMPORTANT: Always provide a string value for each field. If a field is missing:
        - For 'due_date': use a {language_name} translation of "Not specified" 
        - For 'priority': use the {language_name} equivalent of "medium"
        - For 'assignee': use the {language_name} equivalent of "Unassigned"
        
        DO NOT return null values - use appropriate string defaults instead.
        """),
        ("human", """Meeting Transcript: {transcript}
        
        Participants: {participants}
        
        Extract action items from this meeting IN {language_name} ONLY.""")
    ])
    
    action_chain = action_prompt | llm | JsonOutputParser()
    
    try:
        if len(transcript) > 8000:
            print(f"Long transcript detected ({len(transcript)} chars), breaking into chunks")
            chunks = chunk_transcript(transcript)
            
            # Get summaries for each chunk
            chunk_summaries = []
            for i, chunk in enumerate(chunks):
                print(f"Processing chunk {i+1}/{len(chunks)}")
                chunk_result = summary_chain.invoke({
                    "transcript": chunk,
                    "participants": participants,
                    "language_name": language_name
                })
                chunk_summaries.append(chunk_result)
            
            # Merge summaries
            meeting_summary = {
                "summary": " ".join([s.get("summary", "") for s in chunk_summaries]),
                "key_points": [],
                "decisions": []
            }
            
            # Collect all key points and decisions
            for summary in chunk_summaries:
                meeting_summary["key_points"].extend(summary.get("key_points", []))
                meeting_summary["decisions"].extend(summary.get("decisions", []))
        else:
            # Original code for shorter transcripts
            meeting_summary = summary_chain.invoke({
                "transcript": transcript,
                "participants": participants,
                "language_name": language_name
            })
        
        # Get action items
        action_items = action_chain.invoke({
            "transcript": transcript,
            "participants": participants,
            "language_name": language_name
        })
        
        # Format into the expected result structure
        result = {
            "meeting_summary": meeting_summary,
            "action_items": action_items
        }
        
        return result
    
    except Exception as e:
        # If something fails, fall back to the original English function
        print(f"Error generating {language_name} summary: {str(e)}. Falling back to English.")
        return original_summarize_meeting(transcript, participants)
    
def chunk_transcript(transcript, max_chunk_size=8000):
    """
    Split a long transcript into manageable chunks to avoid context window limitations
    
    Args:
        transcript: The full transcript text
        max_chunk_size: Maximum size per chunk in characters
        
    Returns:
        List of transcript chunks
    """
    if len(transcript) <= max_chunk_size:
        return [transcript]
    
    # Try to split at paragraph boundaries
    paragraphs = transcript.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= max_chunk_size:
            if current_chunk:
                current_chunk += '\n\n'
            current_chunk += para
        else:
            if current_chunk:
                chunks.append(current_chunk)
            
            # If a single paragraph is too long, split it at sentence boundaries
            if len(para) > max_chunk_size:
                sentences = para.split('. ')
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 2 <= max_chunk_size:
                        if current_chunk:
                            current_chunk += '. '
                        current_chunk += sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk + '.')
                        current_chunk = sentence
            else:
                current_chunk = para
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def get_language_name(language_code):
    """
    Convert language code or name to a full language name
    
    Args:
        language_code (str): Language code (e.g., 'hi') or language name
        
    Returns:
        str: Full language name (e.g., 'Hindi')
    """
    language_map = {
        # Language codes
        "hi": "Hindi",
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "zh": "Chinese",
        "ja": "Japanese",
        "ru": "Russian",
        "ar": "Arabic",
        "pt": "Portuguese",
        "bn": "Bengali",
        "ur": "Urdu",
        "te": "Telugu",
        "ta": "Tamil",
        "mr": "Marathi",
        "gu": "Gujarati",
        "kn": "Kannada",
        "ml": "Malayalam",
        "pa": "Punjabi",
        
        # Full names (for when a language name is passed instead of code)
        "hindi": "Hindi",
        "english": "English",
        "spanish": "Spanish",
        "french": "French",
        "german": "German",
        "chinese": "Chinese",
        "japanese": "Japanese",
        "russian": "Russian",
        "arabic": "Arabic",
        "portuguese": "Portuguese",
        "bengali": "Bengali",
        "urdu": "Urdu",
        "telugu": "Telugu",
        "tamil": "Tamil",
        "marathi": "Marathi",
        "gujarati": "Gujarati",
        "kannada": "Kannada",
        "malayalam": "Malayalam",
        "punjabi": "Punjabi",
        
        # Handle auto-detect case without defaulting to Hindi
        "auto-detect": "Auto-detected",
        "auto-detected": "Auto-detected",
        "auto": "Auto-detected"
    }
    
    # If we get a code, return the language name; if we get a language name, return it
    try:
        # First, check if this is a specific language code/name we know
        if isinstance(language_code, str):
            code_lower = language_code.lower()
            if code_lower in language_map:
                return language_map[code_lower]
            
            # If it's one of the auto-detect values, log a warning only if it appears
            # we don't have a proper language detection
            if code_lower in ["auto-detect", "auto-detected", "auto"]:
                import logging
                logging.warning(f"Language was auto-detected but no specific language was identified. Defaulting to English.")
                return "Auto-detected (defaulting to English)"
                
            # For unknown codes, just return the code itself
            return language_code
        else:
            # For None or other non-string values, default to English
            return "English"
    except:
        # If language_code is None or any other type that causes errors
        return "English"  # Default to English as a safer default