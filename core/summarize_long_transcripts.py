from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
import re
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("long_transcript_summarizer")

def robust_json_parse(text):
    """
    Attempt to parse JSON from text, with fallback mechanisms for malformed JSON
    
    Args:
        text: Text that should contain JSON
        
    Returns:
        Parsed JSON object or a default structure
    """
    import json
    import re
    import logging
    
    # First, try direct JSON parsing
    try:
        # Try to extract JSON if it's embedded in markdown or other text
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
        if json_match:
            json_str = json_match.group(1)
            return json.loads(json_str)
        
        # Try to find JSON-like structure with curly braces
        json_match = re.search(r'(\{[\s\S]*\})', text)
        if json_match:
            json_str = json_match.group(1)
            return json.loads(json_str)
        
        # If text doesn't have clear JSON markers, try direct parsing
        return json.loads(text)
    except json.JSONDecodeError:
        # If direct parsing fails, try to fix common issues
        
        # Remove any non-JSON text before opening brace and after closing brace
        cleaned_text = re.sub(r'^[^{]*', '', text)
        cleaned_text = re.sub(r'[^}]*$', '', cleaned_text)
        
        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            # If still failing, try to construct a structured response from the text
            logger.warning(f"Failed to parse JSON, creating fallback structure from text")
            
            # Extract what seem to be key points or summary
            lines = text.split('\n')
            key_points = []
            decisions = []
            summary = ""
            action_items = []
            
            for line in lines:
                line = line.strip()
                if line and len(line) > 10:  # Only consider substantive lines
                    # Try to identify what kind of content this is
                    if "summary" in line.lower() or "overview" in line.lower():
                        summary = line.split(":", 1)[1].strip() if ":" in line else line
                    elif any(marker in line.lower() for marker in ["point", "discuss", "topic"]):
                        key_points.append(line.split(":", 1)[1].strip() if ":" in line else line)
                    elif any(marker in line.lower() for marker in ["decid", "decision", "conclude"]):
                        decisions.append(line.split(":", 1)[1].strip() if ":" in line else line)
                    elif any(marker in line.lower() for marker in ["action", "task", "todo", "assign"]):
                        action_items.append({"action": line.split(":", 1)[1].strip() if ":" in line else line})
                    elif not summary and len(line) < 100:
                        # Use first substantial text as summary if nothing else found
                        summary = line
            
            return {
                "summary": summary if summary else "Unable to extract summary from text",
                "key_points": key_points[:5] if key_points else ["Unable to extract key points"],
                "decisions": decisions if decisions else [],
                "action_items": action_items
            }

def chunk_transcript_by_time(transcript_data, chunk_minutes=10):
    """
    Split a transcript into chunks based on time
    
    Args:
        transcript_data: List of transcript segments with timestamps
        chunk_minutes: Size of each chunk in minutes
        
    Returns:
        List of transcript chunks
    """
    chunk_seconds = chunk_minutes * 60
    chunks = []
    current_chunk = []
    chunk_start_time = 0
    
    # Ensure transcript is sorted by start time
    sorted_data = sorted(transcript_data, key=lambda x: x['start_time'])
    
    for segment in sorted_data:
        # If this segment starts after the chunk boundary and we have data,
        # finalize the current chunk and start a new one
        if segment['start_time'] >= chunk_start_time + chunk_seconds and current_chunk:
            chunks.append({
                'start_time': chunk_start_time,
                'end_time': current_chunk[-1]['end_time'],
                'segments': current_chunk
            })
            current_chunk = []
            chunk_start_time = segment['start_time']
        
        current_chunk.append(segment)
    
    # Add the final chunk if there's data
    if current_chunk:
        chunks.append({
            'start_time': chunk_start_time,
            'end_time': current_chunk[-1]['end_time'],
            'segments': current_chunk
        })
    
    return chunks

def format_transcript_chunk(chunk):
    """Format a transcript chunk into text format"""
    formatted_text = []
    for segment in chunk['segments']:
        formatted_text.append(f"Speaker {segment['speaker']}: {segment['text']}")
    return "\n".join(formatted_text)

def summarize_transcript_chunk(chunk_text, language=None, is_final=False):
    """
    Summarize a single transcript chunk
    
    Args:
        chunk_text: Text of the transcript chunk
        language: Optional language code
        is_final: Whether this is the final summary (affects prompt)
        
    Returns:
        Summary object
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Language-specific instructions
    language_instructions = ""
    if language:
        if language == "hi":
            language_instructions = "Generate your response in Hindi language using Hindi script (Devanagari)."
        elif language != "en":
            language_instructions = f"Generate your response in the {language} language."
    
    # Determine the appropriate content based on whether this is a chunk or final summary
    if not is_final:
        # Prompt for individual chunk summary
        system_content = f"""You are an expert meeting analyst. Summarize this chunk of a meeting transcript.
        Focus on:
        1. Key points discussed
        2. Decisions made
        3. Action items mentioned
        
        Format your response as JSON with these fields:
        - summary: A paragraph summarizing this part of the meeting
        - key_points: List of important points (2-4 items)
        - decisions: List of any decisions made
        - action_items: List of action items mentioned, each with "action" and "assignee" if available
        
        Keep your summary concise but include all important information.
        {language_instructions}
        """
        
        human_content = f"Here is a chunk of the meeting transcript:\n\n{chunk_text}"
    else:
        # Prompt for final summary (combining chunk summaries)
        system_content = f"""You are an expert meeting analyst. Create a final summary of a meeting based on these sectional summaries.
        Combine and synthesize the information to provide a coherent overall summary.
        
        Format your response as JSON with these fields:
        - summary: A paragraph summarizing the entire meeting
        - key_points: List of the most important points (3-5 items)
        - decisions: List of all decisions made
        - action_items: List of all action items, each with "action", "assignee", and "due_date" if available
        
        Eliminate redundancies and provide a clear, structured overview.
        {language_instructions}
        """
        
        human_content = f"Here are the sectional summaries of the meeting:\n\n{chunk_text}"
    
    # Create messages directly
    system_message = SystemMessage(content=system_content)
    human_message = HumanMessage(content=human_content)
    
    # Process the text
    try:
        # First try with JSON output parser
        prompt = ChatPromptTemplate.from_messages([system_message, human_message])
        chain = prompt | llm | JsonOutputParser()
        result = chain.invoke({})
        return result
    except Exception as json_error:
        logger.warning(f"JSON parsing error: {json_error}. Attempting recovery...")
        
        try:
            # Fallback to string output and manual parsing
            str_chain = prompt | llm | StrOutputParser()
            
            # Get raw text response
            raw_response = str_chain.invoke({})
            
            # Use robust parsing
            parsed_result = robust_json_parse(raw_response)
            logger.info("Successfully recovered JSON structure")
            return parsed_result
        except Exception as e:
            logger.error(f"Error summarizing chunk, even with recovery: {e}")
            # Return a basic structure in case of error
            return {
                "summary": f"Error processing this section: {str(e)}",
                "key_points": [],
                "decisions": [],
                "action_items": []
            }

def hierarchical_summarize(transcript_data, language=None, progress_callback=None):
    """
    Summarize a long transcript using hierarchical summarization
    
    Args:
        transcript_data: List of transcript segments with timestamps
        language: Optional language code
        progress_callback: Optional callback function for progress updates
        
    Returns:
        Final summary object
    """
    try:
        # Step 1: Split transcript into chunks by time
        if progress_callback:
            progress_callback(10, "Splitting transcript into manageable chunks")
            
        transcript_chunks = chunk_transcript_by_time(transcript_data)
        
        # Step 2: Summarize each chunk
        chunk_summaries = []
        
        for i, chunk in enumerate(transcript_chunks):
            if progress_callback:
                progress_percentage = 20 + (i / len(transcript_chunks) * 50)  # 20-70% for chunk summaries
                progress_callback(progress_percentage, f"Summarizing chunk {i+1}/{len(transcript_chunks)}")
            
            chunk_text = format_transcript_chunk(chunk)
            chunk_summary = summarize_transcript_chunk(chunk_text, language)
            
            # Add time range to the summary
            chunk_summary["time_range"] = {
                "start": chunk["start_time"],
                "end": chunk["end_time"]
            }
            
            chunk_summaries.append(chunk_summary)
        
        # Step 3: Combine chunk summaries
        if progress_callback:
            progress_callback(75, "Creating final summary from all chunks")
        
        # Format the chunk summaries for the final summary
        combined_text = ""
        for i, summary in enumerate(chunk_summaries):
            combined_text += f"=== Section {i+1} ===\n"
            combined_text += f"Summary: {summary['summary']}\n\n"
            
            combined_text += "Key Points:\n"
            for point in summary['key_points']:
                combined_text += f"- {point}\n"
            combined_text += "\n"
            
            if summary['decisions']:
                combined_text += "Decisions:\n"
                for decision in summary['decisions']:
                    combined_text += f"- {decision}\n"
                combined_text += "\n"
            
            if summary['action_items']:
                combined_text += "Action Items:\n"
                for item in summary['action_items']:
                    if isinstance(item, dict):
                        action = item.get('action', 'Unknown action')
                        assignee = item.get('assignee', 'Unassigned')
                        combined_text += f"- {action} (Assigned to: {assignee})\n"
                    else:
                        combined_text += f"- {item}\n"
                combined_text += "\n"
            
            combined_text += "\n"
        
        # Step 4: Create final summary
        final_summary = summarize_transcript_chunk(combined_text, language, is_final=True)
        
        # Step 5: Add metadata
        final_summary["metadata"] = {
            "total_duration_minutes": int((transcript_data[-1]['end_time'] - transcript_data[0]['start_time']) / 60),
            "chunks_analyzed": len(transcript_chunks),
            "language": language or "en"
        }
        
        if progress_callback:
            progress_callback(100, "Hierarchical summarization complete")
            
        return final_summary
    
    except Exception as e:
        logger.error(f"Error in hierarchical summarization: {e}", exc_info=True)
        raise Exception(f"Failed to summarize long transcript: {str(e)}")

def extract_speakers_from_transcript(transcript_data):
    """
    Extract unique speakers from transcript data
    
    Args:
        transcript_data: List of transcript segments
        
    Returns:
        List of unique speaker IDs
    """
    speakers = set()
    for segment in transcript_data:
        speakers.add(segment['speaker'])
    return sorted(list(speakers))

def summarize_long_meeting(transcript_data, language=None, progress_callback=None):
    """
    Main function to summarize a long meeting transcript
    
    Args:
        transcript_data: List of transcript segments with speaker IDs and text
        language: Optional language code
        progress_callback: Optional callback function for progress updates
        
    Returns:
        Dictionary with summary and action items
    """
    if progress_callback:
        progress_callback(0, "Starting long meeting summarization")
    
    # Get speakers from transcript
    speakers = extract_speakers_from_transcript(transcript_data)
    
    # Generate hierarchical summary
    summary_result = hierarchical_summarize(transcript_data, language, progress_callback)
    
    # Format the result in the expected structure for our application
    meeting_summary = {
        "summary": summary_result.get("summary", ""),
        "key_points": summary_result.get("key_points", []),
        "decisions": summary_result.get("decisions", [])
    }
    
    # Process action items to ensure they have the expected structure
    action_items = []
    for item in summary_result.get("action_items", []):
        if isinstance(item, dict):
            # If it's already in the right format, use it as is
            action_items.append({
                "action": item.get("action", ""),
                "assignee": item.get("assignee", "Unassigned"),
                "due_date": item.get("due_date", "Not specified"),
                "priority": item.get("priority", "medium")
            })
        else:
            # Otherwise, try to parse it
            action_text = item
            assignee = "Unassigned"
            
            # Try to extract assignee from text
            for speaker in speakers:
                speaker_name = f"Speaker {speaker}"
                if speaker_name in action_text:
                    assignee = speaker_name
                    break
            
            action_items.append({
                "action": action_text,
                "assignee": assignee,
                "due_date": "Not specified",
                "priority": "medium"
            })
    
    final_result = {
        "meeting_summary": meeting_summary,
        "action_items": action_items,
        "metadata": summary_result.get("metadata", {})
    }
    
    return final_result