from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from services.llm_service import get_llm, create_chat_prompt_template, create_output_parser
import logging
import re

def generate_speaker_summaries(transcript, participants, language=None):
    """
    Generate individual summaries for each speaker in the meeting.
    
    Args:
        transcript (str): The full meeting transcript
        participants (list): List of participant names/identifiers
        language (str, optional): Language code (e.g., 'hi' for Hindi) to generate summaries in
        
    Returns:
        dict: Dictionary mapping each speaker to their summary
    """
    # Initialize the LLM
    llm = get_llm(temperature=0, purpose="summarization")
    
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
    
    # Language-specific instructions
    language_instructions = ""
    if language:
        if language == "hi":
            language_instructions = "Generate your response in Hindi language using Hindi script (Devanagari)."
        elif language == "en":
            language_instructions = "Generate your response in English."
        else:
            # For other languages, specify to respond in that language
            language_instructions = f"Generate your response in the {language} language."
    
    # Create a prompt template for generating speaker summaries with instructions for Ollama
    system_message = f"""You are an expert meeting analyst. Your task is to create a concise 
    summary of what a specific participant contributed to a meeting. Focus on:

    1. Main points they raised
    2. Questions they asked
    3. Action items they took on or assigned
    4. Key decisions they influenced
    5. Their primary concerns or interests

    IMPORTANT: Reply with JSON data containing the following fields:
    - brief_summary: A 1-2 sentence summary of their overall participation
    - key_contributions: List of 2-4 main points they contributed
    - action_items: List of any tasks they agreed to do or assigned
    - questions_raised: List of important questions they asked, if any

    Your response MUST be properly formatted JSON. Just return the JSON object, no other text.

    {language_instructions}
    """

    user_message = """Speaker: {speaker}

    Their contributions:
    {contributions}

    Please summarize this speaker's participation in the meeting."""

    prompt = create_chat_prompt_template(system_message, user_message)
    
    # Create the chain with improved error handling
    json_parser = create_output_parser()
    chain = prompt | llm | json_parser
    
    # Generate summaries for each speaker
    speaker_summaries = {}
    
    for speaker, contributions in speaker_contributions.items():
        if contributions.strip():  # Only process if they have actual contributions
            if len(contributions) > 8000:
                # Handle large contributions by truncation or chunking
                logging.warning(f"Contributions for {speaker} exceed 8000 chars ({len(contributions)}), truncating")
                speaker_contributions[speaker] = contributions[:7500] + "...\n[Content truncated due to length]"
            try:
                summary = chain.invoke({
                    "speaker": speaker,
                    "contributions": contributions,
                })
                
                # Validate that the response is properly formatted
                if not isinstance(summary, dict):
                    raise ValueError(f"Response is not a dictionary: {summary}")
                
                # Ensure all required fields are present
                required_fields = ["key_contributions", "action_items", "questions_raised", "brief_summary"]
                for field in required_fields:
                    if field not in summary:
                        summary[field] = [] if field != "brief_summary" else "No summary available"
                    
                    # Ensure list fields are actually lists
                    if field != "brief_summary" and not isinstance(summary[field], list):
                        if summary[field]:
                            summary[field] = [str(summary[field])]
                        else:
                            summary[field] = []
                
                speaker_summaries[speaker] = summary
                
            except Exception as e:
                logging.error(f"Error generating summary for {speaker}: {str(e)}")
                
                # Attempt to get raw output and parse it manually
                try:
                    # Create a simpler prompt for fallback
                    str_chain = prompt | llm | StrOutputParser()
                    raw_response = str_chain.invoke({
                        "speaker": speaker,
                        "contributions": contributions,
                    })
                    
                    # Try to parse the raw response as JSON
                    summary = robust_json_parse(raw_response)
                    
                    # Ensure all required fields are present
                    required_fields = ["key_contributions", "action_items", "questions_raised", "brief_summary"]
                    for field in required_fields:
                        if field not in summary:
                            summary[field] = [] if field != "brief_summary" else "No summary available"
                        
                        # Ensure list fields are actually lists
                        if field != "brief_summary" and not isinstance(summary[field], list):
                            if summary[field]:
                                summary[field] = [str(summary[field])]
                            else:
                                summary[field] = []
                    
                    speaker_summaries[speaker] = summary
                    logging.info(f"Successfully recovered summary for {speaker} using robust parsing")
                    
                except Exception as fallback_error:
                    # Ultimate fallback - extract summary from raw text
                    try:
                        # Extract useful information from the text
                        lines = raw_response.split('\n')
                        
                        brief_summary = ""
                        key_contributions = []
                        action_items = []
                        questions_raised = []
                        
                        # Track which section we're in
                        current_section = None
                        
                        for line in lines:
                            line = line.strip()
                            if not line:
                                continue
                                
                            # Check for section headers
                            line_lower = line.lower()
                            
                            if "summary" in line_lower and ":" in line:
                                current_section = "summary"
                                brief_summary = line.split(":", 1)[1].strip()
                                continue
                                
                            if "contribution" in line_lower and ":" in line:
                                current_section = "contributions"
                                continue
                                
                            if "action" in line_lower and ":" in line:
                                current_section = "actions"
                                continue
                                
                            if "question" in line_lower and ":" in line:
                                current_section = "questions"
                                continue
                            
                            # Process line based on current section
                            if line.startswith("-") or line.startswith("*") or line.startswith("•") or re.match(r'^\d+\.', line):
                                # Remove bullet/number and get content
                                content = re.sub(r'^[-*•\d.]+\s*', '', line).strip()
                                
                                if current_section == "contributions":
                                    key_contributions.append(content)
                                elif current_section == "actions":
                                    action_items.append(content)
                                elif current_section == "questions":
                                    questions_raised.append(content)
                                elif not brief_summary:  # If no section header was found yet
                                    brief_summary = content
                            elif len(line) > 10 and not brief_summary:
                                brief_summary = line
                        
                        # Create a valid summary structure
                        summary = {
                            "brief_summary": brief_summary or f"Summary for {speaker}",
                            "key_contributions": key_contributions[:3] or ["No specific contributions identified"],
                            "action_items": action_items or [],
                            "questions_raised": questions_raised or []
                        }
                        
                        speaker_summaries[speaker] = summary
                        logging.info(f"Created text-based summary for {speaker}")
                        
                    except Exception as text_error:
                        # Create a basic placeholder summary
                        speaker_summaries[speaker] = {
                            "key_contributions": ["Error processing contributions"],
                            "action_items": [],
                            "questions_raised": [],
                            "brief_summary": f"Error generating summary for {speaker}: {str(e)}"
                        }
    
    return speaker_summaries


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
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
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
            logging.warning(f"Failed to parse JSON, creating fallback structure from text")
            
            # Extract structured fields from the text
            brief_summary = ""
            key_contributions = []
            action_items = []
            questions_raised = []
            
            # Track which section we're in
            current_section = None
            
            # Process line by line
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check for section headers
                line_lower = line.lower()
                
                if any(term in line_lower for term in ["summary", "overview"]):
                    current_section = "summary"
                    if ":" in line:
                        brief_summary = line.split(":", 1)[1].strip()
                    continue
                    
                if any(term in line_lower for term in ["contribution", "point", "main point"]):
                    current_section = "contributions"
                    continue
                    
                if any(term in line_lower for term in ["action", "task", "to do"]):
                    current_section = "actions"
                    continue
                    
                if any(term in line_lower for term in ["question", "asked"]):
                    current_section = "questions"
                    continue
                
                # Process line based on current section
                if line.startswith("-") or line.startswith("*") or line.startswith("•") or re.match(r'^\d+\.', line):
                    # Remove bullet/number and get content
                    content = re.sub(r'^[-*•\d.]+\s*', '', line).strip()
                    
                    if current_section == "contributions" or not current_section:
                        key_contributions.append(content)
                    elif current_section == "actions":
                        action_items.append(content)
                    elif current_section == "questions":
                        questions_raised.append(content)
                    elif not brief_summary:  # If no section is identified but we need a summary
                        brief_summary = content
                elif len(line) > 10 and not brief_summary and current_section == "summary":
                    brief_summary = line
            
            # If we didn't find a summary, use the first substantive line
            if not brief_summary:
                for line in lines:
                    if len(line.strip()) > 15 and len(line.strip()) < 150:
                        brief_summary = line.strip()
                        break
            
            # Create a valid summary structure
            return {
                "brief_summary": brief_summary or "No summary available",
                "key_contributions": key_contributions[:4] or ["No specific contributions identified"],
                "action_items": action_items or [],
                "questions_raised": questions_raised or []
            }