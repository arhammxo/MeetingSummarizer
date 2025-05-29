from services.llm_service import get_llm, get_ollama_llm, create_chat_prompt_template, create_output_parser
import logging

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
    
    # Simplified prompt
    system_message = """Create a JSON summary for this speaker:
{
  "key_contributions": ["contribution1", "contribution2"],
  "action_items": ["action1", "action2"],
  "questions_raised": ["question1"],
  "brief_summary": "1-2 sentence summary"
}

Be concise and factual. {language_instructions}"""

    user_message = """Speaker: {speaker}
Contributions: {contributions}"""

    # Define schema
    speaker_schema = {
        "type": "object",
        "properties": {
            "key_contributions": {"type": "array", "items": {"type": "string"}},
            "action_items": {"type": "array", "items": {"type": "string"}},
            "questions_raised": {"type": "array", "items": {"type": "string"}},
            "brief_summary": {"type": "string"}
        },
        "required": ["key_contributions", "action_items", "questions_raised", "brief_summary"]
    }

    prompt = create_chat_prompt_template(system_message, user_message)
    
    # Create the chain with JSON output
    json_parser = create_output_parser(schema=speaker_schema)
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
                    "language_instructions": language_instructions
                })
                
                # Validate that the response is properly formatted
                if not isinstance(summary, dict):
                    raise ValueError("Response is not a dictionary")
                
                # Ensure all required fields are present
                required_fields = ["key_contributions", "action_items", "questions_raised", "brief_summary"]
                for field in required_fields:
                    if field not in summary:
                        summary[field] = [] if field != "brief_summary" else "No summary available"
                
                speaker_summaries[speaker] = summary
                
            except Exception as e:
                logging.error(f"Error generating summary for {speaker}: {str(e)}")
                
                # Attempt to generate a simpler summary with fewer requirements
                try:
                    # Create a simpler prompt for fallback
                    simple_system = f"Summarize what {speaker} contributed to the meeting in 1-2 sentences."
                    simple_user = f"Speaker contributions:\n{contributions}"
                    simple_prompt = create_chat_prompt_template(simple_system, simple_user)
                    
                    # Get a simple text response without JSON parsing
                    simple_result = simple_prompt | llm
                    simple_summary = simple_result.invoke({})
                    
                    # Format into expected structure
                    speaker_summaries[speaker] = {
                        "key_contributions": ["See brief summary"],
                        "action_items": [],
                        "questions_raised": [],
                        "brief_summary": simple_summary if isinstance(simple_summary, str) else str(simple_summary)
                    }
                except Exception as fallback_error:
                    # Ultimate fallback
                    speaker_summaries[speaker] = {
                        "key_contributions": ["Error processing contributions"],
                        "action_items": [],
                        "questions_raised": [],
                        "brief_summary": f"Error generating summary for {speaker}"
                    }
    
    return speaker_summaries