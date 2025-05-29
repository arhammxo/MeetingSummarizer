from services.llm_service import get_llm, create_chat_prompt_template, create_output_parser, get_ollama_llm
import logging
import json
from config import settings

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
    
    # Create prompt without XML tags that confuse smaller models
    system_message = """You are a meeting analyst. Create a JSON summary for each speaker with these fields:
- key_contributions: List of 2-4 main points
- action_items: List of tasks they agreed to
- questions_raised: List of questions they asked
- brief_summary: 1-2 sentence summary

Return ONLY valid JSON, no other text."""
    
    user_message = """Speaker: {speaker}
Contributions: {contributions}"""
    
    prompt = create_chat_prompt_template(system_message, user_message)
    
    # Create the chain - no schema parameter!
    json_parser = create_output_parser()  # Fixed: No schema parameter
    chain = prompt | llm | json_parser
    
    # Generate summaries for each speaker
    speaker_summaries = {}
    
    for speaker, contributions in speaker_contributions.items():
        if contributions.strip():
            if len(contributions) > 8000:
                logging.warning(f"Contributions for {speaker} exceed 8000 chars, truncating")
                contributions = contributions[:7500] + "...\n[Content truncated]"
            
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
                        temperature=0.1,
                        purpose="summarization",
                        format_schema=schema
                    )
                    
                    # Create chain without JSON parser for structured output
                    structured_chain = prompt | structured_llm
                    result = structured_chain.invoke({
                        "speaker": speaker,
                        "contributions": contributions
                    })
                    
                    # Parse the structured response
                    if hasattr(result, 'content'):
                        summary = json.loads(result.content)
                    else:
                        summary = json.loads(str(result))
                else:
                    # Regular chain with JSON parser
                    summary = chain.invoke({
                        "speaker": speaker,
                        "contributions": contributions
                    })
                
                # Validate required fields
                required_fields = ["key_contributions", "action_items", "questions_raised", "brief_summary"]
                for field in required_fields:
                    if field not in summary:
                        summary[field] = [] if field != "brief_summary" else "No summary available"
                
                speaker_summaries[speaker] = summary
                
            except Exception as e:
                logging.error(f"Error generating summary for {speaker}: {str(e)}")
                
                # Simpler fallback
                try:
                    simple_prompt = f"Summarize {speaker}'s contributions in 2 sentences: {contributions[:1000]}"
                    simple_chain = llm
                    simple_result = simple_chain.invoke(simple_prompt)
                    
                    speaker_summaries[speaker] = {
                        "key_contributions": ["See brief summary"],
                        "action_items": [],
                        "questions_raised": [],
                        "brief_summary": str(simple_result.content if hasattr(simple_result, 'content') else simple_result)
                    }
                except:
                    speaker_summaries[speaker] = {
                        "key_contributions": ["Error processing contributions"],
                        "action_items": [],
                        "questions_raised": [],
                        "brief_summary": f"Error generating summary for {speaker}"
                    }
    
    return speaker_summaries