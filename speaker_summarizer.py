from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

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
    llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key="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")
    
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
    
    # Create a prompt template for generating speaker summaries
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert meeting analyst. Your task is to create a concise 
        summary of what a specific participant contributed to a meeting. Focus on:
        
        1. Main points they raised
        2. Questions they asked
        3. Action items they took on or assigned
        4. Key decisions they influenced
        5. Their primary concerns or interests
        
        Format your response as a JSON object with these keys:
        - "key_contributions": List of 2-4 main points they contributed
        - "action_items": List of any tasks they agreed to do or assigned
        - "questions_raised": List of important questions they asked, if any
        - "brief_summary": A 1-2 sentence summary of their overall participation
        
        Keep your response focused only on this speaker's contributions.
        
        {language_instructions}
        """),
        ("human", """Speaker: {speaker}
        
        Their contributions:
        {contributions}
        
        Please summarize this speaker's participation in the meeting.""")
    ])
    
    # Create the chain with JSON output
    chain = prompt | llm | JsonOutputParser()
    
    # Generate summaries for each speaker
    speaker_summaries = {}
    
    for speaker, contributions in speaker_contributions.items():
        if contributions.strip():  # Only process if they have actual contributions
            try:
                summary = chain.invoke({
                    "speaker": speaker,
                    "contributions": contributions,
                    "language_instructions": language_instructions
                })
                speaker_summaries[speaker] = summary
            except Exception as e:
                # Fallback in case of parsing errors
                speaker_summaries[speaker] = {
                    "key_contributions": ["Error processing contributions"],
                    "action_items": [],
                    "questions_raised": [],
                    "brief_summary": f"Error generating summary for {speaker}: {str(e)}"
                }
    
    return speaker_summaries