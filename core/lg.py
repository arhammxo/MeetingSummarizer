import os
import json
import re
from typing import Dict, List, TypedDict, Literal, Union, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field, validator
import logging
from services.llm_service import get_ollama_llm
from config import settings

# Configure logging for this module
logger = logging.getLogger(__name__)

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
            
            # Add pattern matching for common narrative intros
            narrative_patterns = [
                r'Summary of the Conversation\s*(.*?)(?=\n\n|\Z)',
                r'Analysis\s*:\s*(.*?)(?=\n\n|\Z)',
                r'Meeting Analysis\s*:\s*(.*?)(?=\n\n|\Z)'
            ]

            for pattern in narrative_patterns:
                narrative_match = re.search(pattern, text, re.DOTALL)
                if narrative_match:
                    # Extract content and attempt to structure it
                    narrative_content = narrative_match.group(1).strip()
                    # Process narrative into structured data
                    logger.info(f"Found narrative introduction with pattern: {pattern}")
                    
                    # Extract what seem to be key points or summary
                    lines = narrative_content.split('\n')
                    key_points = []
                    decisions = []
                    summary = ""
                    
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
                            elif not summary and len(line) < 100:
                                # Use first substantial text as summary if nothing else found
                                summary = line
                    
                    return {
                        "summary": summary if summary else "Unable to extract summary from text",
                        "key_points": key_points[:5] if key_points else ["Unable to extract key points"],
                        "decisions": decisions if decisions else [],
                        "action_items": []
                    }
            
            # If no narrative patterns found, fall back to the original approach
            # Extract what seem to be key points or summary
            lines = text.split('\n')
            key_points = []
            decisions = []
            summary = ""
            
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
                    elif not summary and len(line) < 100:
                        # Use first substantial text as summary if nothing else found
                        summary = line
            
            return {
                "summary": summary if summary else "Unable to extract summary from text",
                "key_points": key_points[:5] if key_points else ["Unable to extract key points"],
                "decisions": decisions if decisions else [],
                "action_items": []
            }

# Define the state of our graph
class ActionItem(BaseModel):
    """An action item extracted from the meeting."""
    action: str = Field(description="The action to be taken")
    assignee: str = Field(description="Person responsible for the action", default="Unassigned")
    due_date: str = Field(description="Due date for the action (if mentioned)", default="Not specified")
    priority: str = Field(description="Priority level (high, medium, low)", default="medium")

class MeetingSummary(BaseModel):
    """A concise summary of the meeting."""
    summary: str = Field(description="Concise summary of the meeting")
    key_points: List[str] = Field(description="Key points discussed in the meeting")
    decisions: List[str] = Field(description="Decisions made during the meeting")

class AgentState(TypedDict):
    """The state of our meeting summarizer agent."""
    transcript: str
    participants: List[str]
    language: Optional[str]
    current_step: Literal["initialization", "analyze", "summarize", "extract_actions", "format_output", "complete"]
    analysis: Dict
    meeting_summary: MeetingSummary
    action_items: List[ActionItem]
    final_output: Dict

# Initialize our LLM
def get_llm():
    """Get the language model"""
    from services.llm_service import get_llm as get_llm_service
    return get_llm_service(temperature=0, purpose="summarization")

def merge_analyses(analyses):
    """
    Merge multiple chunk analyses into a single cohesive analysis
    
    Args:
        analyses: List of analysis dictionaries from different chunks
        
    Returns:
        Merged analysis dictionary
    """
    if not analyses:
        return {}
    
    if len(analyses) == 1:
        return analyses[0]
    
    # Initialize with the first analysis
    merged = {
        "meeting_purpose": analyses[0].get("meeting_purpose", ""),
        "main_topics": [],
        "emotional_tone": analyses[0].get("emotional_tone", ""),
        "participation_level": analyses[0].get("participation_level", ""),
        "disagreement_areas": []
    }
    
    # Collect all topics and disagreement areas
    all_topics = set()
    all_disagreements = set()
    
    for analysis in analyses:
        # Add topics
        for topic in analysis.get("main_topics", []):
            all_topics.add(topic)
        
        # Add disagreement areas
        for area in analysis.get("disagreement_areas", []):
            all_disagreements.add(area)
    
    # Update merged analysis
    merged["main_topics"] = list(all_topics)
    merged["disagreement_areas"] = list(all_disagreements)
    
    return merged

# Define the nodes for our graph
def create_analyze_node(language=None):
    """Create the analyze node with simplified prompts"""
    language_instructions = ""
    if language and language != "en":
        language_instructions = f"Output in {language} language."
    
    system_message = SystemMessage(content=f"""Analyze the meeting transcript and return JSON with this exact structure:
{{
  "meeting_purpose": "brief purpose",
  "main_topics": ["topic1", "topic2", "topic3"],
  "emotional_tone": "brief tone description",
  "participation_level": "brief participation description",
  "disagreement_areas": ["area1", "area2"]
}}

Rules:
- Return ONLY valid JSON
- Keep descriptions brief (under 50 words)
- List 3-5 main topics
- List 0-3 disagreement areas
{language_instructions}""")
    
    user_template = """Transcript: {transcript}
Participants: {participants}"""
    
    def analyze_node(state: AgentState) -> AgentState:
        """Analyze the meeting transcript"""
        try:
            transcript = state["transcript"]
            participants = state["participants"]
            
            analysis_schema = {
                "type": "object",
                "properties": {
                    "meeting_purpose": {"type": "string"},
                    "main_topics": {"type": "array", "items": {"type": "string"}},
                    "emotional_tone": {"type": "string"},
                    "participation_level": {"type": "string"},
                    "disagreement_areas": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["meeting_purpose", "main_topics", "emotional_tone", "participation_level", "disagreement_areas"]
            }
            
            # For shorter transcripts
            if len(transcript) < 15000:
                prompt = ChatPromptTemplate.from_messages([
                    system_message,
                    HumanMessage(content=user_template.format(
                        transcript=transcript,
                        participants=", ".join(participants)
                    ))
                ])
                
                # Try structured output first for Ollama
                if settings.LLM_PROVIDER == "ollama" and settings.OLLAMA_USE_STRUCTURED_OUTPUT:
                    try:
                        structured_llm = get_ollama_llm(
                            temperature=0.1,
                            purpose="summarization",
                            format_schema=analysis_schema
                        )
                        result = prompt | structured_llm
                        response = result.invoke({})
                        
                        # Parse response
                        if hasattr(response, 'content'):
                            analysis = json.loads(response.content)
                        else:
                            analysis = json.loads(str(response))
                            
                        return {**state, "analysis": analysis, "current_step": "summarize"}
                    except Exception as e:
                        logger.warning(f"Structured output failed: {e}, falling back to JSON parser")
                
                # Fallback to regular JSON parser
                llm = get_llm()
                chain = prompt | llm | JsonOutputParser()
                analysis = chain.invoke({})
                
                return {**state, "analysis": analysis, "current_step": "summarize"}
            
            # For longer transcripts, chunk processing
            chunks = chunk_transcript(transcript)
            logging.info(f"Splitting transcript into {len(chunks)} chunks")
            
            chunk_analyses = []
            for i, chunk in enumerate(chunks):
                logging.info(f"Analyzing chunk {i+1}/{len(chunks)}")
                
                chunk_prompt = ChatPromptTemplate.from_messages([
                    SystemMessage(content=f"Analyze chunk {i+1} of {len(chunks)}. {system_message.content}"),
                    HumanMessage(content=user_template.format(
                        transcript=chunk,
                        participants=", ".join(participants)
                    ))
                ])
                
                try:
                    # Try structured output first for Ollama
                    if settings.LLM_PROVIDER == "ollama" and settings.OLLAMA_USE_STRUCTURED_OUTPUT:
                        try:
                            structured_llm = get_ollama_llm(
                                temperature=0.1,
                                purpose="summarization",
                                format_schema=analysis_schema
                            )
                            result = chunk_prompt | structured_llm
                            response = result.invoke({})
                            
                            # Parse response
                            if hasattr(response, 'content'):
                                chunk_analysis = json.loads(response.content)
                            else:
                                chunk_analysis = json.loads(str(response))
                                
                            chunk_analyses.append(chunk_analysis)
                            continue
                        except Exception as e:
                            logger.warning(f"Structured output failed for chunk {i+1}: {e}, falling back to JSON parser")
                    
                    # Fallback to regular JSON parser
                    llm = get_llm()
                    chain = chunk_prompt | llm | JsonOutputParser()
                    chunk_analysis = chain.invoke({})
                    chunk_analyses.append(chunk_analysis)
                    
                except Exception as e:
                    logger.warning(f"Chunk {i+1} failed: {e}")
                    chunk_analyses.append({
                        "meeting_purpose": f"Chunk {i+1} analysis failed",
                        "main_topics": ["Error processing chunk"],
                        "emotional_tone": "Unknown",
                        "participation_level": "Unknown",
                        "disagreement_areas": []
                    })
            
            merged_analysis = merge_analyses(chunk_analyses)
            return {**state, "analysis": merged_analysis, "current_step": "summarize"}
            
        except Exception as e:
            logger.error(f"Error in analyze_node: {str(e)}")
            return {**state, "analysis": {
                "meeting_purpose": "Error analyzing transcript",
                "main_topics": ["Analysis failed"],
                "emotional_tone": "Unknown",
                "participation_level": "Unknown",
                "disagreement_areas": []
            }, "current_step": "summarize"}
    
    return analyze_node

def chunk_transcript(transcript, max_chunk_size=15000):  # Increased from 8000
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

def create_summarize_node(language=None):
    """Create the summarize node with simplified prompts"""
    language_instructions = f"Output in {language} language." if language and language != "en" else ""
    
    system_message = SystemMessage(content=f"""Create a meeting summary with this JSON structure:
{{
  "summary": "2-3 sentence overview",
  "key_points": ["point1", "point2", "point3"],
  "decisions": ["decision1", "decision2"]
}}

Keep it concise and factual. {language_instructions}""")
    
    user_template = """Based on this analysis: {analysis}
Transcript: {transcript}
Participants: {participants}"""
    
    def summarize_node(state: AgentState) -> AgentState:
        """Generate a concise summary of the meeting"""
        try:
            summary_schema = {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "key_points": {"type": "array", "items": {"type": "string"}},
                    "decisions": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["summary", "key_points", "decisions"]
            }
            
            prompt = ChatPromptTemplate.from_messages([
                system_message,
                HumanMessage(content=user_template.format(
                    transcript=state["transcript"][:5000],
                    analysis=json.dumps(state["analysis"]),
                    participants=", ".join(state["participants"])
                ))
            ])
            
            if settings.LLM_PROVIDER == "ollama" and settings.OLLAMA_USE_STRUCTURED_OUTPUT:
                llm = get_ollama_llm(temperature=0.1, purpose="summarization", format_schema=summary_schema)
                result = prompt | llm
                response = result.invoke({})
                summary_data = json.loads(response.content)
            else:
                llm = get_llm()
                chain = prompt | llm | JsonOutputParser()
                summary_data = chain.invoke({})
            
            meeting_summary = MeetingSummary(
                summary=summary_data["summary"],
                key_points=summary_data["key_points"],
                decisions=summary_data["decisions"]
            )
            return {**state, "meeting_summary": meeting_summary, "current_step": "extract_actions"}
            
        except Exception as e:
            logger.error(f"Error in summarize_node: {str(e)}")
            meeting_summary = MeetingSummary(
                summary="Error generating summary",
                key_points=["Unable to extract key points"],
                decisions=[]
            )
            return {**state, "meeting_summary": meeting_summary, "current_step": "extract_actions"}
    
    return summarize_node

def create_extract_actions_node(language=None):
    """Create the action extraction node with simplified prompts"""
    language_instructions = f"Output in {language} language." if language and language != "en" else ""
    
    system_message = SystemMessage(content=f"""Extract action items as JSON array:
[
  {{
    "action": "specific action",
    "assignee": "person name or Unassigned",
    "due_date": "date or Not specified",
    "priority": "high/medium/low"
  }}
]

Return empty array [] if no actions found. {language_instructions}""")
    
    user_template = """Find action items in: {transcript}
Participants: {participants}"""
    
    def extract_actions_node(state: AgentState) -> AgentState:
        """Extract action items from the meeting transcript"""
        try:
            actions_schema = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "assignee": {"type": "string"},
                        "due_date": {"type": "string"},
                        "priority": {"type": "string", "enum": ["high", "medium", "low"]}
                    },
                    "required": ["action", "assignee", "due_date", "priority"]
                }
            }
            
            prompt = ChatPromptTemplate.from_messages([
                system_message,
                HumanMessage(content=user_template.format(
                    transcript=state["transcript"][:5000],
                    participants=", ".join(state["participants"])
                ))
            ])
            
            if settings.LLM_PROVIDER == "ollama" and settings.OLLAMA_USE_STRUCTURED_OUTPUT:
                llm = get_ollama_llm(temperature=0.1, purpose="summarization", format_schema=actions_schema)
                result = prompt | llm
                response = result.invoke({})
                action_data = json.loads(response.content)
            else:
                llm = get_llm()
                chain = prompt | llm | JsonOutputParser()
                action_data = chain.invoke({})
            
            action_items = []
            for item in action_data:
                action_items.append(ActionItem(
                    action=item.get("action", ""),
                    assignee=item.get("assignee", "Unassigned"),
                    due_date=item.get("due_date", "Not specified"),
                    priority=item.get("priority", "medium")
                ))
            
            return {**state, "action_items": action_items, "current_step": "format_output"}
            
        except Exception as e:
            logger.error(f"Error in extract_actions_node: {str(e)}")
            return {**state, "action_items": [], "current_step": "format_output"}
    
    return extract_actions_node

def create_format_output_node(language=None):
    """Create the format output node with language-specific instructions"""
    language_instructions = ""
    if language:
        if language == "hi":
            language_instructions = "Respond in Hindi language using Devanagari script."
        elif language != "en":  # For languages other than English
            language_instructions = f"Respond in {language} language."
    
    # 4. Format the final output
    system_message = SystemMessage(content=f"""You are responsible for creating the final meeting summary and action item report.
    Format the provided information into a well-structured, professional report.
    
    Your output should be a JSON object with two sections:
    - meeting_summary: Contains the summary, key points, and decisions
    - action_items: The list of action items with their details
    
    Format your response as JSON. DO NOT include explanatory text before or after the JSON.
    {language_instructions}""")
    
    user_template = """Meeting Summary: {meeting_summary}
    
    Action Items: {action_items}"""
    
    def format_output_node(state: AgentState) -> AgentState:
        """Format the final output with the meeting summary and action items."""
        # Convert Pydantic models to dictionaries for the LLM
        meeting_summary_dict = state["meeting_summary"].model_dump()
        action_items_dict = [item.model_dump() for item in state["action_items"]]
        
        try:
            # Create a one-time template
            prompt = ChatPromptTemplate.from_messages([
                system_message,
                HumanMessage(content=user_template.format(
                    meeting_summary=json.dumps(meeting_summary_dict, indent=2),
                    action_items=json.dumps(action_items_dict, indent=2)
                ))
            ])
            
            llm = get_llm()
            chain = prompt | llm | JsonOutputParser()
            final_output = chain.invoke({})
            return {**state, "final_output": final_output, "current_step": "complete"}
        except Exception as e:
            logger.warning(f"JSON parsing error in format_output_node: {e}. Attempting recovery...")
            
            try:
                # Fall back to string output and robust parsing
                str_chain = prompt | llm | StrOutputParser()
                raw_response = str_chain.invoke({})
                
                # Use robust parsing
                parsed_result = robust_json_parse(raw_response)
                logger.info("Successfully recovered JSON structure for final output")
                return {**state, "final_output": parsed_result, "current_step": "complete"}
            except Exception as recovery_error:
                logger.error(f"Recovery failed: {recovery_error}")
                # Return a simple structure using the existing data
                final_output = {
                    "meeting_summary": meeting_summary_dict,
                    "action_items": action_items_dict
                }
                return {**state, "final_output": final_output, "current_step": "complete"}
    
    return format_output_node

# Define the workflow graph
def create_meeting_summarizer_graph(language=None):
    """Create the LangGraph workflow for meeting summarization with language support."""
    # Initialize the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes with language-specific handlers
    workflow.add_node("analyze", create_analyze_node(language))
    workflow.add_node("summarize", create_summarize_node(language))
    workflow.add_node("extract_actions", create_extract_actions_node(language))
    workflow.add_node("format_output", create_format_output_node(language))
    
    # Define edges
    workflow.add_edge("analyze", "summarize")
    workflow.add_edge("summarize", "extract_actions")
    workflow.add_edge("extract_actions", "format_output")
    workflow.add_edge("format_output", END)
    
    # Set the entry point
    workflow.set_entry_point("analyze")
    
    # Compile the graph
    return workflow.compile()

# Main function to run the meeting summarizer
def summarize_meeting(transcript: str, participants: List[str], language: str = None, additional_context: str = None):
    """Run the meeting summarizer on a transcript and return the summary and action items."""
    # Check for empty inputs
    if not transcript or not transcript.strip():
        raise ValueError("Meeting transcript cannot be empty")
    
    if not participants:
        raise ValueError("Participants list cannot be empty")
    
    # Clean the transcript
    transcript = transcript.strip()
    
    # Create the graph with language support
    graph = create_meeting_summarizer_graph(language)
    
    try:
        # Initialize the state
        initial_state = {
            "transcript": transcript,
            "participants": participants,
            "language": language,
            "additional_context": additional_context,  # Add additional context
            "current_step": "initialization",
            "analysis": {},
            "meeting_summary": MeetingSummary(summary="", key_points=[], decisions=[]),
            "action_items": [],
            "final_output": {}
        }
        
        # Run the graph
        result = graph.invoke(initial_state)
        
        return result["final_output"]
    except Exception as e:
        # Provide meaningful error message
        logger.error(f"Error processing meeting: {str(e)}")
        raise Exception(f"Failed to summarize meeting: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Example meeting transcript
    example_transcript = """
    Alice: Good morning everyone, thanks for joining our product roadmap discussion. Today we need to finalize Q3 priorities.
    
    Bob: I think we should focus on the new user onboarding flow. Our metrics show a 30% drop-off in the first week.
    
    Charlie: That's a good point. The analytics team sent me a report yesterday confirming that issue. I can share it with everyone after the meeting.
    
    Alice: Let's make that priority #1 then. Bob, can you draft a project plan by Friday?
    
    Bob: Yes, I'll have it ready by end of day Friday.
    
    Dave: What about the mobile app redesign? We promised that to customers last quarter.
    
    Alice: You're right, we need to address that too. Let's make it priority #2. Dave, you'll lead that initiative, right?
    
    Dave: Yes, but I'll need support from the design team. Eva, can your team help with this?
    
    Eva: Definitely. We can allocate 2 designers starting next week.
    
    Alice: Great. Charlie, please set up a kickoff meeting for the mobile redesign team for Monday.
    
    Charlie: Will do. I'll send out calendar invites this afternoon.
    
    Alice: Perfect. Are there any other urgent items we need to discuss?
    
    Bob: We should decide on the budget allocation between these two projects.
    
    Alice: Good point. Let's allocate 60% of resources to onboarding and 40% to the mobile redesign. Any objections?
    
    Dave: That works for me.
    
    Eva: Same here.
    
    Charlie: No objections.
    
    Alice: Great, then we're all set. Bob will share the onboarding project plan by Friday, and Charlie will set up the mobile redesign kickoff for Monday. Thanks everyone!
    """
    
    participants = ["Alice", "Bob", "Charlie", "Dave", "Eva"]
    
    # Run the meeting summarizer
    result = summarize_meeting(example_transcript, participants)
    
    # Output the result
    print(json.dumps(result, indent=2))