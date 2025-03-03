import os
import json
from typing import Dict, List, TypedDict, Literal, Union, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field, validator

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
    current_step: Literal["initialization", "analyze", "summarize", "extract_actions", "format_output", "complete"]
    analysis: Dict
    meeting_summary: MeetingSummary
    action_items: List[ActionItem]
    final_output: Dict

# Initialize our LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=os.OPENAI_API_KEY)

# Define the nodes for our graph

# 1. Analyze the transcript
analyze_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert meeting analyst. Analyze the provided meeting transcript 
    and identify the following elements:
    - Meeting purpose
    - Main topics discussed
    - Emotional tone of the meeting
    - Level of participation across attendees
    - Any areas of disagreement or conflict
    
    Provide your analysis in a structured JSON format."""),
    ("human", "Here is the meeting transcript: {transcript}\n\nParticipants: {participants}")
])

analyze_chain = analyze_prompt | llm | JsonOutputParser()

def analyze_node(state: AgentState) -> AgentState:
    """Analyze the meeting transcript to understand context and participants."""
    analysis = analyze_chain.invoke({
        "transcript": state["transcript"],
        "participants": state["participants"]
    })
    return {**state, "analysis": analysis, "current_step": "summarize"}

# 2. Summarize the meeting
summarize_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert meeting summarizer. Based on the meeting transcript and analysis,
    create a concise summary of the meeting that captures the essence of what was discussed and decided.
    
    Follow this structure:
    1. Summary: A 2-3 sentence overview of the meeting
    2. Key Points: Bullet points of important topics discussed
    3. Decisions: Bullet points of decisions made during the meeting
    
    Your response should be a JSON object with the fields 'summary', 'key_points', and 'decisions'.
    Keep the summary concise and focused on what matters."""),
    ("human", """Meeting Transcript: {transcript}
    
    Meeting Analysis: {analysis}
    
    Participants: {participants}""")
])

summarize_chain = summarize_prompt | llm | JsonOutputParser()

def summarize_node(state: AgentState) -> AgentState:
    """Generate a concise summary of the meeting."""
    summary_data = summarize_chain.invoke({
        "transcript": state["transcript"],
        "analysis": state["analysis"],
        "participants": state["participants"]
    })
    meeting_summary = MeetingSummary(
        summary=summary_data["summary"],
        key_points=summary_data["key_points"],
        decisions=summary_data["decisions"]
    )
    return {**state, "meeting_summary": meeting_summary, "current_step": "extract_actions"}

# 3. Extract action items
action_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at identifying action items from meeting transcripts.
    Your task is to extract clear action items from the provided meeting transcript.
    
    For each action item, identify:
    - The specific action to be taken
    - Who is responsible for the action
    - Any mentioned deadline or due date
    - The priority level (high, medium, low) based on context
    
    Return your results as a JSON array of action items. If no action items are mentioned, return an empty array.
    Each action item should have fields: 'action', 'assignee', 'due_date', and 'priority'.
    
    IMPORTANT: Always provide a string value for each field. If a field is missing:
    - For 'due_date': use "Not specified" 
    - For 'priority': use "medium"
    - For 'assignee': use "Unassigned"
    
    DO NOT return null values - use appropriate string defaults instead."""),
    ("human", """Meeting Transcript: {transcript}
    
    Meeting Summary: {summary}
    
    Participants: {participants}""")
])

action_chain = action_prompt | llm | JsonOutputParser()

def extract_actions_node(state: AgentState) -> AgentState:
    """Extract action items from the meeting transcript."""
    action_data = action_chain.invoke({
        "transcript": state["transcript"],
        "summary": state["meeting_summary"],
        "participants": state["participants"]
    })
    
    action_items = []
    for item in action_data:
        # Handle possible None values by ensuring all fields are strings
        action = item.get("action", "")
        assignee = item.get("assignee", "Unassigned")
        
        # Explicitly convert None to strings
        due_date = item.get("due_date")
        if due_date is None or due_date == "":
            due_date = "Not specified"
            
        priority = item.get("priority")
        if priority is None or priority == "":
            priority = "medium"
        
        action_items.append(ActionItem(
            action=action,
            assignee=assignee,
            due_date=due_date,
            priority=priority
        ))
    
    return {**state, "action_items": action_items, "current_step": "format_output"}

# 4. Format the final output
format_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are responsible for creating the final meeting summary and action item report.
    Format the provided information into a well-structured, professional report that can be easily read and shared.
    
    Your output should be a JSON object with two sections:
    1. 'meeting_summary': Contains the summary, key points, and decisions
    2. 'action_items': The list of action items with their details
    
    Make sure the formatting is clean and professional."""),
    ("human", """Meeting Summary: {meeting_summary}
    
    Action Items: {action_items}""")
])

format_chain = format_prompt | llm | JsonOutputParser()

def format_output_node(state: AgentState) -> AgentState:
    """Format the final output with the meeting summary and action items."""
    # Convert Pydantic models to dictionaries for the LLM
    meeting_summary_dict = state["meeting_summary"].model_dump()
    action_items_dict = [item.model_dump() for item in state["action_items"]]
    
    final_output = format_chain.invoke({
        "meeting_summary": meeting_summary_dict,
        "action_items": action_items_dict
    })
    return {**state, "final_output": final_output, "current_step": "complete"}

# Define the workflow graph
def create_meeting_summarizer_graph():
    """Create the LangGraph workflow for meeting summarization."""
    # Initialize the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("extract_actions", extract_actions_node)
    workflow.add_node("format_output", format_output_node)
    
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
def summarize_meeting(transcript: str, participants: List[str]):
    """Run the meeting summarizer on a transcript and return the summary and action items."""
    # Check for empty inputs
    if not transcript or not transcript.strip():
        raise ValueError("Meeting transcript cannot be empty")
    
    if not participants:
        raise ValueError("Participants list cannot be empty")
    
    # Clean the transcript
    transcript = transcript.strip()
    
    # Create the graph
    graph = create_meeting_summarizer_graph()
    
    try:
        # Initialize the state
        initial_state = {
            "transcript": transcript,
            "participants": participants,
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
        print(f"Error processing meeting: {str(e)}")
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