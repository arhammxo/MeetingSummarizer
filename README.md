# Meeting Summarizer and Action List Generator

This project uses LangGraph to create an AI agent that automatically summarizes meeting transcripts and generates action item lists.

## Overview

The Meeting Summarizer and Action List Generator is designed to:

1. **Analyze meeting transcripts** to understand context and participants
2. **Generate concise summaries** of what was discussed
3. **Extract action items** with assignees and deadlines
4. **Format everything** into a clean, shareable report

## Project Structure

- `meeting_summarizer.py`: Core agent implementation using LangGraph
- `app.py`: Streamlit web application for using the meeting summarizer
- `requirements.txt`: Dependencies needed to run the project

## How It Works

The LangGraph workflow consists of four main steps:

1. **Analysis**: Examines the meeting transcript to identify purpose, topics, tone, and participation
2. **Summarization**: Creates a concise summary with key points and decisions
3. **Action Item Extraction**: Identifies action items, assignees, deadlines, and priorities
4. **Output Formatting**: Combines everything into a well-structured, professional report

## Getting Started

### Prerequisites

- Python 3.9+
- An OpenAI API key (for GPT-4o access)

### Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set your OpenAI API key:
   ```
   export OPENAI_API_KEY="your_api_key_here"
   ```

### Running the Application

Launch the Streamlit app:
```
streamlit run app.py
```

The web interface will open, allowing you to paste meeting transcripts and get summaries and action items.

## Example

Here's a sample of what the output looks like:

```json
{
  "meeting_summary": {
    "summary": "This meeting focused on finalizing Q3 priorities, with the team deciding to prioritize the new user onboarding flow (Priority #1) and mobile app redesign (Priority #2). Resource allocation was set at 60% for onboarding and 40% for mobile redesign.",
    "key_points": [
      "Analytics show a 30% drop-off in new user onboarding within the first week",
      "Mobile app redesign was promised to customers last quarter",
      "Design team can allocate 2 designers for the mobile redesign starting next week",
      "Budget allocation was decided at 60% for onboarding and 40% for mobile redesign"
    ],
    "decisions": [
      "New user onboarding flow set as Priority #1",
      "Mobile app redesign set as Priority #2",
      "Resource allocation: 60% to onboarding, 40% to mobile redesign",
      "Bob will lead the onboarding initiative",
      "Dave will lead the mobile redesign initiative"
    ]
  },
  "action_items": [
    {
      "action": "Draft a project plan for the new user onboarding flow",
      "assignee": "Bob",
      "due_date": "Friday (end of day)",
      "priority": "high"
    },
    {
      "action": "Set up a kickoff meeting for the mobile redesign team",
      "assignee": "Charlie",
      "due_date": "This afternoon (for Monday meeting)",
      "priority": "medium"
    },
    {
      "action": "Allocate 2 designers for the mobile redesign project",
      "assignee": "Eva",
      "due_date": "Next week",
      "priority": "medium"
    }
  ]
}
```

## Customization

The system is designed to be extensible. You can customize:

- The prompt templates to modify how analysis is performed
- The structure of the summary and action items
- The UI of the Streamlit application

## Dependencies

- LangGraph
- LangChain
- OpenAI
- Streamlit
- Pydantic

## License

This project is licensed under the MIT License - see the LICENSE file for details.
