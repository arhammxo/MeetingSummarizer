# test_core.py
import os
from dotenv import load_dotenv
from core.lg import summarize_meeting

# Load environment variables
load_dotenv()

# Verify OpenAI API key is set
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("Error: OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
    exit(1)

# Test simple summarization
transcript = """
Alice: Welcome to our meeting about the new product launch.
Bob: I think we should aim for a September release date.
Alice: That sounds good. Can you prepare the marketing materials?
Bob: Yes, I'll have them ready by next Friday.
Alice: Great, then we're all set.
"""

participants = ["Alice", "Bob"]

try:
    result = summarize_meeting(transcript, participants)
    print("Success! Here's the summary:")
    print(f"Summary: {result['meeting_summary']['summary']}")
    print("Key Points:")
    for point in result['meeting_summary']['key_points']:
        print(f"- {point}")
    print("Action Items:")
    for item in result['action_items']:
        print(f"- {item['action']} (Assigned to: {item['assignee']})")
except Exception as e:
    print(f"Error: {str(e)}")