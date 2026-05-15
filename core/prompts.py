# Centralized prompt templates and instructions for the Meeting Summarizer

CONTEXT_INSTRUCTION = """MEETING CONTEXT AND BACKGROUND: {context}
Please use this information to better understand the discussion, identify key topics, and provide a more accurate analysis."""

ANALYZE_SYSTEM_PROMPT = """Analyze the meeting transcript and return JSON with this exact structure:
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
{language_instructions}"""

SUMMARIZE_SYSTEM_PROMPT = """Create a meeting summary with this JSON structure:
{{
  "summary": "2-3 sentence overview",
  "key_points": ["point1", "point2", "point3"],
  "decisions": ["decision1", "decision2"]
}}

Keep it concise and factual. {language_instructions}"""

EXTRACT_ACTIONS_SYSTEM_PROMPT = """Extract action items as JSON array:
[
  {{
    "action": "specific action",
    "assignee": "person name or Unassigned",
    "due_date": "date or Not specified",
    "priority": "high/medium/low"
  }}
]

Return empty array [] if no actions found. {language_instructions}"""
