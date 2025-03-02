import streamlit as st
import json
import traceback
import os
from lg import summarize_meeting

# Check for OpenAI API key
if not os.environ.get("OPENAI_API_KEY"):
    st.warning("⚠️ OpenAI API key not found! Set it with 'export OPENAI_API_KEY=your-key' before running this app.")


st.set_page_config(
    page_title="Meeting Summarizer & Action Item Generator",
    page_icon="📝",
    layout="wide"
)

st.title("Meeting Summarizer & Action Item Generator")
st.subheader("Convert your meeting transcripts into concise summaries and actionable tasks")

with st.form("meeting_form"):
    transcript = st.text_area(
        "Meeting Transcript",
        height=300,
        placeholder="Paste your meeting transcript here..."
    )
    
    participants_input = st.text_input(
        "Participants (comma-separated)",
        placeholder="Alice, Bob, Charlie, Dave, Eva"
    )
    
    submit_button = st.form_submit_button("Generate Summary & Action Items")

if submit_button and transcript and participants_input:
    participants = [p.strip() for p in participants_input.split(",")]
    
    with st.spinner("Processing your meeting transcript..."):
        # Call the meeting summarizer
        try:
            result = summarize_meeting(transcript, participants)
            
            # Display meeting summary
            st.success("✅ Summary generated successfully!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Meeting Summary")
                if "meeting_summary" in result and "summary" in result["meeting_summary"]:
                    st.write(result["meeting_summary"]["summary"])
                else:
                    st.warning("Summary content is missing or malformed")
                
                st.subheader("Key Points")
                if "meeting_summary" in result and "key_points" in result["meeting_summary"]:
                    for point in result["meeting_summary"]["key_points"]:
                        st.markdown(f"- {point}")
                else:
                    st.warning("Key points are missing or malformed")
                
                st.subheader("Decisions Made")
                if "meeting_summary" in result and "decisions" in result["meeting_summary"]:
                    for decision in result["meeting_summary"]["decisions"]:
                        st.markdown(f"- {decision}")
                else:
                    st.warning("Decisions are missing or malformed")
            
            with col2:
                st.subheader("Action Items")
                if "action_items" in result and result["action_items"]:
                    for item in result["action_items"]:
                        with st.expander(f"📌 {item.get('action', 'Unnamed Action')}"):
                            st.markdown(f"**Assignee:** {item.get('assignee', 'Unassigned')}")
                            st.markdown(f"**Due Date:** {item.get('due_date', 'Not specified')}")
                            
                            priority = item.get('priority', 'medium').lower()
                            if priority == "high":
                                priority_color = "🔴 High"
                            elif priority == "medium":
                                priority_color = "🟠 Medium"
                            else:
                                priority_color = "🟢 Low"
                                
                            st.markdown(f"**Priority:** {priority_color}")
                else:
                    st.info("No action items were identified in this meeting.")
            
            # Add download button for JSON export
            st.download_button(
                label="Download Summary & Action Items (JSON)",
                data=json.dumps(result, indent=2),
                file_name="meeting_summary.json",
                mime="application/json"
            )
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.expander("See detailed error trace").write(traceback.format_exc())
            st.info("💡 Tips to fix this error: Check that your OpenAI API key is valid and has sufficient credits. Make sure your meeting transcript is properly formatted with speaker names.")
else:
    st.info("Enter a meeting transcript and list of participants, then click 'Generate Summary & Action Items'.")

# Add sidebar with tips
with st.sidebar:
    st.header("Tips for Best Results")
    st.markdown("""
    - Include speaker names in the transcript (e.g., "Alice: Hello everyone")
    - Make sure all participants are listed in the participants field
    - For more accurate action items, make sure assignments and deadlines are clearly stated
    - Longer transcripts may take more time to process
    """)
    
    st.header("About")
    st.markdown("""
    This tool uses LangGraph and LLMs to analyze meeting transcripts and extract:
    
    1. A concise meeting summary
    2. Key points discussed
    3. Decisions made
    4. Action items with assignees and deadlines
    
    The tool helps teams track action items and ensure accountability.
    """)