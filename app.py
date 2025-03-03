import streamlit as st
import json
import traceback
import os
import io
import re
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

def extract_participants(transcript):
    """
    Extract participant names from the meeting transcript.
    
    Looks for common patterns in meeting transcripts:
    1. "Name (Role):" pattern
    2. "Name:" at the beginning of lines
    3. Names followed by speaking indicators
    
    Args:
        transcript (str): The meeting transcript
        
    Returns:
        list: Unique participant names
    """
    participants = set()
    
    # Pattern 1: Name (Role): Text
    pattern1 = r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s*\([^)]+\):'
    matches1 = re.findall(pattern1, transcript)
    for name in matches1:
        participants.add(name.strip())
    
    # Pattern 2: Name: Text
    pattern2 = r'(?:^|\n)([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s*:'
    matches2 = re.findall(pattern2, transcript)
    for name in matches2:
        participants.add(name.strip())
    
    # If we found participants, return them
    if participants:
        return sorted(list(participants))
    
    # If no participants found, try a more general approach for names
    # This is a fallback method that might catch more names but could include false positives
    words = re.findall(r'\b([A-Z][a-z]+)\b', transcript)
    potential_names = set()
    
    for word in words:
        # Skip common non-name words that start with capital letters
        common_words = {"I", "We", "The", "This", "They", "Monday", "Tuesday", "Wednesday", 
                       "Thursday", "Friday", "Saturday", "Sunday", "January", "February", 
                       "March", "April", "May", "June", "July", "August", "September",
                       "October", "November", "December", "Hello", "Hi", "Thanks", "Yes",
                       "No", "Ok", "Okay", "Perfect", "Great", "Good", "Today", "Tomorrow"}
        if word not in common_words and len(word) > 1:
            potential_names.add(word)
    
    # Only use this method if the others failed and we found potential names
    if not participants and potential_names:
        return sorted(list(potential_names))
    
    return sorted(list(participants))

# Initialize session state for transcript and file
if 'transcript_content' not in st.session_state:
    st.session_state.transcript_content = ""

# Add tabs for different input methods
input_method = st.radio(
    "Choose input method:",
    ["Paste Text", "Upload File"],
    horizontal=True
)

# Handle file upload outside the form
file_content = None
if input_method == "Upload File":
    file_upload = st.file_uploader(
        "Upload Meeting Transcript",
        type=["txt"],
        help="Upload a text file containing your meeting transcript"
    )
    
    if file_upload is not None:
        file_content = file_upload.getvalue().decode("utf-8")
        st.session_state.transcript_content = file_content
        
        # Preview of uploaded file
        with st.expander("Preview Uploaded Transcript", expanded=False):
            st.text(file_content[:1000] + ("..." if len(file_content) > 1000 else ""))
        
        # Detect participants from file
        detected_participants = extract_participants(file_content)
        if detected_participants:
            st.success(f"✅ Detected {len(detected_participants)} participants: {', '.join(detected_participants)}")
        
with st.form("meeting_form"):
    # Show different input methods based on selection
    transcript = ""
    if input_method == "Paste Text":
        transcript = st.text_area(
            "Meeting Transcript",
            height=300,
            placeholder="Paste your meeting transcript here..."
        )
    else:  # Upload File
        if file_content:
            st.success(f"Transcript uploaded! ({len(file_content)} characters)")
        else:
            st.info("Please upload a transcript file above")
    
    # Participants input field
    participants_input = st.text_input(
        "Participants (comma-separated)",
        value=", ".join(extract_participants(file_content)) if input_method == "Upload File" and file_content else "",
        placeholder="Alice, Bob, Charlie, Dave, Eva",
        help="Participants are automatically detected from uploaded files. For pasted text, participants will be detected after submission."
    )
    
    submit_button = st.form_submit_button("Generate Summary & Action Items")

# Process the transcript
if submit_button:
    process_transcript = False
    final_transcript = ""
    participants = []

    # Get the transcript based on input method
    if input_method == "Paste Text" and transcript:
        process_transcript = True
        final_transcript = transcript
        
        # Auto-detect participants from pasted text if not provided
        if not participants_input:
            detected_participants = extract_participants(transcript)
            if detected_participants:
                participants = detected_participants
                st.success(f"✅ Automatically detected participants: {', '.join(participants)}")
            else:
                st.warning("Could not detect participants. Please enter them manually.")
                process_transcript = False
        else:
            participants = [p.strip() for p in participants_input.split(",")]
            
    elif input_method == "Upload File" and file_content:
        process_transcript = True
        final_transcript = file_content
        
        # Use provided participants or auto-detect from file
        if participants_input:
            participants = [p.strip() for p in participants_input.split(",")]
        else:
            detected_participants = extract_participants(file_content)
            if detected_participants:
                participants = detected_participants
                st.success(f"✅ Using detected participants: {', '.join(participants)}")
            else:
                st.warning("Could not detect participants. Please enter them manually.")
                process_transcript = False
    else:
        st.warning("Please provide a meeting transcript (either paste text or upload a file).")

    # Process if we have both transcript and participants
    if process_transcript and final_transcript and participants:
        with st.spinner("Processing your meeting transcript..."):
            # Call the meeting summarizer
            try:
                result = summarize_meeting(final_transcript, participants)
                
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
    elif submit_button and (not final_transcript or not participants):
        st.warning("Both transcript and participants are required to generate a summary.")

# Add sidebar with tips
with st.sidebar:
    st.header("Tips for Best Results")
    st.markdown("""
    - Include speaker names in the transcript (e.g., "Alice: Hello everyone")
    - The app will automatically detect participants from the transcript
    - For more accurate action items, make sure assignments and deadlines are clearly stated
    - Longer transcripts may take more time to process
    """)
    
    # Add information about file uploads and participant detection
    st.header("Supported File Types")
    st.markdown("""
    Currently, the app supports:
    - Plain text files (.txt)
    """)
    
    st.header("Automatic Participant Detection")
    st.markdown("""
    The app recognizes participants based on these patterns:
    - Name (Role): Text
    - Name: Text at the beginning of lines
    
    You can always review and edit the detected participants if needed.
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