import streamlit as st
import json
import traceback
import os
import io
import re
import tempfile
import time
from core.lg import summarize_meeting
from core.audio_processor import process_audio_file
from core.speaker_summarizer import generate_speaker_summaries
from core.long_recording_processor import process_long_audio
from core.summarize_long_transcripts import summarize_long_meeting

# Check for OpenAI API key
#if not os.environ.get("OPENAI_API_KEY"):
 #   st.warning("⚠️ OpenAI API key not found! Set it with 'export OPENAI_API_KEY=your-key' before running this app.")

st.set_page_config(
    page_title="On-Premise Meeting Summarizer",
    page_icon="📝",
    layout="wide"
)

st.title("On-Premise Meeting Summarizer")
st.subheader("Convert your meeting recordings or transcripts into concise summaries and actionable tasks")

def extract_participants(transcript):
    """
    Extract participant names from the meeting transcript.
    Supports multiple languages by looking for patterns rather than specific words.

    Args:
        transcript (str): The meeting transcript

    Returns:
        list: Unique participant names
    """
    participants = set()

    # Pattern 1: Name (Role): Text - works for English and transliterated Hindi
    pattern1 = r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s*\([^)]+\):'
    matches1 = re.findall(pattern1, transcript)
    for name in matches1:
        participants.add(name.strip())

    # Pattern 2: Name: Text - works for English and transliterated Hindi
    pattern2 = r'(?:^|\n)([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s*:'
    matches2 = re.findall(pattern2, transcript)
    for name in matches2:
        participants.add(name.strip())

    # Pattern 3: Speaker X (from audio transcription) - language independent
    pattern3 = r'Speaker[\s_](\d+)'
    matches3 = re.findall(pattern3, transcript)
    for speaker_num in matches3:
        participants.add(f"Speaker {speaker_num}")

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
if 'audio_transcript' not in st.session_state:
    st.session_state.audio_transcript = None
if 'audio_processing_complete' not in st.session_state:
    st.session_state.audio_processing_complete = False
if 'speaker_summaries' not in st.session_state:
    st.session_state.speaker_summaries = None
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Meeting Summary"
if 'meeting_result' not in st.session_state:
    st.session_state.meeting_result = None
if 'detected_language' not in st.session_state:
    st.session_state.detected_language = None
if 'processing_progress' not in st.session_state:
    st.session_state.processing_progress = 0
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = ""
if 'is_long_recording' not in st.session_state:
    st.session_state.is_long_recording = False

# Function to update processing progress
def update_progress(progress, status):
    st.session_state.processing_progress = progress
    st.session_state.processing_status = status

# Function to display meeting summary tab
def display_meeting_summary():
    result = st.session_state.meeting_result
    if not result:
        st.info("No meeting summary available yet.")
        return

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

        # Display metadata if it exists
        if "metadata" in result:
            with st.expander("Meeting Metadata", expanded=False):
                st.write(f"**Total Duration:** {result['metadata'].get('total_duration_minutes', 'Unknown')} minutes")
                st.write(f"**Language:** {result['metadata'].get('language', 'Unknown')}")
                if 'chunks_analyzed' in result['metadata']:
                    st.write(f"**Chunks Analyzed:** {result['metadata']['chunks_analyzed']}")

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

# Function to display speaker summaries tab
def display_speaker_summaries():
    speaker_summaries = st.session_state.speaker_summaries
    if not speaker_summaries:
        st.info("No speaker summaries available yet.")
        return

    st.subheader("Speaker Contributions")

    for speaker, summary in speaker_summaries.items():
        with st.expander(f"🗣️ {speaker}", expanded=False):
            st.markdown(f"**Summary:** {summary.get('brief_summary', 'No summary available')}")

            st.markdown("**Key Contributions:**")
            for contribution in summary.get('key_contributions', []):
                st.markdown(f"- {contribution}")

            if summary.get('action_items'):
                st.markdown("**Action Items:**")
                for item in summary.get('action_items', []):
                    st.markdown(f"- {item}")

            if summary.get('questions_raised'):
                st.markdown("**Questions Raised:**")
                for question in summary.get('questions_raised', []):
                    st.markdown(f"- {question}")

# Add tabs for different input methods
input_method = st.radio(
    "Choose input method:",
    ["Upload Audio", "Paste Text", "Upload Text"],
    horizontal=True
)

# Handle file upload outside the form
file_content = None

if input_method == "Upload Audio":
    st.info("Upload an audio recording of your meeting. The system will transcribe it and identify speakers automatically.")

    # Language selection
    language_options = {
        "Auto-detect": None,
        "English": "en",
        "Hindi": "hi",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Chinese": "zh",
        "Japanese": "ja",
        "Russian": "ru",
        "Arabic": "ar"
    }

    col1, col2 = st.columns(2)

    with col1:
        selected_language = st.selectbox(
            "Select Audio Language",
            options=list(language_options.keys()),
            index=0,
            help="Select the primary language of your audio. Auto-detect works well for many languages but explicit selection may improve accuracy."
        )

    with col2:
        # Option for long recordings
        is_long_recording = st.checkbox(
            "This is a long recording (>15 minutes)",
            value=st.session_state.is_long_recording,
            help="Enable special processing for longer recordings. Recommended for meetings over 15 minutes."
        )
        st.session_state.is_long_recording = is_long_recording

    language_code = language_options[selected_language]

    audio_file = st.file_uploader(
        "Upload Meeting Recording",
        type=["wav", "mp3", "m4a"],
        help="Upload an audio file of your meeting"
    )

    if audio_file is not None and not st.session_state.audio_processing_complete:
        st.info("Audio file detected. Click 'Process Audio' to transcribe and identify speakers.")

        if st.button("Process Audio"):
            # Initialize progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("Starting audio processing...")

            # Reset processing progress
            st.session_state.processing_progress = 0
            st.session_state.processing_status = "Starting audio processing..."
            st.session_state.audio_processing_complete = False

            # Update the UI with progress
            def update_ui_progress():
                progress_bar.progress(st.session_state.processing_progress / 100)
                status_text.text(st.session_state.processing_status)

            try:
                # Save the uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(audio_file.getvalue())
                    audio_path = tmp_file.name

                update_ui_progress()

                if is_long_recording:
                    # Process long audio with chunking
                    transcript_data = process_long_audio(
                        audio_path,
                        language=language_code, 
                        progress_callback=update_progress
                    )
                else:
                    # Process regular audio
                    with st.spinner(f"Processing {selected_language if language_code else 'auto-detected'} audio..."):
                        transcript_data = process_audio_file(audio_path, language=language_code)
                        # Simulate progress for regular processing
                        for i in range(10):
                            update_progress(i * 10, f"Processing audio... {i * 10}%")
                            update_ui_progress()
                            time.sleep(0.2)  # Just for UI feedback

                # Store the detected language
                st.session_state.detected_language = transcript_data.get('language', 'auto-detected')

                # Format the transcript for display and processing
                formatted_transcript = []
                for segment in transcript_data["transcript"]:
                    formatted_transcript.append(f"[{segment['start_time_formatted']}] Speaker {segment['speaker']}: {segment['text']}")

                # Join the formatted transcript lines
                full_transcript = "\n".join(formatted_transcript)

                # Store in session state
                st.session_state.transcript_content = full_transcript
                st.session_state.audio_transcript = transcript_data
                st.session_state.audio_processing_complete = True

                # Clean up the temporary file
                os.unlink(audio_path)

                # Update progress to 100%
                update_progress(100, "Audio processing complete!")
                update_ui_progress()

                # Display success message with language info
                if st.session_state.detected_language:
                    st.success(f"Audio processing complete! Detected language: {st.session_state.detected_language}. The transcript is ready for summarization.")
                else:
                    st.success("Audio processing complete! The transcript is ready for summarization.")

            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")
                st.error(traceback.format_exc())

            # Remove progress indicators after completion
            time.sleep(1)  # Give user time to see 100%
            progress_bar.empty()
            status_text.empty()

    # If audio has been processed, show a preview
    if st.session_state.audio_processing_complete:
        with st.expander("Preview Transcript", expanded=False):
            st.text(st.session_state.transcript_content[:1000] + ("..." if len(st.session_state.transcript_content) > 1000 else ""))

elif input_method == "Upload Text":
    text_file = st.file_uploader(
        "Upload Meeting Transcript",
        type=["txt"],
        help="Upload a text file containing your meeting transcript"
    )

    if text_file is not None:
        file_content = text_file.getvalue().decode("utf-8")
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
    elif input_method == "Upload Text":
        if file_content:
            st.success(f"Transcript uploaded! ({len(file_content)} characters)")
        else:
            st.info("Please upload a transcript file above")
    elif input_method == "Upload Audio":
        if st.session_state.audio_processing_complete:
            st.success(f"Audio processed successfully! {'Language: ' + st.session_state.detected_language if st.session_state.detected_language else ''}")
        else:
            st.info("Please process an audio file above")

    # Determine default participants
    default_participants = ""
    if input_method == "Upload Text" and file_content:
        default_participants = ", ".join(extract_participants(file_content))
    elif input_method == "Upload Audio" and st.session_state.audio_processing_complete:
        speakers = set()
        for segment in st.session_state.audio_transcript["transcript"]:
            speakers.add(f"Speaker {segment['speaker']}")
        default_participants = ", ".join(sorted(list(speakers)))

    # Participants input field
    participants_input = st.text_input(
        "Participants (comma-separated)",
        value=default_participants,
        placeholder="Alice, Bob, Charlie, Dave, Eva",
        help="Participants are automatically detected. You can edit this list if needed."
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

    elif input_method == "Upload Text" and file_content:
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

    elif input_method == "Upload Audio" and st.session_state.audio_processing_complete:
        process_transcript = True
        final_transcript = st.session_state.transcript_content

        # Use provided participants or the detected speakers
        if participants_input:
            participants = [p.strip() for p in participants_input.split(",")]
        else:
            speakers = set()
            for segment in st.session_state.audio_transcript["transcript"]:
                speakers.add(f"Speaker {segment['speaker']}")
            participants = sorted(list(speakers))
            st.success(f"✅ Using detected speakers: {', '.join(participants)}")
    else:
        st.warning("Please provide a meeting transcript (paste text, upload a file, or process audio).")

    # Process if we have both transcript and participants
    if process_transcript and final_transcript and participants:
        # Set up progress tracking for summarization
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Starting summarization...")

        # Reset processing progress
        st.session_state.processing_progress = 0
        st.session_state.processing_status = "Starting summarization..."

        # Update the UI with progress
        def update_ui_progress():
            progress_bar.progress(st.session_state.processing_progress / 100)
            status_text.text(st.session_state.processing_status)

        try:
            # For long recordings, use the hierarchical summarization
            if st.session_state.is_long_recording and input_method == "Upload Audio":
                update_status = lambda p, s: update_progress(p, s)

                # Call the long meeting summarizer with transcript data
                result = summarize_long_meeting(
                    st.session_state.audio_transcript["transcript"],
                    language=st.session_state.detected_language,
                    progress_callback=update_status
                )

                # Update UI after each step
                update_ui_progress()
            else:
                # For regular recordings, use the standard summarizer
                # Call the meeting summarizer with language information
                if input_method == "Upload Audio" and st.session_state.detected_language:
                    # Simulate progress updates for standard processing
                    for i in range(5):
                        update_progress(i * 20, f"Summarizing... {i * 20}%")
                        update_ui_progress()
                        time.sleep(0.3)  # Just for UI feedback

                    result = summarize_meeting(final_transcript, participants,
                                              language=st.session_state.detected_language)
                else:
                    # Simulate progress updates for standard processing
                    for i in range(5):
                        update_progress(i * 20, f"Summarizing... {i * 20}%")
                        update_ui_progress()
                        time.sleep(0.3)  # Just for UI feedback

                    result = summarize_meeting(final_transcript, participants)

            st.session_state.meeting_result = result

            # Update progress to 100%
            update_progress(100, "Summary generated successfully!")
            update_ui_progress()

            # Display success message
            st.success("✅ Summary generated successfully!")

            # Generate speaker summaries
            with st.spinner("Generating speaker-specific summaries..."):
                try:
                    # Generate speaker summaries with language support
                    if input_method == "Upload Audio" and st.session_state.detected_language:
                        speaker_summaries = generate_speaker_summaries(
                            final_transcript,
                            participants,
                            language=st.session_state.detected_language
                        )
                    else:
                        speaker_summaries = generate_speaker_summaries(final_transcript, participants)

                    st.session_state.speaker_summaries = speaker_summaries
                except Exception as e:
                    st.error(f"Error generating speaker summaries: {str(e)}")

            # Add download buttons for JSON export and transcript
            st.download_button(
                label="Download Summary & Action Items (JSON)",
                data=json.dumps({
                    "meeting_summary": result["meeting_summary"],
                    "action_items": result["action_items"],
                    "speaker_summaries": st.session_state.speaker_summaries if st.session_state.speaker_summaries else {},
                    "language": st.session_state.detected_language,
                    "metadata": {
                        "language": st.session_state.detected_language,
                        "input_method": input_method,
                        "participant_count": len(participants),
                        "is_long_recording": st.session_state.is_long_recording
                    }
                }, indent=2),
                file_name="meeting_summary.json",
                mime="application/json"
            )

            # If this was from audio, offer the transcript download as well
            if input_method == "Upload Audio":
                st.download_button(
                    label="Download Transcript (TXT)",
                    data=final_transcript,
                    file_name="meeting_transcript.txt",
                    mime="text/plain"
                )

            # Reset to default tab
            st.session_state.current_tab = "Meeting Summary"

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.expander("See detailed error trace").write(traceback.format_exc())
            st.info("💡 Tips to fix this error: Check that your OpenAI API key is valid and has sufficient credits. Make sure your meeting transcript is properly formatted with speaker names.")

        # Remove progress indicators after completion
        time.sleep(1)  # Give user time to see 100%
        progress_bar.empty()
        status_text.empty()

    elif submit_button and (not final_transcript or not participants):
        st.warning("Both transcript and participants are required to generate a summary.")

# If we have results to display, show the tabs and content
if st.session_state.meeting_result:
    # Create radio buttons for tab selection instead of tabs component
    tab_options = ["Meeting Summary", "Speaker Summaries"]
    selected_tab = st.radio("View", tab_options, index=tab_options.index(st.session_state.current_tab))
    st.session_state.current_tab = selected_tab

    # Display the selected tab content
    if selected_tab == "Meeting Summary":
        display_meeting_summary()
    else:  # Speaker Summaries
        display_speaker_summaries()

# Add sidebar with tips
with st.sidebar:
    st.header("Tips for Best Results")
    st.markdown("""
    - When using audio, ensure clear recording with minimal background noise
    - For Hindi or other non-English audio, select the language for better accuracy
    - For text transcripts, include speaker names (e.g., "Alice: Hello everyone")
    - The app will automatically detect participants from the transcript or audio
    - For more accurate action items, make sure assignments and deadlines are clearly stated
    - For longer meetings (>15 minutes), check the "This is a long recording" option
    """)

    # Add information about long recordings
    st.header("Processing Long Recordings")
    st.markdown("""
    For meetings longer than 15 minutes, we recommend:

    1. Checking the "This is a long recording" option
    2. Being patient during processing (it may take several minutes)
    3. Ensuring your computer doesn't go to sleep during processing

    Long recordings are processed in chunks to handle meetings of any duration.
    """)

    # Add information about file uploads and participant detection
    st.header("Supported File Types")
    st.markdown("""
    - Audio: WAV, MP3, M4A
    - Text: TXT
    """)

    st.header("Supported Languages")
    st.markdown("""
    Audio transcription supports multiple languages including:
    - English
    - Hindi
    - Spanish
    - French
    - German
    - Chinese
    - And many more...

    For best results with non-English audio, select the specific language rather than using auto-detect.
    """)

    st.header("Automatic Participant Detection")
    st.markdown("""
    The app recognizes participants based on these patterns:
    - Name (Role): Text
    - Name: Text at the beginning of lines
    - Speaker labels from audio processing

    You can always review and edit the detected participants if needed.
    """)

    st.header("About")
    st.markdown("""
    This tool uses LangGraph, LLMs, and audio processing to:

    1. Transcribe meeting recordings with speaker identification
    2. Generate a concise meeting summary
    3. Extract key points discussed
    4. Identify decisions made
    5. Compile action items with assignees and deadlines
    6. Create per-speaker contribution summaries

    The tool helps teams track action items and ensure accountability.
    """)