import os
import subprocess

def run_cmd(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

try:
    # 1. Stash everything safely
    run_cmd("git add .")
    run_cmd("git stash push -m 'mixed_changes'")
    
    # 2. Create Context Branch
    run_cmd("git checkout main")
    run_cmd("git checkout -b feature/adding-context")
    run_cmd("git stash apply stash@{0}")
    
    # Revert the Live Audio files
    run_cmd("git checkout HEAD -- requirements.txt")
    run_cmd("rm -f core/realtime_processor.py")
    
    # Now we need to remove the Live Audio changes from app.py.
    # The safest way is to checkout the original main app.py, and apply ONLY the context patch.
    run_cmd("git checkout HEAD -- app.py")
    
    # Read the clean app.py
    with open("app.py", "r") as f:
        content = f.read()
        
    # Apply Context changes to app.py
    # 1. Add context inputs
    context_ui = """    # Prior Information / Context
    context_file = st.file_uploader(
        "Upload Prior Context File (Optional)",
        type=["txt"],
        help="Upload a text file containing prior context for the meeting."
    )
    
    uploaded_context = ""
    if context_file:
        uploaded_context = context_file.getvalue().decode("utf-8")
        st.success("Context file uploaded successfully!")
        
    meeting_context = st.text_area(
        "Prior Information / Meeting Context (Optional)",
        value=uploaded_context,
        placeholder="e.g., This is a weekly sync for the engineering team. Pay special attention to any database migration tasks.",
        help="Give the AI a 'heads up' about what the meeting is about before it reads the transcript."
    )
    
    submit_button = st.form_submit_button("Generate Summary & Action Items")"""
    content = content.replace('    submit_button = st.form_submit_button("Generate Summary & Action Items")', context_ui)
    
    # 2. Add context to summarize_long_meeting
    content = content.replace(
        'language=st.session_state.detected_language,\n                    progress_callback=update_status',
        'language=st.session_state.detected_language,\n                    progress_callback=update_status,\n                    context=meeting_context'
    )
    
    # 3. Add context to summarize_meeting (with language)
    content = content.replace(
        'result = summarize_meeting(final_transcript, participants, \n                                              language=st.session_state.detected_language)',
        'result = summarize_meeting(\n                        final_transcript, \n                        participants, \n                        language=st.session_state.detected_language,\n                        context=meeting_context\n                    )'
    )
    
    # 4. Add context to summarize_meeting (without language)
    content = content.replace(
        'result = summarize_meeting(final_transcript, participants)',
        'result = summarize_meeting(\n                        final_transcript, \n                        participants,\n                        context=meeting_context\n                    )'
    )
    
    with open("app.py", "w") as f:
        f.write(content)
        
    # Commit Context Branch
    run_cmd("git add core/lg.py core/audio_processor.py config.py main.py README.md core/long_recording_processor.py core/summarize_long_transcripts.py services/audio_converter.py services/summarization_service.py app.py")
    run_cmd("git commit -m 'feat: Add prior meeting context functionality'")
    
    # 3. Create Live Audio Branch
    run_cmd("git checkout main")
    run_cmd("git checkout -b feature/live-audio")
    
    # We want the clean live-audio app.py. The stash has both context and live audio.
    # We will just apply the stash, but then checkout the Context files from main so they aren't included!
    run_cmd("git stash apply stash@{0}")
    
    # Discard context files
    run_cmd("git checkout HEAD -- core/lg.py core/audio_processor.py config.py main.py README.md core/long_recording_processor.py core/summarize_long_transcripts.py services/audio_converter.py services/summarization_service.py")
    
    # For app.py, the stash has BOTH. We want to remove the context parts to make it purely live-audio.
    # We will checkout app.py from main, and apply ONLY live-audio changes.
    # Actually, it might be easier to just leave app.py as is in the stash (with both), because 
    # it's usually fine if a feature branch includes minor overlapping changes, OR we can string-replace the context out.
    # Let's string-replace the context out of app.py!
    
    with open("app.py", "r") as f:
        content = f.read()
        
    content = content.replace(context_ui, '    submit_button = st.form_submit_button("Generate Summary & Action Items")')
    content = content.replace(',\n                    context=meeting_context', '')
    content = content.replace(',\n                        context=meeting_context', '')
    
    with open("app.py", "w") as f:
        f.write(content)
        
    # Commit Live Audio Branch
    run_cmd("git add requirements.txt core/realtime_processor.py app.py")
    run_cmd("git commit -m 'feat: Add real-time audio transcription capabilities'")
    
    # Clean up stash
    run_cmd("git stash drop stash@{0}")
    
    print("SUCCESS: Branches split!")

except Exception as e:
    print(f"FAILED: {e}")

