import re
from typing import List, Optional, Dict, Any
import logging

# Configure logging
logger = logging.getLogger("text-service")

def extract_participants(transcript: str) -> List[str]:
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

def parse_timestamp(timestamp_str: str) -> Optional[float]:
    """
    Parse a timestamp string in HH:MM:SS format to seconds
    
    Args:
        timestamp_str: Timestamp string in HH:MM:SS format
        
    Returns:
        Seconds as float or None if invalid format
    """
    try:
        if not timestamp_str:
            return None
            
        # Remove brackets if present
        timestamp_str = timestamp_str.strip('[]')
        
        parts = timestamp_str.split(':')
        if len(parts) == 3:
            # HH:MM:SS format
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        elif len(parts) == 2:
            # MM:SS format
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        else:
            return float(timestamp_str)
    except:
        return None

def extract_timestamps_and_speakers(transcript: str) -> List[Dict[str, Any]]:
    """
    Extract timestamps and speaker information from a transcript
    
    Args:
        transcript: Text transcript with timestamps and speakers
        
    Returns:
        List of segments with speaker, text, and time information
    """
    segments = []
    
    # Pattern for lines with timestamps: [HH:MM:SS] Speaker X: Text
    pattern = r'\[([0-9:]+)\]\s*(\w+(?:\s+\w+)*)\s*:\s*(.+)'
    
    for line in transcript.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        match = re.match(pattern, line)
        if match:
            timestamp_str, speaker, text = match.groups()
            timestamp = parse_timestamp(timestamp_str)
            
            # If we have a valid timestamp and speaker
            if timestamp is not None:
                # Extract speaker number if this is "Speaker X"
                speaker_num = None
                if "Speaker" in speaker:
                    speaker_match = re.search(r'Speaker\s+(\d+)', speaker)
                    if speaker_match:
                        speaker_num = speaker_match.group(1)
                
                segments.append({
                    "speaker": speaker_num if speaker_num else speaker,
                    "text": text.strip(),
                    "start_time": timestamp,
                    # Estimate end time as start + 0.2s per character (rough approximation)
                    "end_time": timestamp + len(text) * 0.2,
                    "is_parsed": True
                })
        else:
            # Try alternate patterns or add as plain text
            segments.append({
                "speaker": "Unknown",
                "text": line,
                "start_time": 0 if not segments else segments[-1].get("end_time", 0),
                "end_time": 0 if not segments else segments[-1].get("end_time", 0) + len(line) * 0.2,
                "is_parsed": False
            })
    
    return segments