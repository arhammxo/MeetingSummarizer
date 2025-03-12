from datetime import datetime
import os
import tempfile
from typing import Optional, Dict, Any, List
import json
import logging

# Configure logging
logger = logging.getLogger("utils-service")

def format_time(seconds: float) -> str:
    """
    Format seconds into HH:MM:SS format
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    return datetime.utcfromtimestamp(seconds).strftime('%H:%M:%S')

def create_temp_dir() -> str:
    """
    Create a temporary directory for file storage
    
    Returns:
        Path to the temporary directory
    """
    temp_dir = tempfile.mkdtemp(prefix="meeting_summarizer_")
    return temp_dir

def save_temp_file(content: bytes, filename: Optional[str] = None, extension: str = ".tmp") -> str:
    """
    Save content to a temporary file
    
    Args:
        content: File content as bytes
        filename: Optional filename (without path)
        extension: File extension if filename not provided
        
    Returns:
        Path to the saved file
    """
    if filename:
        # Use the system temp directory and the provided filename
        filepath = os.path.join(tempfile.gettempdir(), filename)
    else:
        # Create a named temporary file with the specified extension
        fd, filepath = tempfile.mkstemp(suffix=extension)
        os.close(fd)  # Close the file descriptor
    
    # Write the content to the file
    with open(filepath, 'wb') as f:
        f.write(content)
    
    return filepath

def clean_temp_file(filepath: str) -> bool:
    """
    Remove a temporary file
    
    Args:
        filepath: Path to the file to remove
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if os.path.exists(filepath):
            os.unlink(filepath)
            return True
        return False
    except Exception as e:
        logger.error(f"Error removing temp file {filepath}: {str(e)}")
        return False

def serialize_result(result: Dict[str, Any]) -> str:
    """
    Serialize a result dictionary to JSON string
    
    Args:
        result: Result dictionary
        
    Returns:
        JSON string
    """
    return json.dumps(result, default=str, indent=2)

def deserialize_result(json_str: str) -> Dict[str, Any]:
    """
    Deserialize a JSON string to a result dictionary
    
    Args:
        json_str: JSON string
        
    Returns:
        Result dictionary
    """
    return json.loads(json_str)

def get_language_name(language_code: Optional[str]) -> str:
    """
    Get the full language name from a language code
    
    Args:
        language_code: Language code (e.g., 'en', 'hi')
        
    Returns:
        Full language name
    """
    language_map = {
        "en": "English",
        "hi": "Hindi",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "zh": "Chinese",
        "ja": "Japanese",
        "ru": "Russian",
        "ar": "Arabic",
        "auto": "Auto-detected",
        None: "Auto-detected"
    }
    
    return language_map.get(language_code, "Unknown")