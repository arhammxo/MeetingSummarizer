"""
Audio Converter using ffmpeg-python package

This approach provides a more direct and reliable way to convert audio files
using the ffmpeg-python package.
"""

import os
import tempfile
import logging
from pathlib import Path
import ffmpeg  # Make sure to install with: pip install ffmpeg-python

# Configure logging
logger = logging.getLogger("audio-converter")

def convert_audio_to_wav(input_path: str, output_path: str = None, sample_rate: int = 16000):
    """
    Converts an audio file to WAV format with specified sample rate.
    
    Args:
        input_path (str): The path of the input audio file.
        output_path (str): The path of the output wav file. If None, creates a temp file.
        sample_rate (int): The sample rate for the output WAV file (default: 16000 Hz).
        
    Returns:
        str: Path to the converted WAV file.
        
    Raises:
        ffmpeg.Error: If the conversion fails due to an ffmpeg error.
    """
    # Create a temporary output path if none provided
    if output_path is None:
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_path = temp_file.name
        temp_file.close()
    
    try:
        logger.info(f"Converting audio file {input_path} to WAV format at {sample_rate}Hz")
        
        # Convert to 16kHz mono WAV using ffmpeg-python
        (
            ffmpeg
            .input(input_path)
            .output(
                output_path,
                format="wav",       # Output format
                acodec="pcm_s16le", # 16-bit PCM
                ar=sample_rate,     # Sample rate
                ac=1                # Mono (1 channel)
            )
            .global_args('-loglevel', 'error')  # Reduce ffmpeg output
            .global_args('-y')                  # Overwrite output
            .run(capture_stdout=True, capture_stderr=True)
        )
        
        logger.info(f"Successfully converted to WAV: {output_path}")
        return output_path
        
    except ffmpeg.Error as e:
        error_message = f"ffmpeg error: {e.stderr.decode() if e.stderr else str(e)}"
        logger.error(error_message)
        
        # Clean up the output file if it was created
        if os.path.exists(output_path):
            try:
                os.unlink(output_path)
            except:
                pass
                
        raise RuntimeError(error_message)

def is_mp3_file(file_path: str) -> bool:
    """
    Check if a file is an MP3 file based on extension.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        bool: True if file has .mp3 extension
    """
    return Path(file_path).suffix.lower() == '.mp3'

def process_audio_file_for_diarization(file_path: str) -> str:
    """
    Prepare an audio file for diarization by converting to WAV if needed.
    
    Args:
        file_path (str): Path to the audio file
        
    Returns:
        str: Path to the processed file (original or converted)
    """
    if is_mp3_file(file_path):
        logger.info(f"Processing MP3 file for diarization: {file_path}")
        return convert_audio_to_wav(file_path)
    else:
        # For non-MP3 files, just return the original path
        return file_path