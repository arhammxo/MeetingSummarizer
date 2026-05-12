import queue
import threading
import tempfile
import time
import os
import wave
import numpy as np
import logging

try:
    import sounddevice as sd
except ImportError:
    sd = None

import whisper

logger = logging.getLogger("realtime-processor")

class RealTimeTranscriber:
    def __init__(self, sample_rate=16000, chunk_duration=3, model_size="base"):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.model_size = model_size
        
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.transcript_segments = []
        self.full_audio_data = []  # To store the entire recording
        
        self.model = None
        self.record_thread = None
        self.transcribe_thread = None

    def start(self):
        if sd is None:
            raise RuntimeError("sounddevice is not installed or available.")
            
        if self.model is None:
            logger.info(f"Loading Whisper '{self.model_size}' model for real-time transcription...")
            self.model = whisper.load_model(self.model_size)
            
        self.is_recording = True
        self.transcript_segments = []
        self.full_audio_data = []
        
        # Clear queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
                
        self.record_thread = threading.Thread(target=self._record_loop, daemon=True)
        self.transcribe_thread = threading.Thread(target=self._transcribe_loop, daemon=True)
        
        self.record_thread.start()
        self.transcribe_thread.start()
        logger.info("Real-time transcription started.")

    def stop(self):
        self.is_recording = False
        if self.record_thread:
            self.record_thread.join(timeout=2)
        if self.transcribe_thread:
            self.transcribe_thread.join(timeout=2)
        logger.info("Real-time transcription stopped.")

    def get_transcript(self):
        return " ".join(self.transcript_segments)
        
    def _record_loop(self):
        try:
            def callback(indata, frames, time_info, status):
                if status:
                    logger.warning(f"Audio status: {status}")
                if self.is_recording:
                    data = indata.copy()
                    self.audio_queue.put(data)
                    self.full_audio_data.append(data)
            
            with sd.InputStream(samplerate=self.sample_rate, channels=1, callback=callback, blocksize=self.chunk_samples):
                while self.is_recording:
                    time.sleep(0.1)
        except Exception as e:
            logger.error(f"Error in record loop: {e}")
            self.is_recording = False

    def _transcribe_loop(self):
        # We need torch to check if CUDA is available for the FP16 warning
        import torch
        fp16_supported = torch.cuda.is_available()
        
        while self.is_recording or not self.audio_queue.empty():
            try:
                audio_data = self.audio_queue.get(timeout=1)
                
                # Save chunk to temp file for Whisper
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    wav_path = tmp.name
                    with wave.open(wav_path, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(self.sample_rate)
                        wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())
                
                try:
                    result = self.model.transcribe(wav_path, fp16=fp16_supported)
                    text = result['text'].strip()
                    if text:
                        self.transcript_segments.append(text)
                except Exception as e:
                    logger.error(f"Whisper transcription error: {e}")
                finally:
                    try:
                        os.unlink(wav_path)
                    except:
                        pass
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in transcribe loop: {e}")

    def save_full_audio(self, filepath):
        if not self.full_audio_data:
            logger.warning("No audio data to save.")
            return False
            
        try:
            logger.info(f"Saving full audio to {filepath}...")
            # Concatenate all chunks
            full_audio = np.concatenate(self.full_audio_data, axis=0)
            
            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes((full_audio * 32767).astype(np.int16).tobytes())
            return True
        except Exception as e:
            logger.error(f"Error saving full audio: {e}")
            return False
