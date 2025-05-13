import sounddevice as sd
import numpy as np 
import wave 
import time
import threading
import os 

class AudioRecorder:
    def __init__(self, filename="recordings/audio.wav", samplerate=22050, duration=300):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.filename = filename
        self.samplerate = samplerate
        self.duration = duration
        self.recording = False
        self.frames = []
        
    def start(self):
        if self.recording:
            self.stop()
            
        self.recording = True
        self.recording_thread = threading.Thread(target=self._record, daemon=True)
        self.recording_thread.start()
        print("🎤 Micrófono iniciado")

    def _record(self):
        self.frames = []
        start_time = time.time()

        def callback(indata, frames_count, time_info, status):
            if self.recording:
                self.frames.append(indata.copy())

        with sd.InputStream(samplerate=self.samplerate, channels=1, callback=callback, dtype='float32'):
            while self.recording and (time.time() - start_time) < self.duration:
                sd.sleep(100)

        self._save_audio()
        
    def stop(self):
        self.recording = False
        if hasattr(self, 'recording_thread') and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=1)
        print("🎤 Grabación de audio finalizada")
        
    def _save_audio(self):
        if len(self.frames) > 0:
            audio = np.concatenate(self.frames, axis=0)
            audio_int16 = (audio * 32767).astype(np.int16)

            with wave.open(self.filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.samplerate)
                wf.writeframes(audio_int16.tobytes())