import sounddevice as sd
import numpy as np 
import wave 
import time
import librosa
import tensorflow as tf
import threading
import os 
import matplotlib.pyplot as plt
import sys


class AudioRecorder:
    def __init__(self, filename="recordings/audio.wav", samplerate=22050, duration=10, stress_model="models/modelo_deteccion_estres.keras", stress_threshold=0.6, **kwargs):
        self.filename = filename
        self.samplerate = samplerate
        self.duration = duration
        self.recording = False
        self.frames = []
        self.model = None
        self.stress_threshold = stress_threshold  # Umbral ajustable
    
    # Verifica y crea directorios necesarios
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        os.makedirs("models", exist_ok=True)
        
        # Carga el modelo solo si existe
        if os.path.exists(stress_model):
            try:
                self.model = tf.keras.models.load_model(stress_model)
            except Exception as e:
                print(f"Error cargando el modelo: {e}")
                self.model = None
        else:
            print(f"Modelo no encontrado en {stress_model}. El análisis de estrés no estará disponible.")
        
    def start(self):
        """Inicia una nueva grabación en un hilo fresco"""
        if self.recording:
            self.stop()  # Detiene cualquier grabación existente
            
        self.recording = True
        # Creamos un nuevo hilo cada vez
        self.recording_thread = threading.Thread(target=self._record, daemon=True)
        self.recording_thread.start()      

    def _record(self):  # Método para grabar el audio
        self.frames = []  # Reinicia la lista de frames antes de grabar
        start_time = time.time()

        def callback(indata, frames_count, time_info, status):
            """Función de callback para grabar audio en tiempo real"""
            if self.recording:
                self.frames.append(indata.copy())

        with sd.InputStream(samplerate=self.samplerate, channels=1, callback=callback, dtype='float32'):
            while self.recording and (time.time() - start_time) < self.duration:
                sd.sleep(100)  # Hilo de espera para mantener la grabación

        self._save_audio()
        self._analyze_audio()
        
    def stop(self):
        """Detiene la grabación actual"""
        self.recording = False
        if hasattr(self, 'recording_thread') and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=1)  # Espera máxima de 1 segundo
        
    def _save_audio(self):
        audio = np.concatenate(self.frames, axis=0)
        audio_int16 = (audio * 32767).astype(np.int16)

        with wave.open(self.filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.samplerate)
            wf.writeframes(audio_int16.tobytes())
        
        print(f"Audio guardado en {self.filename}")
        
    def _analyze_audio(self):
        if self.model is None:
            print("Modelo no disponible para análisis")
            return
        
        try:
            audio, sr = librosa.load(self.filename, sr=self.samplerate)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            features = np.mean(mfccs.T, axis=0).reshape(1, -1)
            
            prediction = self.model.predict(features, verbose=0)
            stress_level = prediction[0][0]
            
            print(f"\nNivel de estrés detectado: {stress_level:.2f}")

            # Visualización opcional (comentar en producción)
            self._plot_mfccs(mfccs, sr, stress_level)

            if stress_level > self.stress_threshold:
                print(f"⚠️ ¡ALERTA! Estrés vocal detectado (Nivel: {stress_level:.2f})")
                self._on_stress_detected()  # Método para acciones adicionales
            else:
                print(f"✅ Voz normal (Nivel: {stress_level:.2f})")
                
        except Exception as e:
            print(f"Error en análisis: {str(e)}")
            import traceback
            traceback.print_exc()

    def _plot_mfccs(self, mfccs, sr, stress_level):
        """Versión segura para hilos"""
        try:
            plt.switch_backend('Agg')  # Usa backend no interactivo
            fig = plt.figure(figsize=(10, 4))
            librosa.display.specshow(mfccs, sr=sr, x_axis='time')
            plt.colorbar()
            plt.title(f'MFCCs - Nivel: {stress_level:.2f}')
            plt.tight_layout()
            plt.savefig('mfcc_plot.png')  # Guarda en archivo
            plt.close(fig)
            print("Gráfico guardado en mfcc_plot.png")
        except Exception as e:
            print(f"Error al generar gráfico: {e}")

    def _on_stress_detected(self):
        """Método para acciones al detectar estrés"""
        # Ejemplo: enviar notificación, activar alarma, etc.
        pass