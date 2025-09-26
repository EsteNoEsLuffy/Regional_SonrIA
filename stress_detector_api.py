import os
import numpy as np
import librosa
import tensorflow as tf
import tempfile

# Silenciar logs de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class StressDetectorAPI:
    """Clase para detección de estrés vocal"""

    def __init__(self, model_path: str = "models/modelo_deteccion_estres.keras"):
        self.model_path = model_path
        self.sample_rate = 22050
        self.mfcc_features = 40

        # Umbrales para clasificación
        self.stress_thresholds = {
            'bajo': 0.3,
            'medio': 0.6,
            'alto': 0.8
        }

        # Cargar modelo
        self.model = None
        self._load_model()

    def _load_model(self) -> bool:
        """Carga el modelo"""
        try:
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                return True
            return False
        except:
            return False

    def extract_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extrae características MFCC del audio"""
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.mfcc_features)
        return np.mean(mfccs.T, axis=0)

    def _classify_stress_level(self, stress_score: float) -> str:
        """Clasifica el nivel de estrés"""
        if stress_score < self.stress_thresholds['bajo']:
            return "BAJO"
        elif stress_score < self.stress_thresholds['medio']:
            return "MEDIO"
        elif stress_score < self.stress_thresholds['alto']:
            return "ALTO"
        else:
            return "MUY_ALTO"

    def analyze_audio_bytes(self, audio_bytes: bytes, filename: str = "audio.wav") -> dict:
        """Analiza bytes de audio"""
        if self.model is None:
            return {"success": False, "error": "Modelo no disponible"}

        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(filename)[1], delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_file_path = temp_file.name

        try:
            # Cargar audio
            audio, sr = librosa.load(temp_file_path, sr=self.sample_rate)

            # Verificar duración
            duration = len(audio) / sr
            if duration < 1.0:
                return {"success": False, "error": "Audio demasiado corto (min 1s)"}

            # Extraer características
            features = self.extract_features(audio, sr).reshape(1, -1)

            # Predecir
            prediction = self.model.predict(features, verbose=0)
            stress_score = float(prediction[0][0])

            # Clasificar
            stress_level = self._classify_stress_level(stress_score)

            return {
                "success": True,
                "stress_score": round(stress_score, 4),
                "stress_level": stress_level,
                "duration": round(duration, 2)
            }

        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            # Limpiar archivo temporal
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    def is_model_ready(self) -> bool:
        """Verifica si el modelo está listo"""
        return self.model is not None
