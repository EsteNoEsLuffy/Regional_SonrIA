import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silencia mensajes de TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Desactiva mensajes oneDNN

# Configura el logger de TensorFlow
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel('ERROR')

import numpy as np
import librosa
import sounddevice as sd
from tensorflow import keras

# Parámetros globales
DURATION = 10  # segundos para grabación desde micrófono
SAMPLE_RATE = 22050
MODEL_PATH = "modelo_deteccion_estres.keras"

class Stress_Detector:
    def __init__(self):
        if os.path.exists(MODEL_PATH):
            print("Modelo cargado exitosamente.")
            self.model = tf.keras.models.load_model(MODEL_PATH)
        else:
            print("Modelo no encontrado. Entrénalo usando `train_model()` primero.")
            self.model = None

    def extract_features(self, audio, sr):
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        return mfccs_mean

    def record_audio(self):
        print(f"Grabando audio durante {DURATION} segundos...")
        audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()
        return audio

    def predict_from_mic(self):
        if self.model is None:
            print("No hay modelo cargado.")
            return
        audio = self.record_audio()
        features = self.extract_features(audio, SAMPLE_RATE).reshape(1, -1)
        prediction = self.model.predict(features)
        stress_detected = prediction[0][0] > 0.5  # Corregido el nombre
        print("Resultado:", prediction[0][0])
        if stress_detected:
            print("¡Estrés detectado en la voz!")
        else:
            print("Voz normal (sin estrés).")

    def predict_from_file(self, filepath):
        if self.model is None:
            print("No hay modelo cargado.")
            return
        audio, sr = librosa.load(filepath, sr=SAMPLE_RATE)  # Corregido SAMPLE_RATE
        features = self.extract_features(audio, sr).reshape(1, -1)  # Corregido extract_features
        prediction = self.model.predict(features)
        stress_detected = prediction[0][0] > 0.5
        print(f"Resultado para {filepath}: {prediction[0][0]}")
        if stress_detected:
            print("¡Estrés detectado!")
        else:
            print("Voz sin estrés.")    

    def train_model(self):
        print("=== MODO DEBUG ACTIVADO ===")
        
        try:
            # 1. Verificar dataset
            print("Cargando dataset...")
            data = np.load("processed_dataset.npz")
            X_train = data["X"]
            y_train = data["y"]
            print(f"✔ Dataset cargado - Muestras: {X_train.shape[0]}, Características: {X_train.shape[1]}")
            print(f"Distribución de clases: {np.unique(y_train, return_counts=True)}")

            # 2. Modelo
            print("Creando modelo...")
            model = keras.Sequential([
                keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
            )
            print("✔ Modelo compilado")

            # 3. Entrenamiento FORZANDO guardado
            print("Iniciando entrenamiento (guardado forzado)...")
            history = model.fit(
                X_train,
                y_train,
                epochs=30,
                batch_size=8,
                validation_split=0.2,
                verbose=1
            )
            
            # Guardado INCONDICIONAL
            model.save(MODEL_PATH)
            print(f"✅ Modelo guardado en {MODEL_PATH} (precisión: {history.history['accuracy'][-1]:.2%})")
            
            # Debug adicional
            print("\n=== RESUMEN ===")
            print(f"Precisión final: {history.history['accuracy'][-1]:.2%}")
            print(f"Pérdida final: {history.history['loss'][-1]:.4f}")
            if len(np.unique(y_train)) == 1:
                print("⚠️ ALERTA: Todas las muestras pertenecen a la misma clase")
            
            return True
            
        except Exception as e:
            print(f"ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == '__main__':
    detector = Stress_Detector()
    
    # Verifica primero el dataset
    print("=== VERIFICANDO DATASET ===")
    try:
        data = np.load("processed_dataset.npz")
        print(f"Muestras: {len(data['X'])}")
        print(f"Distribución de clases: {np.unique(data['y'], return_counts=True)}")
    except Exception as e:
        print(f"Error en dataset: {e}")
        exit()
    
    # Ejecuta entrenamiento
    print("\n=== INICIANDO ENTRENAMIENTO ===")
    detector.train_model()