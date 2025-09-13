import os
import logging
import numpy as np
import librosa
import sounddevice as sd
import cv2
import threading
import time
import wave
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Configuración de logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

class StressDetectorTester:
    """
    Clase tester completa para el detector de estrés vocal que:
    1. Entrena el modelo con dataset pre-procesado
    2. Graba audio y video simultáneamente
    3. Detecta niveles de estrés (alto, medio, bajo)
    """
    
    def __init__(self, 
                 dataset_path="processed_dataset.npz",
                 model_path="models/modelo_deteccion_estres.keras",
                 duration=10,
                 sample_rate=22050,
                 mfcc_features=40):
        
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.duration = duration
        self.sample_rate = sample_rate
        self.mfcc_features = mfcc_features
        
        # Crear directorios necesarios
        os.makedirs("models", exist_ok=True)
        os.makedirs("recordings", exist_ok=True)
        
        # Inicializar componentes
        self.model = None
        self.video_cap = None
        self.video_writer = None
        self.recording = False
        
        # Umbrales para clasificación de estrés
        self.stress_thresholds = {
            'bajo': 0.3,
            'medio': 0.6,
            'alto': 0.8
        }
        
        # Cargar modelo si existe
        self._load_model()
    
    def _load_model(self):
        """Carga el modelo de detección de estrés si existe"""
        if os.path.exists(self.model_path):
            try:
                self.model = tf.keras.models.load_model(self.model_path)
                print(f"✅ Modelo cargado desde {self.model_path}")
            except Exception as e:
                print(f"❌ Error cargando modelo: {e}")
                self.model = None
        else:
            print("⚠️ Modelo no encontrado. Debes entrenarlo primero.")
            self.model = None
    
    def train_model(self, epochs=50, batch_size=16, validation_split=0.2):
        """
        Entrena el modelo usando el dataset pre-procesado
        
        Args:
            epochs: Número de épocas de entrenamiento
            batch_size: Tamaño del batch
            validation_split: Porcentaje para validación
        """
        print("=== ENTRENAMIENTO DEL MODELO ===")
        
        try:
            # 1. Cargar dataset
            if not os.path.exists(self.dataset_path):
                print(f"❌ Dataset no encontrado en {self.dataset_path}")
                print("Ejecuta primero load_dataset.py para descargar y procesar los datos")
                return False
            
            print("📊 Cargando dataset...")
            data = np.load(self.dataset_path)
            X_train = data["X"]
            y_train = data["y"]
            
            print(f"✅ Dataset cargado:")
            print(f"   - Muestras: {X_train.shape[0]}")
            print(f"   - Características: {X_train.shape[1]}")
            print(f"   - Distribución de clases: {np.unique(y_train, return_counts=True)}")
            
            # 2. Crear y compilar modelo
            print("🏗️ Creando modelo...")
            model = keras.Sequential([
                keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                keras.layers.Dropout(0.4),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(16, activation='relu'),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
            
            print("✅ Modelo compilado")
            print(model.summary())
            
            # 3. Entrenar modelo
            print(f"🚀 Iniciando entrenamiento ({epochs} épocas)...")
            history = model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=1
            )
            
            # 4. Guardar modelo
            model.save(self.model_path)
            print(f"💾 Modelo guardado en {self.model_path}")
            
            # 5. Mostrar resultados
            final_accuracy = history.history['accuracy'][-1]
            final_loss = history.history['loss'][-1]
            
            print("\n=== RESULTADOS DEL ENTRENAMIENTO ===")
            print(f"📈 Precisión final: {final_accuracy:.2%}")
            print(f"📉 Pérdida final: {final_loss:.4f}")
            
            if 'val_accuracy' in history.history:
                val_accuracy = history.history['val_accuracy'][-1]
                print(f"🔍 Precisión de validación: {val_accuracy:.2%}")
            
            # 6. Cargar modelo en la instancia
            self.model = model
            
            return True
            
        except Exception as e:
            print(f"❌ Error durante el entrenamiento: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def start_simultaneous_recording(self):
        """Inicia grabación simultánea de audio y video"""
        if self.recording:
            print("⚠️ Ya hay una grabación en curso")
            return
        
        print(f"🎬 Iniciando grabación simultánea de {self.duration} segundos...")
        self.recording = True
        
        # Configurar video
        self.video_cap = cv2.VideoCapture(0)
        if not self.video_cap.isOpened():
            print("❌ No se pudo acceder a la cámara")
            return
        
        # Configurar resolución y FPS
        self.video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.video_cap.set(cv2.CAP_PROP_FPS, 20)
        
        # Configurar video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_filename = f"recordings/video_{timestamp}.avi"
        self.video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))
        
        # Iniciar hilos de grabación
        self.audio_frames = []
        self.video_thread = threading.Thread(target=self._record_video, daemon=True)
        self.audio_thread = threading.Thread(target=self._record_audio, daemon=True)
        
        self.video_thread.start()
        self.audio_thread.start()
        
        print("✅ Grabación iniciada. Presiona 'q' para detener o espera a que termine automáticamente.")
        
        # Mostrar preview en tiempo real
        self._show_preview()
    
    def _record_video(self):
        """Hilo para grabación de video"""
        start_time = time.time()
        
        while self.recording and (time.time() - start_time) < self.duration:
            ret, frame = self.video_cap.read()
            if ret:
                self.video_writer.write(frame)
                time.sleep(0.05)  # 20 FPS
        
        print("📹 Grabación de video completada")
    
    def _record_audio(self):
        """Hilo para grabación de audio"""
        start_time = time.time()
        
        def callback(indata, frames, time_info, status):
            if self.recording and (time.time() - start_time) < self.duration:
                self.audio_frames.append(indata.copy())
        
        with sd.InputStream(samplerate=self.sample_rate, channels=1, callback=callback, dtype='float32'):
            while self.recording and (time.time() - start_time) < self.duration:
                time.sleep(0.1)
        
        print("🎵 Grabación de audio completada")
    
    def _show_preview(self):
        """Muestra preview en tiempo real durante la grabación"""
        start_time = time.time()
        
        while self.recording and (time.time() - start_time) < self.duration:
            ret, frame = self.video_cap.read()
            if ret:
                # Agregar información de grabación
                cv2.putText(frame, f"Grabando... {int(self.duration - (time.time() - start_time))}s", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow("Grabación en Curso", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            time.sleep(0.05)
        
        cv2.destroyAllWindows()
    
    def stop_recording(self):
        """Detiene la grabación y guarda los archivos"""
        if not self.recording:
            print("⚠️ No hay grabación activa")
            return
        
        print("🛑 Deteniendo grabación...")
        self.recording = False
        
        # Esperar a que terminen los hilos
        if hasattr(self, 'video_thread'):
            self.video_thread.join(timeout=2)
        if hasattr(self, 'audio_thread'):
            self.audio_thread.join(timeout=2)
        
        # Guardar archivos
        self._save_recordings()
        
        # Limpiar recursos
        if self.video_writer:
            self.video_writer.release()
        if self.video_cap:
            self.video_cap.release()
        
        print("✅ Grabación detenida y archivos guardados")
    
    def _save_recordings(self):
        """Guarda los archivos de audio y video"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Guardar audio
        if self.audio_frames:
            audio = np.concatenate(self.audio_frames, axis=0)
            audio_int16 = (audio * 32767).astype(np.int16)
            
            audio_filename = f"recordings/audio_{timestamp}.wav"
            with wave.open(audio_filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_int16.tobytes())
            
            print(f"🎵 Audio guardado: {audio_filename}")
            
            # Analizar audio para detección de estrés
            self._analyze_audio_for_stress(audio_filename)
    
    def _analyze_audio_for_stress(self, audio_filename):
        """Analiza el audio grabado para detectar estrés"""
        if self.model is None:
            print("❌ No hay modelo disponible para análisis")
            return
        
        try:
            print("\n🔍 Analizando audio para detección de estrés...")
            
            # Cargar y procesar audio
            audio, sr = librosa.load(audio_filename, sr=self.sample_rate)
            
            # Extraer características MFCC
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.mfcc_features)
            features = np.mean(mfccs.T, axis=0).reshape(1, -1)
            
            # Predecir nivel de estrés
            prediction = self.model.predict(features, verbose=0)
            stress_score = prediction[0][0]
            
            # Clasificar nivel de estrés
            stress_level = self._classify_stress_level(stress_score)
            
            # Mostrar resultados
            print(f"\n📊 RESULTADOS DEL ANÁLISIS:")
            print(f"   - Puntuación de estrés: {stress_score:.3f}")
            print(f"   - Nivel de estrés: {stress_level}")
            
            # Generar gráfico MFCC
            self._plot_mfccs(mfccs, sr, stress_score, stress_level)
            
            return stress_level, stress_score
            
        except Exception as e:
            print(f"❌ Error en análisis de audio: {e}")
            return None, None
    
    def _classify_stress_level(self, stress_score):
        """Clasifica el nivel de estrés basado en la puntuación"""
        if stress_score < self.stress_thresholds['bajo']:
            return "BAJO (Normal)"
        elif stress_score < self.stress_thresholds['medio']:
            return "MEDIO (Atención)"
        elif stress_score < self.stress_thresholds['alto']:
            return "ALTO (Alerta)"
        else:
            return "MUY ALTO (Crítico)"
    
    def _plot_mfccs(self, mfccs, sr, stress_score, stress_level):
        """Genera y guarda gráfico de MFCCs"""
        try:
            plt.switch_backend('Agg')
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Gráfico de MFCCs
            img1 = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=ax1)
            ax1.set_title(f'MFCCs - Nivel de Estrés: {stress_level}')
            ax1.set_ylabel('Coeficientes MFCC')
            plt.colorbar(img1, ax=ax1)
            
            # Gráfico de barras de características
            mfcc_mean = np.mean(mfccs.T, axis=0)
            ax2.bar(range(len(mfcc_mean)), mfcc_mean)
            ax2.set_title(f'Características MFCC Promedio - Puntuación: {stress_score:.3f}')
            ax2.set_xlabel('Índice de Coeficiente MFCC')
            ax2.set_ylabel('Valor Promedio')
            
            plt.tight_layout()
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            plot_filename = f"recordings/mfcc_analysis_{timestamp}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"📈 Gráfico de análisis guardado: {plot_filename}")
            
        except Exception as e:
            print(f"⚠️ Error generando gráfico: {e}")
    
    def run_complete_test(self):
        """Ejecuta una prueba completa: grabación + análisis"""
        print("🚀 INICIANDO PRUEBA COMPLETA DE DETECCIÓN DE ESTRÉS")
        print("=" * 60)
        
        # Verificar modelo
        if self.model is None:
            print("❌ No hay modelo disponible. Debes entrenarlo primero.")
            print("Usa: tester.train_model()")
            return
        
        # Iniciar grabación
        self.start_simultaneous_recording()
        
        # Esperar a que termine automáticamente
        time.sleep(self.duration + 1)
        
        # Detener grabación
        self.stop_recording()
        
        print("\n✅ PRUEBA COMPLETADA")
        print("=" * 60)
    
    def get_model_info(self):
        """Muestra información del modelo actual"""
        if self.model is None:
            print("❌ No hay modelo cargado")
            return
        
        print("📋 INFORMACIÓN DEL MODELO:")
        print(f"   - Arquitectura: {len(self.model.layers)} capas")
        print(f"   - Entrada: {self.model.input_shape}")
        print(f"   - Salida: {self.model.output_shape}")
        print(f"   - Parámetros totales: {self.model.count_params():,}")
        
        # Mostrar configuración de capas
        for i, layer in enumerate(self.model.layers):
            print(f"   - Capa {i+1}: {layer.__class__.__name__} - {layer.output_shape}")
    
    def cleanup(self):
        """Limpia recursos y cierra conexiones"""
        if self.recording:
            self.stop_recording()
        
        if self.video_cap:
            self.video_cap.release()
        if self.video_writer:
            self.video_writer.release()
        
        cv2.destroyAllWindows()
        print("🧹 Recursos limpiados")


# Función de ejemplo de uso
def main():
    """Función principal de ejemplo"""
    print("🎯 DETECTOR DE ESTRÉS VOCAL - CLASE TESTER")
    print("=" * 50)
    
    # Crear instancia del tester
    tester = StressDetectorTester()
    
    try:
        while True:
            print("\n📋 OPCIONES DISPONIBLES:")
            print("1. Entrenar modelo")
            print("2. Ejecutar prueba completa")
            print("3. Información del modelo")
            print("4. Salir")
            
            choice = input("\nSelecciona una opción (1-4): ").strip()
            
            if choice == "1":
                print("\n🚀 Iniciando entrenamiento...")
                success = tester.train_model(epochs=30, batch_size=16)
                if success:
                    print("✅ Entrenamiento completado exitosamente!")
                else:
                    print("❌ Error en el entrenamiento")
            
            elif choice == "2":
                if tester.model is None:
                    print("❌ Debes entrenar el modelo primero (opción 1)")
                else:
                    tester.run_complete_test()
            
            elif choice == "3":
                tester.get_model_info()
            
            elif choice == "4":
                print("👋 ¡Hasta luego!")
                break
            
            else:
                print("❌ Opción no válida. Intenta de nuevo.")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupción por usuario")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        tester.cleanup()


if __name__ == "__main__":
    main()
