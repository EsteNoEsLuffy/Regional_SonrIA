import os
import sys
import time
import logging
import numpy as np
import librosa
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import tempfile
import uuid
from datetime import datetime
import threading

# Configurar variables de entorno
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configurar logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel('ERROR')

# Importar módulos locales
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from stress_detector import Stress_Detector
from audio_recorder import AudioRecorder
from video_recorder import VideoRecoder
from stress_detector_tester import StressDetectorTester

# Inicializar FastAPI
app = FastAPI(
    title="Stress Detector API",
    description="API para detección de estrés vocal en tiempo real",
    version="1.0.0"
)

# Variables globales
DURATION = 10
SAMPLE_RATE = 22050
MODEL_PATH = "models/modelo_deteccion_estres.keras"

# Inicializar detector de estrés
stress_detector = None
audio_recorder = None
video_recorder = None
stress_tester = None

# Modelos Pydantic
class StressAnalysisRequest(BaseModel):
    duration: Optional[int] = 10
    threshold: Optional[float] = 0.6

class StressAnalysisResponse(BaseModel):
    stress_detected: bool
    stress_score: float
    confidence: float
    stress_level: str
    timestamp: str
    analysis_id: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str

class TextStorageRequest(BaseModel):
    text: str
    title: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[list] = None

class TextStorageResponse(BaseModel):
    success: bool
    message: str
    text_id: str
    file_path: str
    timestamp: str

class TextWithStressRequest(BaseModel):
    text: str
    title: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[list] = None
    threshold: Optional[float] = 0.6

class AudioWithTextRequest(BaseModel):
    text: str
    title: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[list] = None
    threshold: Optional[float] = 0.6

class StressAnalysisData(BaseModel):
    stress_detected: bool
    stress_score: float
    confidence: float
    stress_level: str
    analysis_id: str

class TextWithStressResponse(BaseModel):
    success: bool
    message: str
    text_id: str
    file_path: str
    stress_analysis: StressAnalysisData
    timestamp: str

# Funciones auxiliares
def classify_stress_level(score: float) -> str:
    """Clasifica el nivel de estrés basado en el score"""
    if score < 0.3:
        return "BAJO"
    elif score < 0.6:
        return "MEDIO"
    elif score < 0.8:
        return "ALTO"
    else:
        return "MUY ALTO"

def save_text_to_file(text: str, title: str = None, category: str = None, tags: list = None) -> tuple:
    """Guarda texto en un archivo con metadatos"""
    try:
        # Crear directorio de textos si no existe
        texts_dir = "stored_texts"
        os.makedirs(texts_dir, exist_ok=True)
        
        # Generar ID único y timestamp
        text_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Crear nombre de archivo
        if title:
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_title = safe_title.replace(' ', '_')[:50]  # Limitar longitud
            filename = f"{safe_title}_{timestamp}.txt"
        else:
            filename = f"text_{timestamp}_{text_id[:8]}.txt"
        
        # Ruta completa del archivo
        file_path = os.path.join(texts_dir, filename)
        
        # Crear contenido del archivo con metadatos
        content = f"""=== METADATOS ===
ID: {text_id}
Título: {title or 'Sin título'}
Categoría: {category or 'Sin categoría'}
Tags: {', '.join(tags) if tags else 'Sin tags'}
Fecha: {datetime.now().isoformat()}

=== CONTENIDO ===
{text}
"""
        
        # Guardar archivo
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return text_id, file_path
        
    except Exception as e:
        raise Exception(f"Error guardando texto: {str(e)}")

def save_text_with_stress_analysis(text: str, title: str = None, category: str = None, 
                                 tags: list = None, stress_data: dict = None) -> tuple:
    """Guarda texto en un archivo con metadatos y análisis de estrés"""
    try:
        # Crear directorio de textos si no existe
        texts_dir = "stored_texts"
        os.makedirs(texts_dir, exist_ok=True)
        
        # Generar ID único y timestamp
        text_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Crear nombre de archivo
        if title:
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_title = safe_title.replace(' ', '_')[:50]  # Limitar longitud
            filename = f"{safe_title}_{timestamp}.txt"
        else:
            filename = f"text_stress_{timestamp}_{text_id[:8]}.txt"
        
        # Ruta completa del archivo
        file_path = os.path.join(texts_dir, filename)
        
        # Crear contenido del archivo con metadatos y análisis de estrés
        content = f"""=== METADATOS ===
ID: {text_id}
Título: {title or 'Sin título'}
Categoría: {category or 'Sin categoría'}
Tags: {', '.join(tags) if tags else 'Sin tags'}
Fecha: {datetime.now().isoformat()}

=== ANÁLISIS DE ESTRÉS VOCAL ===
Estrés Detectado: {'SÍ' if stress_data.get('stress_detected') else 'NO'}
Puntuación de Estrés: {stress_data.get('stress_score', 0):.4f}
Confianza: {stress_data.get('confidence', 0):.4f}
Nivel de Estrés: {stress_data.get('stress_level', 'N/A')}
ID de Análisis: {stress_data.get('analysis_id', 'N/A')}

=== CONTENIDO ===
{text}
"""
        
        # Guardar archivo
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return text_id, file_path
        
    except Exception as e:
        raise Exception(f"Error guardando texto con análisis de estrés: {str(e)}")

def analyze_voice_stress_from_audio(audio_file_path: str, threshold: float = 0.6) -> dict:
    """Analiza estrés vocal usando el modelo real de TensorFlow"""
    try:
        if not stress_detector or not stress_detector.model:
            raise Exception("Modelo de estrés no disponible")
        
        # Cargar audio usando librosa (mismo método que tu software)
        audio, sr = librosa.load(audio_file_path, sr=SAMPLE_RATE)
        
        # Extraer características MFCC (mismo método que tu software)
        features = stress_detector.extract_features(audio, sr).reshape(1, -1)
        
        # Predecir usando el modelo real
        prediction = stress_detector.model.predict(features, verbose=0)
        stress_score = float(prediction[0][0])
        
        stress_detected = stress_score > threshold
        confidence = abs(stress_score - 0.5) * 2
        analysis_id = str(uuid.uuid4())
        
        return {
            'stress_detected': stress_detected,
            'stress_score': stress_score,
            'confidence': confidence,
            'stress_level': classify_stress_level(stress_score),
            'analysis_id': analysis_id
        }
        
    except Exception as e:
        raise Exception(f"Error en análisis de estrés vocal: {str(e)}")

def analyze_voice_stress_from_text(text: str, threshold: float = 0.6) -> dict:
    """Simula análisis de estrés basado en el texto (para cuando no hay audio)"""
    try:
        # Esta función simula el análisis de estrés
        # En una implementación real, aquí se analizaría el audio de la voz
        
        # Generar análisis simulado basado en características del texto
        text_length = len(text)
        word_count = len(text.split())
        
        # Simular puntuación de estrés basada en longitud y características del texto
        base_score = 0.3 + (text_length / 1000) * 0.2  # Más texto = más estrés simulado
        if word_count > 50:
            base_score += 0.1
        if any(word in text.lower() for word in ['estrés', 'presión', 'urgencia', 'importante']):
            base_score += 0.2
        
        # Añadir algo de aleatoriedad para simular variación real
        import random
        base_score += random.uniform(-0.1, 0.1)
        base_score = max(0.0, min(1.0, base_score))  # Mantener en rango [0,1]
        
        stress_detected = base_score > threshold
        confidence = abs(base_score - 0.5) * 2
        analysis_id = str(uuid.uuid4())
        
        return {
            'stress_detected': stress_detected,
            'stress_score': base_score,
            'confidence': confidence,
            'stress_level': classify_stress_level(base_score),
            'analysis_id': analysis_id
        }
        
    except Exception as e:
        raise Exception(f"Error en análisis de estrés: {str(e)}")

def initialize_components():
    """Inicializa los componentes del sistema"""
    global stress_detector, audio_recorder, video_recorder, stress_tester
    
    try:
        # Inicializar detector de estrés
        stress_detector = Stress_Detector()
        
        # Inicializar grabadores
        audio_recorder = AudioRecorder(
            stress_model=MODEL_PATH,
            stress_threshold=0.6,
            duration=DURATION
        )
        
        video_recorder = VideoRecoder()
        
        # Inicializar StressDetectorTester (tu clase principal)
        stress_tester = StressDetectorTester(
            dataset_path="processed_dataset.npz",
            model_path=MODEL_PATH,
            duration=DURATION,
            sample_rate=SAMPLE_RATE,
            mfcc_features=40
        )
        
        return True
    except Exception as e:
        print(f"Error inicializando componentes: {e}")
        return False

# Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Endpoint raíz"""
    return {
        "message": "Stress Detector API",
        "version": "1.0.0",
        "status": "active"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Verificar estado del sistema"""
    model_loaded = stress_detector is not None and stress_detector.model is not None
    
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        timestamp=datetime.now().isoformat()
    )

@app.post("/analyze/audio", response_model=StressAnalysisResponse)
async def analyze_audio_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    threshold: float = 0.6
):
    """Analizar archivo de audio para detectar estrés"""
    
    if not stress_detector or not stress_detector.model:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    try:
        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Analizar audio
        audio, sr = librosa.load(temp_file_path, sr=SAMPLE_RATE)
        features = stress_detector.extract_features(audio, sr).reshape(1, -1)
        prediction = stress_detector.model.predict(features, verbose=0)
        
        stress_score = float(prediction[0][0])
        stress_detected = stress_score > threshold
        confidence = abs(stress_score - 0.5) * 2  # Confianza basada en distancia del umbral
        
        analysis_id = str(uuid.uuid4())
        
        # Limpiar archivo temporal
        background_tasks.add_task(os.unlink, temp_file_path)
        
        return StressAnalysisResponse(
            stress_detected=stress_detected,
            stress_score=stress_score,
            confidence=confidence,
            stress_level=classify_stress_level(stress_score),
            timestamp=datetime.now().isoformat(),
            analysis_id=analysis_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando audio: {str(e)}")

@app.post("/analyze/realtime", response_model=StressAnalysisResponse)
async def analyze_realtime(request: StressAnalysisRequest):
    """Analizar audio en tiempo real (simulado)"""
    
    if not stress_detector or not stress_detector.model:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    try:
        # Simular grabación y análisis
        print(f"Grabando audio durante {request.duration} segundos...")
        
        # Grabar audio
        audio = sd.rec(
            int(SAMPLE_RATE * request.duration), 
            samplerate=SAMPLE_RATE, 
            channels=1, 
            dtype='float32'
        )
        sd.wait()
        audio = audio.flatten()
        
        # Extraer características y predecir
        features = stress_detector.extract_features(audio, SAMPLE_RATE).reshape(1, -1)
        prediction = stress_detector.model.predict(features, verbose=0)
        
        stress_score = float(prediction[0][0])
        stress_detected = stress_score > request.threshold
        confidence = abs(stress_score - 0.5) * 2
        
        analysis_id = str(uuid.uuid4())
        
        return StressAnalysisResponse(
            stress_detected=stress_detected,
            stress_score=stress_score,
            confidence=confidence,
            stress_level=classify_stress_level(stress_score),
            timestamp=datetime.now().isoformat(),
            analysis_id=analysis_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en análisis en tiempo real: {str(e)}")

@app.get("/model/info")
async def get_model_info():
    """Obtener información del modelo"""
    
    if not stress_detector or not stress_detector.model:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    try:
        model = stress_detector.model
        return {
            "model_loaded": True,
            "model_path": MODEL_PATH,
            "input_shape": model.input_shape,
            "output_shape": model.output_shape,
            "total_params": model.count_params(),
            "layers": len(model.layers)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo info del modelo: {str(e)}")

@app.post("/record/start")
async def start_recording():
    """Iniciar grabación de audio y video"""
    
    try:
        if audio_recorder:
            audio_recorder.start()
        if video_recorder:
            video_recorder.start()
            
        return {
            "message": "Grabación iniciada",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error iniciando grabación: {str(e)}")

@app.post("/record/stop")
async def stop_recording():
    """Detener grabación de audio y video"""
    
    try:
        if audio_recorder:
            audio_recorder.stop()
        if video_recorder:
            video_recorder.stop()
            
        return {
            "message": "Grabación detenida",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deteniendo grabación: {str(e)}")

@app.post("/store/text", response_model=TextStorageResponse)
async def store_text(request: TextStorageRequest):
    """Almacenar texto con metadatos opcionales"""
    
    try:
        # Validar que el texto no esté vacío
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="El texto no puede estar vacío")
        
        # Guardar texto en archivo
        text_id, file_path = save_text_to_file(
            text=request.text,
            title=request.title,
            category=request.category,
            tags=request.tags
        )
        
        return TextStorageResponse(
            success=True,
            message="Texto almacenado correctamente",
            text_id=text_id,
            file_path=file_path,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error almacenando texto: {str(e)}")

@app.post("/store/text-with-stress-analysis", response_model=TextWithStressResponse)
async def store_text_with_stress_analysis(request: TextWithStressRequest):
    """Almacenar texto con análisis de nivel de estrés en la voz (simulado)"""
    
    try:
        # Validar que el texto no esté vacío
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="El texto no puede estar vacío")
        
        # Analizar estrés vocal basado en el texto (simulado)
        stress_analysis = analyze_voice_stress_from_text(
            text=request.text,
            threshold=request.threshold
        )
        
        # Guardar texto con análisis de estrés
        text_id, file_path = save_text_with_stress_analysis(
            text=request.text,
            title=request.title,
            category=request.category,
            tags=request.tags,
            stress_data=stress_analysis
        )
        
        # Crear objeto de análisis de estrés
        stress_data = StressAnalysisData(**stress_analysis)
        
        return TextWithStressResponse(
            success=True,
            message="Texto almacenado con análisis de estrés vocal (simulado)",
            text_id=text_id,
            file_path=file_path,
            stress_analysis=stress_data,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error almacenando texto con análisis de estrés: {str(e)}")

# ===== ENDPOINTS PARA ESP32-S3 AI CAMERA (SOLO FUNCIONALIDADES EXISTENTES) =====

@app.post("/esp32/receive-audio")
async def receive_audio_from_esp32(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Recibe audio de ESP32-S3 y lo analiza usando StressDetectorTester
    Solo usa funcionalidades existentes del proyecto
    """
    
    if not stress_tester:
        raise HTTPException(status_code=503, detail="StressDetectorTester no disponible")
    
    if not stress_tester.model:
        raise HTTPException(status_code=400, detail="No hay modelo disponible para análisis")
    
    try:
        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Analizar usando el método exacto de tu clase
        stress_level, stress_score = stress_tester._analyze_audio_for_stress(temp_file_path)
        
        # Limpiar archivo temporal
        background_tasks.add_task(os.unlink, temp_file_path)
        
        if stress_level is None or stress_score is None:
            raise HTTPException(status_code=500, detail="Error en el análisis de audio")
        
        # Respuesta con datos que ESP32-S3 puede procesar
        return {
            "success": True,
            "device_id": "ESP32-S3",
            "analysis_result": {
                "stress_score": float(stress_score),
                "stress_level": stress_level,
                "stress_detected": stress_score > stress_tester.stress_thresholds['medio'],
                "confidence": abs(stress_score - 0.5) * 2
            },
            "thresholds": stress_tester.stress_thresholds,
            "timestamp": datetime.now().isoformat(),
            "ready_for_esp32": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando audio de ESP32-S3: {str(e)}")

@app.post("/esp32/send-stress-data")
async def send_stress_data_to_esp32(
    device_id: str,
    stress_score: float,
    stress_level: str,
    timestamp: str
):
    """
    Envía datos de análisis de estrés a ESP32-S3
    Solo confirma recepción de datos existentes
    """
    
    if not stress_tester:
        raise HTTPException(status_code=503, detail="StressDetectorTester no disponible")
    
    try:
        # Validar datos recibidos
        if not (0.0 <= stress_score <= 1.0):
            raise HTTPException(status_code=400, detail="Puntuación de estrés debe estar entre 0.0 y 1.0")
        
        if stress_level not in ["BAJO (Normal)", "MEDIO (Atención)", "ALTO (Alerta)", "MUY ALTO (Crítico)"]:
            raise HTTPException(status_code=400, detail="Nivel de estrés no válido")
        
        # Confirmar recepción de datos
        return {
            "success": True,
            "message": "Datos de estrés recibidos correctamente",
            "device_id": device_id,
            "received_data": {
                "stress_score": stress_score,
                "stress_level": stress_level,
                "timestamp": timestamp
            },
            "server_timestamp": datetime.now().isoformat(),
            "status": "data_received"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando datos de ESP32-S3: {str(e)}")

@app.get("/esp32/model-status")
async def get_model_status_for_esp32():
    """
    Envía estado del modelo a ESP32-S3
    Solo información básica para verificar conectividad
    """
    
    if not stress_tester:
        return {
            "success": False,
            "device_ready": False,
            "model_loaded": False,
            "message": "StressDetectorTester no disponible"
        }
    
    try:
        model_loaded = stress_tester.model is not None
        
        return {
            "success": True,
            "device_ready": True,
            "model_loaded": model_loaded,
            "stress_thresholds": stress_tester.stress_thresholds,
            "sample_rate": stress_tester.sample_rate,
            "mfcc_features": stress_tester.mfcc_features,
            "timestamp": datetime.now().isoformat(),
            "ready_for_esp32": True
        }
        
    except Exception as e:
        return {
            "success": False,
            "device_ready": False,
            "model_loaded": False,
            "error": str(e)
        }

# ===== ENDPOINTS QUE USAN EXACTAMENTE TU STRESSDETECTORTESTER =====

@app.post("/tester/train-model")
async def train_model_tester(epochs: int = 30, batch_size: int = 16):
    """Entrena el modelo usando exactamente el método de StressDetectorTester"""
    
    if not stress_tester:
        raise HTTPException(status_code=503, detail="StressDetectorTester no disponible")
    
    try:
        print(f"🚀 Iniciando entrenamiento con {epochs} épocas...")
        success = stress_tester.train_model(epochs=epochs, batch_size=batch_size)
        
        if success:
            return {
                "success": True,
                "message": "Modelo entrenado exitosamente",
                "epochs": epochs,
                "batch_size": batch_size,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Error en el entrenamiento del modelo")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error entrenando modelo: {str(e)}")

@app.post("/tester/run-complete-test")
async def run_complete_test_tester():
    """Ejecuta una prueba completa usando exactamente el método de StressDetectorTester"""
    
    if not stress_tester:
        raise HTTPException(status_code=503, detail="StressDetectorTester no disponible")
    
    if not stress_tester.model:
        raise HTTPException(status_code=400, detail="No hay modelo disponible. Debes entrenarlo primero.")
    
    try:
        print("🚀 INICIANDO PRUEBA COMPLETA DE DETECCIÓN DE ESTRÉS")
        
        # Ejecutar prueba completa (grabación + análisis)
        stress_tester.run_complete_test()
        
        return {
            "success": True,
            "message": "Prueba completa ejecutada exitosamente",
            "duration": stress_tester.duration,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ejecutando prueba completa: {str(e)}")

@app.get("/tester/model-info")
async def get_model_info_tester():
    """Obtiene información del modelo usando exactamente el método de StressDetectorTester"""
    
    if not stress_tester:
        raise HTTPException(status_code=503, detail="StressDetectorTester no disponible")
    
    if not stress_tester.model:
        raise HTTPException(status_code=400, detail="No hay modelo cargado")
    
    try:
        # Obtener información usando el método de tu clase
        model = stress_tester.model
        
        return {
            "model_loaded": True,
            "model_path": stress_tester.model_path,
            "dataset_path": stress_tester.dataset_path,
            "duration": stress_tester.duration,
            "sample_rate": stress_tester.sample_rate,
            "mfcc_features": stress_tester.mfcc_features,
            "stress_thresholds": stress_tester.stress_thresholds,
            "architecture": {
                "layers": len(model.layers),
                "input_shape": model.input_shape,
                "output_shape": model.output_shape,
                "total_params": model.count_params()
            },
            "layer_info": [
                {
                    "index": i+1,
                    "name": layer.__class__.__name__,
                    "output_shape": layer.output_shape
                }
                for i, layer in enumerate(model.layers)
            ],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo info del modelo: {str(e)}")

@app.post("/tester/analyze-audio-file")
async def analyze_audio_file_tester(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Analiza archivo de audio usando exactamente el método de StressDetectorTester"""
    
    if not stress_tester:
        raise HTTPException(status_code=503, detail="StressDetectorTester no disponible")
    
    if not stress_tester.model:
        raise HTTPException(status_code=400, detail="No hay modelo disponible para análisis")
    
    try:
        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Analizar usando el método exacto de tu clase
        stress_level, stress_score = stress_tester._analyze_audio_for_stress(temp_file_path)
        
        # Limpiar archivo temporal
        background_tasks.add_task(os.unlink, temp_file_path)
        
        if stress_level is None or stress_score is None:
            raise HTTPException(status_code=500, detail="Error en el análisis de audio")
        
        return {
            "success": True,
            "message": "Análisis completado exitosamente",
            "stress_score": float(stress_score),
            "stress_level": stress_level,
            "stress_detected": stress_score > stress_tester.stress_thresholds['medio'],
            "thresholds": stress_tester.stress_thresholds,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando audio: {str(e)}")

@app.post("/tester/start-recording")
async def start_recording_tester():
    """Inicia grabación simultánea usando exactamente el método de StressDetectorTester"""
    
    if not stress_tester:
        raise HTTPException(status_code=503, detail="StressDetectorTester no disponible")
    
    try:
        stress_tester.start_simultaneous_recording()
        
        return {
            "success": True,
            "message": "Grabación iniciada",
            "duration": stress_tester.duration,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error iniciando grabación: {str(e)}")

@app.post("/tester/stop-recording")
async def stop_recording_tester():
    """Detiene grabación usando exactamente el método de StressDetectorTester"""
    
    if not stress_tester:
        raise HTTPException(status_code=503, detail="StressDetectorTester no disponible")
    
    try:
        stress_tester.stop_recording()
        
        return {
            "success": True,
            "message": "Grabación detenida",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deteniendo grabación: {str(e)}")

# Eventos de la aplicación
@app.on_event("startup")
async def startup_event():
    """Inicializar componentes al arrancar la aplicación"""
    print("=== INICIANDO STRESS DETECTOR API ===")
    
    success = initialize_components()
    if success:
        print("✅ Componentes inicializados correctamente")
    else:
        print("⚠️ Algunos componentes no se pudieron inicializar")
    
    print("🚀 API lista para recibir requests")

@app.on_event("shutdown")
async def shutdown_event():
    """Limpiar recursos al cerrar la aplicación"""
    print("=== DETENIENDO STRESS DETECTOR API ===")
    
    if audio_recorder:
        audio_recorder.stop()
    if video_recorder:
        video_recorder.stop()
    
    print("✅ Recursos liberados correctamente")

# Ejecutar con uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main_fastapi:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
