import os
import sys
import time
import logging
import numpy as np
import librosa
import sounddevice as sd
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

def initialize_components():
    """Inicializa los componentes del sistema"""
    global stress_detector, audio_recorder, video_recorder
    
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
