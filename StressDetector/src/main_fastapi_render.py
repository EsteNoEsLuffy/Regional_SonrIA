import os
import sys
import time
import logging
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import tempfile
import uuid
from datetime import datetime

# Configurar variables de entorno
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configurar logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel('ERROR')

# Inicializar FastAPI
app = FastAPI(
    title="Stress Detector API - ESP32 Only",
    description="API para recibir datos de ESP32-S3 AI Camera",
    version="1.0.0"
)

# Variables globales
DURATION = 10
SAMPLE_RATE = 22050
MODEL_PATH = "models/modelo_deteccion_estres.keras"

# Crear directorios necesarios
os.makedirs("recordings", exist_ok=True)
os.makedirs("stored_texts", exist_ok=True)

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

# Variables globales
stress_detector = None

# Funciones auxiliares
def classify_stress_level(score: float) -> str:
    """Clasifica el nivel de estrés basado en el score"""
    if score < 0.3:
        return "BAJO (Normal)"
    elif score < 0.6:
        return "MEDIO (Atención)"
    elif score < 0.8:
        return "ALTO (Alerta)"
    else:
        return "MUY ALTO (Crítico)"

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

def initialize_components():
    """Inicializa los componentes del sistema (sin audio)"""
    global stress_detector
    
    try:
        # Solo intentar cargar el modelo si existe
        if os.path.exists(MODEL_PATH):
            print(f"✅ Modelo encontrado en {MODEL_PATH}")
            stress_detector = tf.keras.models.load_model(MODEL_PATH)
        else:
            print("⚠️ Modelo no encontrado - modo solo recepción")
            stress_detector = None
        
        return True
    except Exception as e:
        print(f"⚠️ Error inicializando modelo: {e}")
        stress_detector = None
        return True  # Continuar aunque no haya modelo

# Endpoints básicos
@app.get("/", response_model=Dict[str, str])
async def root():
    """Endpoint raíz"""
    return {
        "message": "Stress Detector API - ESP32 Ready",
        "version": "1.0.0",
        "status": "active",
        "mode": "ESP32-only"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Verificar estado del sistema"""
    model_loaded = stress_detector is not None
    
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        timestamp=datetime.now().isoformat()
    )

# ===== ENDPOINTS PARA ESP32-S3 AI CAMERA =====

@app.post("/esp32/receive-audio")
async def receive_audio_from_esp32(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Recibe audio de ESP32-S3 y lo almacena
    """
    try:
        # Crear directorio de grabaciones si no existe
        os.makedirs("recordings", exist_ok=True)
        
        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Generar nombre de archivo con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"esp32_audio_{timestamp}.wav"
        final_path = os.path.join("recordings", filename)
        
        # Mover archivo a ubicación final
        import shutil
        shutil.move(temp_file_path, final_path)
        
        return {
            "success": True,
            "message": "Audio recibido de ESP32-S3",
            "device_id": "ESP32-S3",
            "filename": filename,
            "file_path": final_path,
            "file_size": len(content),
            "timestamp": datetime.now().isoformat(),
            "ready_for_esp32": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recibiendo audio de ESP32-S3: {str(e)}")

@app.post("/esp32/send-stress-data")
async def send_stress_data_to_esp32(
    device_id: str,
    stress_score: float,
    stress_level: str,
    timestamp: str
):
    """
    Recibe datos de análisis de estrés de ESP32-S3
    """
    try:
        # Validar datos recibidos
        if not (0.0 <= stress_score <= 1.0):
            raise HTTPException(status_code=400, detail="Puntuación de estrés debe estar entre 0.0 y 1.0")
        
        valid_levels = ["BAJO (Normal)", "MEDIO (Atención)", "ALTO (Alerta)", "MUY ALTO (Crítico)"]
        if stress_level not in valid_levels:
            raise HTTPException(status_code=400, detail=f"Nivel de estrés no válido. Debe ser uno de: {valid_levels}")
        
        # Guardar datos en archivo de log
        log_entry = f"{datetime.now().isoformat()},{device_id},{stress_score},{stress_level},{timestamp}\n"
        with open("esp32_stress_log.csv", "a") as f:
            f.write(log_entry)
        
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
            "status": "data_received",
            "ready_for_esp32": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando datos de ESP32-S3: {str(e)}")

@app.get("/esp32/model-status")
async def get_model_status_for_esp32():
    """
    Envía estado del modelo a ESP32-S3
    """
    try:
        model_loaded = stress_detector is not None
        
        return {
            "success": True,
            "device_ready": True,
            "model_loaded": model_loaded,
            "stress_thresholds": {
                "bajo": 0.3,
                "medio": 0.6,
                "alto": 0.8
            },
            "sample_rate": SAMPLE_RATE,
            "mfcc_features": 40,
            "timestamp": datetime.now().isoformat(),
            "ready_for_esp32": True,
            "mode": "reception_only"
        }
        
    except Exception as e:
        return {
            "success": False,
            "device_ready": False,
            "model_loaded": False,
            "error": str(e),
            "ready_for_esp32": False
        }

# Endpoint adicional para almacenar texto (sin análisis de estrés)
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

# Eventos de la aplicación
@app.on_event("startup")
async def startup_event():
    """Inicializar componentes al arrancar la aplicación"""
    print("=== INICIANDO STRESS DETECTOR API - ESP32 MODE ===")
    
    success = initialize_components()
    if success:
        print("✅ Componentes inicializados correctamente")
    else:
        print("⚠️ Algunos componentes no se pudieron inicializar")
    
    print("🚀 API lista para recibir datos de ESP32-S3")
    print("📱 Modo: Solo recepción (sin audio local)")

@app.on_event("shutdown")
async def shutdown_event():
    """Limpiar recursos al cerrar la aplicación"""
    print("=== DETENIENDO STRESS DETECTOR API ===")
    print("✅ Recursos liberados correctamente")

# Ejecutar con uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main_fastapi_render:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
