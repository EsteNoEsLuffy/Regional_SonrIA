from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import os
from stress_detector_api import StressDetectorAPI

# Crear instancia de FastAPI
app = FastAPI(title="Regional SonrIA - Detector de Estrés Vocal")

# Inicializar detector de estrés
detector = StressDetectorAPI()

# Extensiones soportadas
SUPPORTED_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    """
    Analiza un archivo de audio para detectar estrés vocal
    """
    if not file:
        raise HTTPException(status_code=400, detail="No se proporcionó archivo")

    # Validar extensión
    filename = file.filename.lower()
    file_extension = os.path.splitext(filename)[1]

    if file_extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Formato no soportado. Use: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    # Leer archivo
    file_content = await file.read()
    max_size = 10 * 1024 * 1024  # 10MB

    if len(file_content) > max_size:
        raise HTTPException(status_code=413, detail="Archivo demasiado grande (max 10MB)")

    # Verificar modelo
    if not detector.is_model_ready():
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    try:
        # Analizar audio
        result = detector.analyze_audio_bytes(file_content, filename)

        if not result.get("success", False):
            raise HTTPException(status_code=422, detail=result.get("error", "Error en análisis"))

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.get("/health")
async def health_check():
    """Verifica el estado de la API"""
    return {
        "status": "ok" if detector.is_model_ready() else "error",
        "model_ready": detector.is_model_ready()
    }

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000)