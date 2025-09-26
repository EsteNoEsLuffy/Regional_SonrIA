@echo off
echo ============================================
echo  Regional SonrIA - API de Estres Vocal
echo ============================================
echo.

REM Crear entorno virtual si no existe
if not exist "stress-env" (
    echo Creando entorno virtual...
    python -m venv stress-env
)

REM Activar entorno virtual
call stress-env\Scripts\activate.bat

REM Instalar dependencias
echo Instalando dependencias (esto puede tomar varios minutos)...
echo Descargando TensorFlow y otras librerias...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Fallo al instalar dependencias
    echo Verifica tu conexion a internet y vuelve a intentar
    pause
    exit /b 1
)
echo Dependencias instaladas correctamente.
echo.

REM Verificar modelo
echo Verificando modelo de deteccion...
if not exist "models\modelo_deteccion_estres.keras" (
    echo.
    echo ERROR: Modelo no encontrado en models\modelo_deteccion_estres.keras
    echo.
    echo Para entrenar el modelo:
    echo cd StressDetector\src
    echo python stress_detector_tester.py
    echo.
    pause
    exit /b 1
)

REM Ejecutar API
echo Iniciando servidor en http://localhost:8000
echo Presiona Ctrl+C para detener
echo.
python api.py

