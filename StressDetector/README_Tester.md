# 🎯 StressDetectorTester - Clase Completa para Detección de Estrés Vocal

## 📋 Descripción

La clase `StressDetectorTester` es una implementación completa e integrada que combina todas las funcionalidades del detector de estrés vocal en una sola clase fácil de usar. Permite:

1. **🏋️ Entrenar el modelo** usando el dataset pre-procesado
2. **🎬 Grabar audio y video simultáneamente** por 10 segundos (configurable)
3. **🔍 Extraer características MFCC** del audio grabado
4. **📊 Detectar niveles de estrés** (BAJO, MEDIO, ALTO, MUY ALTO)
5. **📈 Generar visualizaciones** de los análisis

## 🚀 Características Principales

- **Grabación simultánea**: Audio y video se graban al mismo tiempo usando hilos separados
- **Análisis automático**: El audio se analiza automáticamente después de cada grabación
- **Clasificación de estrés**: 4 niveles de estrés con umbrales configurables
- **Visualizaciones**: Gráficos MFCC y análisis de características
- **Interfaz simple**: Métodos fáciles de usar para diferentes escenarios

## 📦 Instalación y Dependencias

Asegúrate de tener instaladas las siguientes librerías:

```bash
pip install tensorflow librosa sounddevice opencv-python matplotlib numpy
```

## 🎮 Uso Básico

### 1. Crear Instancia

```python
from stress_detector_tester import StressDetectorTester

# Crear tester con parámetros por defecto
tester = StressDetectorTester()

# O con parámetros personalizados
tester = StressDetectorTester(
    duration=15,           # 15 segundos de grabación
    sample_rate=44100,     # Mayor calidad de audio
    mfcc_features=50       # Más características MFCC
)
```

### 2. Entrenar el Modelo

```python
# Entrenar con parámetros por defecto
success = tester.train_model()

# O con parámetros personalizados
success = tester.train_model(
    epochs=50,           # 50 épocas
    batch_size=32,       # Batch size de 32
    validation_split=0.3 # 30% para validación
)
```

### 3. Ejecutar Prueba Completa

```python
# Ejecuta grabación + análisis automáticamente
tester.run_complete_test()
```

### 4. Limpiar Recursos

```python
# Siempre limpia al final
tester.cleanup()
```

## 🔧 Métodos Principales

### Entrenamiento
- `train_model(epochs, batch_size, validation_split)`: Entrena el modelo

### Grabación
- `start_simultaneous_recording()`: Inicia grabación de audio y video
- `stop_recording()`: Detiene grabación y guarda archivos
- `run_complete_test()`: Ejecuta prueba completa automáticamente

### Análisis
- `_analyze_audio_for_stress(audio_file)`: Analiza audio para detectar estrés
- `_classify_stress_level(score)`: Clasifica nivel de estrés

### Utilidades
- `get_model_info()`: Muestra información del modelo
- `cleanup()`: Limpia recursos y conexiones

## 📊 Clasificación de Estrés

La clase clasifica el estrés en 4 niveles:

| Nivel | Rango | Descripción |
|-------|-------|-------------|
| **BAJO** | 0.0 - 0.3 | Voz normal, sin estrés |
| **MEDIO** | 0.3 - 0.6 | Atención, posible estrés |
| **ALTO** | 0.6 - 0.8 | Alerta, estrés significativo |
| **MUY ALTO** | 0.8 - 1.0 | Crítico, estrés severo |

Los umbrales son configurables en `self.stress_thresholds`.

## 📁 Archivos Generados

Cada grabación genera:

- **Audio**: `recordings/audio_YYYYMMDD_HHMMSS.wav`
- **Video**: `recordings/video_YYYYMMDD_HHMMSS.avi`
- **Análisis**: `recordings/mfcc_analysis_YYYYMMDD_HHMMSS.png`

## 🎯 Ejemplos de Uso

### Ejemplo 1: Uso Simple

```python
from stress_detector_tester import StressDetectorTester

tester = StressDetectorTester()

# Entrenar modelo
if tester.train_model():
    # Ejecutar prueba
    tester.run_complete_test()

tester.cleanup()
```

### Ejemplo 2: Múltiples Pruebas

```python
tester = StressDetectorTester()

# Entrenar una vez
tester.train_model(epochs=30)

# Múltiples pruebas
for i in range(5):
    print(f"Prueba {i+1}/5")
    tester.run_complete_test()
    time.sleep(2)

tester.cleanup()
```

### Ejemplo 3: Control Manual

```python
tester = StressDetectorTester()

# Entrenar modelo
tester.train_model()

# Control manual de grabación
tester.start_simultaneous_recording()
time.sleep(10)  # Grabar por 10 segundos
tester.stop_recording()

tester.cleanup()
```

## ⚠️ Consideraciones Importantes

1. **Dataset**: Asegúrate de que `processed_dataset.npz` esté disponible
2. **Cámara**: Se requiere acceso a la cámara web
3. **Micrófono**: Se requiere acceso al micrófono
4. **Recursos**: Siempre llama `cleanup()` al final
5. **Primera vez**: Debes entrenar el modelo antes de usarlo

## 🐛 Solución de Problemas

### Error: "No se pudo acceder a la cámara"
- Verifica que la cámara web esté conectada y funcionando
- Cierra otras aplicaciones que puedan estar usando la cámara

### Error: "Dataset no encontrado"
- Ejecuta primero `load_dataset.py` para descargar y procesar los datos
- Verifica que `processed_dataset.npz` esté en el directorio correcto

### Error: "Modelo no disponible"
- Entrena el modelo primero con `train_model()`
- Verifica que el archivo del modelo se haya guardado correctamente

### Grabación lenta o interrumpida
- Verifica que no haya otros procesos usando CPU intensivamente
- Reduce la duración de grabación si es necesario

## 🔮 Funcionalidades Futuras

- [ ] Soporte para múltiples cámaras
- [ ] Análisis en tiempo real
- [ ] Exportación de resultados a CSV/JSON
- [ ] Interfaz gráfica (GUI)
- [ ] Soporte para diferentes formatos de audio/video

## 📞 Soporte

Para problemas o preguntas:
1. Verifica que todas las dependencias estén instaladas
2. Revisa los mensajes de error en la consola
3. Asegúrate de que el dataset esté disponible
4. Verifica permisos de cámara y micrófono

## 🎉 ¡Listo para Usar!

La clase `StressDetectorTester` está diseñada para ser fácil de usar y potente. Con solo unas pocas líneas de código puedes entrenar un modelo de detección de estrés y realizar análisis completos de audio y video.

¡Disfruta detectando el estrés vocal! 🎵🎬
