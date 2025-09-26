---
title: Regional SonrIA API
emoji: 🎧
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# Regional SonrIA - API de Detección de Estrés Vocal

API simple para detectar niveles de estrés en la voz usando FastAPI.

## Inicio Rápido

1. **Ejecutar el script de inicio:**
   ```bash
   start_api.bat
   ```

2. **Enviar archivo de audio:**
   ```python
   import requests

   files = {'file': open('audio.wav', 'rb')}
   response = requests.post('http://localhost:8000/analyze', files=files)
   resultado = response.json()

   print(f"Nivel de estrés: {resultado['stress_level']}")
   ```

## Respuesta de la API

```json
{
  "success": true,
  "stress_score": 0.2345,
  "stress_level": "BAJO",
  "duration": 3.45
}
```

## Niveles de Estrés

- **BAJO**: 0.0 - 0.3 (Voz normal)
- **MEDIO**: 0.3 - 0.6 (Atención)
- **ALTO**: 0.6 - 0.8 (Alerta)
- **MUY_ALTO**: 0.8 - 1.0 (Crítico)

## Formatos Soportados

WAV, MP3, M4A, FLAC, OGG (máx. 10MB)

## Requisitos

- Python 3.8+
- Modelo entrenado en `models/modelo_deteccion_estres.keras`
