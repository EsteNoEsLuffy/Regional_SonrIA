import os
import sys

# Agregar el directorio actual al path para importar stress_detector_api
sys.path.append('.')

try:
    from stress_detector_api import StressDetectorAPI

    # Crear instancia del detector
    detector = StressDetectorAPI()

    if detector.is_model_ready():
        print("=== INFORMACIÓN DEL MODELO ===")
        print(f"Modelo cargado: ✅")
        print(f"Input shape: {detector.model.input_shape}")
        print(f"Output shape: {detector.model.output_shape}")
        print(f"Capas: {len(detector.model.layers)}")
        print(f"Parámetros: {detector.model.count_params()}")

        print("\n=== ARQUITECTURA ===")
        for i, layer in enumerate(detector.model.layers):
            print(f"Capa {i+1}: {layer.__class__.__name__} - {layer.output_shape}")

        print("\n=== UMBRALES DE CLASIFICACIÓN ===")
        print(f"Bajo: < {detector.stress_thresholds['bajo']}")
        print(f"Medio: {detector.stress_thresholds['bajo']} - {detector.stress_thresholds['medio']}")
        print(f"Alto: {detector.stress_thresholds['medio']} - {detector.stress_thresholds['alto']}")
        print(f"Muy Alto: > {detector.stress_thresholds['alto']}")

    else:
        print("❌ Modelo no cargado")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
