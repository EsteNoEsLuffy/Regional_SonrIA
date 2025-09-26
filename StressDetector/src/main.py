# from src.video_recorder import VideoRecoder
# from src.audio_recorder import AudioRecorder
# from src.stress_detector import detect_stress

import sys
import os
import time
import gradio as gr


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from video_recorder import VideoRecoder
from audio_recorder import AudioRecorder
from stress_detector import Stress_Detector

def main():
    video_recorder = None
    audio_recorder = None
    
    try:
        print("=== SISTEMA DE MONITOREO DE ESTRÉS VOCAL ===")
        print("Configurando componentes...")

        # Inicializa componentes
        video_recorder = VideoRecoder()
        audio_recorder = AudioRecorder(
            stress_model="models/modelo_deteccion_estres.keras",
            stress_threshold=0.6,
            duration=10
        )

        if not audio_recorder.model:
            print("\n¡ADVERTENCIA! El análisis de estrés no estará disponible")
            print("Motivo: No se pudo cargar el modelo de detección\n")

        print("\nOpciones:")
        print("- Presiona Enter para nuevo análisis")
        print("- Escribe 'q' para salir\n")

        # Inicia grabaciones
        video_recorder.start()
        audio_recorder.start()

        while True:
            cmd = input(">> ").lower()
            if cmd == "q":
                break
                
            # Reinicia completamente la grabación
            audio_recorder.stop()
            audio_recorder.start()
            
            # Espera el tiempo de grabación + margen
            time.sleep(audio_recorder.duration + 0.5)
        
    except KeyboardInterrupt:
        print("\nInterrupción por usuario")
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Limpieza de recursos
        print("\nDeteniendo sistema...")
        if audio_recorder:
            audio_recorder.stop()
        if video_recorder:
            video_recorder.stop()
        print("Sistema detenido correctamente")

if __name__ == "__main__":
    main()