import os
import time
from video_recorder import VideoRecorder
from audio_recorder import AudioRecorder

def main():
    DURACION_GRABACION = 300  # 5 minutos
     
    # Crear directorio para grabaciones
    os.makedirs("recordings", exist_ok=True)
    
    print("=== SISTEMA DE GRABACIÓN DE AUDIO Y VIDEO ===")
    print("Duración de grabación: 5 minutos")
    
    try:
        # Crear nombre único para esta sesión de grabación
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_path = f"recordings/video_{timestamp}.avi"
        audio_path = f"recordings/audio_{timestamp}.wav"
         
        # Inicializar grabadores
        video_recorder = VideoRecorder(video_path)
        audio_recorder = AudioRecorder(audio_path, duration=DURACION_GRABACION)
        
        grabacion_activa = False
         
        print("\n=== CONTROLES ===")
        print("1. 'i' - Iniciar grabación")
        print("2. 'd' - Detener grabación")
        print("3. 's' - Salir")
        
        while True:
            cmd = input("\n>> ").strip().lower()
            
            if cmd == 's':
                if grabacion_activa:
                    print("Por favor, detén la grabación antes de salir (presiona 'd')")
                    continue
                print("\nFinalizando programa...")
                break
                
            elif cmd == 'i':
                if grabacion_activa:
                    print("Ya hay una grabación en curso")
                    continue
                
                print("\nIniciando grabación de 5 minutos...")
                grabacion_activa = True
                
                # Iniciar grabaciones simultáneas
                video_recorder.start()
                time.sleep(0.5)  # Pequeña pausa para que la cámara esté lista
                audio_recorder.start()
                
                # Mostrar tiempo restante
                start_time = time.time()
                while grabacion_activa and audio_recorder.recording:
                    elapsed = time.time() - start_time
                    remaining = max(0, DURACION_GRABACION - elapsed)
                    minutes = int(remaining // 60)
                    seconds = int(remaining % 60)
                    print(f"\rTiempo restante: {minutes:02d}:{seconds:02d}", end="")
                    time.sleep(0.1)
                
            elif cmd == 'd':
                if not grabacion_activa:
                    print("No hay grabación activa para detener")
                    continue
                
                print("\nDeteniendo grabación...")
                # Detener ambas grabaciones simultáneamente
                audio_recorder.stop()
                video_recorder.stop()
                grabacion_activa = False
                
                print("\nGrabación completada y guardada:")
                print(f"Video: {video_path}")
                print(f"Audio: {audio_path}")
                
                # Crear nueva sesión de grabación
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                video_path = f"recordings/video_{timestamp}.avi"
                audio_path = f"recordings/audio_{timestamp}.wav"
                video_recorder = VideoRecorder(video_path)
                audio_recorder = AudioRecorder(audio_path, duration=DURACION_GRABACION)
            
    except Exception as e:
        print(f"\nError: {str(e)}")
    finally:
        # Limpieza final
        if 'video_recorder' in locals():
            video_recorder.stop()
        if 'audio_recorder' in locals():
            audio_recorder.stop()
        print("\nPrograma finalizado")

if __name__ == "__main__":
    main()
    
    """
    
    Grabará tanto audio como video durante 5 minutos
Mostrará un contador de tiempo restante en formato minutos:segundos
Permitirá detener la grabación en cualquier momento con 's'
Analizará el nivel de estrés al finalizar la grabación
Guardará ambos archivos en la carpeta recordings
Recomendaciones de uso:
Asegúrate de tener suficiente espacio en disco (especialmente para el video)
La ventana de la cámara se mostrará durante toda la grabación
Puedes detener en cualquier momento con 's'
El análisis de estrés se realizará sobre todo el audio grabado
    

Para usar el sistema:


Asegúrate de tener todas las dependencias instaladas:

pip install opencv-python tensorflow numpy librosa sounddevice matplotlib


python main.py
    
    
    
    """
    