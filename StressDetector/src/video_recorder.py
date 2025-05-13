import cv2
import threading
import time
import os


#esta clase nos ayuda a manejar la grabación de video usando openCV
class VideoRecorder:  
    def __init__(self, filename="recordings/video.avi", fps=20.0, resolution=(640, 480)):
        # Crear el directorio si no existe
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        self.filename = filename
        self.fps = fps
        self.resolution = resolution
        self.recording = False
        
        # Inicializar la cámara
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, resolution[0])  # ancho
        self.cap.set(4, resolution[1])  # alto
        
        # Configurar el escritor de video
        self.out = cv2.VideoWriter(
            self.filename,
            cv2.VideoWriter_fourcc(*'XVID'),
            fps,
            resolution
        )
        
    def start(self):
        """Inicia la grabación de video"""
        if not self.cap.isOpened():
            print("❌ Error: No se puede acceder a la cámara")
            return
            
        self.recording = True
        threading.Thread(target=self._record, daemon=True).start()
        print("📹 Cámara iniciada")
        
    def _record(self):
        while self.recording:
            ret, frame = self.cap.read()
            if ret:
                self.out.write(frame)
                cv2.imshow("Grabando...", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    def stop(self):
        """Detiene la grabación de video"""
        self.recording = False
        time.sleep(1)
        if self.cap.isOpened():
            self.cap.release()
        if self.out:
            self.out.release()
        cv2.destroyAllWindows()
        print("📹 Grabación de video finalizada")