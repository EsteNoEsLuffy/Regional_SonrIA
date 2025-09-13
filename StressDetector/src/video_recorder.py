import cv2
import threading
import time


"""
modulo de grabación de video

"""
class VideoRecoder: 
    def __init__(self, filename= "recodings/video.avi", fps=20.0, resolution=(640, 480)):
        
        self.filename=filename
        self.fps=fps
        self.resolution=resolution
        self.cap=cv2.VideoCapture(0)
        self.cap.set(3, resolution[0] ) #ancho
        self.cap.set(4, resolution[1] ) #alto
        self.out= cv2.VideoWriter(self.filename, cv2.VideoWriter_fourcc (*'XVID'), fps, resolution)
        self.recording=False
        
    def start(self):
        self.recording=True
        threading.Thread(target=self._record, daemon=True).start()
        
    def _record(self):
        while self.recording:
            ret, frame= self.cap.read()
            if ret:
                self.out.write(frame)
                cv2.imshow("Grabando...", frame)
            if cv2.waitKey(1)& 0xFF == ord('q'):
                break
            
    def stop (self):
        self.recording=False
        time.sleep(1) #hilos 
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        
        #se guarda en formato .avi