import requests
import os
import librosa
import numpy as np
from tqdm import tqdm  # Para barra de progreso

DATASET_URL = "https://datasets-server.huggingface.co/rows?dataset=macabdul9%2FStressDetection_MIRSD&config=default&split=test&offset=0&length=100"
DOWNLOAD_DIR = "dataset_audios"
SAMPLE_RATE = 22050
MFCC_FEATURES = 40

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def download_and_extract_data():
    try:
        print("Conectando con el servidor...")
        response = requests.get(DATASET_URL)
        response.raise_for_status()  # Verifica errores HTTP
        data = response.json()
        
        X = []
        y = []
        
        print(f"Procesando {len(data['rows'])} muestras...")
        
        for row in tqdm(data["rows"], desc="Descargando audios"):
            try:
                audio_url = row["row"]["audio"][0]["src"]
                label_str = row["row"]["label"]
                
                # Convertir etiqueta a numérico
                label_map = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
                label = label_map.get(label_str.lower(), 0)
                
                filename = os.path.join(DOWNLOAD_DIR, f"{row['row_idx']}.wav")
                
                # Descargar audio
                audio_data = requests.get(audio_url).content
                with open(filename, "wb") as f:
                    f.write(audio_data)
                
                # Procesar audio
                audio, sr = librosa.load(filename, sr=SAMPLE_RATE)
                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=MFCC_FEATURES)
                mfcc_mean = np.mean(mfcc.T, axis=0)
                
                X.append(mfcc_mean)
                y.append(int(label > 2))  # 0-2 = no estrés, 3-5 = estrés
                
            except Exception as e:
                print(f"\nError procesando muestra {row['row_idx']}: {str(e)}")
                continue
        
        # Guardar dataset procesado
        X = np.array(X)
        y = np.array(y)
        np.savez("processed_dataset.npz", X=X, y=y)
        
        print(f"\n✅ Dataset procesado correctamente:")
        print(f"- Muestras: {len(X)}")
        print(f"- Características por muestra: {MFCC_FEATURES}")
        print(f"- Archivos guardados en: {DOWNLOAD_DIR}/")
        print(f"- Dataset final: processed_dataset.npz")
        
    except Exception as e:
        print(f"\nError grave: {str(e)}")
        return False

if __name__ == "__main__":
    download_and_extract_data()