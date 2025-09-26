import numpy as np

# Cargar dataset
data = np.load('processed_dataset.npz')
X = data['X']
y = data['y']

print("=== ANÁLISIS DEL DATASET ===")
print(f"Muestras totales: {X.shape[0]}")
print(f"Características por muestra: {X.shape[1]}")
print(f"Distribución de clases: {np.unique(y, return_counts=True)}")

# Calcular porcentajes
unique, counts = np.unique(y, return_counts=True)
for cls, count in zip(unique, counts):
    percentage = (count / len(y)) * 100
    print(f"Clase {int(cls)}: {count} muestras ({percentage:.1f}%)")

print("\nRango de características:")
print(f"X.min: {X.min():.4f}")
print(f"X.max: {X.max():.4f}")
print(f"X.mean: {X.mean():.4f}")

print("\nPrimeras 10 etiquetas:")
print(y[:10])

print("\nEstadísticas de características:")
print(f"Media: {np.mean(X):.4f}")
print(f"Desviación estándar: {np.std(X):.4f}")
