#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ejemplo de uso de la clase StressDetectorTester
===============================================

Este archivo muestra cómo usar la clase tester para:
1. Entrenar el modelo de detección de estrés
2. Grabar audio y video simultáneamente
3. Detectar niveles de estrés en la voz
"""

from stress_detector_tester import StressDetectorTester
import time

def ejemplo_basico():
    """Ejemplo básico de uso del tester"""
    print("🎯 EJEMPLO BÁSICO DE USO DEL TESTER")
    print("=" * 50)
    
    # Crear instancia del tester
    tester = StressDetectorTester()
    
    # Opción 1: Entrenar el modelo (solo la primera vez)
    print("\n🚀 PASO 1: Entrenando el modelo...")
    print("(Esto puede tomar varios minutos)")
    
    success = tester.train_model(epochs=20, batch_size=16)
    
    if not success:
        print("❌ Error en el entrenamiento. Verifica que el dataset esté disponible.")
        return
    
    print("✅ Modelo entrenado exitosamente!")
    
    # Opción 2: Ejecutar prueba completa
    print("\n🎬 PASO 2: Ejecutando prueba completa...")
    print("Se grabará audio y video por 10 segundos y se analizará el estrés.")
    
    # Pausa para que el usuario se prepare
    input("Presiona Enter cuando estés listo para grabar...")
    
    # Ejecutar prueba
    tester.run_complete_test()
    
    # Limpiar recursos
    tester.cleanup()
    
    print("\n🎉 ¡Ejemplo completado!")

def ejemplo_avanzado():
    """Ejemplo avanzado con más control"""
    print("🚀 EJEMPLO AVANZADO DE USO DEL TESTER")
    print("=" * 50)
    
    # Crear tester con parámetros personalizados
    tester = StressDetectorTester(
        duration=15,  # 15 segundos de grabación
        sample_rate=44100,  # Mayor calidad de audio
        mfcc_features=50  # Más características MFCC
    )
    
    # Mostrar información del modelo
    print("\n📋 Información del modelo:")
    tester.get_model_info()
    
    # Entrenar con parámetros personalizados
    print("\n🏋️ Entrenando modelo con parámetros personalizados...")
    success = tester.train_model(
        epochs=40,
        batch_size=32,
        validation_split=0.3
    )
    
    if success:
        # Múltiples pruebas
        for i in range(3):
            print(f"\n🔄 PRUEBA {i+1}/3")
            input("Presiona Enter para grabar...")
            
            tester.run_complete_test()
            
            if i < 2:  # No esperar después de la última prueba
                time.sleep(2)
    
    tester.cleanup()

def ejemplo_rapido():
    """Ejemplo rápido para probar funcionalidad"""
    print("⚡ EJEMPLO RÁPIDO DEL TESTER")
    print("=" * 50)
    
    tester = StressDetectorTester()
    
    # Verificar si ya existe un modelo entrenado
    if tester.model is None:
        print("⚠️ No hay modelo entrenado. Entrenando uno rápido...")
        success = tester.train_model(epochs=10, batch_size=8)
        if not success:
            print("❌ Error en entrenamiento rápido")
            return
    else:
        print("✅ Modelo ya entrenado encontrado!")
    
    # Ejecutar prueba
    print("\n🎬 Ejecutando prueba rápida...")
    input("Presiona Enter para grabar...")
    
    tester.run_complete_test()
    tester.cleanup()

if __name__ == "__main__":
    print("🎯 SELECCIONA UN EJEMPLO:")
    print("1. Ejemplo básico (recomendado para principiantes)")
    print("2. Ejemplo avanzado (más control y opciones)")
    print("3. Ejemplo rápido (solo funcionalidad)")
    
    try:
        choice = input("\nSelecciona (1-3): ").strip()
        
        if choice == "1":
            ejemplo_basico()
        elif choice == "2":
            ejemplo_avanzado()
        elif choice == "3":
            ejemplo_rapido()
        else:
            print("❌ Opción no válida. Ejecutando ejemplo básico...")
            ejemplo_basico()
            
    except KeyboardInterrupt:
        print("\n\n👋 ¡Hasta luego!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Ejecutando ejemplo básico como respaldo...")
        ejemplo_basico()
