# Tarea 3: Fundamentos de Imágenes - Pregunta 1

Este repositorio contiene la implementación y la experimentación asociada a los algoritmos de procesamiento de imágenes de la Pregunta 1.

---

## Estructura del Repositorio

* `Pregunta 1/Implementacion.py`: Módulo que contiene las funciones base y algoritmos principales de procesamiento de imágenes.
* `Pregunta 1/Experimentacion.py`: Script de experimentación principal para la reproducción de resultados y generación de gráficos/imágenes.
* `Pregunta 1/Experimentacion2.py`: Script secundario para pruebas y análisis adicionales de sensibilidad/parámetros.
* `Pregunta 1/figures_p1/`: Carpeta donde se guardan las imágenes generadas (e.g., constant_0_gaussiano.png).

---

## Requisitos Previos

Asegúrate de tener instalado Python 3.x junto con las siguientes bibliotecas:

    pip install numpy matplotlib scipy opencv-python

---

## Instrucciones de Reproducción de Resultados

### 1. Configuración de Archivos

Navega a la carpeta de la pregunta:

    cd "Pregunta 1"

### 2. Descomentar Experimentos

Para generar las figuras y gráficos específicos en la carpeta figures_p1/:

1. Abre Experimentacion.py (o Experimentacion2.py según el caso).
2. Dirígete al bloque principal al final del archivo (if __name__ == '__main__':).
3. Descomenta las funciones o bloques de prueba asociados al experimento que deseas reproducir.

### 3. Ejecución

Ejecuta el script de experimentación deseado desde la consola:

    python Experimentacion.py

O para los experimentos secundarios:

    python Experimentacion2.py

Las figuras resultantes se actualizarán automáticamente en la carpeta figures_p1/.