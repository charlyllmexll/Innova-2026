# Agrovisión Analytics - Herramienta de Anotación Automatizada 🌵

Este repositorio contiene la **Estación de Etiquetado** de Agrovisión Analytics, desarrollada específicamente para resolver los altos costos de las plataformas de anotación comercial. El objetivo exclusivo de este ecosistema de scripts es automatizar la creación de datasets masivos, limpios y precisos en formato **YOLO**, utilizando visión artificial de última generación.

El dataset generado mediante esta herramienta será el insumo de oro para entrenar el modelo predictivo final.

---

## 🏗️ Arquitectura del Repositorio

La estructura del proyecto sigue un estándar internacional de código abierto para desarrollo en Inteligencia Artificial, separando estrictamente la lógica de procesamiento, la gestión de datos ópticos y los pesos de los modelos:

```text
Innova-2026/
├── core/                # Lógica pura del pipeline de etiquetado (Scripts de Python)
├── data/                # Contenedor unificado de datos y configuraciones del dataset
│   ├── annotated_labels/# Archivos de salida .txt en formato YOLO normalizado
│   ├── raw_inputs/      # Imágenes crudas del dron capturadas a 20 metros de altura
│   ├── reports/         # Reportes de auditoría y diagnóstico generados por el sistema
│   ├── verified_outputs/# Control de calidad visual (Imágenes con siluetas dibujadas)
│   └── data.yaml        # Archivo de traducción de clases e indexación para YOLO
└── models/              # Almacenamiento local de los pesos del modelo de IA (.pth)
```

---

## 🛠️ Requisitos del Sistema y Hardware

Para garantizar la máxima precisión y velocidad en el procesamiento de imágenes de alta resolución, se requiere la siguiente configuración:

*   **Sistema Operativo:** Windows 10/11 (Optimizado para arquitectura local).
*   **Entorno:** Python 3.11 o Python 3.12 (Evitar versiones superiores por incompatibilidad de compilación en librerías matemáticas).
*   **Hardware Recomendado:** Tarjeta gráfica dedicada NVIDIA (Serie RTX o superior) con arquitectura de núcleos **CUDA** activa para aceleración por hardware.
*   **Modelo Base de IA:** Archivo de pesos de Meta AI: `sam_vit_h_4b8939.pth` (2.4 GB), el cual debe colocarse manualmente dentro de la carpeta `models/`, debe descargarse desde el siguiente enlace: "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth".
*   **Librerias y dependencias:** Debe instalar las librerias incluidas en el archivo requirements.txt desde la linea de comandos con la siguiente linea: 
```bash
pip install -r requirements.txt
```
*   **TORCH:** Es indispensable instalar la version de torch para CUDA de lo contrario se usara CPU en lugar de GPU, puede hacerlo con la siguiente linea de comandos(se incluye tambien en los requerimientos):

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

```

---

## 🚀 Manual de Operación del Pipeline (Paso a Paso)

El procesamiento de imágenes para la creación del dataset se ejecuta de forma secuencial. Abra la terminal en la raíz del proyecto y ejecute los comandos estrictamente en el siguiente orden:

### Paso 1: Auditoría de Imágenes de Entrada
```bash
python core/001_Explorar_imagenes_crudas.py
```
*   **¿Qué hace?:** Escanea la carpeta `data/imagenes_crudas/`. Realiza un control de calidad óptico validando las dimensiones de las fotos del dron (capturadas a 20m) y sus formatos (.png, .jpg). Genera un diagnóstico inicial que se guarda en `data/reports/exploration_report.txt` para asegurar que el lote de imágenes sea apto para la IA.

### Paso 2: Segmentación y Anotación Automatizada por IA
```bash
python core/002_Auto_anotar_con_SAM.py
```
*   **¿Qué hace?:** El núcleo automatizado del sistema. Carga el modelo pesado **SAM ViT-Huge** en la GPU (NVIDIA CUDA). Aplica un preprocesamiento de contraste adaptativo (**CLAHE**) para separar las pencas del agave del suelo arcilloso. Ejecuta un grid denso de muestreo (64 puntos) para delimitar la corona y aplica un filtro geométrico estricto (ignora objetos menores a 400px o mayores a 25,000px) para destruir falsos positivos como maleza o piedras. Exporta las coordenadas calculadas en formato YOLO normalizado (`.txt`) dentro de `data/imagenes_anotadas/`.

### Paso 3: Control de Calidad Visual (QA Operador)
```bash
python core/03_verificar.py
```
*   **¿Qué hace?:** Una herramienta de seguridad para el operador. Lee las coordenadas guardadas en el paso anterior y utiliza OpenCV para dibujar físicamente las siluetas y cajas delimitadoras sobre una copia de las imágenes originales, guardando el resultado en `data/imagenes_verificadas/`. Esto permite al usuario auditar visualmente el rendimiento de la IA antes de validar el lote.

### Paso 4: Certificación Final del Dataset
```bash
python core/04_validar.py
```
*   **¿Qué hace?:** El cierre del pipeline técnico. Realiza un cruce de datos integral comprobando que cada imagen en `data/imagenes_crudas/` tenga su correspondiente archivo de etiquetas en `data/imagenes_anotadas/` perfectamente estructurado, sin archivos corruptos y alineado con el mapeo del archivo `data/data.yaml`. Emite el certificado final de exportación en `data/reports/validation_report.txt`, dejando el dataset 100% listo para el entrenamiento del modelo predictivo de Agrovisión Analytics.
