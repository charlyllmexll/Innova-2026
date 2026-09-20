# Agrovision Analytics

Plataforma local para analizar imágenes de agave tomadas con dron. El proyecto tiene dos objetivos relacionados, pero independientes:

1. **Estación de anotación:** preparar, generar, revisar y validar datasets YOLO para entrenar modelos de detección.
2. **Calculadora de peso:** detectar coronas de agave, contar plantas y estimar el peso individual y total por imagen.

La separación permite que el trabajo de preparación de datos avance de forma independiente mientras se desarrolla y calibra el modelo de estimación de peso para su futura integración con un ERP en la nube.

---

## Arquitectura del repositorio

```text
Innova-2026/
├── core/
│   ├── estacion_anotacion/
│   │   ├── estacion_anotacion.py
│   │   ├── 01_explorar_imagenes.py
│   │   ├── 02_generar_anotaciones_sam.py
│   │   ├── 03_verificar_anotaciones.py
│   │   └── 04_validar_dataset.py
│   └── calculadora_peso/
│       ├── calculadora_peso.py
│       └── motor_estimacion.py
├── data/
│   ├── imagenes_crudas/       # Imágenes originales tomadas con dron
│   ├── imagenes_anotadas/     # Etiquetas YOLO en formato .txt
│   ├── imagenes_verificadas/  # Imágenes con cajas para control visual
│   ├── reports/               # Reportes de auditoría y validación
│   ├── estimaciones/          # Resultados CSV de la calculadora de peso
│   └── data.yaml              # Configuración de clases para YOLO
├── models/                    # Pesos locales de los modelos .pth
├── requirements.txt
└── README.md
```

Las rutas se resuelven desde la raíz del proyecto, por lo que los comandos pueden ejecutarse desde la carpeta `Innova-2026/` sin depender de la carpeta de trabajo actual.

---

## Requisitos del sistema

- **Sistema operativo:** Windows 10 u 11.
- **Python:** 3.11 o 3.12. Se recomienda evitar versiones superiores si alguna dependencia de PyTorch u OpenCV no cuenta todavía con ruedas compatibles.
- **RAM:** suficiente para trabajar con imágenes de alta resolución.
- **GPU recomendada:** NVIDIA RTX o superior con CUDA para acelerar SAM.
- **CPU:** el sistema puede ejecutarse en CPU, pero la generación automática de máscaras será considerablemente más lenta.
- **Espacio en disco:** el modelo SAM ViT-H ocupa aproximadamente 2.4 GB, además del espacio de las imágenes y resultados.

---

## Instalación

Abra una terminal en la raíz del proyecto y ejecute:

```bash
pip install -r requirements.txt
```

El archivo de requisitos incluye PyTorch, OpenCV, NumPy, CustomTkinter y Segment Anything.

### Instalación de PyTorch con CUDA

Si utiliza una GPU NVIDIA, instale una versión de PyTorch compatible con CUDA. Por ejemplo:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Después compruebe que PyTorch detecta CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Si el resultado es `False`, la aplicación puede funcionar en CPU, pero no tendrá aceleración CUDA.

---

## Modelo SAM

La estación de anotación utiliza **Segment Anything (SAM) ViT-Huge** para generar máscaras y apoyar el marcado de las coronas de agave.

Descargue el archivo oficial desde:

<https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth>

Después coloque el archivo exactamente en:

```text
models/sam_vit_h_4b8939.pth
```

La aplicación y el script de anotación buscan el modelo en esa ubicación. No es necesario modificar una ruta absoluta de usuario.

> **Nota de memoria:** SAM ViT-H es un modelo pesado. En GPU con poca memoria puede producir un error de memoria insuficiente durante la generación automática de máscaras. En ese caso conviene reducir la resolución de las imágenes, usar una configuración menos densa o evaluar un modelo SAM más pequeño.

---

## Estación de anotación

Esta estación está orientada a crear datasets para entrenamiento. Permite cargar lotes, navegar entre imágenes, aplicar CLAHE, marcar coronas con SAM, guardar etiquetas YOLO y ejecutar el pipeline de revisión.

### Interfaz gráfica

Desde la raíz del proyecto:

```bash
python core/estacion_anotacion/estacion_anotacion.py
```

Las anotaciones manuales o asistidas se guardan en `data/imagenes_anotadas/` como archivos `.txt` compatibles con YOLO.

### Pipeline automático del dataset

Ejecute los pasos en este orden:

#### Paso 1. Auditoría de imágenes de entrada

```bash
python core/estacion_anotacion/01_explorar_imagenes.py
```

Escanea `data/imagenes_crudas/`, verifica que las imágenes se puedan leer y revisa sus dimensiones. Genera `data/reports/exploration_report.txt`.

#### Paso 2. Segmentación y anotación automática con SAM

```bash
python core/estacion_anotacion/02_generar_anotaciones_sam.py
```

Carga SAM ViT-Huge, aplica preprocesamiento CLAHE y genera anotaciones YOLO a partir de las máscaras detectadas. Los archivos `.txt` se guardan en `data/imagenes_anotadas/`.

#### Paso 3. Control de calidad visual

```bash
python core/estacion_anotacion/03_verificar_anotaciones.py
```

Lee las etiquetas YOLO y dibuja las cajas sobre las imágenes originales. Los resultados se guardan en `data/imagenes_verificadas/` para revisar visualmente el conteo y la ubicación de las detecciones.

#### Paso 4. Certificación del dataset

```bash
python core/estacion_anotacion/04_validar_dataset.py
```

Comprueba que las imágenes y sus archivos de etiquetas estén emparejados. Genera `data/reports/validation_report.txt` con el dictamen del lote.

---

## Calculadora de peso

Esta estación representa el flujo de negocio del proyecto: procesar imágenes de dron para estimar el tamaño de la corona, contar plantas y calcular un peso aproximado por planta.

### Interfaz gráfica

```bash
python core/calculadora_peso/calculadora_peso.py
```

La interfaz permite:

- cargar las imágenes de `data/imagenes_crudas/`;
- navegar entre las imágenes del lote;
- consultar el número de plantas detectadas por imagen;
- consultar el peso promedio por planta;
- consultar el peso total estimado de cada imagen.

### Ejecución por línea de comandos

```bash
python core/calculadora_peso/motor_estimacion.py
```

También puede especificar las rutas y la escala de la imagen:

```bash
python core/calculadora_peso/motor_estimacion.py --input data/imagenes_crudas --output data/estimaciones --scale 0.05
```

Parámetros disponibles:

- `--input`: carpeta de imágenes de entrada.
- `--output`: carpeta donde se guardará el CSV.
- `--scale`: metros por píxel. Debe ajustarse según la altura del dron y la cámara.
- `--min-area`: área mínima en píxeles para aceptar una detección.

El resultado se exporta como:

```text
data/estimaciones/estimacion_peso_agave.csv
```

---

## Precisión y calibración

La fórmula actual de peso es un modelo inicial basado en el diámetro equivalente de la corona. Es útil para validar el flujo completo y obtener resultados comparables, pero todavía debe calibrarse con datos reales de campo.

Para mejorar la precisión se necesitarán, como mínimo:

1. diámetro de corona medido en campo;
2. peso real de las plantas cosechadas;
3. altura y parámetros de la cámara del dron;
4. varias muestras tomadas en diferentes tamaños y condiciones de iluminación.

Con esos datos se podrá ajustar la relación entre tamaño de corona y peso antes de enviar resultados a un ERP.

---

## Flujo recomendado

```text
Imágenes de dron
      |
      +--> Estación de anotación --> Dataset YOLO --> Entrenamiento futuro
      |
      +--> Calculadora de peso --> Conteo y peso estimado --> Exportación CSV / ERP
```
