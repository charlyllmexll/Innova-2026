import os
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# =====================================================================
# 1. CONFIGURACIÓN DE INFRAESTRUCTURA Y HARDWARE
# =====================================================================
# Definimos las rutas exactas alineadas a su nueva arquitectura de carpetas
RUTA_MODELO_SAM    = r"C:\Users\charl\Downloads\sam_vit_h_4b8939.pth" # El archivo pesado de 2.4 GB
TIPO_MODELO_SAM    = "vit_h"                         # Indicamos que es la versión Huge (Máxima precisión)
DIR_IMAGENES_CRUDAS = "data/imagenes_crudas"          # De aquí leemos las fotos
DIR_ETIQUETAS_YOLO  = "data/imagenes_anotadas"        # Aquí se guardan los .txt para YOLO

# Nos aseguramos de que la carpeta de salida exista en el disco
os.makedirs(DIR_ETIQUETAS_YOLO, exist_ok=True)

print("[AGROVISIÓN CORE] Inicializando el motor de Inteligencia Artificial...")

# Forzar el uso de la tarjeta de video dedicada NVIDIA (CUDA) si los drivers están listos
DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Cargando pesos en el dispositivo: {DISPOSITIVO.upper()}")

# Cargamos la arquitectura de SAM y le inyectamos los pesos pesados
sam = sam_model_registry[TIPO_MODELO_SAM](checkpoint=RUTA_MODELO_SAM)
sam.to(device=DISPOSITIVO)

# MEJORA 1: Configuración Ultra Densa del Muestreo de SAM
# Modificamos los parámetros internos del algoritmo para agaves a 20m de altura
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=64,           # Sube el muestreo a 4,096 puntos (Antes 32/1,024) para pencas finas
    pred_iou_thresh=0.88,         # Exigimos un 88% de confianza en la consistencia geométrica
    stability_score_thresh=0.95    # Solo acepta máscaras estables ante sutiles variaciones de brillo
)

# =====================================================================
# 2. PIPELINE DE PROCESAMIENTO EN BUCLE (IMÁGENES)
# =====================================================================
archivos_en_carpeta = os.listdir(DIR_IMAGENES_CRUDAS)
imagenes_validas = [f for f in archivos_en_carpeta if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(f"[INFO] Iniciando el procesamiento de {len(imagenes_validas)} imágenes...")

for index, nombre_img in enumerate(imagenes_validas, start=1):
    ruta_imagen = os.path.join(DIR_IMAGENES_CRUDAS, nombre_img)
    print(f"\n[{index}/{len(imagenes_validas)}] Triturando -> {nombre_img}")
    
    img_original = cv2.imread(ruta_imagen)
    if img_original is None:
        print(f"[ADVERTENCIA] No se pudo leer la imagen: {nombre_img}")
        continue
        
    alto_img, ancho_img, _ = img_original.shape
    
    # MEJORA 2: Filtro de Contraste Adaptativo CLAHE (Preprocesamiento Óptico)
    # Convertimos a LAB para manipular únicamente la luz, sin alterar el color real
    lab = cv2.cvtColor(img_original, cv2.COLOR_BGR2LAB)
    capa_l, capa_a, capa_b = cv2.split(lab)
    
    # El ecualizador divide la foto en bloques de 8x8 píxeles aumentando el contraste local
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    capa_l_optimizada = clahe.apply(capa_l)
    
    # Fusionamos de vuelta y regresamos al formato estándar de OpenCV (BGR)
    lab_optimizado = cv2.merge((capa_l_optimizada, capa_a, capa_b))
    img_para_ia = cv2.cvtColor(lab_optimizado, cv2.COLOR_LAB2BGR)
    
    # Entregamos la imagen con el contraste mejorado al cerebro de SAM
    print("[SAM] Ejecutando segmentación geométrica...")
    mascaras_detectadas = mask_generator.generate(img_para_ia)
    
    # Preparamos el archivo de salida .txt que leerá YOLO en el futuro
    nombre_base = os.path.splitext(nombre_img)[0]
    ruta_salida_txt = os.path.join(DIR_ETIQUETAS_YOLO, f"{nombre_base}.txt")
    
    lineas_yolo = []
    
    # Analizamos objeto por objeto lo que encontró la IA
    for obj in mascaras_detectadas:
        # MEJORA 3: Filtro Geométrico por Tamaño Real (Destrucción de Falsos Positivos)
        area_en_pixeles = obj['area']
        
        LIMITE_MIN_HIJUELO = 400
        LIMITE_MAX_MADURO  = 25000
        
        # Si el objeto es muy chico (maleza/piedras) o gigante (árbol/sombra), se ignora
        if not (LIMITE_MIN_HIJUELO <= area_en_pixeles <= LIMITE_MAX_MADURO):
            continue
            
        # Si pasó el filtro, extraemos su caja delimitadora (Bounding Box)
        # OpenCV entrega: [X_mínimo, Y_mínimo, Ancho_caja, Alto_caja] en píxeles absolutos
        x_min, y_min, ancho_caja, alto_caja = obj['bbox']
        
        # TRANSFORMACIÓN MATEMÁTICA: Normalización al formato estricto de YOLO
        # YOLO exige: ID_Clase, Centro_X, Centro_Y, Ancho, Alto (Todo mapeado de 0.0 a 1.0)
        centro_x = (x_min + (ancho_caja / 2)) / ancho_img
        centro_y = (y_min + (alto_caja / 2)) / alto_img
        ancho_normalizado = ancho_caja / ancho_img
        alto_normalizado  = alto_caja / alto_img
        
        # Clase 0 = 'agave' (Configurado de forma idéntica en su data.yaml)
        linea_formateada = f"0 {centro_x:.6f} {centro_y:.6f} {ancho_normalizado:.6f} {alto_normalizado:.6f}\n"
        lineas_yolo.append(linea_formateada)
        
    # Escribimos los resultados limpios en el archivo .txt correspondiente
    with open(ruta_salida_txt, "w", encoding="utf-8") as f_txt:
        f_txt.writelines(lineas_yolo)
        
    print(f"[ÉXITO] Generadas {len(lineas_yolo)} etiquetas válidas de agave para esta imagen.")

print("\n[PROCESO COMPLETADO] La estación de etiquetado procesó el lote correctamente.")
