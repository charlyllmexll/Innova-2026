# ============================================================
#  SCRIPT 2: Verificación visual de anotaciones YOLO
#
#  ¿Para qué sirve?
#  Después de correr el script de auto-anotación, necesitas
#  verificar que las máscaras y bounding boxes quedaron bien.
#  Este script dibuja las anotaciones encima de las imágenes
#  para que puedas revisarlas visualmente antes de entrenar.
#
#  ¿Qué genera?
#  - Una carpeta "verified/" con imágenes anotadas visualmente
#  - Cada imagen muestra: bounding boxes en verde,
#    máscaras de segmentación semitransparentes en azul,
#    y el conteo de plantas detectadas
#  - Un reporte de cuántas anotaciones tiene cada imagen
#
#  ¿Cuándo usarlo?
#  1. Primero: corre el script de anotación sobre 5-10 imágenes
#  2. Luego: corre ESTE script para revisar el resultado
#  3. Ajusta MIN_AREA, MAX_AREA, MIN_SOLIDITY si hace falta
#  4. Cuando estés conforme, procesa las 500 imágenes
# ============================================================

import os
import cv2
import numpy as np
from tqdm import tqdm

# ─────────────────────────────────────────────
#  CONFIGURACIÓN
# ─────────────────────────────────────────────

DATASET_DIR  = r"D:\Nube Mega\Trabajo\programitas\python\Marcado de imagenes\dataset"      # carpeta generada por el script de anotación
OUTPUT_DIR   = r"D:\Nube Mega\Trabajo\programitas\python\Marcado de imagenes\verified"     # carpeta donde se guardarán las imágenes verificadas
SPLIT        = "train"        # "train" o "val" — qué split revisar
MAX_IMAGES   = 20             # cuántas imágenes verificar (None = todas)

# Colores en formato BGR (Blue, Green, Red) — así trabaja OpenCV internamente
COLOR_BBOX   = (0, 255, 0)    # verde para bounding boxes
COLOR_MASK   = (255, 100, 0)  # azul para máscaras
COLOR_TEXT   = (255, 255, 255)# blanco para texto
COLOR_BG     = (0, 0, 0)      # negro para fondo del texto
MASK_ALPHA   = 0.35           # transparencia de la máscara (0=invisible, 1=sólido)

# ─────────────────────────────────────────────
#  LECTURA DE ANOTACIONES YOLO
# ─────────────────────────────────────────────

def parse_yolo_line(line, img_w, img_h):
    """
    Convierte una línea del archivo .txt de YOLO a coordenadas reales en píxeles.
    
    Formato YOLO-seg:
        clase cx cy bw bh x1 y1 x2 y2 ... xn yn
    
    Todo está normalizado entre 0 y 1 (relativo al tamaño de la imagen).
    Para obtener píxeles reales multiplicamos por el ancho/alto de la imagen.
    
    Ejemplo:
        línea: "0 0.5 0.5 0.1 0.15 0.45 0.42 0.55 0.42 ..."
        clase: 0  (agave)
        cx, cy: centro del bounding box = 0.5 * img_w, 0.5 * img_h
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None  # línea malformada

    clase = int(parts[0])

    # Bounding box: las primeras 4 coordenadas después de la clase
    cx = float(parts[1]) * img_w
    cy = float(parts[2]) * img_h
    bw = float(parts[3]) * img_w
    bh = float(parts[4]) * img_h

    # Convertir de formato (cx, cy, w, h) a (x1, y1, x2, y2)
    # que es lo que necesita cv2.rectangle()
    x1 = int(cx - bw / 2)
    y1 = int(cy - bh / 2)
    x2 = int(cx + bw / 2)
    y2 = int(cy + bh / 2)

    bbox = (x1, y1, x2, y2)

    # Polígono de segmentación: el resto de los valores (pares x,y)
    seg_values = list(map(float, parts[5:]))

    # Convertir a coordenadas reales en píxeles
    # zip(seg_values[::2], seg_values[1::2]) agrupa los valores de dos en dos:
    # [x1_norm, y1_norm, x2_norm, y2_norm] → [(x1_norm, y1_norm), (x2_norm, y2_norm)]
    polygon = [
        (int(x * img_w), int(y * img_h))
        for x, y in zip(seg_values[::2], seg_values[1::2])
    ]

    return {
        "clase":   clase,
        "bbox":    bbox,
        "polygon": polygon,
    }


def load_annotations(label_path, img_w, img_h):
    """Lee el archivo .txt completo y parsea todas las anotaciones."""
    if not os.path.exists(label_path):
        return []

    annotations = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ann = parse_yolo_line(line, img_w, img_h)
            if ann:
                annotations.append(ann)

    return annotations


# ─────────────────────────────────────────────
#  DIBUJADO DE ANOTACIONES
# ─────────────────────────────────────────────

def draw_mask(image, polygon, color, alpha):
    """
    Dibuja una máscara semitransparente sobre la imagen.
    
    OpenCV no soporta transparencia directa en imágenes BGR,
    así que usamos el truco de la superposición (overlay):
    
    1. Copiamos la imagen original
    2. Dibujamos la máscara sólida sobre la copia (cv2.fillPoly)
    3. Mezclamos original + copia con cv2.addWeighted:
       resultado = original * (1 - alpha) + copia * alpha
    
    Esto simula transparencia sin necesidad de formato BGRA.
    """
    if len(polygon) < 3:
        return image  # necesitamos al menos 3 puntos para un polígono

    overlay = image.copy()
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(overlay, [pts], color)

    # Mezclar con transparencia
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def draw_annotations(image, annotations):
    """
    Dibuja todas las anotaciones de una imagen:
    máscaras → luego bounding boxes → luego texto.
    
    El orden importa: dibujamos máscaras primero porque están debajo,
    los bounding boxes van encima de las máscaras,
    y el texto va al final para que siempre sea legible.
    """
    result = image.copy()

    # Primera pasada: máscaras semitransparentes
    for ann in annotations:
        result = draw_mask(result, ann["polygon"], COLOR_MASK, MASK_ALPHA)

    # Segunda pasada: bounding boxes y texto
    for i, ann in enumerate(annotations):
        x1, y1, x2, y2 = ann["bbox"]

        # Bounding box
        cv2.rectangle(result, (x1, y1), (x2, y2), COLOR_BBOX, 2)

        # Etiqueta con número de planta
        label = f"agave #{i+1}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Fondo negro detrás del texto para legibilidad
        cv2.rectangle(result, (x1, y1 - lh - 6), (x1 + lw + 4, y1), COLOR_BG, -1)
        cv2.putText(result, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)

    # Contador total en esquina superior izquierda
    total_label = f"Plantas detectadas: {len(annotations)}"
    (tw, th), _ = cv2.getTextSize(total_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(result, (8, 8), (tw + 16, th + 16), COLOR_BG, -1)
    cv2.putText(result, total_label, (12, th + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2, cv2.LINE_AA)

    return result


# ─────────────────────────────────────────────
#  PROCESAMIENTO PRINCIPAL
# ─────────────────────────────────────────────

def main():
    print(f"\n{'='*55}")
    print("  VERIFICACIÓN VISUAL DE ANOTACIONES")
    print(f"{'='*55}\n")

    images_dir = os.path.join(DATASET_DIR, "images", SPLIT)
    labels_dir = os.path.join(DATASET_DIR, "labels", SPLIT)

    if not os.path.isdir(images_dir):
        print(f"[ERROR] No se encontró: {images_dir}")
        print("Asegúrate de haber corrido primero el script de anotación.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Obtener lista de imágenes anotadas
    valid_ext = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith(valid_ext)
    ])

    if MAX_IMAGES:
        image_files = image_files[:MAX_IMAGES]

    print(f"Verificando {len(image_files)} imágenes del split '{SPLIT}'...\n")

    total_anns  = 0
    images_ok   = 0
    images_empty = 0

    for filename in tqdm(image_files):
        img_path = os.path.join(images_dir, filename)
        base_name = os.path.splitext(filename)[0]
        lbl_path  = os.path.join(labels_dir, f"{base_name}.txt")

        img = cv2.imread(img_path)
        if img is None:
            print(f"  [!] No se pudo leer: {img_path}")
            continue

        h, w = img.shape[:2]
        annotations = load_annotations(lbl_path, w, h)

        if annotations:
            images_ok += 1
            total_anns += len(annotations)
        else:
            images_empty += 1

        # Dibujar y guardar
        result = draw_annotations(img, annotations)
        out_path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(out_path, result)

    # Reporte final
    print(f"\n── RESUMEN ──────────────────────────────────────────")
    print(f"  Imágenes procesadas    : {len(image_files)}")
    print(f"  Con anotaciones        : {images_ok}")
    print(f"  Sin anotaciones        : {images_empty}")
    print(f"  Total de plantas       : {total_anns}")
    if images_ok > 0:
        print(f"  Promedio por imagen    : {total_anns / images_ok:.1f} plantas")
    print(f"\n✓ Imágenes verificadas guardadas en '{OUTPUT_DIR}/'")
    print(f"  Revisa las imágenes y ajusta los parámetros si es necesario.")


if __name__ == "__main__":
    main()