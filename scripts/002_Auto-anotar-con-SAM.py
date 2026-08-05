# ============================================================
#  Auto-anotación de agaves con SAM + Tiling
#  Salida: dataset en formato YOLO (bounding boxes + máscaras)
# ============================================================
#
#  Instalación de dependencias:
#  pip install torch torchvision opencv-python numpy tqdm
#  pip install git+https://github.com/facebookresearch/segment-anything.git
#
#  Descarga el peso del modelo SAM (vit_b para 4GB VRAM):
#  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
#
# ============================================================

import os
import cv2
import numpy as np
from tqdm import tqdm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ─────────────────────────────────────────────
#  CONFIGURACIÓN
# ─────────────────────────────────────────────

IMAGES_DIR     = r"C:\Users\charl\Downloads\wetransfer_agave16_2026-06-19_1616\Agave16"  # carpeta con tus imágenes originales
OUTPUT_DIR     = r"D:\Nube Mega\Trabajo\programitas\python\Marcado de imagenes\dataset"
SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"
DEVICE         = "cuda"

# ── Tiling ──────────────────────────────────
# En lugar de pasar la imagen completa (5280x3956) a SAM,
# la dividimos en secciones más pequeñas para que SAM pueda
# ver cada agave con suficiente detalle.
TILE_SIZE    = 1024   # tamaño de cada sección en píxeles
TILE_OVERLAP = 256    # solapamiento entre secciones para no perder plantas en los bordes
                      # (una planta que cae en el borde de un tile aparece en el siguiente)

# ── Filtros por área (ahora en coordenadas de TILE, no de imagen completa) ──
# A 1024px de tile, un agave mediano ocupa entre 3000 y 80000 píxeles
MIN_AREA     = 3000
MAX_AREA     = 80000
MIN_SOLIDITY = 0.3

# ── Filtro de color HSV ──────────────────────
# El agave azul tequilero tiene tono verde-azulado con baja saturación.
# Esto elimina postes, árboles de hoja ancha, suelo y otros objetos.
USE_COLOR_FILTER = True
COLOR_HUE_MIN    = 50    # tono mínimo (verde)
COLOR_HUE_MAX    = 130   # tono máximo (azul-verde)
COLOR_SAT_MIN    = 15    # saturación mínima
COLOR_SAT_MAX    = 180   # saturación máxima
COLOR_VAL_MIN    = 30    # brillo mínimo (descartar sombras)
COLOR_VAL_MAX    = 210   # brillo máximo
COLOR_MIN_RATIO  = 0.25  # mínimo 25% de píxeles en rango de color

# ── Filtro de forma (aspecto) ────────────────
# Un agave visto desde arriba tiende a ser aproximadamente circular.
# Descartamos objetos muy alargados (postes, tuberías, sombras largas).
MAX_ASPECT_RATIO = 3.0   # ancho/alto máximo del bounding box

# ── Parámetros SAM para 4GB VRAM ─────────────
POINTS_PER_SIDE        = 32   # más puntos porque ahora el tile es pequeño
PRED_IOU_THRESH        = 0.80
STABILITY_SCORE_THRESH = 0.88
BATCH_SIZE             = 16

# ── NMS (Non-Maximum Suppression) ────────────
# Cuando dos tiles se solapan, la misma planta puede detectarse dos veces.
# NMS elimina detecciones duplicadas comparando su solapamiento (IoU).
IOU_THRESHOLD = 0.4   # si dos máscaras se solapan más de 40%, conserva solo la mayor

# ─────────────────────────────────────────────
#  INICIALIZACIÓN
# ─────────────────────────────────────────────

def setup_output_dirs():
    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)


def load_sam_model():
    print(f"Cargando SAM en {DEVICE}...")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=DEVICE)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=POINTS_PER_SIDE,
        pred_iou_thresh=PRED_IOU_THRESH,
        stability_score_thresh=STABILITY_SCORE_THRESH,
        points_per_batch=BATCH_SIZE,
    )
    return mask_generator


# ─────────────────────────────────────────────
#  FILTROS DE MÁSCARAS
# ─────────────────────────────────────────────

def compute_solidity(mask):
    """
    Solidez = área / área del convex hull.
    Los agaves tienen forma de roseta: solidez media (0.3–0.75).
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return 0
    largest   = max(contours, key=cv2.contourArea)
    hull_area = cv2.contourArea(cv2.convexHull(largest))
    if hull_area == 0:
        return 0
    return cv2.contourArea(largest) / hull_area


def check_color(mask_array, img_rgb):
    """
    Verifica que los píxeles de la máscara sean del color verde-grisáceo del agave.
    Usa espacio HSV para ser robusto ante cambios de iluminación.
    Descarta postes, lonas, tuberías y pasto (diferente tono/saturación).
    """
    img_hsv     = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    mask_pixels = img_hsv[mask_array.astype(bool)]
    if len(mask_pixels) == 0:
        return False

    h, s, v = mask_pixels[:, 0], mask_pixels[:, 1], mask_pixels[:, 2]
    in_range = (
        (h >= COLOR_HUE_MIN) & (h <= COLOR_HUE_MAX) &
        (s >= COLOR_SAT_MIN) & (s <= COLOR_SAT_MAX) &
        (v >= COLOR_VAL_MIN) & (v <= COLOR_VAL_MAX)
    )
    return (in_range.sum() / len(mask_pixels)) >= COLOR_MIN_RATIO


def check_aspect_ratio(mask):
    """
    Descarta objetos muy alargados (postes, tuberías, sombras largas).
    Un agave desde arriba es aproximadamente circular, aspect ratio ≈ 1.
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return False
    _, _, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    if h == 0:
        return False
    ratio = max(w, h) / min(w, h)  # siempre >= 1
    return ratio <= MAX_ASPECT_RATIO


def is_valid_agave(mask, img_rgb=None):
    """
    Aplica todos los filtros en orden de menor a mayor costo computacional.
    """
    if not (MIN_AREA < mask["area"] < MAX_AREA):
        return False
    if compute_solidity(mask["segmentation"]) < MIN_SOLIDITY:
        return False
    if not check_aspect_ratio(mask["segmentation"]):
        return False
    if USE_COLOR_FILTER and img_rgb is not None:
        if not check_color(mask["segmentation"], img_rgb):
            return False
    return True


# ─────────────────────────────────────────────
#  TILING
# ─────────────────────────────────────────────

def get_tiles(img_h, img_w, tile_size, overlap):
    """
    Calcula las coordenadas (x1, y1, x2, y2) de cada tile.

    ¿Por qué solapamiento?
    Si una planta cae exactamente en el borde entre dos tiles,
    sin solapamiento quedaría cortada y SAM no la detectaría.
    Con solapamiento, aparece completa en al menos un tile.

    El stride (paso) es tile_size - overlap:
    - tile_size=1024, overlap=256 → stride=768
    - El siguiente tile empieza 768px después del anterior
    - Los últimos 256px del tile anterior se repiten en el siguiente
    """
    stride = tile_size - overlap
    tiles  = []
    y = 0
    while y < img_h:
        x = 0
        while x < img_w:
            x2 = min(x + tile_size, img_w)
            y2 = min(y + tile_size, img_h)
            x1 = max(0, x2 - tile_size)
            y1 = max(0, y2 - tile_size)
            tiles.append((x1, y1, x2, y2))
            if x2 == img_w:
                break
            x += stride
        if y2 == img_h:
            break
        y += stride
    return tiles


def masks_to_global(masks, tile_x1, tile_y1, img_w, img_h):
    """
    Convierte las máscaras de coordenadas locales del tile
    a coordenadas globales de la imagen completa.

    ¿Por qué es necesario?
    SAM trabaja en el sistema de coordenadas del tile (0,0 = esquina del tile).
    Necesitamos traducir cada máscara al sistema de la imagen completa
    para poder combinar detecciones de distintos tiles.

    Lo hacemos creando una máscara del tamaño de la imagen completa
    y "pegando" la máscara del tile en la posición correcta.
    """
    global_masks = []
    for m in masks:
        # Crear máscara vacía del tamaño de la imagen completa
        global_seg = np.zeros((img_h, img_w), dtype=bool)
        local_seg  = m["segmentation"]
        th, tw     = local_seg.shape

        # Calcular dimensiones reales del tile en la imagen
        actual_h = min(th, img_h - tile_y1)
        actual_w = min(tw, img_w - tile_x1)

        # Pegar la máscara local en la posición global correspondiente
        global_seg[tile_y1:tile_y1+actual_h, tile_x1:tile_x1+actual_w] = \
            local_seg[:actual_h, :actual_w]

        new_mask = dict(m)
        new_mask["segmentation"] = global_seg
        new_mask["area"]         = int(global_seg.sum())
        global_masks.append(new_mask)

    return global_masks


# ─────────────────────────────────────────────
#  NMS — ELIMINAR DUPLICADOS
# ─────────────────────────────────────────────

def compute_iou(mask_a, mask_b):
    """
    IoU (Intersection over Union) entre dos máscaras binarias.

    IoU = área de intersección / área de unión
    - IoU = 1.0 → máscaras idénticas
    - IoU = 0.0 → sin solapamiento
    - IoU > IOU_THRESHOLD → consideramos que son la misma planta

    Es la métrica estándar para comparar detecciones en visión por computadora.
    """
    intersection = (mask_a & mask_b).sum()
    union        = (mask_a | mask_b).sum()
    if union == 0:
        return 0.0
    return intersection / union


def apply_nms(masks):
    """
    Non-Maximum Suppression: elimina máscaras duplicadas.

    Algoritmo:
    1. Ordenar máscaras por score de confianza (de mayor a menor)
    2. Tomar la máscara con mayor score → conservarla
    3. Descartar todas las que tengan IoU > IOU_THRESHOLD con ella
    4. Repetir con la siguiente máscara no descartada

    Esto garantiza que cada planta quede representada por una sola máscara,
    la de mayor confianza entre todas las detecciones solapadas.
    """
    if not masks:
        return []

    # Ordenar por predicted_iou descendente
    masks = sorted(masks, key=lambda m: m.get("predicted_iou", 0), reverse=True)

    kept      = []
    discarded = set()

    for i, m in enumerate(masks):
        if i in discarded:
            continue
        kept.append(m)
        seg_i = m["segmentation"]
        for j in range(i + 1, len(masks)):
            if j in discarded:
                continue
            if compute_iou(seg_i, masks[j]["segmentation"]) > IOU_THRESHOLD:
                discarded.add(j)

    return kept


# ─────────────────────────────────────────────
#  CONVERSIÓN A FORMATO YOLO
# ─────────────────────────────────────────────

def mask_to_yolo(mask_array, img_w, img_h):
    """
    Convierte una máscara binaria al formato YOLO-seg:
    clase cx cy bw bh x1 y1 x2 y2 ... xn yn  (todos normalizados 0-1)
    """
    contours, _ = cv2.findContours(
        mask_array.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    largest    = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    bw = w / img_w
    bh = h / img_h

    epsilon    = 0.005 * cv2.arcLength(largest, True)
    simplified = cv2.approxPolyDP(largest, epsilon, True)

    seg_points = []
    for pt in simplified:
        seg_points.extend([pt[0][0] / img_w, pt[0][1] / img_h])

    if len(seg_points) < 6:
        return None

    seg_str = " ".join(f"{v:.6f}" for v in seg_points)
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {seg_str}"


# ─────────────────────────────────────────────
#  PROCESAMIENTO PRINCIPAL
# ─────────────────────────────────────────────

def process_image(image_path, mask_generator, split="train"):
    """
    Procesa una imagen completa usando tiling:
    1. Divide la imagen en tiles de TILE_SIZE x TILE_SIZE
    2. Pasa cada tile a SAM
    3. Convierte las detecciones a coordenadas globales
    4. Aplica NMS para eliminar duplicados del solapamiento
    5. Guarda el resultado en formato YOLO
    """
    import torch

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"  [!] No se pudo leer: {image_path}")
        return 0

    img_rgb        = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_h, img_w   = img_rgb.shape[:2]
    tiles          = get_tiles(img_h, img_w, TILE_SIZE, TILE_OVERLAP)
    all_masks      = []

    for (x1, y1, x2, y2) in tiles:
        tile_rgb = img_rgb[y1:y2, x1:x2]

        torch.cuda.empty_cache()
        raw_masks = mask_generator.generate(tile_rgb)

        # Filtrar en coordenadas del tile (más rápido que en coordenadas globales)
        valid = [m for m in raw_masks if is_valid_agave(m, tile_rgb)]

        # Convertir a coordenadas globales antes de acumular
        global_masks = masks_to_global(valid, x1, y1, img_w, img_h)
        all_masks.extend(global_masks)

    # Eliminar duplicados generados por el solapamiento de tiles
    all_masks = apply_nms(all_masks)

    if not all_masks:
        return 0

    lines = []
    for mask in all_masks:
        line = mask_to_yolo(mask["segmentation"], img_w, img_h)
        if line:
            lines.append(line)

    if not lines:
        return 0

    base_name    = os.path.splitext(os.path.basename(image_path))[0]
    out_img_path = os.path.join(OUTPUT_DIR, "images", split, os.path.basename(image_path))
    out_lbl_path = os.path.join(OUTPUT_DIR, "labels", split, f"{base_name}.txt")

    cv2.imwrite(out_img_path, img_bgr)
    with open(out_lbl_path, "w") as f:
        f.write("\n".join(lines))

    return len(lines)


def generate_yaml():
    yaml_content = f"""path: {os.path.abspath(OUTPUT_DIR)}
train: images/train
val: images/val

nc: 1
names:
  0: agave
"""
    with open(os.path.join(OUTPUT_DIR, "data.yaml"), "w") as f:
        f.write(yaml_content)


# ─────────────────────────────────────────────
#  PUNTO DE ENTRADA
# ─────────────────────────────────────────────

def main():
    setup_output_dirs()
    mask_generator = load_sam_model()

    valid_ext  = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    image_files = [
        os.path.join(IMAGES_DIR, f)
        for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith(valid_ext)
    ]

    if not image_files:
        print(f"No se encontraron imágenes en '{IMAGES_DIR}'")
        return

    np.random.shuffle(image_files)
    split_idx = int(len(image_files) * 0.8)
    splits = {
        "train": image_files[:split_idx],
        "val":   image_files[split_idx:]
    }

    total_annotations = 0

    for split_name, files in splits.items():
        print(f"\nProcesando {split_name} ({len(files)} imágenes)...")
        for img_path in tqdm(files):
            count = process_image(img_path, mask_generator, split=split_name)
            total_annotations += count

    generate_yaml()

    print(f"\n✓ Dataset generado en '{OUTPUT_DIR}/'")
    print(f"✓ Total de anotaciones: {total_annotations}")
    print(f"✓ Configura tu entrenamiento con: {OUTPUT_DIR}/data.yaml")


if __name__ == "__main__":
    main()