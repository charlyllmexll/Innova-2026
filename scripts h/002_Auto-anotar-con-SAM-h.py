# ============================================================
#  Auto-anotación de agaves con SAM + Tiling
#  Salida: dataset en formato YOLO (bounding boxes + máscaras)
# ============================================================
#
#  Instalación de dependencias:
#  pip install torch torchvision opencv-python numpy tqdm
#  pip install git+https://github.com/facebookresearch/segment-anything.git
#
#  Descarga el peso del modelo SAM (vit_h):
#  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
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

IMAGES_DIR     = r"./Fotos"  # carpeta con tus imágenes originales
OUTPUT_DIR     = r"./Fotos/dataset h"
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"
DEVICE         = "cuda"

# ── Tiling ──────────────────────────────────
TILE_SIZE    = 1024
TILE_OVERLAP = 256

# ── Filtros de área (en coordenadas de tile) ──
MIN_AREA         = 1500
MAX_AREA         = 140000
MIN_SOLIDITY     = 0.35
MAX_ASPECT_RATIO = 2.5

# ── Filtro de color HSV ──────────────────────
USE_COLOR_FILTER = True
COLOR_HUE_MIN    = 50
COLOR_HUE_MAX    = 130
COLOR_SAT_MIN    = 15
COLOR_SAT_MAX    = 180
COLOR_VAL_MIN    = 30
COLOR_VAL_MAX    = 210
COLOR_MIN_RATIO  = 0.25

# ── Parámetros SAM ───────────────────────────
POINTS_PER_SIDE        = 32
PRED_IOU_THRESH        = 0.80
STABILITY_SCORE_THRESH = 0.88
BATCH_SIZE             = 16

# ── NMS ─────────────────────────────────────
IOU_THRESHOLD = 0.4

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
    return SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=POINTS_PER_SIDE,
        pred_iou_thresh=PRED_IOU_THRESH,
        stability_score_thresh=STABILITY_SCORE_THRESH,
        points_per_batch=BATCH_SIZE,
    )


# ─────────────────────────────────────────────
#  FILTROS
# ─────────────────────────────────────────────

def compute_solidity(mask):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return 0
    largest   = max(contours, key=cv2.contourArea)
    hull_area = cv2.contourArea(cv2.convexHull(largest))
    return cv2.contourArea(largest) / hull_area if hull_area > 0 else 0


def check_color(mask_array, img_rgb):
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
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return False
    _, _, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    if min(w, h) == 0:
        return False
    return (max(w, h) / min(w, h)) <= MAX_ASPECT_RATIO


def is_valid_agave(mask, img_rgb=None):
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
#  EXTRACCIÓN LIGERA DE DETECCIONES
# ─────────────────────────────────────────────

def extract_detection(mask, tile_x1, tile_y1, img_w, img_h, iou_score):
    """
    En lugar de guardar la máscara completa (20MB por máscara),
    extraemos solo el polígono simplificado y el bounding box
    en coordenadas globales. Esto consume ~100 veces menos memoria.

    También calculamos un bbox_global para el NMS, que opera
    sobre rectángulos en lugar de máscaras completas — mucho más rápido.
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    largest    = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Traducir a coordenadas globales sumando el offset del tile
    gx = x + tile_x1
    gy = y + tile_y1

    # Simplificar polígono y traducir a coordenadas globales
    epsilon    = 0.005 * cv2.arcLength(largest, True)
    simplified = cv2.approxPolyDP(largest, epsilon, True)
    polygon_global = [(pt[0][0] + tile_x1, pt[0][1] + tile_y1) for pt in simplified]

    if len(polygon_global) < 3:
        return None

    return {
        "bbox":    (gx, gy, w, h),          # (x, y, w, h) en coords globales
        "polygon": polygon_global,            # lista de (x, y) en coords globales
        "area":    int(mask.sum()),
        "score":   iou_score,
    }


# ─────────────────────────────────────────────
#  NMS SOBRE BOUNDING BOXES (sin máscaras completas)
# ─────────────────────────────────────────────

def bbox_iou(a, b):
    """
    IoU entre dos bounding boxes (x, y, w, h).
    Más eficiente que comparar máscaras completas pixel a pixel.
    """
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = b[0] + b[2], b[1] + b[3]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = a[2] * a[3]
    area_b = b[2] * b[3]
    union  = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def apply_nms(detections):
    """
    NMS sobre bounding boxes en lugar de máscaras completas.
    Mismo algoritmo pero ~100x más rápido y sin consumo de RAM extra.
    """
    if not detections:
        return []

    detections = sorted(detections, key=lambda d: d["score"], reverse=True)
    kept      = []
    discarded = set()

    for i, det in enumerate(detections):
        if i in discarded:
            continue
        kept.append(det)
        for j in range(i + 1, len(detections)):
            if j in discarded:
                continue
            if bbox_iou(det["bbox"], detections[j]["bbox"]) > IOU_THRESHOLD:
                discarded.add(j)

    return kept


# ─────────────────────────────────────────────
#  CONVERSIÓN A FORMATO YOLO
# ─────────────────────────────────────────────

def detection_to_yolo(det, img_w, img_h):
    """
    Convierte una detección (bbox + polígono en coords globales)
    al formato YOLO-seg normalizado:
    clase cx cy bw bh x1 y1 x2 y2 ... xn yn
    """
    x, y, w, h = det["bbox"]

    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    bw = w / img_w
    bh = h / img_h

    seg_points = []
    for (px, py) in det["polygon"]:
        seg_points.extend([px / img_w, py / img_h])

    if len(seg_points) < 6:
        return None

    seg_str = " ".join(f"{v:.6f}" for v in seg_points)
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {seg_str}"


# ─────────────────────────────────────────────
#  TILING
# ─────────────────────────────────────────────

def get_tiles(img_h, img_w, tile_size, overlap):
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


# ─────────────────────────────────────────────
#  PROCESAMIENTO PRINCIPAL
# ─────────────────────────────────────────────

def process_image(image_path, mask_generator, split="train"):
    import torch

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"  [!] No se pudo leer: {image_path}")
        return 0

    img_rgb      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_h, img_w = img_rgb.shape[:2]
    tiles        = get_tiles(img_h, img_w, TILE_SIZE, TILE_OVERLAP)
    all_dets     = []   # lista ligera: solo bbox + polígono, sin máscaras completas

    for (x1, y1, x2, y2) in tiles:
        tile_rgb = img_rgb[y1:y2, x1:x2]

        torch.cuda.empty_cache()
        raw_masks = mask_generator.generate(tile_rgb)

        for m in raw_masks:
            if not is_valid_agave(m, tile_rgb):
                continue
            det = extract_detection(
                m["segmentation"], x1, y1, img_w, img_h,
                m.get("predicted_iou", 0)
            )
            if det:
                all_dets.append(det)

        # Liberar máscaras del tile inmediatamente — no las necesitamos más
        del raw_masks

    # NMS sobre bounding boxes (ligero, sin máscaras completas)
    all_dets = apply_nms(all_dets)

    if not all_dets:
        return 0

    lines = [detection_to_yolo(d, img_w, img_h) for d in all_dets]
    lines = [l for l in lines if l]

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

    valid_ext   = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
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
