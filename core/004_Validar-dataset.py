# ============================================================
#  SCRIPT 3: Validación de integridad del dataset YOLO
#
#  ¿Para qué sirve?
#  Antes de iniciar el entrenamiento, este script verifica que
#  el dataset esté bien estructurado y no haya errores que
#  causen fallos silenciosos durante el entrenamiento.
#
#  Problemas comunes que detecta:
#  - Imágenes sin archivo de etiquetas correspondiente
#  - Archivos de etiquetas vacíos o malformados
#  - Coordenadas fuera del rango [0, 1]
#  - Polígonos con menos de 3 puntos (inválidos)
#  - Desbalance extremo entre train y val
#  - data.yaml ausente o con rutas incorrectas
#
#  ¿Qué genera?
#  - Reporte en consola
#  - validation_report.txt con detalles completos
#  - Lista de archivos problemáticos para corregir manualmente
# ============================================================

import os
import yaml
import numpy as np
from collections import defaultdict

# ─────────────────────────────────────────────
#  CONFIGURACIÓN
# ─────────────────────────────────────────────

DATASET_DIR   = r"./Fotos/dataset h"      # carpeta generada por el script de anotación
REPORT_FILE   = "validation_report-h.txt"

# ─────────────────────────────────────────────
#  VALIDACIÓN DE LÍNEAS YOLO
# ─────────────────────────────────────────────

def validate_yolo_line(line, line_num):
    """
    Valida una sola línea del archivo de etiquetas YOLO-seg.
    
    Reglas del formato:
    - Primer valor: clase (entero >= 0)
    - Siguientes 4: cx, cy, bw, bh (floats entre 0 y 1)
    - El resto: pares x,y del polígono (floats entre 0 y 1)
    - Mínimo de puntos para un polígono: 3 pares = 6 valores
    - Total mínimo de valores en la línea: 1 + 4 + 6 = 11
    
    Devuelve: (es_válida, mensaje_de_error)
    """
    parts = line.strip().split()

    # ── Verificar mínimo de valores ──
    if len(parts) < 11:
        return False, f"Línea {line_num}: muy pocos valores ({len(parts)}, mínimo 11)"

    # ── Verificar que todos los valores sean numéricos ──
    try:
        valores = list(map(float, parts))
    except ValueError:
        return False, f"Línea {line_num}: contiene valores no numéricos"

    clase = int(valores[0])
    bbox  = valores[1:5]   # cx, cy, bw, bh
    seg   = valores[5:]    # puntos del polígono

    # ── Clase válida ──
    if clase < 0:
        return False, f"Línea {line_num}: clase negativa ({clase})"

    # ── Bounding box en rango [0, 1] ──
    for i, v in enumerate(bbox):
        if not (0.0 <= v <= 1.0):
            nombres = ["cx", "cy", "bw", "bh"]
            return False, f"Línea {line_num}: {nombres[i]}={v:.4f} fuera de rango [0,1]"

    # ── Polígono: número par de valores ──
    if len(seg) % 2 != 0:
        return False, f"Línea {line_num}: polígono con número impar de coordenadas"

    # ── Polígono: mínimo 3 puntos ──
    n_points = len(seg) // 2
    if n_points < 3:
        return False, f"Línea {line_num}: polígono con solo {n_points} puntos (mínimo 3)"

    # ── Puntos del polígono en rango [0, 1] ──
    for i, v in enumerate(seg):
        if not (0.0 <= v <= 1.0):
            coord = "x" if i % 2 == 0 else "y"
            return False, f"Línea {line_num}: coordenada de polígono {coord}={v:.4f} fuera de rango"

    return True, None


def validate_label_file(label_path):
    """
    Valida un archivo .txt completo de etiquetas.
    
    Devuelve un diccionario con:
    - is_valid: si el archivo en general es válido
    - n_annotations: cuántas anotaciones válidas tiene
    - errors: lista de mensajes de error encontrados
    """
    result = {
        "is_valid":      True,
        "n_annotations": 0,
        "errors":        [],
    }

    if not os.path.exists(label_path):
        result["is_valid"] = False
        result["errors"].append("Archivo no existe")
        return result

    with open(label_path, "r") as f:
        lines = [l for l in f.readlines() if l.strip()]  # ignorar líneas vacías

    if not lines:
        # Un archivo vacío puede ser válido si la imagen no tenía agaves,
        # pero lo marcamos para revisión manual
        result["errors"].append("Archivo vacío (imagen sin anotaciones)")
        return result

    for i, line in enumerate(lines, start=1):
        valid, error_msg = validate_yolo_line(line, i)
        if valid:
            result["n_annotations"] += 1
        else:
            result["is_valid"] = False
            result["errors"].append(error_msg)

    return result


# ─────────────────────────────────────────────
#  VALIDACIÓN DEL DATA.YAML
# ─────────────────────────────────────────────

def validate_yaml(yaml_path):
    """
    Verifica que el archivo data.yaml exista y tenga los campos requeridos.
    
    YOLO necesita obligatoriamente:
    - path: ruta base del dataset
    - train: ruta relativa al split de entrenamiento
    - val:   ruta relativa al split de validación
    - nc:    número de clases
    - names: lista/diccionario de nombres de clases
    """
    errors = []

    if not os.path.exists(yaml_path):
        return False, ["data.yaml no encontrado"]

    try:
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return False, [f"Error al leer YAML: {e}"]

    required_keys = ["path", "train", "val", "nc", "names"]
    for key in required_keys:
        if key not in data:
            errors.append(f"Falta el campo requerido: '{key}'")

    if "nc" in data and "names" in data:
        nc    = data["nc"]
        names = data["names"]
        n_names = len(names) if isinstance(names, (list, dict)) else 0
        if nc != n_names:
            errors.append(f"nc={nc} pero hay {n_names} nombres definidos")

    return len(errors) == 0, errors


# ─────────────────────────────────────────────
#  VALIDACIÓN COMPLETA DEL DATASET
# ─────────────────────────────────────────────

def validate_split(split_name):
    """
    Valida un split completo (train o val).
    
    Para cada imagen verifica que exista su .txt correspondiente,
    y valida el contenido de cada .txt.
    
    Devuelve estadísticas y listas de archivos problemáticos.
    """
    images_dir = os.path.join(DATASET_DIR, "images", split_name)
    labels_dir = os.path.join(DATASET_DIR, "labels", split_name)

    valid_ext = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith(valid_ext)
    ]) if os.path.isdir(images_dir) else []

    stats = {
        "total_images":       len(image_files),
        "missing_labels":     [],
        "empty_labels":       [],
        "invalid_labels":     [],
        "total_annotations":  0,
        "annotations_per_img": [],
    }

    for filename in image_files:
        base = os.path.splitext(filename)[0]
        lbl_path = os.path.join(labels_dir, f"{base}.txt")
        result = validate_label_file(lbl_path)

        if not os.path.exists(lbl_path):
            stats["missing_labels"].append(filename)
        elif not result["errors"] or result["n_annotations"] > 0:
            stats["total_annotations"] += result["n_annotations"]
            stats["annotations_per_img"].append(result["n_annotations"])
        
        if result["errors"] and "vacío" in result["errors"][0]:
            stats["empty_labels"].append(filename)
        elif not result["is_valid"]:
            stats["invalid_labels"].append((filename, result["errors"]))

    return stats


# ─────────────────────────────────────────────
#  REPORTE PRINCIPAL
# ─────────────────────────────────────────────

def main():
    print(f"\n{'='*55}")
    print("  VALIDACIÓN DEL DATASET YOLO")
    print(f"{'='*55}\n")

    lines_out = []

    def log(text=""):
        print(text)
        lines_out.append(text)

    # ── Validar data.yaml ──
    yaml_path = os.path.join(DATASET_DIR, "data.yaml")
    yaml_ok, yaml_errors = validate_yaml(yaml_path)

    log("── data.yaml ────────────────────────────────────────")
    if yaml_ok:
        log("  ✓ data.yaml válido")
    else:
        log("  ✗ Problemas en data.yaml:")
        for err in yaml_errors:
            log(f"    → {err}")

    # ── Validar splits ──
    all_stats = {}
    for split in ["train", "val"]:
        split_dir = os.path.join(DATASET_DIR, "images", split)
        if not os.path.isdir(split_dir):
            log(f"\n  [!] No se encontró el split '{split}' — omitiendo")
            continue

        log(f"\n── Split: {split} ───────────────────────────────────")
        stats = validate_split(split)
        all_stats[split] = stats

        log(f"  Total de imágenes      : {stats['total_images']}")
        log(f"  Sin archivo de etiqueta: {len(stats['missing_labels'])}")
        log(f"  Etiquetas vacías       : {len(stats['empty_labels'])}")
        log(f"  Etiquetas con errores  : {len(stats['invalid_labels'])}")
        log(f"  Total de anotaciones   : {stats['total_annotations']}")

        if stats["annotations_per_img"]:
            arr = np.array(stats["annotations_per_img"])
            log(f"  Plantas por imagen     : mín={arr.min()}  máx={arr.max()}  prom={arr.mean():.1f}")

        if stats["missing_labels"]:
            log(f"\n  [!] Imágenes sin etiqueta:")
            for f in stats["missing_labels"][:10]:
                log(f"      {f}")
            if len(stats["missing_labels"]) > 10:
                log(f"      ... y {len(stats['missing_labels']) - 10} más")

        if stats["invalid_labels"]:
            log(f"\n  [!] Etiquetas con errores:")
            for fname, errs in stats["invalid_labels"][:5]:
                log(f"      {fname}:")
                for e in errs:
                    log(f"        → {e}")

    # ── Balance train/val ──
    if "train" in all_stats and "val" in all_stats:
        log("\n── Balance train / val ──────────────────────────────")
        n_train = all_stats["train"]["total_images"]
        n_val   = all_stats["val"]["total_images"]
        total   = n_train + n_val
        if total > 0:
            pct_train = n_train / total * 100
            pct_val   = n_val   / total * 100
            log(f"  train: {n_train} imágenes ({pct_train:.1f}%)")
            log(f"  val:   {n_val}   imágenes ({pct_val:.1f}%)")

            if pct_val < 10:
                log("  [!] Advertencia: val tiene menos del 10% — considera aumentarlo")
            elif pct_val > 35:
                log("  [!] Advertencia: val tiene más del 35% — podrías mover más a train")
            else:
                log("  ✓ Balance train/val dentro del rango recomendado (65-90% / 10-35%)")

    # ── Veredicto final ──
    log(f"\n{'='*55}")
    has_errors = any(
        len(s["missing_labels"]) + len(s["invalid_labels"]) > 0
        for s in all_stats.values()
    )
    if has_errors:
        log("  ✗ Dataset tiene problemas — revisa los errores arriba")
        log("    antes de iniciar el entrenamiento.")
    else:
        log("  ✓ Dataset válido — listo para entrenar")

    # Guardar reporte
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out))

    print(f"\n  Reporte guardado en: {REPORT_FILE}")


if __name__ == "__main__":
    main()
