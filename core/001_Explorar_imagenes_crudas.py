# ============================================================
#  SCRIPT 1: Exploración del dataset de imágenes
#  
#  ¿Para qué sirve este script?
#  Antes de correr SAM sobre 500 imágenes, necesitas conocer
#  tu material: resoluciones, tamaños de archivo, si hay
#  imágenes corruptas, y sobre todo — qué tan grande aparece
#  un agave en píxeles. Ese último dato es clave para calibrar
#  MIN_AREA y MAX_AREA en el script de anotación.
#
#  ¿Qué genera?
#  - Reporte en consola con estadísticas generales
#  - Una imagen de muestra con una cuadrícula de recortes
#    aleatorios para que estimes visualmente el tamaño de los agaves
#  - Un archivo exploration_report.txt con todo el reporte
# ============================================================

import os
import cv2
import numpy as np
import random
from datetime import datetime

# ─────────────────────────────────────────────
#  CONFIGURACIÓN
# ─────────────────────────────────────────────

IMAGES_DIR   = r"./Fotos"  # carpeta con tus imágenes originales
REPORT_FILE  = "exploration_report-h.txt"  # archivo de salida del reporte
SAMPLE_GRID  = "sample_grid-h.jpg"         # imagen de muestra que genera el script
GRID_COLS    = 4                         # columnas en la cuadrícula de muestra
GRID_ROWS    = 3                         # filas en la cuadrícula de muestra
THUMB_SIZE   = (400, 400)                # tamaño de cada miniatura en la cuadrícula

# ─────────────────────────────────────────────
#  FUNCIONES DE ANÁLISIS
# ─────────────────────────────────────────────

def get_image_files(directory):
    """
    Escanea el directorio y devuelve solo archivos de imagen válidos.
    
    os.listdir() devuelve TODOS los archivos, incluyendo .txt, .yaml, etc.
    Por eso filtramos por extensión antes de intentar abrirlos.
    """
    valid_ext = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(valid_ext)
    ]
    return sorted(files)  # sorted() para orden consistente entre ejecuciones


def analyze_image(path):
    """
    Lee una imagen y extrae sus metadatos básicos.
    
    cv2.imread() devuelve None si el archivo está corrupto o no es una imagen
    válida — por eso siempre verificamos antes de usar el resultado.
    
    img.shape devuelve (alto, ancho, canales):
    - Alto y ancho en píxeles
    - Canales: 3 para RGB/BGR, 1 para escala de grises
    
    os.path.getsize() devuelve el tamaño en bytes, dividimos entre 1024*1024
    para convertirlo a megabytes.
    """
    img = cv2.imread(path)
    if img is None:
        return None  # imagen corrupta o ilegible

    h, w, c = img.shape
    size_mb = os.path.getsize(path) / (1024 * 1024)

    return {
        "path":    path,
        "width":   w,
        "height":  h,
        "channels": c,
        "size_mb": round(size_mb, 2),
        "aspect":  round(w / h, 2),
    }


def print_and_save(text, file_handle):
    """Imprime en consola Y guarda en el archivo de reporte simultáneamente."""
    print(text)
    file_handle.write(text + "\n")


# ─────────────────────────────────────────────
#  GENERACIÓN DE CUADRÍCULA DE MUESTRA
# ─────────────────────────────────────────────

def build_sample_grid(image_files, cols=4, rows=3, thumb_size=(400, 400)):
    """
    Crea una imagen compuesta con miniaturas aleatorias de tu dataset.
    
    ¿Por qué es útil?
    Ver un muestreo aleatorio te ayuda a:
    - Identificar variaciones de iluminación entre imágenes
    - Ver si hay imágenes fuera de foco o con artefactos
    - Estimar visualmente el tamaño típico de los agaves
    
    ¿Cómo funciona?
    1. Seleccionamos imágenes al azar con random.sample()
    2. Redimensionamos cada una a thumb_size con cv2.resize()
    3. Las pegamos en una "canvas" (imagen vacía negra) calculando
       la posición de cada miniatura según su fila y columna
    """
    n_samples = min(cols * rows, len(image_files))
    selected  = random.sample(image_files, n_samples)

    cell_w, cell_h = thumb_size
    canvas = np.zeros((cell_h * rows, cell_w * cols, 3), dtype=np.uint8)

    for idx, path in enumerate(selected):
        img = cv2.imread(path)
        if img is None:
            continue

        row = idx // cols   # división entera: qué fila le toca
        col = idx % cols    # módulo: qué columna le toca

        thumb = cv2.resize(img, (cell_w, cell_h))

        # Calcular coordenadas de inicio (y1:y2, x1:x2) en el canvas
        y1, y2 = row * cell_h, (row + 1) * cell_h
        x1, x2 = col * cell_w, (col + 1) * cell_w
        canvas[y1:y2, x1:x2] = thumb

        # Añadir etiqueta con el nombre del archivo sobre la miniatura
        label = os.path.basename(path)[:25]  # truncar si es muy largo
        cv2.putText(
            canvas, label,
            (x1 + 5, y1 + 20),           # posición del texto
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,                          # tamaño de fuente
            (255, 255, 255),               # color blanco
            1, cv2.LINE_AA
        )

    return canvas


# ─────────────────────────────────────────────
#  REPORTE PRINCIPAL
# ─────────────────────────────────────────────

def main():
    print(f"\n{'='*55}")
    print("  EXPLORACIÓN DE IMÁGENES")
    print(f"{'='*55}\n")

    # Verificar que existe la carpeta
    if not os.path.isdir(IMAGES_DIR):
        print(f"[ERROR] No se encontró la carpeta '{IMAGES_DIR}'")
        print("Asegúrate de que el script está en el mismo directorio que tu carpeta de imágenes.")
        return

    image_files = get_image_files(IMAGES_DIR)

    if not image_files:
        print(f"[ERROR] No se encontraron imágenes en '{IMAGES_DIR}'")
        return

    # Analizar cada imagen
    print(f"Analizando {len(image_files)} archivos...")
    results   = []
    corrupted = []

    for path in image_files:
        info = analyze_image(path)
        if info:
            results.append(info)
        else:
            corrupted.append(path)

    # ── Calcular estadísticas ──
    widths    = [r["width"]   for r in results]
    heights   = [r["height"]  for r in results]
    sizes_mb  = [r["size_mb"] for r in results]
    resolutions = set(f"{r['width']}x{r['height']}" for r in results)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print_and_save(f"Reporte generado: {timestamp}", f)
        print_and_save(f"Directorio analizado: {os.path.abspath(IMAGES_DIR)}\n", f)

        print_and_save("── RESUMEN GENERAL ──────────────────────────────────", f)
        print_and_save(f"  Total de archivos encontrados : {len(image_files)}", f)
        print_and_save(f"  Imágenes válidas              : {len(results)}", f)
        print_and_save(f"  Imágenes corruptas/ilegibles  : {len(corrupted)}", f)

        print_and_save("\n── RESOLUCIONES ─────────────────────────────────────", f)
        if len(resolutions) == 1:
            print_and_save(f"  Todas las imágenes son {list(resolutions)[0]}", f)
        else:
            print_and_save(f"  Se encontraron {len(resolutions)} resoluciones distintas:", f)
            for res in sorted(resolutions):
                count = sum(1 for r in results if f"{r['width']}x{r['height']}" == res)
                print_and_save(f"    {res} → {count} imágenes", f)

        print_and_save(f"\n  Ancho  — mín: {min(widths)}px  máx: {max(widths)}px  promedio: {int(np.mean(widths))}px", f)
        print_and_save(f"  Alto   — mín: {min(heights)}px  máx: {max(heights)}px  promedio: {int(np.mean(heights))}px", f)

        print_and_save("\n── TAMAÑO DE ARCHIVOS ───────────────────────────────", f)
        print_and_save(f"  Mínimo  : {min(sizes_mb):.2f} MB", f)
        print_and_save(f"  Máximo  : {max(sizes_mb):.2f} MB", f)
        print_and_save(f"  Promedio: {np.mean(sizes_mb):.2f} MB", f)
        print_and_save(f"  Total   : {sum(sizes_mb):.2f} MB", f)

        if corrupted:
            print_and_save("\n── IMÁGENES CORRUPTAS ───────────────────────────────", f)
            for path in corrupted:
                print_and_save(f"  [!] {path}", f)

        # ── Estimación de área para calibrar SAM ──
        avg_w = int(np.mean(widths))
        avg_h = int(np.mean(heights))
        img_area = avg_w * avg_h

        print_and_save("\n── GUÍA PARA CALIBRAR MIN_AREA / MAX_AREA ───────────", f)
        print_and_save(f"  Área total promedio de imagen: {img_area:,} píxeles", f)
        print_and_save("", f)
        print_and_save("  Si un agave ocupa aproximadamente este % de la imagen:", f)
        for pct in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
            area_px = int(img_area * pct / 100)
            print_and_save(f"    {pct:5.1f}% → {area_px:>8,} píxeles cuadrados", f)
        print_and_save("", f)
        print_and_save("  Usa estos valores como referencia para ajustar", f)
        print_and_save("  MIN_AREA y MAX_AREA en el script de anotación.", f)

        print_and_save(f"\n{'='*55}", f)
        print_and_save(f"Reporte guardado en: {REPORT_FILE}", f)
        print_and_save(f"Cuadrícula de muestra guardada en: {SAMPLE_GRID}", f)

    # Generar cuadrícula visual
    print(f"\nGenerando cuadrícula de muestra...")
    grid = build_sample_grid(image_files, GRID_COLS, GRID_ROWS, THUMB_SIZE)
    cv2.imwrite(SAMPLE_GRID, grid)
    print(f"✓ Guardada en '{SAMPLE_GRID}' — ábrela para revisar tus imágenes visualmente.")


if __name__ == "__main__":
    main()
