import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np

RUTA_BASE_PROYECTO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# =============================================================
# 1) Concepto clave del proyecto
# =============================================================
# Este script vuelve al objetivo principal del proyecto:
#   estimar el peso de las piñas del agave a partir del tamaño
#   de la corona visible en imágenes tomadas con dron.
#
# La idea práctica es simple:
#   1. detectar la corona del agave
#   2. medir su área o diámetro
#   3. convertir ese tamaño a una estimación de peso
#   4. exportar el resultado a CSV para luego integrarlo a un ERP
#
# IMPORTANTE:
#   Este es un MVP funcional. La relación peso-vs-coroa debe
#   calibrarse con datos reales de campo para ser precisa.
#   Por ahora usamos una fórmula base para generar resultados.
# =============================================================


def obtener_archivos_imagenes(directorio_entrada: str):
    """Devuelve una lista ordenada de imágenes válidas dentro del directorio."""
    if not os.path.exists(directorio_entrada):
        return []

    extensiones = (".png", ".jpg", ".jpeg", ".bmp")
    archivos = []
    for nombre in os.listdir(directorio_entrada):
        ruta = os.path.join(directorio_entrada, nombre)
        if os.path.isfile(ruta) and nombre.lower().endswith(extensiones):
            archivos.append(ruta)
    return sorted(archivos)


def aplicar_clahe(img_bgr):
    """Mejora el contraste local para resaltar la corona verde."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_mejorado = clahe.apply(l)
    lab_mejorado = cv2.merge((l_mejorado, a, b))
    return cv2.cvtColor(lab_mejorado, cv2.COLOR_LAB2BGR)


def segmentar_corona(img_bgr):
    """Segmenta zonas verdes que se parecen a la corona del agave."""
    img = aplicar_clahe(img_bgr)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Rango verde para agave. Ajusta si tu imagen tiene más brillo o sombras.
    lower_green = np.array([25, 40, 40], dtype=np.uint8)
    upper_green = np.array([90, 255, 255], dtype=np.uint8)
    mascara_verde = cv2.inRange(hsv, lower_green, upper_green)

    # Eliminar ruido pequeño
    kernel = np.ones((5, 5), np.uint8)
    mascara_verde = cv2.morphologyEx(mascara_verde, cv2.MORPH_OPEN, kernel)
    mascara_verde = cv2.morphologyEx(mascara_verde, cv2.MORPH_CLOSE, kernel)

    contornos, _ = cv2.findContours(mascara_verde, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contornos, mascara_verde


def calcular_diametro_equivalente(contorno, area_px):
    """Calcula un diámetro equivalente para la forma detectada."""
    if area_px <= 0:
        return 0.0
    return float(np.sqrt((4 * area_px) / np.pi))


def estimar_peso_kg(diametro_cm: float, area_m2: float, coeficiente: float = 0.00065, exponente: float = 2.3):
    """
    Fórmula base para estimar peso desde el tamaño de la corona.

    El modelo empírico inicial es:
        peso_kg = coeficiente * diametro_cm^exponente

    Esto es un punto de partida. Debe calibrarse con mediciones reales de campo.
    """
    if diametro_cm <= 0:
        return 0.0
    return float(coeficiente * (diametro_cm ** exponente))


def procesar_imagen(ruta_imagen: str, metros_por_pixel: float = 0.05, min_area_px: int = 500):
    """
    Procesa una imagen y devuelve una lista de detecciones de coronas con peso estimado.

    metros_por_pixel: escala de la imagen. Ejemplo: 0.05 significa que 1 px = 5 cm.
    min_area_px: área mínima para considerar una corona válida.
    """
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        return []

    contornos, mascara = segmentar_corona(imagen)
    detecciones = []

    for contorno in contornos:
        area_px = cv2.contourArea(contorno)
        if area_px < min_area_px:
            continue

        x, y, w, h = cv2.boundingRect(contorno)
        diametro_px = calcular_diametro_equivalente(contorno, area_px)
        diametro_m = diametro_px * metros_por_pixel
        diametro_cm = diametro_m * 100.0

        area_m2 = area_px * (metros_por_pixel ** 2)
        peso_kg = estimar_peso_kg(diametro_cm, area_m2)

        detecciones.append({
            "imagen": os.path.basename(ruta_imagen),
            "planta_numero": len(detecciones) + 1,
            "area_px": round(float(area_px), 2),
            "diametro_px": round(float(diametro_px), 2),
            "diametro_cm": round(float(diametro_cm), 2),
            "area_m2": round(float(area_m2), 4),
            "peso_kg_estimado": round(float(peso_kg), 2),
            "bbox_x": int(x),
            "bbox_y": int(y),
            "bbox_w": int(w),
            "bbox_h": int(h),
        })

    return detecciones


def exportar_csv(registros, ruta_csv: str):
    """Exporta una lista de registros a CSV para ERP o análisis posterior."""
    ruta_destino = Path(ruta_csv)
    ruta_destino.parent.mkdir(parents=True, exist_ok=True)

    campos = [
        "imagen",
        "planta_numero",
        "area_px",
        "diametro_px",
        "diametro_cm",
        "area_m2",
        "peso_kg_estimado",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
    ]

    with open(ruta_destino, "w", newline="", encoding="utf-8") as archivo_csv:
        writer = csv.DictWriter(archivo_csv, fieldnames=campos)
        writer.writeheader()
        for registro in registros:
            writer.writerow(registro)


def procesar_lote(directorio_entrada: str, directorio_salida: str, metros_por_pixel: float = 0.05):
    """Procesa todas las imágenes del lote y guarda CSV final."""
    rutas = obtener_archivos_imagenes(directorio_entrada)
    if not rutas:
        print(f"[ERROR] No se encontraron imágenes en: {directorio_entrada}")
        return []

    registros = []
    for ruta in rutas:
        print(f"[INFO] Procesando: {os.path.basename(ruta)}")
        detecciones = procesar_imagen(ruta, metros_por_pixel=metros_por_pixel)
        if detecciones:
            registros.extend(detecciones)
        else:
            print(f"[WARN] No se detectó ninguna corona válida en: {ruta}")

    salida_csv = os.path.join(directorio_salida, "estimacion_peso_agave.csv")
    exportar_csv(registros, salida_csv)
    print(f"[OK] Se generó el archivo: {salida_csv}")
    return registros


def main():
    parser = argparse.ArgumentParser(description="Estima peso de piñas de agave desde imágenes de dron.")
    parser.add_argument("--input", default=os.path.join(RUTA_BASE_PROYECTO, "data", "imagenes_crudas"), help="Carpeta con imágenes de dron")
    parser.add_argument("--output", default=os.path.join(RUTA_BASE_PROYECTO, "data", "estimaciones"), help="Carpeta donde guardar los resultados")
    parser.add_argument("--scale", type=float, default=0.05, help="Metros por pixel. Ajusta con la altura del dron.")
    parser.add_argument("--min-area", type=int, default=500, help="Área mínima de una corona para considerar una detección válida")
    args = parser.parse_args()

    print("[AGROVISIÓN] Iniciando estimación de peso para agave...")
    print(f"[INFO] Entrada: {args.input}")
    print(f"[INFO] Escala estimada: {args.scale} m/px")

    registros = procesar_lote(args.input, args.output, metros_por_pixel=args.scale)
    print(f"[RESUMEN] Total de detecciones: {len(registros)}")


if __name__ == "__main__":
    main()
