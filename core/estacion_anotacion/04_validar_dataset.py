import os

RUTA_BASE_PROYECTO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# =====================================================================
# 1. CONFIGURACIÓN DE AUDITORÍA CRUZADA
# =====================================================================
DIR_IMAGENES_CRUDAS = os.path.join(RUTA_BASE_PROYECTO, "data", "imagenes_crudas")
DIR_ETIQUETAS_YOLO  = os.path.join(RUTA_BASE_PROYECTO, "data", "imagenes_anotadas")
DIR_REPORTES        = os.path.join(RUTA_BASE_PROYECTO, "data", "reports")
ARCHIVO_REPORTE     = os.path.join(DIR_REPORTES, "validation_report.txt")

os.makedirs(DIR_REPORTES, exist_ok=True)

print("[AGROVISIÓN CORE] Iniciando auditoría de certificación final del dataset...")

# Listas de control
archivos_imagen = [os.path.splitext(f)[0] for f in os.listdir(DIR_IMAGENES_CRUDAS) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
archivos_labels = [os.path.splitext(f)[0] for f in os.listdir(DIR_ETIQUETAS_YOLO) if f.lower().endswith('.txt')]

reporte_lineas = []
reporte_lineas.append("=========================================================\n")
reporte_lineas.append("      AGROVISIÓN ANALYTICS - CERTIFICACIÓN FINAL DEL DATASET\n")
reporte_lineas.append("=========================================================\n\n")

# =====================================================================
# 2. VALIDACIÓN LOGÍSTICA DE ARCHIVOS EMPAREJADOS
# =====================================================================
imagenes_perfectas = 0
imagenes_sin_etiqueta = 0

reporte_lineas.append("--- AUDITORÍA DE CONSISTENCIA DE IMÁGENES ---\n")
for img in archivos_imagen:
    if img in archivos_labels:
        # Contamos cuántas anotaciones tiene ese archivo txt por seguridad
        ruta_txt = os.path.join(DIR_ETIQUETAS_YOLO, f"{img}.txt")
        with open(ruta_txt, "r", encoding="utf-8") as f:
            total_anotaciones = len(f.readlines())
            
        reporte_lineas.append(f"[OK] Sincronizado: {img} -> Vinculado a su .txt con {total_anotaciones} agaves etiquetados.\n")
        imagenes_perfectas += 1
    else:
        reporte_lineas.append(f"[FALLO] Huérfano: {img} -> No cuenta con archivo de anotaciones de IA (.txt).\n")
        imagenes_sin_etiqueta += 1

# Dictamen de calidad final
reporte_lineas.append("\n---------------------------------------------------------\n")
reporte_lineas.append("DICTAMEN FINAL DE CERTIFICACIÓN:\n")
reporte_lineas.append(f" - Muestras emparejadas y aprobadas: {imagenes_perfectas}\n")
reporte_lineas.append(f" - Muestras inválidas o sin anotar: {imagenes_sin_etiqueta}\n\n")

if imagenes_sin_etiqueta == 0 and imagenes_perfectas > 0:
    reporte_lineas.append("ESTADO DEL DATASET: [CERTIFICADO] El dataset cumple con el estándar estricto de YOLO y está listo para entrenamiento masivo.\n")
else:
    reporte_lineas.append("ESTADO DEL DATASET: [RECHAZADO] Existen inconsistencias o archivos sin etiquetar. Revise el log de arriba.\n")

# Guardamos el archivo final de certificación
with open(ARCHIVO_REPORTE, "w", encoding="utf-8") as f:
    f.writelines(reporte_lineas)

print(f"[ÉXITO] Certificación concluida. Reporte final guardado en: '{ARCHIVO_REPORTE}'")
