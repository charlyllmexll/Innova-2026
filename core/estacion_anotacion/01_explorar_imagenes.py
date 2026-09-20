import os
import cv2

RUTA_BASE_PROYECTO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# =====================================================================
# 1. CONFIGURACIÓN EXPLICATIVA DE RUTAS
# =====================================================================
# Definimos las rutas usando carpetas relativas desde la raíz del proyecto.
# Modifique estos nombres si en su explorador de archivos local se llaman distinto.

DIR_IMAGENES_CRUDAS = os.path.join(RUTA_BASE_PROYECTO, "data", "imagenes_crudas")
DIR_REPORTES        = os.path.join(RUTA_BASE_PROYECTO, "data", "reports")
ARCHIVO_REPORTE     = os.path.join(DIR_REPORTES, "exploration_report.txt")

# Aseguramos de forma automática que la carpeta de reportes exista en el disco duro.
# Si no existe, Python la crea para evitar errores de escritura.
os.makedirs(DIR_REPORTES, exist_ok=True)

# =====================================================================
# 2. AUDITORÍA ÓPTICA DEL DATASET
# =====================================================================
print("[AGROVISIÓN CORE] Iniciando auditoría de imágenes de entrada...")

# Validamos si la carpeta de imágenes crudas existe antes de intentar leerla
if not os.path.exists(DIR_IMAGENES_CRUDAS):
    print(f"[ERROR CRÍTICO] No se encontró la carpeta: '{DIR_IMAGENES_CRUDAS}'")
    print("Por favor, créela y coloque las fotos del dron de 20 metros ahí antes de continuar.")
    exit()

# Escaneamos la carpeta y filtramos únicamente archivos con extensiones de imagen válidas
archivos_en_carpeta = os.listdir(DIR_IMAGENES_CRUDAS)
imagenes_validas = [f for f in archivos_en_carpeta if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

total_imagenes = len(imagenes_validas)
print(f"[INFO] Se detectaron {total_imagenes} archivos de imagen potenciales para procesamiento.")

# Listas internas para clasificar los hallazgos del control de calidad
reporte_lineas = []
imagenes_aptas = 0
imagenes_rechazadas = 0

reporte_lineas.append("=========================================================\n")
reporte_lineas.append("        AGROVISIÓN ANALYTICS - REPORTE DE AUDITORÍA ÓPTICA\n")
reporte_lineas.append("=========================================================\n\n")
reporte_lineas.append(f"Total de archivos analizados: {total_imagenes}\n\n")

# Bucle principal: Analizamos la física y dimensiones de cada foto
for index, img_name in enumerate(imagenes_validas, start=1):
    ruta_completa = os.path.join(DIR_IMAGENES_CRUDAS, img_name)
    
    # Intentamos abrir la imagen usando OpenCV para comprobar que no esté corrupta
    img = cv2.imread(ruta_completa)
    
    if img is None:
        reporte_lineas.append(f"[{index:03d}] RECHAZADA: {img_name} -> Archivo corrupto o ilegible por OpenCV.\n")
        imagenes_rechazadas += 1
        continue
        
    # Extraemos las dimensiones reales en píxeles (Alto, Ancho, Canales de color)
    alto, ancho, canales = img.shape
    
    # CONTROL DE CALIDAD: Como volamos a 20m, las fotos deben tener alta resolución.
    # Si la foto es muy pequeña (menor a 1000px), el GSD milimétrico se pierde y SAM fallará.
    if ancho < 1000 or alto < 1000:
        reporte_lineas.append(f"[{index:03d}] RECHAZADA: {img_name} -> Dimensiones insuficientes ({ancho}x{alto}). Pierde GSD milimétrico.\n")
        imagenes_rechazadas += 1
    else:
        reporte_lineas.append(f"[{index:03d}] APTA: {img_name} -> Dimensiones: {ancho}x{alto} píxeles | Canales: {canales}\n")
        imagenes_aptas += 1

# Resumen final de la auditoría
reporte_lineas.append("\n---------------------------------------------------------\n")
reporte_lineas.append("RESUMEN FINAL DEL LOTE:\n")
reporte_lineas.append(f" - Imágenes aptas para procesamiento por IA: {imagenes_aptas}\n")
reporte_lineas.append(f" - Imágenes rechazadas por control de calidad: {imagenes_rechazadas}\n")

# =====================================================================
# 3. ESCRITURA EN CARPETA DE REPORTES
# =====================================================================
# Guardamos todo el diagnóstico en nuestro nuevo destino ordenado
with open(ARCHIVO_REPORTE, "w", encoding="utf-8") as f:
    f.writelines(reporte_lineas)

print(f"[ÉXITO] Auditoría concluida. Reporte detallado guardado en: '{ARCHIVO_REPORTE}'")
print(f"[RESULTADO] {imagenes_aptas} imágenes listas para la fase de anotación automática.")
