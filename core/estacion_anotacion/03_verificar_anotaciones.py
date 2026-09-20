import os
import cv2

RUTA_BASE_PROYECTO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# =====================================================================
# 1. CONFIGURACIÓN DE CARPETAS DE CONTROL DE CALIDAD
# =====================================================================
DIR_IMAGENES_CRUDAS = os.path.join(RUTA_BASE_PROYECTO, "data", "imagenes_crudas")
DIR_ETIQUETAS_YOLO  = os.path.join(RUTA_BASE_PROYECTO, "data", "imagenes_anotadas")
DIR_VERIFICADAS     = os.path.join(RUTA_BASE_PROYECTO, "data", "imagenes_verificadas")

# Nos aseguramos de que la carpeta de salida visual exista
os.makedirs(DIR_VERIFICADAS, exist_ok=True)

print("[AGROVISIÓN CORE] Generando reportes visuales de control de calidad (QA)...")

# Buscamos los archivos de etiquetas generados por la IA
if not os.path.exists(DIR_ETIQUETAS_YOLO):
    print(f"[ERROR] No existe la carpeta de anotaciones: '{DIR_ETIQUETAS_YOLO}'")
    exit()

archivos_txt = [f for f in os.listdir(DIR_ETIQUETAS_YOLO) if f.lower().endswith('.txt')]

if not archivos_txt:
    print("[AVISO] No se encontraron archivos de etiquetas .txt para verificar.")
    exit()

# =====================================================================
# 2. DIBUJO GEOMÉTRICO CON OPENCV
# =====================================================================
for index, nombre_txt in enumerate(archivos_txt, start=1):
    nombre_base = os.path.splitext(nombre_txt)[0]
    
    # Buscamos la imagen original correspondiente (puede ser .jpg, .png o .jpeg)
    ruta_img = None
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        posible_ruta = os.path.join(DIR_IMAGENES_CRUDAS, f"{nombre_base}{ext}")
        if os.path.exists(posible_ruta):
            ruta_img = posible_ruta
            break
            
    if not ruta_img:
        print(f"[ADVERTENCIA] No se encontró la imagen original para las etiquetas: {nombre_txt}")
        continue
        
    # Cargamos la imagen original limpiamente
    img = cv2.imread(ruta_img)
    alto_img, ancho_img, _ = img.shape
    
    # Leemos las coordenadas normalizadas del formato YOLO
    ruta_txt = os.path.join(DIR_ETIQUETAS_YOLO, nombre_txt)
    with open(ruta_txt, "r", encoding="utf-8") as f:
        lineas = f.readlines()
        
    conteo_coronas = 0
    
    for linea in lineas:
        partes = linea.strip().split()
        if len(partes) != 5:
            continue
            
        clase, centro_x, centro_y, ancho_norm, alto_norm = map(float, partes)
        
        # OPERACIÓN MATEMÁTICA INVERSA: Desnormalizar de 0.0-1.0 a píxeles absolutos de pantalla
        w_caja = int(ancho_norm * ancho_img)
        h_caja = int(alto_norm * alto_img)
        x_min  = int((centro_x * ancho_img) - (w_caja / 2))
        y_min  = int((centro_y * alto_img) - (h_caja / 2))
        
        # Dibujamos un rectángulo verde (grosor de 2px) rodeando la corona del agave
        cv2.rectangle(img, (x_min, y_min), (x_min + w_caja, y_min + h_caja), (0, 255, 0), 2)
        
        # Opcional: Colocar un pequeño texto indicando la clase o conteo
        conteo_coronas += 1

    # Colocamos un banner informativo de Agrovisión en la esquina superior izquierda de la foto
    cv2.putText(img, f"Agaves Detectados por IA: {conteo_coronas}", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    
    # Guardamos la imagen resultante con los gráficos encima
    ruta_salida_visual = os.path.join(DIR_VERIFICADAS, f"VERIF_{nombre_base}.jpg")
    cv2.imwrite(ruta_salida_visual, img)
    print(f"[{index}/{len(archivos_txt)}] Renderizado exitoso -> VERIF_{nombre_base}.jpg ({conteo_coronas} agaves)")

print("\n[PROCESO COMPLETADO] El reporte visual de control de calidad está listo.")
