import os
import re
import cv2  # Importamos OpenCV de forma segura para la lectura base
import customtkinter as ctk
from PIL import Image, ImageTk  # Puente de traducción de matrices a imágenes visuales
import torch
import numpy as np                                               # Motor de aceleración por GPU CUDA
from segment_anything import sam_model_registry, SamPredictor  # Modelos y Predictor Quirúrgico de Meta
import subprocess  # Permite invocar submódulos del pipeline en terminales secundarias
import threading  # Motor para ejecutar procesos pesados sin congelar la interfaz gráfica

# Configuración estética global de la interfaz al estilo Agrovisión Analytics
ctk.set_appearance_mode("Dark")       
ctk.set_default_color_theme("green")   

class AppEtiquetadoAgave(ctk.CTk):
    def __init__(self):
        super().__init__()

        # =====================================================================
        # 1. PROPIEDADES DE LA VENTANA PRINCIPAL Y RUTAS
        # =====================================================================
        self.title("Agrovisión Analytics - Estación de Etiquetado Híbrida v1.3")
        self.geometry("1250x850")
        self.minsize(1000, 650)
        
        # Arquitectura de carpetas del sistema local
        self.ruta_base_proyecto = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.dir_imagenes = os.path.join(self.ruta_base_proyecto, "data", "imagenes_crudas")
        self.img_opencv_original = None
        self.img_opencv_filtrada = None
        self.ancho_render = 1
        self.alto_render = 1
        self.escala_zoom = 1.0           
        self.centro_zoom_x = 0           
        self.centro_zoom_y = 0           
        self.ruta_sam = r"C:\Users\charl\Downloads\sam_vit_h_4b8939.pth"  
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  
        self.predictor = None  
        self.lineas_yolo_acumuladas = [] 
        self.mascara_maestra_acumulada = None  
        
        # Control del lote de fotos aéreas
        self.lista_fotos = []
        self.foto_actual_index = 0
        
        # =====================================================================
        # 2. DISEÑO DEL PANEL LATERAL DE CONTROLES
        # =====================================================================
        self.panel_lateral = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.panel_lateral.pack(side="left", fill="y", padx=0, pady=0)

        # Logotipo corporativo
        self.logo_label = ctk.CTkLabel(
            self.panel_lateral, 
            text="AGROVISIÓN", 
            font=ctk.CTkFont(size=22, weight="bold")
        )
        self.logo_label.pack(padx=20, pady=(20, 5))
        
        self.sub_label = ctk.CTkLabel(
            self.panel_lateral, 
            text="ANALYTICS CORE", 
            text_color="#4ADE80", 
            font=ctk.CTkFont(size=11, weight="bold")
        )
        self.sub_label.pack(padx=20, pady=(0, 20))

        # Sección: Control de Datos y Lotes
        self.btn_cargar = ctk.CTkButton(
            self.panel_lateral, 
            text="📁 Seleccionar Lote", 
            font=ctk.CTkFont(weight="bold"), 
            command=self.cargar_lote_imagenes  
        )
        self.btn_cargar.pack(padx=20, pady=10, fill="x")

        # Botones de navegación entre fotos del predio
        self.frame_navegacion = ctk.CTkFrame(self.panel_lateral, fg_color="transparent")
        self.frame_navegacion.pack(padx=20, pady=5, fill="x")
        
        self.btn_atras = ctk.CTkButton(
            self.frame_navegacion, text="◀ Atrás", width=100, 
            command=self.foto_anterior, state="disabled" 
        )
        self.btn_atras.pack(side="left", expand=True, padx=(0, 5))
        
        self.btn_sig = ctk.CTkButton(
            self.frame_navegacion, text="Sig ▶", width=100, 
            command=self.foto_siguiente, state="disabled" 
        )
        self.btn_sig.pack(side="right", expand=True, padx=(5, 0))

        # Selector del Backend Híbrido
        self.lbl_modo = ctk.CTkLabel(
            self.panel_lateral, 
            text="⚙️ Seteo del Backend", 
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.lbl_modo.pack(padx=20, pady=(15, 5), anchor="w")
        
        self.switch_modo = ctk.CTkSwitch(
            self.panel_lateral, 
            text="Modo Automático Masivo", 
            font=ctk.CTkFont(size=12),
            command=self.alternar_modo_operacion
        )
        self.switch_modo.pack(padx=20, pady=5, anchor="w")

        # NUEVO: Botón de ejecución en bloque para procesamiento secuencial masivo
        self.btn_ejecutar_pipeline = ctk.CTkButton(
            self.panel_lateral,
            text="🚀 Ejecutar Pipeline IA",
            fg_color="#3B82F6",
            hover_color="#1D4ED8",
            font=ctk.CTkFont(weight="bold"),
            command=self.iniciar_hilo_pipeline,
            state="disabled"  # Se activa únicamente en Modo Automático Masivo
        )
        self.btn_ejecutar_pipeline.pack(padx=20, pady=5, fill="x")

        # Sección de Filtros Ópticos
        self.lbl_filtros = ctk.CTkLabel(
            self.panel_lateral, 
            text="🖼️ Filtros Ópticos", 
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.lbl_filtros.pack(padx=20, pady=(15, 5), anchor="w")

        self.switch_clahe = ctk.CTkSwitch(
            self.panel_lateral, 
            text="Filtro Contraste CLAHE", 
            font=ctk.CTkFont(size=12),
            command=self.actualizar_visor_filtros
        )
        self.switch_clahe.pack(padx=20, pady=5, anchor="w")
        self.switch_clahe.select()

        # NUEVO: Botón para costura homográfica de ortomosaicos en la parcela
        self.btn_ortomosaico = ctk.CTkButton(
            self.panel_lateral,
            text="🧩 Generar Ortomosaico",
            fg_color="#8B5CF6",
            hover_color="#6D28D9",
            font=ctk.CTkFont(weight="bold"),
            command=self.iniciar_hilo_ortomosaico
        )
        self.btn_ortomosaico.pack(padx=20, pady=10, fill="x")

        # =====================================================================
        # NUEVO: COMPONENTES VISUALES DE MONITOREO Y PROGRESO (BARRA DE ESTADO)
        # =====================================================================
        self.lbl_progreso = ctk.CTkLabel(
            self.panel_lateral,
            text="Estado: Estación Lista",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#E2E8F0"
        )
        self.lbl_progreso.pack(padx=20, pady=(20, 2), anchor="w")

        self.barra_progreso = ctk.CTkProgressBar(self.panel_lateral)
        self.barra_progreso.pack(padx=20, pady=5, fill="x")
        self.barra_progreso.set(0.0)

        # Botón de Guardado del Dataset YOLO (.txt)
        self.btn_guardar = ctk.CTkButton(
            self.panel_lateral, 
            text="💾 Guardar Anotaciones (.txt)", 
            fg_color="#10B981", 
            hover_color="#059669", 
            font=ctk.CTkFont(weight="bold"), 
            command=self.guardar_etiquetas_yolo
        )
        self.btn_guardar.pack(padx=20, pady=20, fill="x", side="bottom")

        # Indicador de Hardware
        self.lbl_hardware = ctk.CTkLabel(
            self.panel_lateral, 
            text=f"Hardware Detectado: {self.device.upper()}", 
            font=ctk.CTkFont(size=11), 
            text_color="#94A3B8"
        )
        self.lbl_hardware.pack(padx=20, pady=5, side="bottom", anchor="w")

        # =====================================================================
        # 3. DISEÑO DEL PANEL CENTRAL (VISOR INTERACTIVO PÚBLICO)
        # =====================================================================
        self.panel_visor = ctk.CTkFrame(self, fg_color="#0F172A")
        self.panel_visor.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.canvas_foto = ctk.CTkLabel(
            self.panel_visor, 
            text="[ ESTACIÓN AGROVISIÓN ACTIVE ]\n\nHaga clic en 'Seleccionar Lote' para buscar fotos reales\nen la carpeta data/imagenes_crudas.", 
            text_color="#64748B", 
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.canvas_foto.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Mapeo de eventos del mouse
        self.canvas_foto.bind("<Button-1>", lambda event: self.capturar_clic_operador(event))

        # Mapeo del teclado para control del operador
        self.bind("<plus>", lambda event: self.ejecutar_zoom(1.2))   
        self.bind("<minus>", lambda event: self.ejecutar_zoom(0.8))  
        self.bind("<r>", lambda event: self.reiniciar_zoom())        
        self.focus_set() 

    # =====================================================================
    # 4. LÓGICA ACTIVA - MANEJO DE IMÁGENES Y NAVEGACIÓN
    # =====================================================================
    def cargar_lote_imagenes(self):
        """Escanea físicamente el directorio para buscar las fotos del dron"""
        if not os.path.exists(self.dir_imagenes):
            self.canvas_foto.configure(
                text=f"[FALLO TÉCNICO]\nNo se encontró la carpeta: {self.dir_imagenes}\nPor favor créela en el directorio del proyecto."
            )
            return

        archivos = os.listdir(self.dir_imagenes)
        self.lista_fotos = [f for f in archivos if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if self.lista_fotos:
            self.foto_actual_index = 0
            self.btn_atras.configure(state="normal")
            self.btn_sig.configure(state="normal")
            self.actualizar_texto_visor_bloque1()
        else:
            self.canvas_foto.configure(
                text=f"[LOTE VACÍO]\nCarpeta encontrada, pero no contiene imágenes (.png, .jpg).\nColoque las fotos del dron ahí."
            )

    def foto_siguiente(self):
        """Avanza al siguiente archivo de forma cíclica"""
        if self.lista_fotos:
            self.foto_actual_index = (self.foto_actual_index + 1) % len(self.lista_fotos)
            self.reiniciar_zoom_silencioso()
            self.actualizar_texto_visor_bloque1()

    def foto_anterior(self):
        """Retrocede al archivo anterior de forma cíclica"""
        if self.lista_fotos:
            self.foto_actual_index = (self.foto_actual_index - 1) % len(self.lista_fotos)
            self.reiniciar_zoom_silencioso()
            self.actualizar_texto_visor_bloque1()

    def reiniciar_zoom_silencioso(self):
        """Limpia el estado geométrico del visor sin re-renderizar la pantalla inmediatamente"""
        self.escala_zoom = 1.0
        self.zoom_coords = None

    def actualizar_texto_visor_bloque1(self):
        """Lee la imagen actual del lote, procesa filtros ópticos y la manda a pantalla"""
        self.lineas_yolo_acumuladas = []
        self.mascara_maestra_acumulada = None

        if not self.lista_fotos:
            return

        nombre_archivo = self.lista_fotos[self.foto_actual_index]
        ruta_completa = os.path.join(self.dir_imagenes, nombre_archivo)
        self.img_opencv_original = cv2.imread(ruta_completa)

        if self.img_opencv_original is None:
            print(f"[ERROR] No se pudo leer la imagen: {nombre_archivo}")
            return

        # Aplicación del contraste adaptativo local CLAHE
        if self.switch_clahe.get():
            lab = cv2.cvtColor(self.img_opencv_original, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            self.img_opencv_filtrada = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
        else:
            self.img_opencv_filtrada = self.img_opencv_original.copy()

        # Acoplamiento del embedding a la memoria de la GPU
        self.acoplar_imagen_a_ia(self.img_opencv_filtrada)

        # Renderizado inicial
        self.renderizar_matriz_en_pantalla_con_capas(self.img_opencv_filtrada)
        print(f"[MOTOR ÓPTICO] Renderizado exitoso para: {nombre_archivo} | Filtro CLAHE: {self.switch_clahe.get()}")

    def actualizar_visor_filtros(self):
        """Permite actualizar el renderizado de inmediato al cambiar el switch del filtro"""
        if self.lista_fotos:
            self.actualizar_texto_visor_bloque1()

    # =====================================================================
    # 5. INTEGRACIÓN DE INTELIGENCIA ARTIFICIAL (SAM CORE)
    # =====================================================================
    def inicializar_modelo_sam(self):
        """Carga el modelo ViT-Huge de Meta en frío en la GPU y activa el Predictor"""
        if not os.path.exists(self.ruta_sam):
            self.canvas_foto.configure(
                text=f"[FALLO DE INFRAESTRUCTURA]\n\nNo se encontró el archivo de pesos:\n{self.ruta_sam}\nPor favor verifique la ruta especificada."
            )
            return

        print(f"[IA CORE] Despertando hardware... Cargando pesos de SAM en {self.device.upper()}...")
        try:
            sam_architecture = sam_model_registry["vit_h"](checkpoint=self.ruta_sam)
            sam_architecture.to(device=self.device)
            self.predictor = SamPredictor(sam_architecture)
            print("[IA CORE] ¡Cerebro de SAM acoplado exitosamente a la interfaz gráfica!")
        except Exception as e:
            print(f"[ERROR DE CARGA] Ocurrió un problema al montar la IA: {e}")

    def acoplar_imagen_a_ia(self, matriz_bgr):
        """Le entrega la matriz actual de OpenCV a SAM para que precalcule las texturas"""
        if self.predictor is not None:
            print("[SAM] Precalculando mapa de texturas de la imagen actual (Embedding)...")
            self.predictor.set_image(matriz_bgr)

    # =====================================================================
    # 6. PIPELINE AUTOMÁTICO Y CAMBIO DE INTERFAZ
    # =====================================================================
    def alternar_modo_operacion(self):
        """Controla el comportamiento y los estados de los botones según el modo seleccionado"""
        if self.switch_modo.get():
            self.btn_ejecutar_pipeline.configure(state="normal")
            self.canvas_foto.configure(text="[MODO AUTOMÁTICO MASIVO ACTIVADO]\n\nPresione el botón 'Ejecutar Pipeline IA' en el panel lateral\npara procesar secuencialmente todo el lote.")
        else:
            self.btn_ejecutar_pipeline.configure(state="disabled")
            self.barra_progreso.set(0.0)
            self.lbl_progreso.configure(text="Estado: Estación Lista", text_color="#E2E8F0")
            self.actualizar_texto_visor_bloque1()

    def resolver_ruta_script_pipeline(self, nombre_script):
        """Busca un script del pipeline tanto por nombre exacto como por variantes con guiones/underscores."""
        if not nombre_script:
            return None

        rutas_a_probar = [
            os.path.join(self.ruta_base_proyecto, "core", "estacion_anotacion", nombre_script),
            os.path.join(os.getcwd(), "core", "estacion_anotacion", nombre_script),
            os.path.join(os.getcwd(), "core", nombre_script),
            os.path.join(os.getcwd(), nombre_script),
            nombre_script,
        ]

        for ruta in rutas_a_probar:
            if os.path.isfile(ruta):
                return ruta

        carpeta_core = os.path.join(self.ruta_base_proyecto, "core", "estacion_anotacion")
        if os.path.isdir(carpeta_core):
            nombre_normalizado = re.sub(r"[^a-z0-9]", "", nombre_script.lower())
            for nombre_archivo in os.listdir(carpeta_core):
                if os.path.isfile(os.path.join(carpeta_core, nombre_archivo)):
                    if re.sub(r"[^a-z0-9]", "", nombre_archivo.lower()) == nombre_normalizado:
                        return os.path.join(carpeta_core, nombre_archivo)

        return None

    def iniciar_hilo_pipeline(self):
        """Detona un hilo secundario para el pipeline, manteniendo viva la UI"""
        hilo = threading.Thread(target=self.hilo_ejecutar_pipeline)
        hilo.daemon = True
        hilo.start()

    def hilo_ejecutar_pipeline(self):
        """Lógica técnica en segundo plano para la llamada en cadena de scripts"""
        scripts_pipeline = [
            "01_explorar_imagenes.py",
            "02_generar_anotaciones_sam.py",
            "03_verificar_anotaciones.py",
            "04_validar_dataset.py"
        ]
        total_pasos = len(scripts_pipeline)
        print("🚀 Iniciando pipeline en segundo plano...")

        for index, script in enumerate(scripts_pipeline):
            porcentaje = index / total_pasos
            self.barra_progreso.set(porcentaje)
            self.lbl_progreso.configure(text=f"Corriendo: {script}", text_color="#3B82F6")
            self.canvas_foto.configure(text=f"⚙️ Procesando: {script}\n\nPor favor espere. La aplicación sigue respondiendo.")

            ruta_final = self.resolver_ruta_script_pipeline(script)
            if not ruta_final:
                print(f"❌ Error: {script} no fue encontrado en las rutas del predio.")
                self.lbl_progreso.configure(text="❌ Error: Script no hallado", text_color="#EF4444")
                self.canvas_foto.configure(text=f"[FALLO CRÍTICO EN PIPELINE]\n\nNo se localizó el archivo: {script}\nAsegúrese de que esté en la raíz o en la carpeta core/.")
                return

            resultado = subprocess.run(["python", ruta_final], capture_output=True, text=True)
            if resultado.returncode != 0:
                print(f"💥 Error en {script}: {resultado.stderr}")
                self.lbl_progreso.configure(text="💥 Error en proceso", text_color="#EF4444")
                self.canvas_foto.configure(text=f"[ERROR EN PIPELINE]\nFallo interno en {script}.\nDetalles técnicos impresos en consola.")
                return

        self.barra_progreso.set(1.0)
        self.lbl_progreso.configure(text="✅ Pipeline Finalizado", text_color="#10B981")
        self.canvas_foto.configure(text="✅ ¡Lote Completo Procesado con Éxito!\n\nLos archivos de etiquetas YOLO e inventariado han sido almacenados.")

    def ejecutar_pipeline_completo_local(self):
        """
        Ejecuta de forma consecutiva la cadena de scripts del core del proyecto
        mediante procesos aislados del sistema operativo.
        """
        ruta_carpeta_core = os.path.join(self.ruta_base_proyecto, "core")
        scripts_pipeline = [
            "01_explorar_imagenes.py",
            "02_generar_anotaciones_sam.py",
            "03_verificar_anotaciones.py",
            "04_validar_dataset.py"
        ]
        self.canvas_foto.configure(text="⚙️ Procesando lote completo con Inteligencia Artificial...\nPor favor espere. Revise los logs de la terminal de Python.")
        self.update()
        print("🚀 Iniciando ejecución en cadena del pipeline de IA...")

        for script in scripts_pipeline:
            ruta_script = self.resolver_ruta_script_pipeline(script)
            if not ruta_script:
                print(f"❌ Error: El componente {script} no se encontró en el disco.")
                self.canvas_foto.configure(text=f"[FALLO PIPELINE]\nNo se encontró el script: {script}")
                return

            print(f"⚙️ Corriendo módulo: {script}...")
            resultado = subprocess.run(["python", ruta_script], capture_output=True, text=True)
            if resultado.returncode != 0:
                print(f"💥 Error crítico dentro del script {script}:")
                print(resultado.stderr)
                self.canvas_foto.configure(text=f"[ERROR EN PIPELINE]\nFallo en {script}.\nDetalle en consola.")
                return

        print("🎯 ¡Pipeline completado con éxito!")
        self.canvas_foto.configure(text="✅ ¡Lote Procesado e Inventariado de Forma Masiva!\n\nLos reportes y marcas YOLO automatizados han sido guardados en el disco.")

    # =====================================================================
    # 7. MOTOR DE RECONSTRUCCIÓN ESPACIAL (ORTOMOSAICOS)
    # =====================================================================
    def iniciar_hilo_ortomosaico(self):
        """Detona un hilo secundario para la costura de imágenes, evitando que se congele la app"""
        hilo = threading.Thread(target=self.hilo_generar_ortomosaico)
        hilo.daemon = True
        hilo.start()

    def hilo_generar_ortomosaico(self):
        if not self.lista_fotos or len(self.lista_fotos) < 2:
            self.canvas_foto.configure(text="[FALLO DE TRASLAPE]\nSe necesitan por lo menos 2 capturas aéreas para alinear el terreno.")
            return

        self.barra_progreso.set(0.3)
        self.lbl_progreso.configure(text="🧩 Uniendo Parcelas...", text_color="#8B5CF6")
        self.canvas_foto.configure(text="🧩 Analizando matrices homográficas y uniendo tomas...\nLa barra lateral muestra el avance. No toque la ventana.")

        imagenes_matriz = []
        for foto in self.lista_fotos:
            ruta = os.path.join(self.dir_imagenes, foto)
            img = cv2.imread(ruta)
            if img is not None:
                img_res = cv2.resize(img, (640, 480))
                imagenes_matriz.append(img_res)

        costurero_rpas = cv2.Stitcher_create(cv2.Stitcher_SCANS)
        self.barra_progreso.set(0.6)
        estado, ortomosaico_final = costurero_rpas.stitch(imagenes_matriz)

        if estado == cv2.Stitcher_OK:
            dir_salida = os.path.join(self.ruta_base_proyecto, "data", "output")
            os.makedirs(dir_salida, exist_ok=True)
            ruta_cache = os.path.join(dir_salida, "ortomosaico_parcela.png")
            cv2.imwrite(ruta_cache, ortomosaico_final)
            self.img_opencv_filtrada = ortomosaico_final.copy()
            self.img_opencv_original = ortomosaico_final.copy()
            self.reiniciar_zoom_silencioso()
            self.renderizar_matriz_en_pantalla_con_capas(ortomosaico_final)
            self.barra_progreso.set(1.0)
            self.lbl_progreso.configure(text="✅ Ortomosaico Listo", text_color="#10B981")
        else:
            print(f"❌ Error OpenCV Stitcher: {estado}")
            self.barra_progreso.set(0.0)
            self.lbl_progreso.configure(text="❌ Fallo en Costura", text_color="#EF4444")
            self.canvas_foto.configure(text=f"[FALLO GEOMÉTRICO - CÓDIGO {estado}]\n\nLas imágenes cargadas no se pudieron alinear debido a bajo traslape\no variaciones extremas en las sombras del cultivo.")

    def generar_y_desplegar_ortomosaico(self):
        """Toma todas las imágenes del lote y realiza una costura de alta precisión con OpenCV"""
        self.iniciar_hilo_ortomosaico()

    # =====================================================================
    # 8. MÓDULO DE INTERACCIÓN POR CLICS E INFERENCIA PRE CREADA
    # =====================================================================
    def capturar_clic_operador(self, event):
        """Segmenta con SAM, mapea clics bajo zoom y acumula marcas de forma persistente"""
        if self.img_opencv_original is None or self.predictor is None:
            return

        alto_real, ancho_real, _ = self.img_opencv_original.shape
        # Traducción geométrica de coordenadas considerando el factor de zoom activo
        if hasattr(self, 'zoom_coords') and self.zoom_coords is not None:
            x1_recorte, y1_recorte, x2_recorte, y2_recorte = self.zoom_coords
            ancho_recorte = x2_recorte - x1_recorte
            alto_recorte = y2_recorte - y1_recorte
            x_real = int(x1_recorte + (event.x * (ancho_recorte / self.ancho_render)))
            y_real = int(y1_recorte + (event.y * (alto_recorte / self.alto_render)))
        else:
            x_real = int(event.x * (ancho_real / self.ancho_render))
            y_real = int(event.y * (alto_real / self.alto_render))

        # Ajuste de límites seguros
        x_real = max(0, min(x_real, ancho_real - 1))
        y_real = max(0, min(y_real, alto_real - 1))
        print(f"[CLIC EN REGISTRO] Coordenada Real -> X: {x_real}, Y: {y_real}")

        puntos_input = np.array([[x_real, y_real]])
        etiquetas_input = np.array([1])
        masks, scores, logits = self.predictor.predict(
            point_coords=puntos_input,
            point_labels=etiquetas_input,
            multimask_output=True
        )
        index_optimo = np.argmin(scores) if np.max(scores) > 0.9 else 0
        mascara_optima = masks[index_optimo]

        # Extracción matemática del Bounding Box
        indices_y, indices_x = np.where(mascara_optima)
        if len(indices_x) == 0 or len(indices_y) == 0:
            print("[IA] Alerta: SAM no pudo estructurar contornos válidos en este píxel.")
            return

        x_min, x_max = np.min(indices_x), np.max(indices_x)
        y_min, y_max = np.min(indices_y), np.max(indices_y)
        ancho_caja_abs = x_max - x_min
        alto_caja_abs = y_max - y_min

        # Acumulación de las máscaras para evitar que se borren marcas pasadas
        if self.mascara_maestra_acumulada is None:
            self.mascara_maestra_acumulada = np.zeros((alto_real, ancho_real), dtype=bool)
        self.mascara_maestra_acumulada = np.logical_or(self.mascara_maestra_acumulada, mascara_optima)

        # Normalización estricta YOLO
        centro_x_norm = (x_min + (ancho_caja_abs / 2)) / ancho_real
        centro_y_norm = (y_min + (alto_caja_abs / 2)) / alto_real
        ancho_norm = ancho_caja_abs / ancho_real
        alto_norm = alto_caja_abs / alto_real
        linea_yolo = f"0 {centro_x_norm:.6f} {centro_y_norm:.6f} {ancho_norm:.6f} {alto_norm:.6f}\n"
        self.lineas_yolo_acumuladas.append(linea_yolo)

        # Dibujo analítico sobre la imagen
        img_con_graficos = self.img_opencv_filtrada.copy()
        color_verde = np.array([0, 255, 0], dtype=np.uint8)
        img_con_graficos[self.mascara_maestra_acumulada] = (img_con_graficos[self.mascara_maestra_acumulada] * 0.5 + color_verde * 0.5).astype(np.uint8)
        cv2.rectangle(img_con_graficos, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
        cv2.circle(img_con_graficos, (x_real, y_real), 8, (0, 0, 255), -1)

        # Mantenimiento preciso del encuadre si existiese zoom activo
        if hasattr(self, 'zoom_coords') and self.zoom_coords is not None:
            x1, y1, x2, y2 = self.zoom_coords
            matriz_final_render = img_con_graficos[y1:y2, x1:x2]
        else:
            matriz_final_render = img_con_graficos

        self.renderizar_matriz_en_pantalla_con_capas(matriz_final_render)
        print(f"[RENDER] Anotaciones vivas en esta sesión: {len(self.lineas_yolo_acumuladas)} agaves.")

    def guardar_etiquetas_yolo(self):
        """Toma las líneas acumuladas en memoria y escribe el archivo .txt oficial de YOLO"""
        if not self.lista_fotos or not self.lineas_yolo_acumuladas:
            print("[SISTEMA] No hay anotaciones válidas creadas para guardar en este archivo.")
            return

        nombre_foto = self.lista_fotos[self.foto_actual_index]
        nombre_base = os.path.splitext(nombre_foto)[0]
        dir_salida = os.path.join(self.ruta_base_proyecto, "data", "imagenes_anotadas")
        os.makedirs(dir_salida, exist_ok=True)
        ruta_txt_final = os.path.join(dir_salida, f"{nombre_base}.txt")

        try:
            with open(ruta_txt_final, "w", encoding="utf-8") as f_txt:
                f_txt.writelines(self.lineas_yolo_acumuladas)
            print(f"[EXPORTADOR] ¡ÉXITO! Guardado en: {ruta_txt_final}")
            self.lbl_progreso.configure(text="💾 Anotaciones Guardadas", text_color="#10B981")
        except Exception as e:
            print(f"[ERROR EXPORTADOR] No se pudo escribir en disco: {e}")

    # =====================================================================
    # 9. SISTEMA DE ZOOM DINÁMICO
    # =====================================================================
    def ejecutar_zoom(self, factor):
        """Modifica la escala de zoom guardando la posición del cursor del mouse"""
        x_mouse = self.winfo_pointerx() - self.panel_visor.winfo_rootx()
        y_mouse = self.winfo_pointery() - self.panel_visor.winfo_rooty()
        self.centro_zoom_x = max(0, min(x_mouse / self.ancho_render, 1.0))
        self.centro_zoom_y = max(0, min(y_mouse / self.alto_render, 1.0))
        self.escala_zoom = max(1.0, min(self.escala_zoom * factor, 8.0))
        self.actualizar_visor_con_zoom()

    def reiniciar_zoom(self):
        self.reiniciar_zoom_silencioso()
        print("[ZOOM] Vista restablecida al tamaño original.")
        if self.lista_fotos:
            self.actualizar_texto_visor_bloque1()

    def actualizar_visor_con_zoom(self):
        """Recorta la imagen basándose en las coordenadas del zoom activo y guarda el encuadre"""
        if self.img_opencv_filtrada is None:
            return

        alto_orig, ancho_orig, _ = self.img_opencv_filtrada.shape
        nuevo_ancho = int(ancho_orig / self.escala_zoom)
        nuevo_alto = int(alto_orig / self.escala_zoom)
        foco_x = int(ancho_orig * (self.centro_zoom_x if self.escala_zoom > 1.0 else 0.5))
        foco_y = int(alto_orig * (self.centro_zoom_y if self.escala_zoom > 1.0 else 0.5))
        x1 = max(0, foco_x - int(nuevo_ancho / 2))
        y1 = max(0, foco_y - int(nuevo_alto / 2))
        if x1 + nuevo_ancho > ancho_orig:
            x1 = ancho_orig - nuevo_ancho
        if y1 + nuevo_alto > alto_orig:
            y1 = alto_orig - nuevo_alto
        x2 = min(ancho_orig, x1 + nuevo_ancho)
        y2 = min(alto_orig, y1 + nuevo_alto)
        self.zoom_coords = (x1, y1, x2, y2)
        matriz_recortada = self.img_opencv_filtrada[y1:y2, x1:x2]
        self.renderizar_matriz_en_pantalla_con_capas(matriz_recortada)

    def renderizar_matriz_en_pantalla_con_capas(self, matriz_bgr):
        """Método auxiliar encargado de la transformación de matrices OpenCV a objetos legibles por la GUI"""
        img_rgb = cv2.cvtColor(matriz_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil.thumbnail((800, 500))
        self.ancho_render, self.alto_render = img_pil.size
        img_tk = ImageTk.PhotoImage(img_pil)
        self.canvas_foto.configure(image=img_tk, text="")
        self.canvas_foto.image = img_tk

    # =====================================================================
    # 10. DISPARADOR DE LA APLICACIÓN
    # =====================================================================
if __name__ == "__main__":
    app = AppEtiquetadoAgave()
    app.inicializar_modelo_sam()
    app.mainloop()
        