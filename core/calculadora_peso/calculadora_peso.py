import os
import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk

from motor_estimacion import procesar_imagen


ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")


class AppEstimacionAgave(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Agrovisión - Estimador de Peso de Piñas de Agave")
        self.geometry("1200x760")
        self.minsize(900, 600)

        self.dir_imagenes = os.path.join(os.path.dirname(__file__), "..", "..", "data", "imagenes_crudas")
        self.dir_imagenes = os.path.abspath(self.dir_imagenes)
        self.img_actual = None
        self.resultados = []
        self.archivos = []
        self.indice = 0
        self.resumen_por_imagen = []

        self.frame_lateral = ctk.CTkFrame(self, width=300)
        self.frame_lateral.pack(side="left", fill="y", padx=10, pady=10)

        ctk.CTkLabel(self.frame_lateral, text="AGROVISIÓN", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=(20, 5))
        ctk.CTkLabel(self.frame_lateral, text="ESTIMADOR DE PESO",
                     text_color="#4ADE80", font=ctk.CTkFont(size=12, weight="bold")).pack(pady=(0, 20))

        self.btn_cargar = ctk.CTkButton(self.frame_lateral, text="📁 Cargar lote", command=self.cargar_lote)
        self.btn_cargar.pack(padx=20, pady=10, fill="x")

        self.frame_nav = ctk.CTkFrame(self.frame_lateral, fg_color="transparent")
        self.frame_nav.pack(padx=20, pady=5, fill="x")

        self.btn_anterior = ctk.CTkButton(self.frame_nav, text="◀ Anterior", command=self.anterior_imagen, state="disabled")
        self.btn_anterior.pack(side="left", expand=True, padx=(0, 5))

        self.btn_siguiente = ctk.CTkButton(self.frame_nav, text="Siguiente ▶", command=self.siguiente_imagen, state="disabled")
        self.btn_siguiente.pack(side="right", expand=True, padx=(5, 0))

        self.btn_procesar = ctk.CTkButton(self.frame_lateral, text="⚙️ Procesar imágenes", command=self.procesar_lote)
        self.btn_procesar.pack(padx=20, pady=10, fill="x")

        self.lbl_resumen = ctk.CTkLabel(self.frame_lateral, text="Resumen: sin datos", font=ctk.CTkFont(size=12, weight="bold"), text_color="#E2E8F0")
        self.lbl_resumen.pack(padx=20, pady=(5, 10), anchor="w")

        self.txt_estado = ctk.CTkTextbox(self.frame_lateral, height=150)
        self.txt_estado.pack(padx=20, pady=(20, 10), fill="both", expand=True)

        self.frame_visor = ctk.CTkFrame(self, fg_color="#0F172A")
        self.frame_visor.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.label_imagen = ctk.CTkLabel(self.frame_visor, text="Sin imagen cargada", text_color="#94A3B8")
        self.label_imagen.pack(fill="both", expand=True, padx=20, pady=20)

    def cargar_lote(self):
        if not os.path.exists(self.dir_imagenes):
            self.mostrar_estado("No existe la carpeta de imágenes: " + self.dir_imagenes)
            return

        archivos = [f for f in os.listdir(self.dir_imagenes) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if not archivos:
            self.mostrar_estado("No hay imágenes en la carpeta de entrada.")
            return

        self.mostrar_estado(f"Se cargaron {len(archivos)} imágenes.")
        self.archivos = archivos
        self.indice = 0
        self.actualizar_controles_navegacion()
        self.mostrar_imagen(self.archivos[self.indice])

    def anterior_imagen(self):
        if not self.archivos:
            return
        self.indice = (self.indice - 1) % len(self.archivos)
        self.actualizar_controles_navegacion()
        self.mostrar_imagen(self.archivos[self.indice])
        self.actualizar_resumen_actual()

    def siguiente_imagen(self):
        if not self.archivos:
            return
        self.indice = (self.indice + 1) % len(self.archivos)
        self.actualizar_controles_navegacion()
        self.mostrar_imagen(self.archivos[self.indice])
        self.actualizar_resumen_actual()

    def actualizar_controles_navegacion(self):
        tiene_imagenes = len(self.archivos) > 0
        self.btn_anterior.configure(state="normal" if tiene_imagenes and len(self.archivos) > 1 else "disabled")
        self.btn_siguiente.configure(state="normal" if tiene_imagenes and len(self.archivos) > 1 else "disabled")

    def mostrar_imagen(self, nombre_archivo):
        ruta = os.path.join(self.dir_imagenes, nombre_archivo)
        img = cv2.imread(ruta)
        if img is None:
            return

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil.thumbnail((900, 650))
        img_tk = ImageTk.PhotoImage(img_pil)
        self.label_imagen.configure(image=img_tk, text="")
        self.label_imagen.image = img_tk

    def mostrar_estado(self, texto):
        self.txt_estado.insert("end", texto + "\n")
        self.txt_estado.see("end")

    def actualizar_resumen_actual(self):
        if not self.archivos:
            self.lbl_resumen.configure(text="Resumen: sin datos")
            return

        nombre_actual = self.archivos[self.indice]

        resumen = "Resumen: sin datos"
        for item in self.resumen_por_imagen:
            if item["imagen"] == nombre_actual:
                resumen = (
                    f"Imagen: {nombre_actual}\n"
                    f"Plantas: {item['plantas']}\n"
                    f"Peso por planta: {item['peso_promedio_kg']:.2f} kg\n"
                    f"Peso total: {item['peso_total_kg']:.2f} kg"
                )
                break

        self.lbl_resumen.configure(text=resumen)

    def procesar_lote(self):
        if not hasattr(self, "archivos"):
            self.mostrar_estado("Primero debes cargar un lote.")
            return

        self.mostrar_estado("[INICIO] Procesando imágenes para estimar peso...")
        self.resumen_por_imagen = []
        registros_totales = []

        for nombre in self.archivos:
            ruta = os.path.join(self.dir_imagenes, nombre)
            detecciones = procesar_imagen(ruta, metros_por_pixel=0.05)
            plantas = len(detecciones)
            peso_total = sum(d["peso_kg_estimado"] for d in detecciones)
            peso_promedio = peso_total / plantas if plantas else 0.0

            self.resumen_por_imagen.append({
                "imagen": nombre,
                "plantas": plantas,
                "peso_total_kg": peso_total,
                "peso_promedio_kg": peso_promedio,
            })

            self.mostrar_estado(f"{nombre}: {plantas} plantas detectadas | peso total estimado: {peso_total:.2f} kg")
            for d in detecciones:
                registros_totales.append(d)

        if not registros_totales:
            self.mostrar_estado("[WARNING] No se detectó ninguna corona válida.")
            self.lbl_resumen.configure(text="Resumen: sin plantas detectadas")
            return

        total_peso = sum(r["peso_kg_estimado"] for r in registros_totales)
        self.mostrar_estado(f"[RESUMEN] Peso estimado total: {total_peso:.2f} kg")
        self.actualizar_resumen_actual()


if __name__ == "__main__":
    app = AppEstimacionAgave()
    app.mainloop()
