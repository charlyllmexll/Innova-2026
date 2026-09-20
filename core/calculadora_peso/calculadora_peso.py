import os
import cv2
import customtkinter as ctk
from PIL import Image, ImageTk
from tkinter import filedialog

from motor_estimacion import exportar_csv, procesar_imagen


ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")


class AppEstimacionAgave(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Agrovision - Estimador de Peso de Agaves")
        self.geometry("1250x820")
        self.minsize(1000, 650)

        self.ruta_base_proyecto = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.dir_imagenes = os.path.join(self.ruta_base_proyecto, "data", "imagenes_crudas")
        self.dir_estimaciones = os.path.join(self.ruta_base_proyecto, "data", "estimaciones")
        self.archivos = []
        self.indice = 0
        self.resumen_por_imagen = []
        self.detecciones_por_imagen = {}
        self.registros_totales = []

        self.frame_lateral = ctk.CTkFrame(self, width=330)
        self.frame_lateral.pack(side="left", fill="y", padx=10, pady=10)

        ctk.CTkLabel(
            self.frame_lateral,
            text="AGROVISION",
            font=ctk.CTkFont(size=24, weight="bold"),
        ).pack(pady=(20, 5))
        ctk.CTkLabel(
            self.frame_lateral,
            text="CALCULADORA DE PESO",
            text_color="#4ADE80",
            font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(pady=(0, 20))

        self.btn_cargar = ctk.CTkButton(
            self.frame_lateral,
            text="Cargar lote",
            command=self.cargar_lote,
        )
        self.btn_cargar.pack(padx=20, pady=10, fill="x")

        self.frame_nav = ctk.CTkFrame(self.frame_lateral, fg_color="transparent")
        self.frame_nav.pack(padx=20, pady=5, fill="x")

        self.btn_anterior = ctk.CTkButton(
            self.frame_nav,
            text="Anterior",
            command=self.anterior_imagen,
            state="disabled",
        )
        self.btn_anterior.pack(side="left", expand=True, padx=(0, 5))

        self.btn_siguiente = ctk.CTkButton(
            self.frame_nav,
            text="Siguiente",
            command=self.siguiente_imagen,
            state="disabled",
        )
        self.btn_siguiente.pack(side="right", expand=True, padx=(5, 0))

        self.btn_procesar = ctk.CTkButton(
            self.frame_lateral,
            text="Procesar imágenes",
            command=self.procesar_lote,
        )
        self.btn_procesar.pack(padx=20, pady=10, fill="x")

        self.btn_exportar = ctk.CTkButton(
            self.frame_lateral,
            text="Exportar CSV",
            command=self.exportar_resultados_csv,
            state="disabled",
        )
        self.btn_exportar.pack(padx=20, pady=(0, 10), fill="x")

        self.lbl_imagen_actual = ctk.CTkLabel(
            self.frame_lateral,
            text="Imagen 0 de 0",
            text_color="#94A3B8",
        )
        self.lbl_imagen_actual.pack(padx=20, pady=(5, 0), anchor="w")

        self.lbl_resumen = ctk.CTkLabel(
            self.frame_lateral,
            text="Imagen actual: sin datos",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#E2E8F0",
            justify="left",
            anchor="w",
        )
        self.lbl_resumen.pack(padx=20, pady=(5, 10), anchor="w")

        self.txt_plantas = ctk.CTkTextbox(self.frame_lateral, height=190)
        self.txt_plantas.pack(padx=20, pady=(0, 10), fill="x")
        self.txt_plantas.configure(state="disabled")

        self.txt_estado = ctk.CTkTextbox(self.frame_lateral, height=130)
        self.txt_estado.pack(padx=20, pady=(10, 10), fill="both", expand=True)

        self.frame_visor = ctk.CTkFrame(self, fg_color="#0F172A")
        self.frame_visor.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.label_imagen = ctk.CTkLabel(
            self.frame_visor,
            text="Sin imagen cargada",
            text_color="#94A3B8",
        )
        self.label_imagen.pack(fill="both", expand=True, padx=20, pady=20)

    def cargar_lote(self):
        if not os.path.exists(self.dir_imagenes):
            self.mostrar_estado(f"No existe la carpeta de imágenes: {self.dir_imagenes}")
            return

        self.archivos = sorted(
            nombre
            for nombre in os.listdir(self.dir_imagenes)
            if nombre.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        )
        if not self.archivos:
            self.mostrar_estado("No hay imágenes en la carpeta de entrada.")
            return

        self.indice = 0
        self.resumen_por_imagen = []
        self.detecciones_por_imagen = {}
        self.registros_totales = []
        self.btn_exportar.configure(state="disabled")
        self.actualizar_controles_navegacion()
        self.mostrar_estado(f"Se cargaron {len(self.archivos)} imágenes.")
        self.mostrar_imagen(self.archivos[self.indice])
        self.actualizar_resumen_actual()

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
        tiene_imagenes = bool(self.archivos)
        hay_navegacion = tiene_imagenes and len(self.archivos) > 1
        estado = "normal" if hay_navegacion else "disabled"
        self.btn_anterior.configure(state=estado)
        self.btn_siguiente.configure(state=estado)
        numero_actual = self.indice + 1 if tiene_imagenes else 0
        self.lbl_imagen_actual.configure(
            text=f"Imagen {numero_actual} de {len(self.archivos)}"
        )

    def mostrar_imagen(self, nombre_archivo):
        ruta = os.path.join(self.dir_imagenes, nombre_archivo)
        imagen = cv2.imread(ruta)
        if imagen is None:
            self.mostrar_estado(f"No se pudo leer la imagen: {nombre_archivo}")
            return

        detecciones = self.detecciones_por_imagen.get(nombre_archivo, [])
        imagen_marcada = self.dibujar_detecciones(imagen, detecciones)
        imagen_rgb = cv2.cvtColor(imagen_marcada, cv2.COLOR_BGR2RGB)
        imagen_pil = Image.fromarray(imagen_rgb)
        imagen_pil.thumbnail((900, 700))
        imagen_tk = ImageTk.PhotoImage(imagen_pil)
        self.label_imagen.configure(image=imagen_tk, text="")
        self.label_imagen.image = imagen_tk

    def dibujar_detecciones(self, imagen, detecciones):
        imagen_marcada = imagen.copy()
        for deteccion in detecciones:
            x = deteccion["bbox_x"]
            y = deteccion["bbox_y"]
            w = deteccion["bbox_w"]
            h = deteccion["bbox_h"]
            numero = deteccion["planta_numero"]
            etiqueta = f"Planta {numero}: {deteccion['peso_kg_estimado']:.2f} kg"

            cv2.rectangle(
                imagen_marcada,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                3,
            )
            cv2.putText(
                imagen_marcada,
                etiqueta,
                (x, max(30, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        return imagen_marcada

    def actualizar_resumen_actual(self):
        if not self.archivos:
            self.lbl_resumen.configure(text="Imagen actual: sin datos")
            self.actualizar_lista_plantas([])
            return

        nombre_actual = self.archivos[self.indice]
        resumen = "Imagen actual: sin datos"
        detecciones_actuales = self.detecciones_por_imagen.get(nombre_actual, [])

        for item in self.resumen_por_imagen:
            if item["imagen"] == nombre_actual:
                resumen = (
                    f"Imagen: {nombre_actual}\n"
                    f"Plantas detectadas: {item['plantas']}\n"
                    f"Peso total: {item['peso_total_kg']:.2f} kg\n"
                    f"Peso promedio: {item['peso_promedio_kg']:.2f} kg"
                )
                break

        self.lbl_resumen.configure(text=resumen)
        self.actualizar_lista_plantas(detecciones_actuales)

    def actualizar_lista_plantas(self, detecciones):
        self.txt_plantas.configure(state="normal")
        self.txt_plantas.delete("1.0", "end")
        if detecciones:
            self.txt_plantas.insert("end", "Peso por número de planta\n\n")
            for deteccion in detecciones:
                self.txt_plantas.insert(
                    "end",
                    f"Planta {deteccion['planta_numero']:02d}: "
                    f"{deteccion['peso_kg_estimado']:.2f} kg | "
                    f"diámetro: {deteccion['diametro_cm']:.2f} cm\n",
                )
        else:
            self.txt_plantas.insert(
                "end",
                "Procesa el lote para ver el peso por planta.",
            )
        self.txt_plantas.configure(state="disabled")

    def mostrar_estado(self, texto):
        self.txt_estado.insert("end", texto + "\n")
        self.txt_estado.see("end")

    def procesar_lote(self):
        if not self.archivos:
            self.mostrar_estado("Primero debes cargar un lote.")
            return

        self.mostrar_estado("[INICIO] Procesando imágenes para estimar peso...")
        self.resumen_por_imagen = []
        self.detecciones_por_imagen = {}
        self.registros_totales = []

        for nombre in self.archivos:
            ruta = os.path.join(self.dir_imagenes, nombre)
            detecciones = procesar_imagen(ruta, metros_por_pixel=0.05)
            self.detecciones_por_imagen[nombre] = detecciones
            plantas = len(detecciones)
            peso_total = sum(d["peso_kg_estimado"] for d in detecciones)
            peso_promedio = peso_total / plantas if plantas else 0.0

            self.resumen_por_imagen.append(
                {
                    "imagen": nombre,
                    "plantas": plantas,
                    "peso_total_kg": peso_total,
                    "peso_promedio_kg": peso_promedio,
                }
            )
            self.registros_totales.extend(detecciones)
            self.mostrar_estado(
                f"{nombre}: {plantas} plantas detectadas | "
                f"peso total estimado: {peso_total:.2f} kg"
            )

        if not self.registros_totales:
            self.mostrar_estado("[WARNING] No se detectó ninguna corona válida.")
            self.btn_exportar.configure(state="disabled")
        else:
            self.btn_exportar.configure(state="normal")
            self.mostrar_estado("[OK] Revisión disponible por imagen.")

        self.mostrar_imagen(self.archivos[self.indice])
        self.actualizar_resumen_actual()

    def exportar_resultados_csv(self):
        if not self.registros_totales:
            self.mostrar_estado("Primero debes procesar el lote.")
            return

        os.makedirs(self.dir_estimaciones, exist_ok=True)
        ruta_csv = filedialog.asksaveasfilename(
            title="Guardar estimaciones CSV",
            initialdir=self.dir_estimaciones,
            initialfile="estimacion_peso_agave.csv",
            defaultextension=".csv",
            filetypes=[("Archivo CSV", "*.csv")],
        )
        if not ruta_csv:
            return

        exportar_csv(self.registros_totales, ruta_csv)
        self.mostrar_estado(f"[OK] CSV exportado: {ruta_csv}")


if __name__ == "__main__":
    app = AppEstimacionAgave()
    app.mainloop()
