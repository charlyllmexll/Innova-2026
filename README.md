# Marcado de imágenes con SAM

Este proyecto automatiza la anotación de agaves usando Segment Anything Model (SAM) y genera datasets en formato YOLO.

## Estructura

- scripts/: scripts para explorar imágenes, generar anotaciones y validar el dataset.
- scripts h/: versión alternativa con ajustes para un segundo conjunto de datos.
- dataset/: salida generada por los scripts de anotación.
- verified/: resultados verificados.

## Requisitos


#Modelo LLM sam vit h
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

Instala las dependencias con:
```bash
pip install -r requeriments.txt
```

## Uso

1. Ajusta las rutas de entrada/salida en los scripts.
2. Ejecuta los scripts en este orden:
   - 001_Explorar-imagenes.py
   - 002_Auto-anotar-con-SAM.py
   - 003_Verificar.py
   - 004_Validar-dataset.py

## Nota

Los pesos de SAM y los datasets generados se excluyen del repositorio por tamaño.
