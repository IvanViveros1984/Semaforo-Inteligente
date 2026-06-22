# Sistema de Semáforo Inteligente con Detección de Objetos

> Proyecto Final – Diplomado en Tecnologías para Inteligencia Artificial
> Universidad del Bío-Bío – 2026

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8s-orange)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)]()
[![Flask](https://img.shields.io/badge/Demo-Flask%20Web-lightgrey)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

**Autores:** Fanny Balchen · Diego Mardones · Iván Viveros
**Entrega:** 30 de junio de 2026

---

## Descripción general

Sistema end-to-end que combina detección de objetos con YOLOv8 y lógica de control adaptativa para gestionar una intersección urbana de 4 caminos en doble sentido. Detecta en tiempo real **peatones**, **vehículos** y **vehículos de emergencia con balizas** a través de dos cámaras IP, y decide el estado del semáforo aplicando 4 reglas jerárquicas con priorización de emergencias.

El sistema incluye una **demo web interactiva** (Flask + MJPEG streaming) para validar el pipeline completo sin hardware físico.

---

## Tabla de contenidos

1. [Arquitectura del sistema](#arquitectura-del-sistema)
2. [Fases del semáforo](#fases-del-semáforo)
3. [Reglas de priorización](#reglas-de-priorización)
4. [Modelo de detección](#modelo-de-detección)
5. [Robustez climática](#robustez-climática)
6. [Resultados](#resultados)
7. [Estructura del repositorio](#estructura-del-repositorio)
8. [Demo web](#demo-web)
9. [Instalación y ejecución](#instalación-y-ejecución)
10. [Datasets](#datasets)
11. [Parámetros de entrenamiento](#parámetros-de-entrenamiento)
12. [Consideraciones éticas](#consideraciones-éticas)
13. [Referencias](#referencias)

---

## Arquitectura del sistema

El sistema opera sobre una intersección con **dos ejes de detección** cubiertos por dos cámaras IP:

```
  cam_sur (Sur→Norte)          cam_este (Este→Oeste)
        │                               │
        ▼                               ▼
  YOLOv8s (GPU)            YOLOv8s (mismo modelo)
        │                               │
        └──────────┬────────────────────┘
                   ▼
      DeteccionPorCamara (peatones, vehículos, emergencia)
                   │
                   ▼
     ┌─────────────────────────────────┐
     │   ControladorInterseccion       │
     │   · 4 reglas jerárquicas        │
     │   · 5 fases de semáforo         │
     │   · Ventana de estabilidad      │
     │   · Latch de emergencia         │
     └──────────────┬──────────────────┘
                    ▼
      FASE_SN │ FASE_EO │ TRANSICION │ EMERGENCIA │ FALLA_SEGURA
```

---

## Fases del semáforo

El `ControladorInterseccion` opera con 5 fases definidas en el enum `FaseSemaforo`:

| Fase | Semáforo SN | Semáforo EO | Trigger | Duración |
|---|---|---|---|---|
| **FASE_SN** | Verde | Rojo | Ciclo o densidad NS | 20–30 s |
| **FASE_EO** | Rojo | Verde | Ciclo o densidad EO | 20–30 s |
| **TRANSICION** | Amarillo (lado activo) | Rojo / Amarillo | Entre **todos** los cambios de fase | 3 s (75 frames) |
| **EMERGENCIA** | Verde en aprox. ambulancia | Rojo resto | Vehículo emergencia detectado | Hasta despeje |
| **FALLA_SEGURA** | Amarillo | Amarillo | Cámara sin visibilidad | Mientras dure |

> La fase **TRANSICION** se aplica en **todos** los cambios de fase, incluyendo la entrada y salida de EMERGENCIA. Nunca se pasa de verde a rojo directamente.

---

## Reglas de priorización

Las reglas se evalúan en cada frame en orden de prioridad decreciente:

| Prioridad | Regla | Condición | Acción |
|---|---|---|---|
| **0** | Falla segura | Cámara con visibilidad crítica | Amarillo parpadeante en todo |
| **1** | Emergencia | 1 frame con conf ≥ 0.50 (latch) | TRANSICION → EMERGENCIA (verde en aprox.) |
| **2** | Densidad asimétrica | Diferencia ≥ 5 vehículos entre SN y EO | TRANSICION → fase de mayor densidad |
| **3** | Ciclo normal | Timer ≥ 750 frames (30 s) | TRANSICION → fase alternada |

### Lógica de emergencia (latch + debounce de salida)

- **Activación:** el primer frame con conf ≥ 0.50 activa el latch **inmediatamente** e inicia TRANSICION (3 s amarillo) → EMERGENCIA.
- **Persistencia:** mientras el latch esté activo, el sistema permanece en EMERGENCIA aunque haya frames sin detección (videos de baja calidad o parpadeos).
- **Salida:** tras **5 segundos consecutivos** (125 frames a 25 fps) sin detectar emergencia, el latch se libera → TRANSICION (3 s amarillo) → fase anterior restaurada.
- **Timer:** al salir de emergencia el timer de fase se reinicia a 0, forzando el mínimo de 20 s antes del próximo cambio.

---

## Modelo de detección

**YOLOv8s** (Ultralytics) con transfer learning desde pesos COCO.

**Clases detectadas:**

| ID | Clase | Descripción |
|---|---|---|
| 0 | Peatón | Personas en cruces o esperando |
| 1 | Vehículo | Autos, camionetas, camiones, motos, buses |
| 2 | Emergencia | Vehículos con balizas encendidas (ambulancias, bomberos, carabineros) |

**Justificación de YOLOv8s:**
- Inferencia > 25 FPS en GPU GTX 1650 (4 GB VRAM)
- Arquitectura anchor-free, sin hiperparámetros de anclas
- API Ultralytics con exportación a ONNX / TensorRT para edge
- Transfer learning desde COCO reduce datos propios necesarios en ~100×

---

## Robustez climática

Concepción (Biobío) presenta neblina densa frecuente en mañanas invernales. Tres mecanismos complementarios:

### 1. Augmentation climático offline (Albumentations)

| Transformación | Probabilidad | Efecto |
|---|---|---|
| `RandomFog` | 30 % | Niebla leve a densa |
| `RandomRain` | 20 % | Gotas y trazos de lluvia |
| `RandomBrightnessContrast` | 35 % | Amanecer, atardecer, noche |
| `RandomShadow` | 20 % | Sombras dinámicas |
| `MotionBlur` / `GaussianBlur` | 20 % | Vibración y desenfoque |
| `ISONoise` | 15 % | Ruido de baja luz |

### 2. Detector de baja visibilidad (< 1 ms por frame)

| Métrica | Mide | Umbral crítico |
|---|---|---|
| Varianza del Laplaciano | Nitidez / borrosidad | < 50 |
| Contraste RMS | Diferencia claro/oscuro | < 20 |
| Brillo medio | Luminancia global | < 40 o > 215 |

### 3. Falla segura (REGLA 0)

Si alguna cámara reporta visibilidad crítica → `FALLA_SEGURA` (amarillo parpadeante). Inspirada en IEC 61508: *"un semáforo predecible es siempre preferible a uno inteligente pero impredecible"*.

---

## Resultados

### Métricas de detección (modelo `exp2_yolov8s_50ep`)

| Métrica | Objetivo | Obtenido |
|---|---|---|
| mAP@0.5 global | ≥ 0.85 | **0.620** |
| mAP@0.5:0.95 global | ≥ 0.60 | **0.471** |
| Precisión global | — | **0.667** |
| Recall global | ≥ 0.90 | **0.614** |
| Velocidad inferencia (GPU GTX 1650) | ≥ 25 FPS | **≥ 25 FPS** |

### Comparativa de experimentos

| Experimento | Épocas | imgsz | mAP@0.5 | Precisión | Recall |
|---|---|---|---|---|---|
| exp1_yolov8s (baseline) | 15 | 416 | 0.5646 | 0.561 | 0.541 |
| **exp2_yolov8s_50ep** | **50** | **640** | **0.620 (+9.8 %)** | **0.667** | **0.614** |

> El recall de la clase emergencia es la métrica más crítica. Un falso negativo implica no priorizar una ambulancia, con potenciales consecuencias graves.

---

## Estructura del repositorio

```
Versión 2/
├── controlador_interseccion.py     ← lógica del semáforo (5 fases, 4 reglas, latch)
├── semaforo_inteligenteV2.ipynb    ← notebook principal (entrenamiento, EDA, evaluación)
│
├── demo/                           ← aplicación web de demostración
│   ├── app.py                      ← servidor Flask (inferencia + API HTTP)
│   ├── templates/
│   │   └── index.html              ← interfaz web (semáforos SVG, MJPEG, SSE)
│   └── uploads/                    ← videos cargados en tiempo de ejecución
│
├── dataset_combinado/              ← dataset combinado (Open Images V7 + Emergency xockh)
│   ├── images/{train,val,test}/
│   ├── labels/{train,val,test}/
│   └── data.yaml
│
├── runs/semaforo/
│   ├── exp1_yolov8s/               ← resultados baseline (15 épocas, imgsz=416)
│   └── exp2_yolov8s_50ep/          ← modelo final (50 épocas, imgsz=640)
│       └── weights/best.pt         ← pesos del modelo entrenado
│
├── videos/                         ← videos de prueba para la demo
│   ├── Este2_Ambu1_Este2.mp4       ← secuencia: tráfico → ambulancia → tráfico
│   ├── Ambu1.mp4
│   ├── Este2.mp4
│   └── ...
│
├── Proyecto_Final_Semaforo_Inteligente_v2.docx   ← informe técnico
├── Presentacion_Semaforo_Inteligente_v2.pptx     ← presentación
├── requirements.txt
└── README.md
```

> **Nota:** `dataset_combinado/` (3 GB) y `runs/` (356 MB) no se incluyen en el repositorio Git. Ver sección [Datasets](#datasets) para reproducir.

---

## Demo web

La demo web permite validar el pipeline completo en tiempo real sin hardware físico.

### Características

- **Streaming MJPEG**: dos feeds de video simultáneos con bounding boxes y HUD dibujados con OpenCV
- **Semáforos SVG animados**: verde / amarillo / rojo con efecto de resplandor (glow) según la fase activa
- **Server-Sent Events (SSE)**: actualización del estado cada 150 ms (fase, motivo, FPS, detecciones)
- **Badge GPU/CPU**: indica el dispositivo activo; fallback automático a CPU si CUDA no está disponible
- **Carga de videos**: subida de archivos MP4/AVI para cada cámara directamente desde el navegador

### Iniciar la demo

```bash
# Desde la raíz del repositorio
python3 demo/app.py
```

El servidor precarga el modelo YOLOv8 al arrancar y queda disponible en **http://localhost:5000**.

### Pasos para ejecutar la demo

1. Abrir **http://localhost:5000** en el navegador
2. Cargar video en **CAM SUR** → usar `videos/Este2_Ambu1_Este2.mp4` (demuestra tráfico → emergencia → restauración)
3. Cargar video en **CAM ESTE** → usar `videos/Este2.mp4` o cualquier video de tráfico
4. Presionar **Iniciar**
5. Observar las fases: `FASE_SN` → amarillo (`TRANSICION`) → `EMERGENCIA` → amarillo → fase restaurada

### Rutas de la API

| Ruta | Método | Descripción |
|---|---|---|
| `/` | GET | Página principal |
| `/upload/1`, `/upload/2` | POST | Cargar video por cámara |
| `/start` | POST | Iniciar procesamiento |
| `/stop` | POST | Detener procesamiento |
| `/reset` | POST | Limpiar estado completo |
| `/stream/cam1`, `/stream/cam2` | GET | Stream MJPEG continuo |
| `/events` | GET | Server-Sent Events (JSON cada 150 ms) |

---

## Instalación y ejecución

### Requisitos

- Python 3.10+
- GPU NVIDIA con CUDA 12.x (recomendado). En CPU funciona con fps reducido.
- Driver NVIDIA 595+

### Instalación

```bash
git clone https://github.com/<usuario>/semaforo-inteligente.git
cd semaforo-inteligente

python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### `requirements.txt`

```
ultralytics>=8.3.0
torch>=2.0
torchvision>=0.15
opencv-python>=4.8
flask>=3.0
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.13
scikit-learn>=1.3
albumentations>=1.4.0
jupyter>=1.0
roboflow>=1.1
fiftyone>=0.24
```

### Ejecutar el notebook (entrenamiento y evaluación)

```bash
jupyter notebook semaforo_inteligenteV2.ipynb
```

| Sección | Contenido |
|---|---|
| 0 | Setup e instalación de dependencias |
| 1 | Importación de librerías |
| 2 | Descarga y carga del dataset |
| 3 | EDA y visualización de clases |
| 4 | Split estratificado 70/20/10 |
| 5 | Augmentation climático con Albumentations |
| 5B | Detector de baja visibilidad |
| 6 | Carga del modelo YOLOv8s |
| 7 | Entrenamiento con transfer learning |
| 8 | Evaluación: matriz de confusión, mAP, P, R |
| 9 | Testeo sobre imágenes nuevas |
| 10 | Lógica del semáforo (ControladorInterseccion) |
| 11 | Demo end-to-end con video |
| 12 | Conclusiones |

---

## Datasets

Todos bajo licencia **Creative Commons Attribution 4.0 (CC BY 4.0)** — uso académico y comercial permitidos con atribución.

| Dataset | Imágenes | Aporte |
|---|---|---|
| Open Images V7 (Google) | ~10.000 (subset filtrado) | Peatones + vehículos generales |
| Emergency Vehicles xockh (Roboflow) | 798 | Vehículos de emergencia con/sin balizas |
| COCO (vía transfer learning) | 330.000 | Pesos preentrenados `yolov8s.pt` |
| Augmentation offline (Albumentations) | +6.522 adicionales | Robustez climática sintética |

> El dataset combinado (`dataset_combinado/`, ~3 GB) no se incluye en Git. Para reproducir, ejecutar las secciones 2–4 del notebook que descargan y preparan automáticamente los datos.

---

## Parámetros de entrenamiento

| Hiperparámetro | Valor | Justificación |
|---|---|---|
| Modelo base | YOLOv8s | Balance precisión / velocidad para GPU GTX 1650 |
| Pesos iniciales | yolov8s.pt (COCO) | Transfer learning — reduce datos necesarios |
| Épocas | 50 | Convergencia con early stopping (patience=10) |
| imgsz | 640 | Estándar YOLOv8, mejor detección de objetos pequeños |
| Batch | 16 | Ajustado a 4 GB VRAM |
| Optimizer | AdamW | Convergencia rápida con scheduler coseno |
| LR inicial | 0.01 | Default Ultralytics |
| Mosaic | 1.0 | Augmentation principal |
| MixUp | 0.15 | Regularización suave |
| Seed | 42 | Reproducibilidad |

---

## Consideraciones éticas

- **Privacidad:** el sistema procesa video localmente sin persistencia. Si se almacenan imágenes, aplicar anonimización de rostros y matrículas (Ley 19.628, Chile).
- **Sesgo:** el dataset podría subrepresentar ciertos grupos (ciclistas, personas con movilidad reducida, vehículos de emergencia locales). Auditar desempeño por subgrupo.
- **Falla segura:** ante cualquier fallo o visibilidad crítica, el sistema revierte al ciclo de tiempo fijo (FALLA_SEGURA). Principio IEC 61508.
- **Transparencia:** cada decisión del controlador registra la regla aplicada, los conteos y las métricas de visibilidad — auditable.
- **Impacto ambiental:** el entrenamiento de 50 épocas en GPU tiene un costo energético. Uso de early stopping para minimizarlo.

---

## Referencias

1. Chollet, F. (2017). *Deep Learning with Python.* Manning Publications.
2. Jocher, G., Chaurasia, A., y Qiu, J. (2023). *Ultralytics YOLOv8.* https://github.com/ultralytics/ultralytics
3. Lin, T.-Y. et al. (2014). *Microsoft COCO: Common Objects in Context.* ECCV.
4. Raj, P., Soundarabai, P. B., Augustine, D. P. (2024). *Machine Intelligence. Computer Vision and Natural Language Processing.* CRC Press.
5. Redmon, J. et al. (2016). *You Only Look Once: Unified, Real-Time Object Detection.* CVPR.
6. Vaiyapuri, T. et al. (2024). *A YOLOv8-Based Approach for Smart City Solutions: Emergency Vehicle Detection.* J. Electrical Systems.
7. Wu, X. et al. (2025). *Improved YOLOv8 for vehicle and pedestrian detection in urban traffic.* PLOS ONE.
8. Buslaev, A. et al. (2020). *Albumentations: Fast and Flexible Image Augmentations.* Information.
9. IEC 61508 (2010). *Functional safety of E/E/PE safety-related systems.*
10. Ley 19.628 sobre Protección de la Vida Privada. República de Chile.
11. Material de clases del Diplomado en Tecnologías para IA, Universidad del Bío-Bío, 2026.

---

**Diplomado:** Tecnologías para Inteligencia Artificial – Universidad del Bío-Bío 2026
