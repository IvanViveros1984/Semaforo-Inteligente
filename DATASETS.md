# Guía de Datasets para el Proyecto

> Documento complementario al proyecto **Sistema de Semáforo Inteligente**.
> Listado de datasets **totalmente libres** (CC BY 4.0 o equivalentes) que cubren las 3 clases requeridas.

## ⚡ Lo esencial primero

**¿Todos los datasets recomendados son gratuitos y de libre uso?**
**Sí.** Esta guía usa únicamente datasets con licencias **CC BY 4.0** o equivalentes, lo que permite:
- ✅ Descarga gratuita sin pago.
- ✅ Uso para tu proyecto del diplomado.
- ✅ Publicar el código y modelo en GitHub.
- ✅ Uso comercial futuro si quisieras (requisito: dar atribución).

**¿Se descargan directamente desde el código del notebook?** Sí, con los snippets ya incluidos.

**Nota sobre OpenCV:** OpenCV no es un dataset, es una **librería de visión por computador** (la usamos en el notebook como `import cv2` para leer imágenes y dibujar bounding boxes). No proporciona datasets. Probablemente confundiste el nombre con **Open Images** de Google, que sí es un dataset enorme y libre — está incluido aquí.

---

## Datasets seleccionados (todos CC BY 4.0)

### Dataset 1 — Open Images V7 (Google) ⭐ recomendado principal

**Por qué:** dataset enorme, totalmente libre (CC BY 4.0 para anotaciones y CC BY 2.0 para imágenes), con todas las clases que necesitas:
- `Person` → tu clase peatón
- `Car`, `Truck`, `Bus`, `Motorcycle`, `Van` → tu clase vehículo
- **`Ambulance`** → contribuye a tu clase de emergencia

| Atributo | Valor |
|---|---|
| Imágenes totales | ~9 millones |
| Con bounding boxes | 1,9 millones |
| Clases | 600 (incluye las que necesitas) |
| Licencia anotaciones | CC BY 4.0 (Google) |
| Licencia imágenes | CC BY 2.0 (Flickr) |
| Comercial | ✅ Sí (con atribución) |
| URL | https://storage.googleapis.com/openimages/web/index.html |
| Integración Ultralytics | Nativa con `yolo train data=open-images-v7.yaml` |

### Cómo descargarlo

**Opción A — Integración nativa con Ultralytics:**

```python
from ultralytics import YOLO

# Ultralytics incluye un YAML para Open Images V7
# Atención: el dataset completo son 561 GB. Mejor usar la opción B.
```

**Opción B — Filtrar solo las clases que necesitas con FiftyOne (recomendada):**

```python
# pip install fiftyone
import fiftyone as fo
import fiftyone.zoo as foz

# Descargar SOLO las clases relevantes para el proyecto
clases_relevantes = ["Person", "Car", "Truck", "Bus", "Motorcycle", "Van", "Ambulance"]

dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],
    classes=clases_relevantes,
    max_samples=10000,   # limitar tamaño total
)
# Resultado: ~10K imágenes con las clases que necesitas
# Exportar a formato YOLO:
dataset.export(
    export_dir="/content/openimages_yolo",
    dataset_type=fo.types.YOLOv5Dataset,
    classes=clases_relevantes,
)
```

**Opción C — Subset preprocesado en Roboflow Universe (rápido para empezar):**

```python
from roboflow import Roboflow
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace('google-research').project('open-images-v7-i6dsx')
dataset = project.version(1).download('yolov8')
```

### Remapeo de clases para el proyecto

```python
# Open Images V7 tiene 600 clases. Solo nos interesan estas:
REMAP_OPENIMAGES = {
    # Mapeo por NOMBRE de clase (FiftyOne y Roboflow usan nombres)
    # Si tu pipeline usa IDs, ajustar según el data.yaml descargado
    'Person':     0,  # → peaton
    'Car':        1,  # → vehiculo
    'Truck':      1,  # → vehiculo
    'Bus':        1,  # → vehiculo
    'Motorcycle': 1,  # → vehiculo
    'Van':        1,  # → vehiculo
    'Ambulance':  2,  # → vehiculo_emergencia
    # Resto de clases (598 más) se descartan automáticamente
}
```

---

### Dataset 2 — COCO 2017

**Por qué:** dataset de referencia mundial en detección. Anotaciones bajo CC BY 4.0.

| Atributo | Valor |
|---|---|
| Imágenes train | 118.000 |
| Imágenes val | 5.000 |
| Clases | 80 (incluye Person, Car, Truck, Bus, Motorcycle) |
| Licencia anotaciones | CC BY 4.0 |
| Licencia imágenes | Mayoritariamente CC BY 2.0 (Flickr) |
| Comercial | ✅ Anotaciones sí. Imágenes: la mayoría sí, algunas son CC BY-NC |
| URL | https://cocodataset.org |

**Matiz importante (verificado):** las anotaciones de COCO son CC BY 4.0 sin restricciones. Las **imágenes** vienen de Flickr y la mayoría son CC BY 2.0, pero **algunas son CC BY-NC** (no comercial). Para tu proyecto académico no hay problema; para uso comercial estricto habría que filtrar las imágenes por subtipo de licencia.

**¡Importante!** YOLOv8 ya viene **preentrenado en COCO**. En el notebook usamos `yolov8s.pt`, que son los pesos resultantes de entrenar en COCO. Es decir, **ya estamos aprovechando COCO indirectamente** vía transfer learning. No necesitamos reentrenar con COCO desde cero.

### Clases útiles de COCO para el proyecto

| ID COCO | Clase | Mapeo |
|---|---|---|
| 0 | person | → peaton |
| 1 | bicycle | (opcional) → vehiculo |
| 2 | car | → vehiculo |
| 3 | motorcycle | → vehiculo |
| 5 | bus | → vehiculo |
| 7 | truck | → vehiculo |

**Limitación:** COCO **no tiene clase "ambulancia"** ni "vehículo de emergencia con balizas". Por eso necesitamos el Dataset 3.

---

### Dataset 3 — Emergency Vehicles xockh (Roboflow) ⭐ para emergencias

**Por qué:** la única fuente con licencia CC BY 4.0 totalmente libre que distingue vehículos de emergencia **con balizas encendidas vs apagadas** — exactamente la lógica que tu enunciado pide.

| Atributo | Valor |
|---|---|
| Imágenes | 798 |
| Clases | `ambulance_off`, `ambulance_on`, `firetruck_off`, `firetruck_on` |
| Licencia | CC BY 4.0 ✅ |
| Comercial | ✅ Sí (con atribución) |
| URL | https://universe.roboflow.com/yolo-fn1iu/emergency-vehicles-detection-xockh-af7sr |

### Cómo descargarlo

```python
from roboflow import Roboflow
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace('yolo-fn1iu').project('emergency-vehicles-detection-xockh-af7sr')
dataset = project.version(1).download('yolov8')
```

### Remapeo recomendado

```python
REMAP_EMERGENCIA = {
    0: 1,  # ambulance_off  → vehiculo (sin balizas, es un vehículo normal)
    1: 2,  # ambulance_on   → vehiculo_emergencia (balizas encendidas)
    2: 1,  # firetruck_off  → vehiculo
    3: 2,  # firetruck_on   → vehiculo_emergencia
}
```

Este remapeo es **especialmente útil** porque alinea el dataset directamente con tu lógica: solo los vehículos de emergencia con balizas encendidas se priorizan en el semáforo.

---

## Estrategia recomendada (todo CC BY 4.0)

### Mezcla propuesta

| Dataset | Aporta | Imágenes filtradas |
|---|---|---|
| **Open Images V7** (subset filtrado por clases) | Peatones + vehículos + algunas ambulancias | ~5.000–10.000 |
| **Emergency Vehicles xockh** | Vehículos de emergencia con/sin balizas | 798 |
| **Total estimado** | | **~6.000–11.000 imágenes** |

Esta mezcla cubre las 3 clases con licencia completamente libre y suficiente volumen para entrenar YOLOv8s con transfer learning.

### Para robustez climática (sin recurrir a datasets no-comerciales)

Como **Foggy Cityscapes** y **BDD100K** son no-comerciales, en su lugar:

1. **Augmentation sintético con Albumentations** (ya en el notebook): `RandomFog`, `RandomRain`, `RandomBrightnessContrast`. Genera imágenes con niebla artificial sin depender de datasets restringidos.

2. **Filtrar Open Images V7 por escenas con neblina**: V7 incluye etiquetas de escena/contexto. Hay clases relacionadas con "Fog" o condiciones climáticas que se pueden filtrar con FiftyOne.

3. **(Opcional)** Capturar imágenes propias en días de neblina en Concepción y etiquetarlas con [Roboflow Annotate](https://roboflow.com/annotate) o [LabelImg](https://github.com/heartexlabs/labelImg). Mejor opción a largo plazo.

---

## Pipeline de combinación (totalmente libre)

```python
import os, shutil, random
from pathlib import Path

REMAPS = {
    'open-images-v7-subset': {
        # Reemplazar nombres por IDs reales según el data.yaml descargado
        'Person':     0,
        'Car':        1,
        'Truck':      1,
        'Bus':        1,
        'Motorcycle': 1,
        'Van':        1,
        'Ambulance':  2,
    },
    'emergency-vehicles-detection-xockh-af7sr-1': {
        0: 1,  # ambulance_off → vehiculo
        1: 2,  # ambulance_on  → vehiculo_emergencia
        2: 1,  # firetruck_off → vehiculo
        3: 2,  # firetruck_on  → vehiculo_emergencia
    },
}

# Usar la función remapear_y_unificar() que ya está en el notebook (Sección 2).
# Ejemplo:
# remapear_y_unificar({
#     'open-images-v7-subset':                  './datasets/openimages_yolo',
#     'emergency-vehicles-detection-xockh-af7sr-1': './datasets/Emergency-Vehicles-1',
# }, destino='./dataset_combinado', remaps=REMAPS)
```

---

## Verificación visual obligatoria

**Después de combinar, antes de entrenar**, ejecutar la celda de verificación visual del notebook sobre el dataset combinado. Si las etiquetas no calzan con los objetos, revisar los `REMAPS`.

---

## Datasets descartados (para mantener "totalmente libre")

Estos datasets son técnicamente accesibles pero **no cumplen el criterio de "totalmente libre" que pediste**:

| Dataset | Razón de descarte |
|---|---|
| CrowdHuman | Licencia original: solo investigación académica, no comercial |
| Foggy Cityscapes | Deriva de Cityscapes (no comercial) |
| BDD100K | No comercial, requiere registro académico |
| KITTI | CC BY-NC-SA (no comercial) |
| Cityscapes | No comercial |
| ImageNet | Solo investigación |
| Emergency Vehicle Priority (ADHA en Roboflow) | Licencia no especificada explícitamente — revisar antes de usar |

Si en el futuro decides usar alguno de estos para mejorar el desempeño (caso típicamente válido para un proyecto académico), revisar la licencia exacta en su sitio oficial.

---

## Citaciones BibTeX

```bibtex
@inproceedings{kuznetsova2020openimages,
  title  = {The Open Images Dataset V4: Unified image classification, object detection, and visual relationship detection at scale},
  author = {Kuznetsova, Alina and Rom, Hassan and Alldrin, Neil and Uijlings, Jasper and Krasin, Ivan and Pont-Tuset, Jordi and Kamali, Shahab and Popov, Stefan and Malloci, Matteo and Kolesnikov, Alexander and Duerig, Tom and Ferrari, Vittorio},
  journal = {International Journal of Computer Vision (IJCV)},
  year   = {2020}
}

@inproceedings{lin2014microsoft,
  title  = {Microsoft COCO: Common Objects in Context},
  author = {Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2014}
}

@misc{emergency-vehicles-xockh,
  title  = {Emergency Vehicles Detection Dataset},
  howpublished = {\url{https://universe.roboflow.com/yolo-fn1iu/emergency-vehicles-detection-xockh-af7sr}},
  journal = {Roboflow Universe},
  publisher = {Roboflow},
  year   = {2024},
  note   = {Licensed under CC BY 4.0}
}
```

---

## Resumen ejecutivo

✅ **Open Images V7** + **Emergency Vehicles xockh** te da las 3 clases, todo bajo **CC BY 4.0**, descargable directamente, gratis, sin restricciones de uso comercial (con atribución).

⚠️ **OpenCV NO es un dataset**, es la librería de visión por computador que ya usamos en el código.

📋 **Para robustez climática:** usar el augmentation sintético con Albumentations que ya está en el notebook, en lugar de datasets de niebla no-comerciales.

🎯 **Atribución requerida:** las citas BibTeX están en la sección de Referencias del documento Word.
