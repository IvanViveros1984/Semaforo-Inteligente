#======================================
#  Importaciones de Librerías
#======================================

import streamlit as st
from PIL import Image
import time
from ultralytics import YOLO
from pathlib import Path


#======================================
#  Configuración de página
#======================================

st.set_page_config(
    page_title="Semáforo Inteligente IA",
    page_icon="🚦",
    layout="wide"
)

st.markdown("""
<style>
header {visibility: hidden;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.block-container {
    padding-top: 0rem !important;
    padding-bottom: 0.5rem !important;
}

.semaforo-texto {
    font-size: 54px;
    text-align: center;
    line-height: 1.25;
}

.estado-semaforo {
    font-size: 18px;
    font-weight: bold;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


#======================================
#  RUTA DE IMÁGENES
#======================================

CARPETA_IMAGENES = r"C:\diplomado\Programas\template\imagenes"

EXTENSIONES_VALIDAS = ["*.jpg", "*.jpeg", "*.png"]


#======================================
#  Carga de imágenes desde carpeta
#======================================

@st.cache_data
def cargar_banco_imagenes(carpeta):
    carpeta = Path(carpeta)
    imagenes = {}

    if not carpeta.exists():
        return imagenes

    archivos = []

    for extension in EXTENSIONES_VALIDAS:
        archivos.extend(carpeta.glob(extension))

    archivos = sorted(archivos)

    for archivo in archivos:
        try:
            imagenes[archivo.name] = Image.open(archivo).convert("RGB")
        except Exception:
            pass

    return imagenes


#======================================
#  Carga del Modelo
#======================================

@st.cache_resource
def cargar_modelo():
    return YOLO("semaforo.pt")


modelo = cargar_modelo()


#======================================
#  Título Frontend reducido
#======================================

col_titulo, col_logo = st.columns([6, 1])

with col_titulo:
    st.header("🚦 Semáforo Inteligente con IA")
    st.caption("Diplomado en Tecnologías de IA - Universidad del Bío-Bío")
    st.markdown(
        "*Fanny Balchen · Diego Mardones · Iván Viveros*"
    )

with col_logo:
    st.image("V-Lab.png", width=140)


#===============================================
#  Agrupación de clases
#===============================================

def clasificar_deteccion(nombre_clase):
    clase = nombre_clase.lower().strip()

    clases_emergencia = [
        "vehiculo_emergencia", "vehículo_emergencia",
        "emergencia", "ambulancia", "ambulance",
        "bomberos", "carro bomba",
        "policia", "policía", "police"
    ]

    clases_peatones = [
        "peaton", "peatón", "pedestrian", "persona", "person"
    ]

    clases_vehiculos = [
        "vehiculo", "vehículo", "auto", "car", "camioneta",
        "camion", "camión", "truck", "bus", "motorcycle", "moto"
    ]

    if clase in clases_emergencia:
        return "vehiculo_emergencia"

    if clase in clases_peatones:
        return "peaton"

    if clase in clases_vehiculos:
        return "vehiculo"

    return "otro"


#===============================================
#  Ejecución del modelo
#===============================================

def modelo_dummy(imagenes, progress_bar, progress_text):
    resultados = []
    total = len(imagenes)

    for i, img in enumerate(imagenes):
        numero_imagen = i + 1
        progress_text.text(f"Procesando Calle {numero_imagen}/{total}")

        pred = modelo.predict(img, verbose=False)

        total_vehiculos = 0
        total_peatones = 0
        total_vehiculo_emergencia = 0

        for r in pred:

            motos_boxes = []

            # Primero se guardan las coordenadas de todas las motos detectadas.
            for box in r.boxes:
                clase_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # En YOLO COCO, la clase 3 corresponde a motocicleta.
                # Si tu modelo propio usa otro número para moto, ajusta aquí.
                if clase_id == 3:
                    motos_boxes.append((x1, y1, x2, y2))

            # Luego se cuentan vehículos, peatones reales y emergencias.
            for box in r.boxes:
                clase_id = int(box.cls[0])
                nombre_clase = r.names[clase_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                categoria = clasificar_deteccion(nombre_clase)

                if categoria == "vehiculo":
                    total_vehiculos += 1

                elif categoria == "vehiculo_emergencia":
                    total_vehiculo_emergencia += 1

                elif categoria == "peaton":

                    # Centro de la persona detectada
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    sobre_moto = False

                    # Si el centro de la persona cae dentro del rectángulo de una moto,
                    # se considera conductor o pasajero, no peatón.
                    for mx1, my1, mx2, my2 in motos_boxes:
                        if mx1 <= cx <= mx2 and my1 <= cy <= my2:
                            sobre_moto = True
                            break

                    # Solo se suma como peatón si NO está sobre una moto.
                    if not sobre_moto:
                        total_peatones += 1

        resultados.append({
            "imagen": numero_imagen,
            "vehiculos": total_vehiculos,
            "peatones": total_peatones,
            "vehiculo_emergencia": total_vehiculo_emergencia
        })

        progreso = int((numero_imagen / total) * 100)
        progress_bar.progress(progreso)

    progress_text.text("Proceso finalizado")
    return resultados


def ejecutar_modelo(imagenes):
    inicio = time.time()

    progress_text = st.empty()
    progress_bar = st.progress(0)

    with st.spinner("Ejecutando modelo..."):
        st.session_state.resultados = modelo_dummy(
            imagenes,
            progress_bar,
            progress_text
        )

    fin = time.time()
    st.session_state.tiempo = fin - inicio


#=======================================================
#  Resultados visuales usando componentes nativos
#=======================================================

def mostrar_tarjeta_resultado(resultado):
    """
    Tarjeta compacta de resultados.
    Muestra toda la información en una sola línea para ahorrar espacio vertical.
    """
    with st.container(border=True):
        st.write(
            f"📍 Calle {resultado['imagen']} | "
            f"🚗 {resultado['vehiculos']} | "
            f"🚶 {resultado['peatones']} | "
            f"🚑 {resultado['vehiculo_emergencia']}"
        )


def calcular_prioridad_calles(resultados):
    """
    Reglas aplicadas:
    1. Prioridad máxima: vehículos de emergencia.
    2. Si no hay emergencia, prioridad por densidad de vehículos.
    3. Si no hay prioridad vehicular, prioridad por densidad de peatones mayor a 5.
    4. Si no se detecta nada, funciona el ciclo automático normal.
    """
    if not resultados:
        return 0, "Esperando detección"

    # 1. Prioridad máxima: vehículos de emergencia
    for r in resultados:
        if r["vehiculo_emergencia"] > 0:
            return r["imagen"], f"Prioridad 1: emergencia en Calle {r['imagen']}"

    # 2. Prioridad por densidad de vehículos
    calle_mas_vehiculos = max(resultados, key=lambda r: r["vehiculos"])

    if calle_mas_vehiculos["vehiculos"] > 0:
        return calle_mas_vehiculos["imagen"], \
               f"Prioridad 2: mayor densidad vehicular en Calle {calle_mas_vehiculos['imagen']}"

    # 3. Prioridad por densidad de peatones mayor a 5
    if len(resultados) >= 2:
        peatones_calle1 = resultados[0]["peatones"]
        peatones_calle2 = resultados[1]["peatones"]

        if peatones_calle1 > 5 or peatones_calle2 > 5:

            if peatones_calle1 > peatones_calle2:
                return 10, "Prioridad 3: mayor densidad peatonal en Calle 1"

            if peatones_calle2 > peatones_calle1:
                return 20, "Prioridad 3: mayor densidad peatonal en Calle 2"

            return 30, "Prioridad 3: alta densidad peatonal en ambas calles"

    elif len(resultados) == 1:
        if resultados[0]["peatones"] > 5:
            return 10, "Prioridad 3: alta densidad peatonal en Calle 1"

    # 4. Sin prioridad: ciclo automático normal
    return 0, "Funcionamiento automático: ciclo normal"


def obtener_estados_semaforos(prioridad):
    if prioridad == 1:
        return "VERDE", "ROJO"

    if prioridad == 2:
        return "ROJO", "VERDE"

    if prioridad == 10:
        return "ROJO", "VERDE"

    if prioridad == 20:
        return "VERDE", "ROJO"

    if prioridad == 30:
        return "ROJO", "ROJO"

    # Ciclo automático cuando no hay detecciones
    tiempo_ciclo = int(time.time()) % 72

    if tiempo_ciclo < 30:
        return "VERDE", "ROJO"

    if tiempo_ciclo < 34:
        return "AMARILLO", "ROJO"

    if tiempo_ciclo < 36:
        return "ROJO", "ROJO"

    if tiempo_ciclo < 66:
        return "ROJO", "VERDE"

    if tiempo_ciclo < 70:
        return "ROJO", "AMARILLO"

    return "ROJO", "ROJO"


def mostrar_luces_semaforo(estado):
    st.write("🔴" if estado == "ROJO" else "⚫")
    st.write("🟡" if estado == "AMARILLO" else "⚫")
    st.write("🟢" if estado == "VERDE" else "⚫")


def mostrar_semaforos_dobles(estado_calle1, estado_calle2, mensaje):
    st.markdown("#### Semáforos")

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        with st.container(border=True):
            st.caption("🚦 Calle 1")
            mostrar_luces_semaforo(estado_calle1)

    with col_s2:
        with st.container(border=True):
            st.caption("🚦 Calle 2")
            mostrar_luces_semaforo(estado_calle2)

    st.caption(mensaje)


#===============================================
#  Variables de sesión
#===============================================

if "resultados" not in st.session_state:
    st.session_state.resultados = None

if "tiempo" not in st.session_state:
    st.session_state.tiempo = None

if "imagenes_actuales" not in st.session_state:
    st.session_state.imagenes_actuales = None

if "nombres_actuales" not in st.session_state:
    st.session_state.nombres_actuales = ["", ""]


#============================================
#  Columnas de presentación Streamlit
#============================================

col1, col2, col3 = st.columns([1.5, 0.7, 1.5])

with col1:
    st.markdown("#### 📷 Entrada")

    st.caption("Carpeta de imágenes:")
    st.code(CARPETA_IMAGENES)

    banco_imagenes = cargar_banco_imagenes(CARPETA_IMAGENES)

    if not banco_imagenes:
        st.error(
            "No se encontraron imágenes. Crea la carpeta 'imagenes' dentro de "
            "C:\\diplomado\\Programas\\template y agrega archivos JPG, JPEG o PNG."
        )
        uploaded_files = []

    else:
        nombres_imagenes = list(banco_imagenes.keys())

        st.markdown("##### Selección manual")

        seleccion_calle1 = st.selectbox(
            "Imagen para Calle 1",
            nombres_imagenes,
            key="seleccion_calle1"
        )

        seleccion_calle2 = st.selectbox(
            "Imagen para Calle 2",
            nombres_imagenes,
            key="seleccion_calle2"
        )

        uploaded_files = [
            banco_imagenes[seleccion_calle1],
            banco_imagenes[seleccion_calle2]
        ]

        st.session_state.imagenes_actuales = uploaded_files
        st.session_state.nombres_actuales = [seleccion_calle1, seleccion_calle2]

        st.markdown("##### Botones rápidos")

        st.caption(
            "Estos botones toman las imágenes según el orden alfabético de la carpeta. "
            "Puedes renombrarlas como 01_, 02_, 03_ para controlar el orden."
        )

        col_b1, col_b2 = st.columns(2)

        with col_b1:
            if st.button("1️⃣ Usar imágenes 1 y 2", use_container_width=True):
                if len(nombres_imagenes) >= 2:
                    st.session_state.imagenes_actuales = [
                        banco_imagenes[nombres_imagenes[0]],
                        banco_imagenes[nombres_imagenes[1]]
                    ]
                    st.session_state.nombres_actuales = [
                        nombres_imagenes[0],
                        nombres_imagenes[1]
                    ]
                    ejecutar_modelo(st.session_state.imagenes_actuales)
                    st.rerun()
                else:
                    st.warning("Necesitas al menos 2 imágenes en la carpeta.")

            if st.button("2️⃣ Usar imágenes 3 y 4", use_container_width=True):
                if len(nombres_imagenes) >= 4:
                    st.session_state.imagenes_actuales = [
                        banco_imagenes[nombres_imagenes[2]],
                        banco_imagenes[nombres_imagenes[3]]
                    ]
                    st.session_state.nombres_actuales = [
                        nombres_imagenes[2],
                        nombres_imagenes[3]
                    ]
                    ejecutar_modelo(st.session_state.imagenes_actuales)
                    st.rerun()
                else:
                    st.warning("Necesitas al menos 4 imágenes en la carpeta.")

        with col_b2:
            if st.button("3️⃣ Usar imágenes 5 y 6", use_container_width=True):
                if len(nombres_imagenes) >= 6:
                    st.session_state.imagenes_actuales = [
                        banco_imagenes[nombres_imagenes[4]],
                        banco_imagenes[nombres_imagenes[5]]
                    ]
                    st.session_state.nombres_actuales = [
                        nombres_imagenes[4],
                        nombres_imagenes[5]
                    ]
                    ejecutar_modelo(st.session_state.imagenes_actuales)
                    st.rerun()
                else:
                    st.warning("Necesitas al menos 6 imágenes en la carpeta.")

            if st.button("4️⃣ Usar imágenes 7 y 8", use_container_width=True):
                if len(nombres_imagenes) >= 8:
                    st.session_state.imagenes_actuales = [
                        banco_imagenes[nombres_imagenes[6]],
                        banco_imagenes[nombres_imagenes[7]]
                    ]
                    st.session_state.nombres_actuales = [
                        nombres_imagenes[6],
                        nombres_imagenes[7]
                    ]
                    ejecutar_modelo(st.session_state.imagenes_actuales)
                    st.rerun()
                else:
                    st.warning("Necesitas al menos 8 imágenes en la carpeta.")

        st.markdown("##### Vista previa")

        if st.session_state.imagenes_actuales:
            cols_img = st.columns(2)

            with cols_img[0]:
                img1 = st.session_state.imagenes_actuales[0].copy()
                img1.thumbnail((300, 220))
                st.image(
                    img1,
                    caption=f"Calle 1: {st.session_state.nombres_actuales[0]}",
                    use_container_width=True
                )

            with cols_img[1]:
                img2 = st.session_state.imagenes_actuales[1].copy()
                img2.thumbnail((300, 220))
                st.image(
                    img2,
                    caption=f"Calle 2: {st.session_state.nombres_actuales[1]}",
                    use_container_width=True
                )


with col2:
    st.markdown("#### ⚙️ Ejecución")

    if st.session_state.imagenes_actuales:
        if st.button("Ejecutar Modelo", use_container_width=True):
            ejecutar_modelo(st.session_state.imagenes_actuales)

    if st.session_state.tiempo is not None:
        with st.container(border=True):
            st.caption("⏱ Tiempo ejecución")
            st.write(f"**{st.session_state.tiempo:.2f} s**")


with col3:
    st.markdown("#### 📊 Resultados")

    if st.session_state.resultados:
        for resultado in st.session_state.resultados:
            mostrar_tarjeta_resultado(resultado)

        prioridad, mensaje = calcular_prioridad_calles(
            st.session_state.resultados
        )

        estado_calle1, estado_calle2 = obtener_estados_semaforos(
            prioridad
        )

        mostrar_semaforos_dobles(
            estado_calle1,
            estado_calle2,
            mensaje
        )
    else:
        st.info("Sin resultados aún")


# Si no hay detecciones, se actualiza el ciclo automático cada segundo
if st.session_state.resultados:
    prioridad_actual, _ = calcular_prioridad_calles(st.session_state.resultados)

    if prioridad_actual == 0:
        time.sleep(1)
        st.rerun()

st.markdown("---")
st.caption("Diplomado IA : Plantilla base para proyectos finales")
