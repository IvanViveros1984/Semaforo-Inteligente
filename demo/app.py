import os, sys, time, threading, json, cv2, torch
import numpy as np
torch.set_num_threads(12)   # maximizar cores CPU para inferencia
from flask import Flask, render_template, request, Response, jsonify
from werkzeug.utils import secure_filename

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from ultralytics import YOLO
from controlador_interseccion import (
    ControladorInterseccion, Camara, DeteccionPorCamara, FaseSemaforo
)

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def _detect_device():
    if not torch.cuda.is_available():
        return 'cpu'
    try:
        torch.zeros(1).cuda()   # prueba real antes de confirmar GPU
        return 'cuda'
    except Exception:
        return 'cpu'

DEVICE      = _detect_device()
BEST_PT     = os.path.join(PROJECT_ROOT, 'runs/semaforo/exp2_yolov8s_50ep/weights/best.pt')
FALLBACK_PT = os.path.join(PROJECT_ROOT, 'yolov8s.pt')
MODEL_PATH  = BEST_PT if os.path.exists(BEST_PT) else FALLBACK_PT

CLASES      = {0: 'Peatón', 1: 'Vehículo', 2: 'Emergencia'}
COLORES_BGR = {0: (0, 220, 220), 1: (50, 160, 255), 2: (40, 40, 255)}
ALLOWED_EXT = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

# ── Estado compartido entre threads ──────────────────────────
state = {
    'running':  False,
    'frame1':   None,
    'frame2':   None,
    'decision': {
        'fase':    'ESPERANDO',
        'motivo':  'Carga los videos y presiona Iniciar',
        'cam_prio': None,
    },
    'stats': {'fps': 0, 'det1': 0, 'det2': 0, 'frames': 0, 'emergencia': False},
    'video1': None,
    'video2': None,
}
_lock        = threading.Lock()
_model       = None
_proc_thread = None

# ── Modelo ────────────────────────────────────────────────────
def get_model():
    global _model
    if _model is None:
        print(f'[INFO] Cargando modelo desde {MODEL_PATH} en {DEVICE}...')
        _model = YOLO(MODEL_PATH)
    return _model

# ── Utilidades de imagen ──────────────────────────────────────
def blank_frame(text='Sin señal', w=640, h=360):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (15, 20, 35)
    cv2.putText(img, text, (w//2 - 80, h//2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 80, 120), 2)
    _, buf = cv2.imencode('.jpg', img)
    return buf.tobytes()

def to_jpeg(frame, quality=82):
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()

def draw_hud(frame, cam_label, n_peat, n_veh, hay_emerg, border_color):
    h, w = frame.shape[:2]
    # Borde de color según estado semáforo
    cv2.rectangle(frame, (0, 0), (w-1, h-1), border_color, 6)
    # Panel superior
    cv2.rectangle(frame, (0, 0), (w, 46), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, 0), (w, 46), border_color, 1)
    cv2.putText(frame, cam_label, (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (220, 230, 255), 2)
    # Panel inferior
    cv2.rectangle(frame, (0, h-42), (w, h), (0, 0, 0), -1)
    det_text = f'Peatones: {n_peat}   Vehiculos: {n_veh}'
    cv2.putText(frame, det_text, (12, h-14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 200, 220), 2)
    if hay_emerg:
        cv2.rectangle(frame, (w-180, 0), (w, 46), (0, 0, 200), -1)
        cv2.putText(frame, '  EMERGENCIA', (w-178, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame

_active_device = DEVICE   # puede cambiar a 'cpu' si CUDA crashea en caliente

def infer_frame(modelo, frame, conf=0.30):
    global _active_device
    imgsz = 320 if _active_device == 'cpu' else 640
    try:
        result = modelo.predict(frame, conf=conf, device=_active_device, verbose=False, imgsz=imgsz)[0]
    except Exception as e:
        if 'CUDA' in str(e) or 'cuda' in str(e):
            print(f'[WARN] CUDA crash detectado, cambiando a CPU: {e}')
            _active_device = 'cpu'
            imgsz = 320
            result = modelo.predict(frame, conf=conf, device='cpu', verbose=False, imgsz=imgsz)[0]
        else:
            raise
    ann = frame.copy()
    n_peat, n_veh, hay_emerg = 0, 0, False
    if result.boxes is not None and len(result.boxes):
        for box in result.boxes:
            cls  = int(box.cls[0])
            cval = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            color = COLORES_BGR.get(cls, (200, 200, 200))
            cv2.rectangle(ann, (x1, y1), (x2, y2), color, 2)
            label = f"{CLASES.get(cls,'?')} {cval:.0%}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            cv2.rectangle(ann, (x1, y1-lh-8), (x1+lw+4, y1), color, -1)
            cv2.putText(ann, label, (x1+2, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1)
            if cls == 0: n_peat += 1
            elif cls == 1: n_veh += 1
            elif cls == 2:
                if cval >= 0.50:   # umbral alto para emergencia, evita falsos positivos
                    hay_emerg = True
                n_veh += 1
    return ann, n_peat, n_veh, hay_emerg

def fase_to_colors(fase, cam_prio, fase_anterior=None, emerg_cam=None, fase_siguiente=None):
    """Retorna (color_borde_sur, color_borde_este) en BGR."""
    verde = (0, 200, 80)
    rojo  = (40, 40, 220)
    amar  = (0, 180, 220)
    if fase == 'FALLA_SEGURA':
        return amar, amar
    if fase == 'EMERGENCIA':
        if cam_prio == 'cam_sur':
            return verde, rojo
        return rojo, verde
    if fase == 'FASE_SN':
        return verde, rojo
    if fase == 'FASE_EO':
        return rojo, verde
    if fase == 'TRANSICION':
        if fase_anterior == 'FASE_SN':
            return amar, rojo
        if fase_anterior == 'FASE_EO':
            return rojo, amar
        if fase_anterior == 'EMERGENCIA':
            # La cam que tenía verde en emergencia pasa a amarillo (si cambia)
            sn_tenia = emerg_cam == 'cam_sur'
            sn_recibe = fase_siguiente == 'FASE_SN'
            if sn_tenia and not sn_recibe:
                return amar, rojo
            if not sn_tenia and sn_recibe:
                return rojo, amar
            return (verde, rojo) if sn_recibe else (rojo, verde)
    return amar, amar

# ── Thread de procesamiento ───────────────────────────────────
def processing_loop():
    with _lock:
        v1 = state['video1']
        v2 = state['video2']

    if not v1 or not v2:
        return

    modelo      = get_model()
    controlador = ControladorInterseccion()
    cap1        = cv2.VideoCapture(v1)
    cap2        = cv2.VideoCapture(v2)
    fps_src     = cap1.get(cv2.CAP_PROP_FPS) or 25
    max_fps     = 10 if DEVICE == 'cpu' else 25
    target_dt   = 1.0 / min(fps_src, max_fps)

    frame_count = 0
    t_start     = time.time()

    with _lock:
        state['running'] = True

    print(f'[INFO] Procesando en {DEVICE}  |  FPS objetivo: {fps_src:.0f}')

    while True:
        with _lock:
            if not state['running']:
                break

        t0 = time.time()

        ret1, f1 = cap1.read()
        ret2, f2 = cap2.read()
        if not ret1:
            cap1.set(cv2.CAP_PROP_POS_FRAMES, 0); ret1, f1 = cap1.read()
        if not ret2:
            cap2.set(cv2.CAP_PROP_POS_FRAMES, 0); ret2, f2 = cap2.read()
        if not ret1 or not ret2:
            break

        f1 = cv2.resize(f1, (640, 360))
        f2 = cv2.resize(f2, (640, 360))

        ann1, p1, v1d, e1 = infer_frame(modelo, f1)
        ann2, p2, v2d, e2 = infer_frame(modelo, f2)

        detecciones = {
            Camara.SUR:  DeteccionPorCamara(Camara.SUR,  p1, v1d, e1),
            Camara.ESTE: DeteccionPorCamara(Camara.ESTE, p2, v2d, e2),
        }
        dec = controlador.decidir(detecciones)
        fase     = dec['fase'].value
        cam_prio = dec.get('camara_prioritaria')
        cam_prio_str  = cam_prio.value if cam_prio else None
        fase_sig      = dec.get('fase_siguiente')
        fase_sig_str  = fase_sig.value if fase_sig else None
        fase_ant      = dec.get('fase_anterior')
        fase_ant_str  = fase_ant.value if fase_ant else None
        emerg_cam     = dec.get('emerg_cam')  # cam con prioridad durante emergencia

        c_sur, c_este = fase_to_colors(fase, cam_prio_str, fase_ant_str, emerg_cam, fase_sig_str)

        ann1 = draw_hud(ann1, 'CAM SUR  (Sur -> Norte)', p1, v1d, e1, c_sur)
        ann2 = draw_hud(ann2, 'CAM ESTE (Este -> Oeste)', p2, v2d, e2, c_este)

        frame_count += 1
        fps = frame_count / max(time.time() - t_start, 0.001)

        with _lock:
            state['frame1']   = to_jpeg(ann1)
            state['frame2']   = to_jpeg(ann2)
            state['decision'] = {
                'fase':           fase,
                'motivo':         dec.get('motivo', ''),
                'cam_prio':       cam_prio_str,
                'fase_siguiente': fase_sig_str,
                'fase_anterior':  fase_ant_str,
                'emerg_cam':      emerg_cam,
            }
            state['stats'] = {
                'fps':       round(fps, 1),
                'det1':      p1 + v1d,
                'det2':      p2 + v2d,
                'frames':    frame_count,
                'emergencia': e1 or e2,
            }

        elapsed = time.time() - t0
        wait    = target_dt - elapsed
        if wait > 0:
            time.sleep(wait)

    cap1.release()
    cap2.release()
    with _lock:
        state['running'] = False
    print('[INFO] Procesamiento terminado.')

# ── Flask routes ──────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', device=DEVICE, model=os.path.basename(MODEL_PATH))

@app.route('/upload/<int:cam>', methods=['POST'])
def upload(cam):
    f = request.files.get('video')
    if not f:
        return jsonify({'ok': False, 'error': 'Sin archivo'})
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({'ok': False, 'error': f'Formato no soportado: {ext}'})
    fname = secure_filename(f'cam{cam}{ext}')
    path  = os.path.join(UPLOAD_FOLDER, fname)
    f.save(path)
    with _lock:
        state[f'video{cam}'] = path
    return jsonify({'ok': True, 'name': f.filename})

@app.route('/start', methods=['POST'])
def start():
    global _proc_thread
    with _lock:
        if state['running']:
            return jsonify({'ok': False, 'error': 'Ya en ejecución'})
        if not state['video1'] or not state['video2']:
            return jsonify({'ok': False, 'error': 'Falta cargar un video'})
    _proc_thread = threading.Thread(target=processing_loop, daemon=True)
    _proc_thread.start()
    return jsonify({'ok': True})

@app.route('/stop', methods=['POST'])
def stop():
    with _lock:
        state['running'] = False
    return jsonify({'ok': True})

@app.route('/reset', methods=['POST'])
def reset():
    with _lock:
        state['running']  = False
        state['frame1']   = None
        state['frame2']   = None
        state['video1']   = None
        state['video2']   = None
        state['decision'] = {'fase': 'ESPERANDO', 'motivo': 'Carga videos y presiona Iniciar', 'cam_prio': None}
        state['stats']    = {'fps': 0, 'det1': 0, 'det2': 0, 'frames': 0, 'emergencia': False}
    return jsonify({'ok': True})

def _mjpeg(cam_key, label):
    placeholder = blank_frame(label)
    while True:
        with _lock:
            frame = state.get(cam_key)
        data = frame if frame else placeholder
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + data + b'\r\n')
        time.sleep(0.04)

@app.route('/stream/cam1')
def stream_cam1():
    return Response(_mjpeg('frame1', 'Sin señal — Cámara 1'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream/cam2')
def stream_cam2():
    return Response(_mjpeg('frame2', 'Sin señal — Cámara 2'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/events')
def events():
    def generate():
        while True:
            with _lock:
                payload = {
                    'decision': state['decision'],
                    'stats':    state['stats'],
                    'running':  state['running'],
                    'device':   _active_device,
                }
            yield f'data: {json.dumps(payload)}\n\n'
            time.sleep(0.15)
    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

@app.route('/restart-gpu', methods=['POST'])
def restart_gpu():
    import subprocess, shlex
    script = (
        'sleep 1; '
        'fuser -k 5000/tcp 2>/dev/null; '
        'sleep 2; '
        f'cd "{os.path.dirname(os.path.abspath(__file__))}/.." && '
        f'nohup python3 demo/app.py >> /tmp/demo.log 2>&1 &'
    )
    subprocess.Popen(['bash', '-c', script])
    return jsonify({'ok': True})

if __name__ == '__main__':
    print(f'[INFO] Dispositivo: {DEVICE}')
    print(f'[INFO] Modelo:      {MODEL_PATH}')
    print('[INFO] Precargando modelo YOLO...')
    get_model()
    print('[INFO] Modelo listo. Abre http://localhost:5000 en tu navegador')
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
