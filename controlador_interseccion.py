from enum import Enum
from collections import deque, Counter
from dataclasses import dataclass

class Camara(Enum):
    SUR  = 'cam_sur'
    ESTE = 'cam_este'

class FaseSemaforo(Enum):
    FASE_SN      = 'FASE_SN'
    FASE_EO      = 'FASE_EO'
    TRANSICION   = 'TRANSICION'
    EMERGENCIA   = 'EMERGENCIA'
    FALLA_SEGURA = 'FALLA_SEGURA'

@dataclass
class DeteccionPorCamara:
    camara: Camara
    n_peatones: int = 0
    n_vehiculos: int = 0
    hay_emergencia: bool = False
    visibilidad_ok: bool = True

@dataclass
class ConfigInterseccion:
    umbral_densidad_vehicular: int  = 5
    duracion_minima_frames: int     = 500   # 20s x 25fps
    duracion_normal_frames: int     = 750   # 30s x 25fps
    ventana_estabilidad_frames: int = 15
    debounce_salida_emergencia: int = 125   # 5s x 25fps sin detección para salir
    duracion_transicion_frames: int = 75    # 3s x 25fps de amarillo entre fases

class ControladorInterseccion:
    def __init__(self, config=None):
        self.config = config or ConfigInterseccion()
        self.fase_actual = FaseSemaforo.FASE_SN
        self._frame_actual = 0
        self._frame_inicio_fase = 0
        self._frames_sin_emergencia = 0
        self._frames_guardados = 0  # preserva el timer durante emergencia
        self._emergency_latched = False      # True desde 1er frame detectado
        self._last_emergency_cam = None    # última cámara que vio emergencia
        self._fase_siguiente = None        # fase destino durante TRANSICION
        self._fase_anterior  = None        # fase de origen durante TRANSICION
        self._fase_antes_emergencia = None # fase real activa antes de la emergencia
        self.historial = deque(maxlen=self.config.ventana_estabilidad_frames)

    def _cambiar_fase(self, nueva_fase):
        self._fase_anterior = self.fase_actual
        self.fase_actual = nueva_fase
        self._frame_inicio_fase = self._frame_actual
        self.historial.clear()

    def decidir(self, detecciones):
        self._frame_actual += 1
        frames_en_fase = self._frame_actual - self._frame_inicio_fase

        # REGLA 0 - falla segura
        camaras_ciegas = sum(1 for d in detecciones.values() if not d.visibilidad_ok)
        if camaras_ciegas >= 1:
            if self.fase_actual != FaseSemaforo.FALLA_SEGURA:
                self._cambiar_fase(FaseSemaforo.FALLA_SEGURA)
            return {'fase': self.fase_actual,
                    'motivo': f'{camaras_ciegas} camara(s) sin visibilidad'}

        # REGLA 1 - emergencia: latch inmediato, salida tras 5s sin detección
        hay_emerg_actual = any(d.hay_emergencia and d.visibilidad_ok for d in detecciones.values())

        if hay_emerg_actual:
            self._emergency_latched     = True
            self._frames_sin_emergencia = 0
            for cam, det in detecciones.items():
                if det.hay_emergencia and det.visibilidad_ok:
                    self._last_emergency_cam = cam
                    break
        else:
            self._frames_sin_emergencia += 1
            if self._frames_sin_emergencia >= self.config.debounce_salida_emergencia:
                self._emergency_latched  = False
                # _last_emergency_cam se conserva hasta que termine la transición de salida

        # Ya en EMERGENCIA: mantener mientras latch activo
        if self.fase_actual == FaseSemaforo.EMERGENCIA:
            if self._emergency_latched:
                secs_restantes = (self.config.debounce_salida_emergencia - self._frames_sin_emergencia) / 25
                return {'fase': FaseSemaforo.EMERGENCIA,
                        'camara_prioritaria': self._last_emergency_cam,
                        'motivo': f'Emergencia activa (cierre en >{secs_restantes:.0f}s sin detección)',
                        'inmediata': True}
            # Latch liberado: salir via transición amarilla (respeta mínimo y amarillo)
            fase_restaurar = self._fase_antes_emergencia or FaseSemaforo.FASE_SN
            self._fase_antes_emergencia = None
            self._frames_guardados      = 0   # la fase restaurada empieza desde cero
            self._fase_siguiente = fase_restaurar
            self._cambiar_fase(FaseSemaforo.TRANSICION)  # _fase_anterior = EMERGENCIA
            frames_en_fase = self._frame_actual - self._frame_inicio_fase  # = 0

        # Emergencia detectada: interrumpir cualquier estado y activar transición amarilla
        ya_en_transicion_a_emerg = (self.fase_actual == FaseSemaforo.TRANSICION
                                    and self._fase_siguiente == FaseSemaforo.EMERGENCIA)
        if self._emergency_latched and self.fase_actual != FaseSemaforo.EMERGENCIA and not ya_en_transicion_a_emerg:
            # Guardar la fase real (antes de cualquier transición pendiente)
            if self.fase_actual == FaseSemaforo.TRANSICION:
                self._fase_antes_emergencia = self._fase_anterior  # fase antes de la transición interrumpida
            else:
                self._fase_antes_emergencia = self.fase_actual
            self._frames_guardados = self._frame_actual - self._frame_inicio_fase
            self._fase_siguiente   = FaseSemaforo.EMERGENCIA
            self._cambiar_fase(FaseSemaforo.TRANSICION)
            frames_en_fase = self._frame_actual - self._frame_inicio_fase  # = 0

        # TRANSICION amarilla entre fases
        if self.fase_actual == FaseSemaforo.TRANSICION:
            if frames_en_fase >= self.config.duracion_transicion_frames:
                viene_de_emergencia = self._fase_anterior == FaseSemaforo.EMERGENCIA
                next_fase = self._fase_siguiente
                self._cambiar_fase(next_fase)
                self._fase_siguiente = None
                if viene_de_emergencia:
                    self._last_emergency_cam = None  # ahora sí se puede limpiar
                frames_en_fase = self._frame_actual - self._frame_inicio_fase  # = 0
                if self.fase_actual == FaseSemaforo.EMERGENCIA:
                    return {'fase': FaseSemaforo.EMERGENCIA,
                            'camara_prioritaria': self._last_emergency_cam,
                            'motivo': 'Emergencia activa',
                            'inmediata': True}
                # Fase normal: continúa al bloqueo minimo con timer en 0
            else:
                t     = frames_en_fase / 25
                t_max = self.config.duracion_transicion_frames / 25
                return {'fase': FaseSemaforo.TRANSICION,
                        'fase_anterior':  self._fase_anterior,
                        'fase_siguiente': self._fase_siguiente,
                        'emerg_cam': self._last_emergency_cam.value if self._last_emergency_cam else None,
                        'motivo': f'Transicion amarilla ({t:.1f}s / {t_max:.0f}s)'}

        # Sin emergencia: volver a FASE_SN preservando el timer anterior
        if self.fase_actual == FaseSemaforo.FALLA_SEGURA:
            self.fase_actual = FaseSemaforo.FASE_SN
            self.historial.clear()
            self._fase_siguiente      = None
            self._fase_anterior       = None
            self._last_emergency_cam  = None
            self._frame_inicio_fase = self._frame_actual - self._frames_guardados
            self._frames_guardados  = 0
            frames_en_fase = self._frame_actual - self._frame_inicio_fase

        # Bloqueo minimo
        if frames_en_fase < self.config.duracion_minima_frames:
            t     = frames_en_fase / 25
            t_min = self.config.duracion_minima_frames / 25
            return {'fase': self.fase_actual,
                    'motivo': f'Fase activa ({t:.0f}s / min {t_min:.0f}s)'}

        # REGLA 2 - densidad asimetrica
        veh_sn = detecciones[Camara.SUR].n_vehiculos
        veh_eo = detecciones[Camara.ESTE].n_vehiculos
        if abs(veh_sn - veh_eo) >= self.config.umbral_densidad_vehicular:
            propuesta = FaseSemaforo.FASE_SN if veh_sn > veh_eo else FaseSemaforo.FASE_EO
            return self._aplicar_con_estabilidad(propuesta,
                     motivo=f'Alta demanda (SN={veh_sn} EO={veh_eo})')

        # REGLA 3 - fin de ciclo normal
        if frames_en_fase >= self.config.duracion_normal_frames:
            siguiente = FaseSemaforo.FASE_EO if self.fase_actual == FaseSemaforo.FASE_SN else FaseSemaforo.FASE_SN
            return self._aplicar_con_estabilidad(siguiente,
                     motivo=f'Fin de ciclo ({frames_en_fase/25:.0f}s)')

        return {'fase': self.fase_actual,
                'motivo': f'Ciclo normal ({frames_en_fase/25:.0f}s / {self.config.duracion_normal_frames/25:.0f}s)'}

    def _aplicar_con_estabilidad(self, fase_propuesta, motivo=''):
        self.historial.append(fase_propuesta)
        if len(self.historial) >= self.config.ventana_estabilidad_frames:
            mas_comun, conteo = Counter(self.historial).most_common(1)[0]
            if conteo >= self.config.ventana_estabilidad_frames * 0.6:
                if mas_comun != self.fase_actual and self.fase_actual != FaseSemaforo.TRANSICION:
                    self._fase_siguiente = mas_comun
                    self._cambiar_fase(FaseSemaforo.TRANSICION)
        en_transicion = self.fase_actual == FaseSemaforo.TRANSICION
        return {'fase': self.fase_actual, 'motivo': motivo,
                'fase_siguiente': self._fase_siguiente if en_transicion else None,
                'fase_anterior':  self._fase_anterior  if en_transicion else None}
