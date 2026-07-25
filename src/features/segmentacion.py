"""Segmentación de video continuo en señas individuales por detección de pausa.

Ataca la causa raíz del problema encontrado en demo/app_gradio.py: el modelo
se entrenó y se midió (F1=0.4349) clasificando clips YA segmentados como una
sola seña — nunca aprendió "qué gloss ocurre en este instante de un video
continuo". Antes, la demo pedía una clasificación cada N_FRAMES/2 frames sin
importar el contenido, lo cual no tiene relación con dónde empieza/termina
una seña real.

Esta clase corta el stream de landmarks en el límite real más disponible sin
reentrenar el modelo: la pausa/reposo de las manos entre una seña y la
siguiente. Cada segmento resultante se re-muestrea a 30 frames (igual que en
entrenamiento) y se clasifica una sola vez, como un clip aislado — la misma
tarea para la que el modelo fue medido.

Límite conocido: funciona para señante que hace pausas breves entre señas
(el caso de uso real de comunicación asistida). NO resuelve narración fluida
sin pausas (p.ej. video narrado profesionalmente de corrido) — eso es
reconocimiento continuo gloss-a-gloss (CTC/seq2seq), un problema de
investigación aparte, no un ajuste de segmentación.
"""

import numpy as np


class SegmentadorPausas:
    def __init__(self,
                 umbral_movimiento: float = 0.005,
                 frames_pausa: int = 6,
                 min_frames_seña: int = 3,
                 max_frames_seña: int = 90):
        """
        umbral_movimiento : desplazamiento medio (landmarks normalizados 0-1)
                             de manos entre frames consecutivos por debajo del
                             cual se considera "reposo". Calibrado con datos
                             reales (2026-07-16): se midió el movimiento cuadro
                             a cuadro en 15 clips AEC reales (213 frames) — el
                             reposo genuino (p5-p10 de la distribución) está en
                             0.000-0.006, mientras que señas lentas pero reales
                             (ej. "vaca", "tú") tienen mediana ~0.007-0.02. El
                             valor anterior (0.008) cortaba de más en señas
                             lentas; 0.005 se acerca más al reposo real sin
                             invadir el rango de señas de poco movimiento.
        frames_pausa       : cuántos frames seguidos de reposo confirman el
                             fin de una seña (evita cortar por micro-pausas).
        min_frames_seña    : mínimo de frames de contenido (sin contar la
                             pausa final) para considerar que hubo una seña
                             real. Bajo a propósito — señas rápidas de clips
                             ya recortados pueden durar muy pocos frames.
        max_frames_seña    : tope duro — evita que una racha sin pausa (p.ej.
                             narración continua) crezca indefinidamente sin
                             nunca cerrar un segmento.

        Importante: el buffer acumula TODOS los frames que llegan, se midan
        o no como "movimiento" — el contador de pausa solo decide CUÁNDO
        cerrar el segmento, no qué frames guardar. La primera versión
        descartaba cualquier frame clasificado como pausa, lo que vaciaba
        clips cortos enteros cuando el umbral de movimiento no calzaba bien
        con esa muestra puntual (ruido de MediaPipe, sensibilidad del
        landmark, etc.) — ver commit que corrige esto.
        """
        self.umbral_movimiento = umbral_movimiento
        self.frames_pausa = frames_pausa
        self.min_frames_seña = min_frames_seña
        self.max_frames_seña = max_frames_seña
        self._buffer = []
        self._contador_pausa = 0
        self._prev_kp = None

    def push(self, kp: np.ndarray):
        """kp: landmarks del frame actual, [75,3] (ver src/features/landmarks.py).
        Devuelve la lista de frames del segmento cuando se cierra una seña
        (por pausa sostenida o por tope de longitud), o None si sigue
        acumulando."""
        movimiento = self._calc_movimiento(kp)
        self._prev_kp = kp
        self._buffer.append(kp)

        if movimiento < self.umbral_movimiento:
            self._contador_pausa += 1
        else:
            self._contador_pausa = 0

        contenido_real = len(self._buffer) - self._contador_pausa
        cierre_por_pausa = (self._contador_pausa >= self.frames_pausa
                             and contenido_real >= self.min_frames_seña)
        cierre_por_tope  = len(self._buffer) >= self.max_frames_seña

        if cierre_por_pausa or cierre_por_tope:
            # recorta los frames de pausa que quedaron al final (ya son
            # reposo, no aportan a la seña que se está cerrando)
            segmento = self._buffer[:-self._contador_pausa] if (cierre_por_pausa and self._contador_pausa) else self._buffer
            self._buffer = []
            self._contador_pausa = 0
            return segmento
        return None

    def __len__(self):
        """Frames acumulados en el segmento actual (para reportar progreso,
        p.ej. en el estado 'buffering' de un cliente en tiempo real)."""
        return len(self._buffer)

    def flush(self):
        """Cierra y devuelve lo acumulado al terminar el stream (video/cámara
        se corta a mitad de una seña, sin pausa que la cierre formalmente).
        None si no quedó prácticamente nada."""
        segmento = self._buffer if len(self._buffer) >= self.min_frames_seña else None
        self._buffer = []
        self._contador_pausa = 0
        return segmento

    def _calc_movimiento(self, kp: np.ndarray) -> float:
        if self._prev_kp is None:
            return 999.0  # primer frame: no hay pausa posible todavía
        # Solo manos (índices 0:42) — la pose del torso es más estable y
        # menos informativa del límite real entre una seña y la siguiente.
        diff = kp[:42, :2] - self._prev_kp[:42, :2]
        return float(np.mean(np.linalg.norm(diff, axis=1)))
