"""Post-procesamiento lingüístico — MÓDULO 3 de docs/PROMPT_MAESTRO_LSP_SISTEMA_COMPLETO.md.

Suavizado temporal por voto mayoritario: evita que una predicción espuria de
un solo frame quede confirmada en la transcripción. La corrección gramatical
SOV→SVO con BERT descrita en el mismo documento no tiene diseño ni código
existente en el proyecto — queda fuera de alcance, ver ROADMAP_FRONTEND_Y_EVALUACION.md.
"""

from collections import deque, Counter
from typing import Optional


class SuavizadorTemporal:
    """Confirma una seña solo cuando gana el voto mayoritario de las últimas
    `ventana` predicciones crudas, y solo si difiere de la última confirmada."""

    def __init__(self, ventana: int = 3):
        self.ventana = ventana
        self._historial = deque(maxlen=ventana)
        self._ultima_confirmada: Optional[str] = None

    def push(self, seña: str) -> Optional[str]:
        self._historial.append(seña)
        if len(self._historial) < self.ventana:
            return None
        confirmada = Counter(self._historial).most_common(1)[0][0]
        if confirmada != self._ultima_confirmada:
            self._ultima_confirmada = confirmada
            return confirmada
        return None

    def reset(self):
        self._historial.clear()
        self._ultima_confirmada = None
