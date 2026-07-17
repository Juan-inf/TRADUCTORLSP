# Roadmap diferido: Frontend React y evaluación WER/BLEU continua

**Fecha:** 2026-07-11
**Alcance:** documenta dos piezas del objetivo "sistema integral" que se investigaron pero se dejaron explícitamente fuera de esta pasada de implementación (ver `ANALISIS_S27_OBJETIVOS_PROYECTO.md` y el plan de esta sesión). No modifica ningún informe ni notebook existente.

En esta pasada se conectó `bilstm_s27` (F1=0.4349, latencia 0.72ms) al backend real (`api/main.py`), se corrigió `demo/app_gradio.py` (checkpoint, normalización faltante, TTS de navegador), se agregó suavizado temporal (`src/nlp/postproceso.py`) y Docker. Lo que sigue abajo son las dos piezas grandes que quedan pendientes, con evidencia concreta de por qué no se abordaron ahora.

---

## 1. Frontend React (Fase 2)

**Por qué no se hizo ahora:** stack completamente distinto (JavaScript/TypeScript, no Python), sin ninguna base existente en el repo — ni `package.json`, ni `.tsx`, ni configuración de build. Es un proyecto de varios días, no una corrección puntual. Mientras tanto, `demo/app_gradio.py` ya cubre cámara en vivo + overlay de landmarks + traducción + TTS y ya está conectado al modelo real.

**Lo que ya está especificado** (no hay que redecidir arquitectura, solo construirla) en `docs/PROMPT_MAESTRO_LSP_SISTEMA_COMPLETO.md`:
- Estructura de archivos completa: líneas 1813-1860 (`frontend/src/components/{CameraPanel,TranslationPanel,ControlBar,MetricsBadge}.tsx`, `hooks/{useWebSocket,useCamera,useTTS}.ts`, `utils/{drawSkeleton,postprocess}.ts`).
- Mockup de layout y flujo de componentes: líneas 954-1010.
- Snippet de `useWebSocket.ts` ya escrito: líneas ~1820-1860.

**Corrección importante encontrada al investigar:** ese snippet apunta a un endpoint `ws://localhost:8000/ws/traducir/{clientId}` que **no es el que existe**. El endpoint real, implementado y funcional en `api/main.py`, es:

```
GET (WebSocket) /predict/stream
Cliente → {"frame": "<base64 JPEG>", "include_landmarks": true}
Servidor → {"clase": "...", "texto_castellano": "...", "confidence": 0.92, "latency_ms": 14, "top3": [...]}
         → {"status": "buffering", "frames_collected": N, "frames_needed": 30}  (mientras junta 30 frames)
```

Cuando se construya el frontend, `useWebSocket.ts` debe apuntar a `/predict/stream` con este protocolo (JSON con frame en base64), no al endpoint del doc original.

**Definición de "hecho" para esta fase:** cámara en el navegador → canvas con overlay de skeleton → texto traducido en vivo vía WebSocket → botón TTS (ya no hace falta reinventar TTS, `useTTS.ts` puede ser tan simple como el `speechSynthesis` ya usado en `demo/app_gradio.py`).

---

## 2. Evaluación WER/BLEU continua (Fase 3)

**El mismatch encontrado:** el modelo actual (S27, y todos los anteriores) hace **clasificación de clip completo** — cada "Historias vinetas (N)" es una sola clase de salida entre las 96. El ground truth en `data/SRT/SRT_SEGMENTED_SIGN/` (27 archivos `.srt`, confirmado alineado 1:1 con `data/Keypoints/pkl/Historias_vinetas_N/` para N ∈ {2,3,4,5,6,8,9,11,12,13,14,15,17,18,19,20,21,22,23,24,25,27,29,30,37,42,43}) es una **secuencia de glosas palabra por palabra** por clip (ej. `"Historias vinetas (23).srt"`: 298 cues en ~4 minutos, cada cue una glosa individual como `CHICA`, `MAMÁ`, `DIJO`, `TENER`).

Comparar "1 predicción de clase" contra "298 palabras de ground truth" no es una medición real de WER/BLEU — daría un número (WER cercano a 1.0) que no dice nada útil sobre el sistema. Por eso se documenta como fuera de alcance en vez de calcularlo igual (decisión explícita, ver conversación de esta sesión).

**Qué se necesitaría antes de que la métrica tenga sentido:**
1. Un pipeline de **reconocimiento continuo**: ventana deslizante sobre el video completo (no solo 30 frames fijos), con un mecanismo de segmentación (por cue de SRT, o por detección de pausas/transiciones entre señas) que produzca una *secuencia* de glosas predichas, no una sola clase.
2. Alineación temporal entre las predicciones y los timestamps del SRT (`HH:MM:SS,mmm --> HH:MM:SS,mmm`) para saber qué predicción corresponde a qué cue.
3. Recién ahí, `jiwer` (WER) y `sacrebleu`/`nltk` (BLEU) — ninguno de los dos está en `requirements.txt` todavía — tendrían algo significativo que comparar.

Esto es, en la práctica, un cambio de paradigma del modelo (de clasificación a reconocimiento continuo/secuencial), no una tarea de evaluación aislada — por eso se trata como una fase de investigación propia, no como una corrección de esta sesión.

---

## Estado de archivos tocados en esta sesión (para referencia)

- `src/features/landmarks.py` (nuevo, compartido api+demo)
- `src/nlp/postproceso.py` (nuevo, suavizado temporal)
- `api/main.py` (reescrito: landmarks + bilstm_s27, antes era pixels + checkpoint inexistente)
- `demo/app_gradio.py` (checkpoint s27, normalización agregada, TTS navegador, textos actualizados)
- `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `requirements-api.txt` (nuevos)

No se tocó: `INFORME_RENDIMIENTO_S26.*`, los notebooks de informes, `ANALISIS_S27_OBJETIVOS_PROYECTO.md`, `spaces/` (deploy HF Spaces separado, sigue con su propio modelo hasta que se actualice explícitamente).
