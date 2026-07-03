# Protocolo de Grabación — Dataset LSP (Lengua de Señas Peruana)

**Proyecto:** Sistema Integral LSP → Castellano  
**Versión:** 1.0 | Julio 2026  
**Compatible con:** Pipeline MediaPipe Holistic · 30 frames · 150 dims

---

## Por qué este protocolo

El modelo actual tiene mediana de **4 muestras por clase** y el 57 % de las 479 clases
tienen menos de 5 grabaciones. Con esos volúmenes ninguna arquitectura generaliza.
Este protocolo busca llevar cada clase a **≥ 50 muestras de ≥ 5 señantes distintos**,
que es el umbral mínimo para superar F1-macro > 0.10 de forma estable.

---

## 1. Participantes (señantes)

### Quiénes pueden participar

| Perfil | Prioridad | Motivo |
|--------|-----------|--------|
| Persona sorda nativa de LSP | **Alta** | Producción natural de la seña |
| Intérprete certificado de LSP | Alta | Producción precisa y consistente |
| Estudiante avanzado de LSP (≥ 2 años) | Media | Volumen extra |
| Oyente con curso básico de LSP | Baja | Solo para abecedario o señas muy simples |

### Requisitos por sesión

- **Mínimo 5 señantes distintos** por grupo de grabación
- Diversidad obligatoria dentro del grupo:
  - Al menos 2 mujeres y 2 hombres
  - Al menos 1 persona mayor de 40 años
  - Al menos 1 persona menor de 25 años
  - Si es posible: 1 señante zurdo
- **No grabar el mismo señante dos veces en el mismo día** (evita sesgo de fatiga)

### Consentimiento

Antes de grabar, cada participante firma un **consentimiento informado** que incluye:
- Uso de las grabaciones para entrenamiento de modelos de IA
- Anonimización (no se publica el video original, solo los keypoints extraídos)
- Derecho a retirar su participación y borrar sus datos

---

## 2. Equipamiento

### Cámara

| Parámetro | Valor requerido | Notas |
|-----------|----------------|-------|
| Resolución | ≥ 1280 × 720 (HD) | Full HD preferido (1920×1080) |
| FPS | **30 fps exactos** | El pipeline resamplea a 30 frames |
| Formato | MP4 (H.264) | Compatible con OpenCV |
| Estabilización | Tripié o soporte fijo | Sin cámara en mano |

Opciones válidas: webcam HD, cámara de celular fija en soporte, cámara DSLR/mirrorless.

### Posición de cámara

```
        [CÁMARA]
            |
            |  1.5 – 2.0 m
            |
       [SEÑANTE]
    (de pie o sentado)
```

- **Altura:** lente al nivel del pecho del señante (no desde arriba ni desde abajo)
- **Ángulo:** frontal, ±10° máximo de desviación lateral
- **Encuadre:** cabeza completa + ambas manos visibles en todo momento, con 15 cm de margen en cada lado

### Iluminación

- Luz **frontal o lateral difusa** — nunca a contraluz
- Mínimo 300 lux sobre el señante (equivalente a oficina bien iluminada)
- Sin sombras duras sobre las manos o la cara
- Sesiones variadas: grabar en al menos **3 condiciones de luz distintas**:
  - Luz artificial estable (fluorescente o LED)
  - Luz natural (ventana lateral, sin sol directo)
  - Luz mixta (natural + artificial)

### Fondo

- Preferido: fondo liso de color **azul, gris o verde claro**
- Aceptable: pared de color uniforme
- Evitar: fondos con mucho movimiento, patrones complejos, espejos
- Grabar en al menos **2 fondos distintos** por señante

### Ropa del señante

- Manga corta o camiseta sin mangas (manos y muñecas visibles)
- Color que contraste con el tono de piel (evitar beige/rosado)
- Sin mangas amplias que tapen las muñecas
- Sin anillos, pulseras anchas o guantes (interfieren con la detección de manos)

---

## 3. Clases a grabar

### Prioridad 1 — Glosas de uso frecuente (vacías de datos)

Clases con < 10 muestras actuales. Grabar **50 repeticiones mínimo** cada una,
distribuidas entre ≥ 5 señantes.

Ejemplos de glosas prioritarias del dataset actual:
`AGUA, CASA, GRACIAS, HOLA, SÍ, NO, AYUDA, DOCTOR, ENFERMERA,
DOLOR, HOSPITAL, MEDICINA, COMER, BEBER, BAÑO, DINERO, TRABAJO,
FAMILIA, MADRE, PADRE, HIJO, ESCUELA, MAESTRO, APRENDER`

### Prioridad 2 — Abecedario completo (A–Z)

El abecedario ya tiene muestras pero con poca diversidad de señantes.
Meta: **100 repeticiones de cada letra**, de ≥ 10 señantes distintos.

### Prioridad 3 — Frases cortas (para sistema completo)

Frases de 3–5 señas en secuencia, grabadas sin pausa entre señas:
- `HOLA + ¿CÓMO ESTÁS?`
- `NECESITO + AYUDA + MÉDICO`
- `¿DÓNDE + ESTÁ + EL + BAÑO?`

---

## 4. Procedimiento de grabación por sesión

### Preparación (10 min antes)

1. Verificar encuadre: el señante extiende ambos brazos — deben quedar dentro del frame
2. Verificar iluminación: no deben verse sombras sobre las manos en el monitor
3. Grabar **clip de prueba de 10 segundos** y revisar que MediaPipe detecta manos
4. Pedir al señante que se siente/pare cómodamente — sin tensión en hombros

### Por cada seña a grabar

```
┌─────────────────────────────────────────────────────────────┐
│  SECUENCIA POR TOMA                                         │
│                                                             │
│  1. Señante en posición neutral (manos abajo, relajadas)   │
│  2. Operador dice "LISTO" y comienza la grabación          │
│  3. Señante realiza la seña UNA VEZ de forma natural       │
│  4. Señante vuelve a posición neutral                       │
│  5. Operador detiene grabación                              │
│  6. Repetir desde paso 1                                    │
└─────────────────────────────────────────────────────────────┘
```

**Una seña = un archivo MP4 separado.** No grabar múltiples repeticiones en un solo video.

### Repeticiones requeridas por sesión

| Por señante | Por seña | Total mínimo |
|-------------|---------|-------------|
| 10 repeticiones | 1 seña por archivo | 50 archivos si hay 5 señantes |

- Entre repeticiones: **pausa de 2–3 segundos** (señante vuelve a posición neutral)
- Cada 20 señas: descanso de 5 minutos (evita fatiga que altera la forma de la seña)
- Máximo 30 minutos de grabación continua por señante

### Variaciones intencionadas a capturar

Dentro de las 10 repeticiones de un señante, variar naturalmente:
- **Velocidad:** 3 repeticiones lentas, 4 normales, 3 rápidas
- **Dominancia de mano:** si la seña acepta variación, grabar algunas con la mano no dominante
- No pedir variaciones artificiales — deben surgir de forma natural

---

## 5. Nombrado de archivos y estructura de carpetas

### Estructura de directorios

```
data/
└── NuevasGrabaciones/
    ├── glosas_nuevas/
    │   ├── AGUA/
    │   │   ├── AGUA_S01_001.mp4   ← señante 1, repetición 1
    │   │   ├── AGUA_S01_002.mp4
    │   │   ├── AGUA_S02_001.mp4   ← señante 2, repetición 1
    │   │   └── ...
    │   ├── CASA/
    │   └── ...
    └── abecedario_nuevo/
        ├── A/
        │   ├── A_S01_001.mp4
        │   └── ...
        └── ...
```

### Convención de nombre de archivo

```
<CLASE>_<SEÑANTE_ID>_<REPETICION>.mp4

Ejemplo: AGUA_S03_007.mp4
  AGUA      → nombre de la seña (en mayúsculas, sin tildes en el nombre del archivo)
  S03       → ID del señante (S01, S02, ... S99)
  007       → número de repetición (001 a 999)
```

### Registro de señantes (archivo señantes.csv)

Mantener un CSV con los datos de cada señante:

```csv
signer_id,nombre,edad,genero,condicion,region,años_lsp,fecha_sesion
S01,Ana G.,34,F,sorda_nativa,Lima,30,2026-07-05
S02,Carlos R.,28,M,interprete,Lima,8,2026-07-05
S03,María T.,52,F,sorda_nativa,Arequipa,48,2026-07-06
```

El campo `signer_id` se convierte en el campo `groups` del dataset NPZ, que es lo que usa
el test HE3 para verificar que el modelo generaliza a señantes no vistos.

---

## 6. Control de calidad — verificar antes de guardar

### Criterios de rechazo (eliminar el clip)

| Problema | Criterio |
|----------|----------|
| Manos fuera de encuadre | Alguna mano sale del frame durante la seña |
| Oclusión severa | Una mano tapa a la otra > 30 % del tiempo |
| Mala iluminación | Manos o cara con sombra que tapa puntos clave |
| Seña incorrecta | El señante se interrumpe, corrige o ríe durante la seña |
| Clip muy corto | Menos de 10 frames con manos visibles |
| Sin posición neutral | El señante no regresó a posición neutral antes de empezar |

### Verificación rápida con MediaPipe (script de control)

Después de cada sesión, correr:

```bash
.venv311/bin/python3 scripts/verificar_grabaciones.py data/NuevasGrabaciones/
```

El script reporta qué clips tienen problemas de detección de manos y permite
descartar antes de extraer keypoints.

---

## 7. Sesión tipo — ejemplo de un día de grabación

| Hora | Actividad | Duración |
|------|-----------|---------|
| 09:00 | Preparar setup, verificar cámara e iluminación | 15 min |
| 09:15 | Explicar el proceso a los señantes, firmar consentimientos | 15 min |
| 09:30 | Clip de prueba por señante, ajustar encuadre | 10 min |
| 09:40 | Grabación — Bloque 1: 15 glosas × 10 repeticiones × 5 señantes | 60 min |
| 10:40 | Descanso | 15 min |
| 10:55 | Grabación — Bloque 2: 15 glosas más | 60 min |
| 11:55 | Verificación de calidad (script) | 20 min |
| 12:15 | Fin de sesión |  |

**Producción por día:** ~150 glosas × 5 señantes × 10 reps = **750 clips válidos**  
**Meta para F1 > 0.10:** ~50 glosas × 50 muestras = 2,500 clips adicionales (≈ 4 días)

---

## 8. Integración al pipeline existente

Una vez grabados y verificados los clips, el flujo de integración es:

```bash
# 1. Extraer keypoints de los MP4 nuevos
.venv311/bin/python3 scripts/extract_glosas_keypoints.py \
    --source data/NuevasGrabaciones/glosas_nuevas/ \
    --output data/Keypoints/nuevas_pkl/

# 2. Reconstruir dataset S12 unificando todo
.venv310/bin/python3 scripts/build_dataset_s11.py   # adaptar a S12

# 3. Re-entrenar modelos con el dataset expandido
.venv310/bin/python3 scripts/train_s11.py           # adaptar a S12
```

El campo `groups` del NPZ se llena automáticamente con el `signer_id` extraído
del nombre del archivo — por eso la convención de nombrado es crítica.

---

## 9. Metas cuantitativas por sprint

| Sprint | Meta de datos | F1-test esperado |
|--------|--------------|-----------------|
| S11 actual | 6,855 muestras, mediana 4/clase | ~0.024 |
| S12 | +2,500 clips (top 50 glosas a 50 muestras) | ~0.08–0.12 |
| S13 | +5,000 clips (100 glosas a 100 muestras) | ~0.20–0.30 |
| Sistema robusto | 50,000+ clips (500 glosas a 100+ muestras) | ~0.50+ |

---

## 10. Checklist por sesión de grabación

```
□ Cámara en tripié, lente al nivel del pecho del señante
□ Encuadre verificado: ambas manos visibles al extender brazos
□ Iluminación: sin sombras en manos ni cara
□ Fondo liso y contrastante
□ Ropa del señante: manga corta, color contrastante, sin accesorios en manos
□ Consentimiento informado firmado
□ Archivo señantes.csv actualizado con el signer_id correcto
□ Clip de prueba revisado con MediaPipe antes de empezar
□ Nombres de archivo siguiendo convención: <CLASE>_<S##>_<###>.mp4
□ Script de verificación corrido al terminar la sesión
□ Clips rechazados eliminados antes de extraer keypoints
```
