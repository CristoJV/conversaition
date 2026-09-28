# ConversAItion

Herramienta local para **transcribir conversaciones, identificar hablantes y visualizar la conversación** de forma similar a un chat.

Utiliza modelos de NVIDIA Nemotron para:

- Transcribir audio.
- Detectar distintos hablantes.
- Asociar cada fragmento de texto con su hablante.
- Generar un JSON estructurado con timestamps.
- Visualizar la conversación desde un visor HTML local.
- Renombrar los participantes desde el propio visor.

El procesamiento puede realizarse **completamente en local**.

---

## Flujo de trabajo

```text
audio.wav
    │
    ▼
conversaition
    │
    ├── Nemotron Diarization
    └── Nemotron ASR
    │
    ▼
transcription.json
    │
    ▼
viewer.html
    │
    ▼
Conversación visual
```

El archivo JSON es la **fuente de verdad** de la transcripción.

El visor HTML únicamente utiliza ese JSON para representar la conversación.

---

# Requisitos

El proyecto utiliza:

- Python 3.12
- `uv`
- PyTorch
- CUDA
- NVIDIA NeMo Speech
- FFmpeg para preparar audios cuando sea necesario

Es recomendable utilizar una GPU NVIDIA compatible con CUDA.

---

# Instalación

Clona el proyecto y entra en el directorio:

```bash
git clone <repository-url>
cd conversaition
```

Instala las dependencias:

```bash
uv sync
```

Comprueba que PyTorch detecta la GPU:

```bash
uv run python -c "
import torch
print('CUDA:', torch.cuda.is_available())
print(
    'GPU:',
    torch.cuda.get_device_name()
    if torch.cuda.is_available()
    else 'N/A'
)
"
```

Deberías obtener algo similar a:

```text
CUDA: True
GPU: NVIDIA GeForce RTX ...
```

# Modelos

ConversAItion utiliza dos modelos:

## Diarización

```text
nvidia/Nemotron-3-Diarization
```

## Transcripción

```text
nvidia/nemotron-3.5-asr-streaming-0.6b
```

Los modelos se descargan una vez y después pueden utilizarse localmente.

Crea el directorio:

```bash
mkdir -p models
```

Descarga el modelo de diarización:

```bash
hf download \
    nvidia/Nemotron-3-Diarization \
    Nemotron-3-Diarization.nemo \
    --local-dir models
```

Descarga el modelo ASR:

```bash
hf download \
    nvidia/nemotron-3.5-asr-streaming-0.6b \
    nemotron-3.5-asr-streaming-0.6b.nemo \
    --local-dir models
```

La estructura debería quedar aproximadamente así:

```text
.
├── models/
│   ├── Nemotron-3-Diarization.nemo
│   └── nemotron-3.5-asr-streaming-0.6b.nemo
│
├── src/
│   └── conversaition/
│       ├── __init__.py
│       ├── transcribe.py
│       └── static/
│           └── viewer.html
│
├── pyproject.toml
├── uv.lock
└── README.md
```

---

# Preparar el audio

El transcriptor espera audio:

```text
WAV
16 kHz
Mono
```

Si tu grabación está en MP3, estéreo o utiliza otra frecuencia de muestreo, puedes convertirla con FFmpeg.

Por ejemplo:

```bash
ffmpeg \
    -i recording.mp3 \
    -ac 1 \
    -ar 16000 \
    -c:a pcm_s16le \
    recording.wav
```

También sirve para normalizar un WAV existente:

```bash
ffmpeg \
    -i original.wav \
    -ac 1 \
    -ar 16000 \
    -c:a pcm_s16le \
    recording.wav
```

---

# Transcribir una conversación

ConversAItion instala el comando:

```bash
conversaition
```

Puedes ejecutarlo a través de `uv`:

```bash
uv run conversaition \
    -i recording.wav \
    -o recording.json
```

También puedes indicar explícitamente los modelos:

```bash
uv run conversaition \
    -i recording.wav \
    --diar-model models/Nemotron-3-Diarization.nemo \
    --asr-model models/nemotron-3.5-asr-streaming-0.6b.nemo \
    --language es-ES \
    --max-speakers 4 \
    -o recording.json
```

Para ver las opciones disponibles:

```bash
uv run conversaition --help
```

Si tienes el entorno virtual activado, también puedes ejecutar directamente:

```bash
conversaition --help
```

---

# Resultado

La transcripción se guarda en el archivo indicado mediante:

```text
-o
```

Por ejemplo:

```bash
uv run conversaition \
    -i recording.wav \
    -o recording.json
```

generará:

```text
recording.json
```

La transcripción **no se imprime completa por terminal**.

El resultado principal es el JSON.

---

# Formato del JSON

El archivo generado tiene una estructura similar a:

```json
[
  {
    "speaker": "speaker_0",
    "start": 0.48,
    "end": 3.12,
    "text": "Hola, buenos días."
  },
  {
    "speaker": "speaker_1",
    "start": 3.36,
    "end": 5.91,
    "text": "Buenos días, ¿qué tal?"
  },
  {
    "speaker": "speaker_0",
    "start": 6.16,
    "end": 11.04,
    "text": "Bien, quería comentarte una cosa."
  }
]
```

Cada segmento contiene:

- `speaker`: identificador del hablante.
- `start`: instante de inicio en segundos.
- `end`: instante de finalización en segundos.
- `text`: texto transcrito.

El JSON debe considerarse la **fuente principal de los datos**.

Esto permite crear distintos visores o exportaciones sin necesidad de volver a procesar el audio.


# Visualizar una conversación

El visor está incluido en:

```text
src/conversaition/static/viewer.html
```

Puedes abrirlo directamente con:

```bash
xdg-open src/conversaition/static/viewer.html
```

También puedes abrir el archivo manualmente desde tu navegador.

El visor no necesita servidor web.

---

# Cargar una transcripción

Una vez abierto el visor:

1. Pulsa **Cargar JSON**.
2. Selecciona el archivo generado por ConversAItion.

Por ejemplo:

```text
recording.json
```

También puedes arrastrar el archivo JSON directamente sobre el visor.

La conversación aparecerá representada en forma de chat.

---

# Cambiar los nombres de los participantes

Inicialmente, el modelo identifica a los hablantes con nombres técnicos:

```text
speaker_0
speaker_1
speaker_2
...
```

El visor detecta automáticamente todos los participantes presentes en la conversación.

En la parte superior aparecerán campos como:

```text
speaker_0    [ speaker_0 ]
speaker_1    [ speaker_1 ]
```

Puedes sustituirlos por nombres reales:

```text
speaker_0    [ Pedro ]
speaker_1    [ María ]
```

La conversación se actualizará inmediatamente para mostrar:

```text
Pedro
Hola, buenos días.

                    María
                    Buenos días, ¿qué tal?

Pedro
Quería comentarte una cosa.
```

Estos cambios son únicamente de visualización.

El JSON original no se modifica.

---

# Privacidad

ConversAItion está diseñado para poder ejecutarse localmente.

Los modelos se cargan desde archivos:

```text
models/*.nemo
```

y no es necesario utilizar servicios externos durante la inferencia.

Para forzar el modo offline de Hugging Face puedes utilizar:

```bash
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_OFFLINE=1
```

Después puedes ejecutar normalmente:

```bash
uv run conversaition \
    -i recording.wav \
    -o recording.json
```

El visor HTML también funciona completamente en local.

La carga del JSON se realiza utilizando las APIs locales del navegador:

```text
JSON local
    │
    ▼
File API
    │
    ▼
JSON.parse()
    │
    ▼
Visor
```

No necesita backend ni servidor.

# Uso rápido

El flujo habitual completo es:

## 1. Convertir el audio

```bash
ffmpeg \
    -i recording.mp3 \
    -ac 1 \
    -ar 16000 \
    -c:a pcm_s16le \
    recording.wav
```

## 2. Transcribir

```bash
uv run conversaition \
    -i recording.wav \
    -o recording.json
```

## 3. Abrir el visor

```bash
xdg-open src/conversaition/static/viewer.html
```

## 4. Cargar el JSON

Selecciona:

```text
record
