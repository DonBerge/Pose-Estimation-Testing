# Taekwondo Pose Estimation (Real Time)

MVP de estimacion de pose para una sola persona, pensado para movimientos de taekwondo en tiempo real.

## Caracteristicas

- Entrada de video desde webcam local o stream externo (URL).
- Deteccion de pose con MediaPipe.
- Esqueleto simplificado (sin detalle fino de manos, pies y cara).
- Overlay en vivo de esqueleto sobre la grabacion.
- Metricas basicas de rendimiento (FPS y latencia).
- Reconexion basica de stream para mayor estabilidad.

## Requisitos

- Python 3.10+
- Linux (probado para este entorno)

## Instalacion

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecucion

### Webcam local

```bash
python main.py --source 0
```

### Stream externo (ejemplo RTSP)

```bash
python main.py --source rtsp://usuario:password@IP:PUERTO/ruta
```

### Stream externo (ejemplo HTTP/MJPEG)

```bash
python main.py --source http://IP:PUERTO/video
```

## Parametros utiles

- `--width`, `--height`: resolucion de trabajo (default 640x480).
- `--fps`: fps objetivo para captura.
- `--smoothing-alpha`: suavizado temporal de keypoints (0.0-1.0, default 0.85, valores más altos = más estabilidad).
- `--visibility-threshold`: umbral de confianza de keypoint (default 0.5).
- `--keypoint-mode`: cantidad de keypoints a mostrar.
	- `minimal`: solo torso + codos + rodillas.
	- `default`: modo actual del proyecto.
	- `upper`: modo extra recomendado para foco en tren superior.
	- `full`: todos los keypoints disponibles.
- `--skeleton-color`: color del esqueleto. Opciones: `yellow` (default), `red`, `green`, `blue`, `white`, `cyan`, `magenta`, o RGB personalizado `R,G,B` (ej: `255,0,0` para rojo).
- `--record-movements`: guarda el movimiento del esqueleto en un archivo JSONL.
- `--no-perf-overlay`: desactiva texto de FPS/latencia.

**Ejemplos:**

```bash
# Amarillo (por defecto)
python main.py --source 0

# Esqueleto rojo
python main.py --source 0 --skeleton-color red

# Esqueleto verde
python main.py --source 0 --skeleton-color green

# Modo minimal (torso + codos + rodillas)
python main.py --source 0 --keypoint-mode minimal

# Modo full (todos los keypoints)
python main.py --source 0 --keypoint-mode full

# Esqueleto con RGB personalizado (morado: 128,0,128)
python main.py --source 0 --skeleton-color "128,0,128"

# Máxima estabilidad con esqueleto azul
python main.py --source 0 --smoothing-alpha 0.9 --visibility-threshold 0.6 --width 480 --height 360 --skeleton-color blue

# Grabar movimiento de esqueleto
python main.py --source 0 --record-movements recordings/sesion_01.jsonl
```

## Reproducir movimientos grabados

```bash
# Reproducir grabacion
python replay_movements.py --input recordings/sesion_01.jsonl

# Reproducir mas rapido
python replay_movements.py --input recordings/sesion_01.jsonl --speed 1.5

# Forzar modo de keypoints y color en reproduccion
python replay_movements.py --input recordings/sesion_01.jsonl --keypoint-mode full --skeleton-color cyan

# Reproducir en loop
python replay_movements.py --input recordings/sesion_01.jsonl --loop
```

## Controles

- Presiona `q` para salir.

## Keypoints incluidos

- nariz
- hombros (izq/der)
- codos (izq/der)
- munecas (izq/der)
- caderas (izq/der)
- rodillas (izq/der)
- talones (izq/der)
- metatarsos (izq/der)

## Siguiente paso recomendado

Integrar una app de celular que emita RTSP estable en la misma red local, luego ajustar resolucion y suavizado para tu hardware.
