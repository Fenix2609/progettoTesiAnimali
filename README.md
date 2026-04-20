# Animal Detector Live

Riconoscimento in tempo reale di animali (custom + generici COCO) e persone,
con supporto per file video, webcam e telecamere IP via RTSP.

## Struttura del progetto

```
animal_detector_live/
├── main.py              # Entry point (argparse)
├── config.py            # Costanti, soglie, colori, percorsi
├── detector.py          # DualDetector (custom + base YOLO)
├── video_capture.py     # Cattura video threaded (anti-lag)
├── recorder.py          # Registrazione output con ffmpeg
├── requirements.txt     # Dipendenze Python
├── models/              # ← METTI QUI I TUOI MODELLI
│   ├── best.pt          #    modello fine-tuned
│   └── data.yaml        #    classi del dataset
└── output/              # ← video annotati salvati qui
```
 
## Setup

### 1. Installa le dipendenze
  
```bash
pip install -r requirements.txt
```

### 2. Copia i modelli

Dalla tua cartella Colab (`runs/detect/tesi_yolov8n/weights/`):
- copia `best.pt` in `models/best.pt`
- copia `data.yaml` (quello del dataset) in `models/data.yaml`

### 3. Verifica ffmpeg

```bash
ffmpeg -version
```

Se non è installato:
- **Ubuntu/Debian**: `sudo apt install ffmpeg`
- **macOS**: `brew install ffmpeg`
- **Windows**: scarica da https://ffmpeg.org e aggiungi al PATH

## Uso

### File video (.mp4)

```bash
python main.py --source video_test.mp4
```

### Webcam

```bash
python main.py --source 0
```

### Telecamera IP (RTSP)

```bash
python main.py --source "rtsp://admin:password@192.168.1.10:554/stream1"
```

### Opzioni utili

| Flag | Descrizione |
|------|-------------|
| `--source`, `-s` | Sorgente video (obbligatoria) |
| `--model`, `-m` | Percorso al best.pt custom |
| `--output`, `-o` | Percorso file di output |
| `--record`, `-r` | In live, registra il video annotato |
| `--no-display` | Non mostrare la finestra video |
| `--headless` | Server senza GUI (log su console) |
| `--device`, `-d` | Forza `cuda` o `cpu` |
| `--skip N` | Processa 1 frame ogni N+1 (velocizza su CPU) |

### Esempi combinati

```bash
# Webcam + registra output
python main.py -s 0 --record

# File video, solo processing senza finestra
python main.py -s video.mp4 --no-display

# RTSP su server headless con skip frame
python main.py -s "rtsp://..." --headless --skip 2

# Usa GPU esplicitamente
python main.py -s 0 --device cuda

# Processa veloce su CPU (1 frame ogni 3)
python main.py -s video.mp4 --skip 2 --device cpu
```

## Colori dei bounding box

- **Verde**: animale del tuo dataset (bat, belka, monty…)
- **Rosso**: persona
- **Arancione**: animale generico COCO (cane, gatto, pecora…)

## Comandi in finestra

- **Q** o **ESC**: esci dal programma

## Note tecniche

- Il thread di cattura video gira separatamente dall'inference,
  così la camera non accumula ritardo.
- Per stream RTSP, la riconnessione è automatica se la connessione cade.
- Il modello base (yolov8n.pt) viene scaricato automaticamente da ultralytics
  al primo avvio se non è presente.
- Su CPU, i modelli nano (yolov8n) girano a circa 10-15 FPS.
  Con `--skip 1` raddoppi la velocità effettiva.
