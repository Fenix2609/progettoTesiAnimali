# ── config.py ─────────────────────────────────────────────────
# Configurazione centralizzata del progetto.
# Modifica qui i percorsi, le sog lie e i colori senza toccare la logica.
# ──────────────────────────────────────────────────────────────

import os

# ── Percorsi modelli ─────────────────────────────────────────
# best.pt = modello fine-tuned sui tuoi animali (bat, belka, monty…)
# Metti il file nella cartella models/ accanto a questo file.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CUSTOM_MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
BASE_MODEL_PATH = "yolov8n.pt"  # scaricato automaticamente da ultralytics

# ── Classi del tuo dataset custom ─────────────────────────────
# Mappa indice → nome come nel tuo data.yaml.
# Se preferisci, il programma le legge automaticamente da data.yaml
# (vedi detector.py). Altrimenti impostale manualmente qui:
YAML_PATH = os.path.join(BASE_DIR, "models", "data.yaml")

# Se data.yaml non è disponibile, usa questa mappa di fallback. # questa cosa non mi sembra avere senso -------------------------------
# Aggiorna con le tue classi reali.
CLASSI_CUSTOM_FALLBACK = {
    0: "argo",
    1: "bobby",
    2: "lessi",
    3: "nero",
    4: "rex",
    5: "rocky"
}

# ── Classi COCO di interesse (modello base) ───────────────────
# Solo persone e animali comuni. Indici COCO standard.
CLASSI_COCO = {
    0:  "persona",
    15: "gatto",
    16: "cane",
    18: "pecora"
}

# ── Soglie di rilevamento ─────────────────────────────────────
CONFIDENZA_MINIMA = 0.50   # sotto questa confidenza, il box viene ignorato
SOGLIA_STABILE    = 0.70   # sopra questa, il frame contribuisce alla storia
FRAMES_STORIA     = 10     # quanti frame tenere per la stabilizzazione
SOGLIA_IOU        = 0.30   # IoU minimo per considerare due box sovrapposti

# ── Tracking avanzato ────────────────────────────────────────
# (queste costanti sono usate in detector.py)
# CONFERMA_MIN_VOTI:    quanti voti stabili servono per "confermare" un'identità
# CONF_MINIMA_CONFERMATO: soglia ridotta per soggetti già confermati
# MEMORIA_FRAME_MAX:    quanti frame un soggetto perso resta in memoria
# MEMORIA_DISTANZA_MAX: pixel max (centro-centro) per ri-associare un soggetto
# Se vuoi modificare questi valori, fallo direttamente in detector.py.

# ── Colori BGR per i box ──────────────────────────────────────
COLORE_CUSTOM    = (0, 200, 0)     # verde  → animale del tuo dataset
COLORE_PERSONA   = (0, 0, 255)     # rosso  → persona
COLORE_GENERICO  = (0, 140, 255)   # arancione → animale generico COCO

# ── Output video (per modalità file) ─────────────────────────
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FFMPEG_PRESET = "fast"

# ── Live display ──────────────────────────────────────────────
FINESTRA_NOME = "Animal Detector Live"
MAX_DISPLAY_WIDTH = 1280   # ridimensiona la finestra se il frame è più largo

# ── Display buffer (modalità file) ────────────────────────────
# Frazione della RAM libera da usare per il buffer di frame annotati.
# Es. 0.35 = usa al massimo il 35% della RAM libera al momento dell'avvio.
BUFFER_RAM_FRACTION = 0.35
# Secondi di pre-fill prima di iniziare il display.
# Il display aspetta che il buffer abbia almeno questi secondi di frame pronti.
BUFFER_PREFILL_SECONDS = 3
