# ── detector.py ───────────────────────────────────────────────
# Classe DualDetector: due modelli YOLO (custom + COCO base).
#
# Funzionalità:
#   - Stabilizzazione temporale delle classi (voto a maggioranza)
#   - Unicità identità: due passate per assegnazione corretta
#   - Memoria spaziale: riassociazione soggetti persi
#   - Confidenza adattiva: soggetti confermati non spariscono
#   - Notifiche ingresso/uscita dal campo visivo
# ─────────────────────────────────────────────────────────────

import os
import yaml
from collections import defaultdict, Counter
from ultralytics import YOLO
import cv2

from config import (
    CUSTOM_MODEL_PATH, BASE_MODEL_PATH, YAML_PATH,
    CLASSI_CUSTOM_FALLBACK, CLASSI_COCO,
    CONFIDENZA_MINIMA, SOGLIA_STABILE, FRAMES_STORIA, SOGLIA_IOU,
    COLORE_CUSTOM, COLORE_PERSONA, COLORE_GENERICO,
)

# ── Costanti tracking avanzato ────────────────────────────────
CONFERMA_MIN_VOTI = 5
MEMORIA_FRAME_MAX = 60
MEMORIA_DISTANZA_MAX = 150
CONF_MINIMA_CONFERMATO = 0.30

# Un soggetto è considerato "uscito dal bordo" se il suo ultimo box
# era entro questa distanza (pixel) dal bordo del frame.
MARGINE_BORDO = 40
# Quanti frame consecutivi di assenza prima di dichiarare "uscito"
# (se NON era al bordo, potrebbe essere solo un glitch del tracker)
FRAME_ASSENZA_USCITA = 15


class DualDetector:

    def __init__(self, custom_model_path=None, base_model_path=None, device=None):
        custom_path = custom_model_path or CUSTOM_MODEL_PATH
        base_path = base_model_path or BASE_MODEL_PATH

        if not os.path.exists(custom_path):
            raise FileNotFoundError(
                f"Modello custom non trovato: {custom_path}\n"
                f"Copia il tuo best.pt nella cartella models/"
            )

        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[DualDetector] Device: {device}")
        print(f"[DualDetector] Modello custom: {custom_path}")
        print(f"[DualDetector] Modello base:   {base_path}")

        self.model_custom = YOLO(custom_path)
        self.model_base = YOLO(base_path)
        self.device = device

        self.classi_custom = self._carica_classi_custom()
        print(f"[DualDetector] Classi custom: {self.classi_custom}")

        # ── Stato tracking ────────────────────────────────────
        self.storia_classi = defaultdict(list)
        self.classe_stabile = {}
        self.confermato = {}
        self.ultima_posizione = {}       # tid → (cx, cy, frame_num)
        self.ultimo_box = {}             # tid → (x1, y1, x2, y2)
        self.identita_lock = {}          # nome_classe → tid

        self._frame_num = 0
        self._frame_w = 0
        self._frame_h = 0

        # Memoria soggetti persi
        self.soggetti_persi = []

        # Track ID visti nel frame precedente
        self._tid_precedenti = set()

        # ── Stato notifiche ingresso/uscita ───────────────────
        # soggetti_presenti[nome] = True se attualmente a schermo
        # (un avviso di ingresso è già stato mandato)
        self.soggetti_presenti = {}
        # frame_ultimo_visto[nome] = ultimo frame in cui è stato visto
        self._frame_ultimo_visto = {}
        # Coda eventi: list di dict {"tipo": "ingresso"|"uscita", "nome": str}
        # svuotata da main.py dopo ogni frame
        self.eventi = []

    def _carica_classi_custom(self):
        if os.path.exists(YAML_PATH):
            with open(YAML_PATH, "r") as f:
                data = yaml.safe_load(f)
            return {i: nome for i, nome in enumerate(data["names"])}
        else:
            print(f"[DualDetector] data.yaml non trovato, uso classi di fallback")
            return CLASSI_CUSTOM_FALLBACK

    def reset_tracking(self):
        self.storia_classi.clear()
        self.classe_stabile.clear()
        self.confermato.clear()
        self.ultima_posizione.clear()
        self.ultimo_box.clear()
        self.identita_lock.clear()
        self.soggetti_persi.clear()
        self._tid_precedenti.clear()
        self.soggetti_presenti.clear()
        self._frame_ultimo_visto.clear()
        self.eventi.clear()
        self._frame_num = 0

    def process_frame(self, frame):
        """
        Processa un singolo frame BGR.

        Returns:
            frame_annotato, detections
            (controlla anche self.eventi per notifiche ingresso/uscita)
        """
        self._frame_num += 1
        self._frame_h, self._frame_w = frame.shape[:2]
        self.eventi.clear()

        detections = []
        box_custom_coords = []

        # ══════════════════════════════════════════════════════
        # PASSATA 1: raccogli tutti i track custom del frame,
        # aggiorna storia e stabilizzazione, ma NON assegnare
        # ancora le etichette finali.
        # ══════════════════════════════════════════════════════
        raw_tracks = []   # lista di (tid, cls_id, conf, box, cx, cy)
        tid_correnti = set()

        res_custom = self.model_custom.track(
            frame, persist=True, verbose=False, device=self.device
        )[0]

        if res_custom.boxes.id is not None:
            for box, track_id in zip(res_custom.boxes, res_custom.boxes.id):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                tid = int(track_id)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                soglia = (CONF_MINIMA_CONFERMATO
                          if self.confermato.get(tid)
                          else CONFIDENZA_MINIMA)
                if conf < soglia:
                    continue

                tid_correnti.add(tid)
                self.ultima_posizione[tid] = (cx, cy, self._frame_num)
                self.ultimo_box[tid] = (x1, y1, x2, y2)

                # Stabilizzazione temporale
                if conf >= SOGLIA_STABILE:
                    self.storia_classi[tid].append(cls_id)
                    if len(self.storia_classi[tid]) > FRAMES_STORIA:
                        self.storia_classi[tid].pop(0)

                if self.storia_classi[tid]:
                    voto, conteggio = Counter(
                        self.storia_classi[tid]
                    ).most_common(1)[0]
                    self.classe_stabile[tid] = voto
                    if conteggio >= CONFERMA_MIN_VOTI:
                        self.confermato[tid] = True

                # Recupera da memoria se tid è nuovo
                if tid not in self.classe_stabile:
                    recuperato = self._recupera_da_memoria(cx, cy)
                    if recuperato is not None:
                        nome_rec, cls_rec = recuperato
                        self.classe_stabile[tid] = cls_rec
                        self.confermato[tid] = True
                        self.storia_classi[tid] = [cls_rec] * CONFERMA_MIN_VOTI

                raw_tracks.append((tid, cls_id, conf, (x1, y1, x2, y2), cx, cy))

        # ══════════════════════════════════════════════════════
        # PASSATA 2: assegnazione etichette con unicità.
        # Prima assegna i tid GIA' confermati e che GIA' possiedono
        # un lock, poi i nuovi.
        # ══════════════════════════════════════════════════════

        # Pulisci lock di tid che non sono più presenti
        nomi_da_rilasciare = []
        for nome, proprietario_tid in self.identita_lock.items():
            if proprietario_tid not in tid_correnti:
                nomi_da_rilasciare.append(nome)
        for nome in nomi_da_rilasciare:
            del self.identita_lock[nome]

        # Ordina: prima i tid che hanno già un lock, poi gli altri
        def priorita_tid(entry):
            tid = entry[0]
            # tid con lock esistente → priorità 0 (prima)
            for nome, prop_tid in self.identita_lock.items():
                if prop_tid == tid:
                    return 0
            # tid confermato senza lock → priorità 1
            if self.confermato.get(tid):
                return 1
            # non confermato → priorità 2
            return 2

        raw_tracks_sorted = sorted(raw_tracks, key=priorita_tid)

        # Set di nomi già assegnati in questo frame
        nomi_assegnati_frame = set()

        nomi_visibili_frame = set()  # per notifiche ingresso/uscita

        for (tid, cls_id, conf, box_coords, cx, cy) in raw_tracks_sorted:
            cls_finale = self.classe_stabile.get(tid, cls_id)
            nome_classe = self.classi_custom.get(cls_finale, f"classe_{cls_finale}")

            # ── Unicità: controlla se qualcun altro ha già questo nome ──
            proprietario = self.identita_lock.get(nome_classe)

            if proprietario is not None and proprietario != tid:
                # Qualcun altro possiede il nome E quel qualcuno è
                # nel frame (perché i lock di assenti sono già stati puliti,
                # ma quel tid potrebbe non essere ancora stato processato).
                # Controlla se il proprietario è in tid_correnti:
                if proprietario in tid_correnti:
                    # Il nome è già preso → fallback alla classe grezza
                    nome_classe = self.classi_custom.get(cls_id, f"classe_{cls_id}")
                else:
                    # Proprietario non più presente → prendi il lock
                    self.identita_lock[nome_classe] = tid
            elif nome_classe in nomi_assegnati_frame and proprietario != tid:
                # Un altro tid nello stesso frame ha già preso questo nome
                # (caso in cui due tid non-lockati vogliono lo stesso nome)
                nome_classe = self.classi_custom.get(cls_id, f"classe_{cls_id}")
            else:
                if self.confermato.get(tid):
                    self.identita_lock[nome_classe] = tid

            nomi_assegnati_frame.add(nome_classe)

            x1, y1, x2, y2 = box_coords
            box_custom_coords.append(box_coords)

            # Traccia nomi confermati visibili (per notifiche)
            if self.confermato.get(tid):
                nomi_visibili_frame.add(nome_classe)

            detections.append({
                "box": box_coords,
                "label": f"[{tid}] {nome_classe}",
                "conf": conf,
                "type": "custom",
                "colore": COLORE_CUSTOM,
            })

        # ── Gestione soggetti persi ───────────────────────────
        tid_persi = self._tid_precedenti - tid_correnti
        for tid in tid_persi:
            if tid in self.classe_stabile and self.confermato.get(tid):
                cls_id = self.classe_stabile[tid]
                nome = self.classi_custom.get(cls_id, f"classe_{cls_id}")
                pos = self.ultima_posizione.get(tid)
                if pos is not None:
                    cx, cy, _ = pos
                    self.soggetti_persi.append({
                        "nome": nome,
                        "cls_id": cls_id,
                        "cx": cx, "cy": cy,
                        "frame_perso": self._frame_num,
                        "tid_originale": tid,
                    })

        self.soggetti_persi = [
            s for s in self.soggetti_persi
            if (self._frame_num - s["frame_perso"]) <= MEMORIA_FRAME_MAX
        ]

        self._tid_precedenti = tid_correnti

        # ── Modello BASE ──────────────────────────────────────
        classi_coco_ids = list(CLASSI_COCO.keys())
        res_base = self.model_base.predict(
            frame, verbose=False, classes=classi_coco_ids,
            conf=CONFIDENZA_MINIMA, device=self.device,
        )[0]

        for box in res_base.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if self._sovrapposto(x1, y1, x2, y2, box_custom_coords):
                continue

            nome_classe = CLASSI_COCO.get(cls_id, f"animale_{cls_id}")
            colore = COLORE_PERSONA if cls_id == 0 else COLORE_GENERICO

            nomi_visibili_frame.add(nome_classe)

            detections.append({
                "box": (x1, y1, x2, y2),
                "label": nome_classe,
                "conf": conf,
                "type": "base",
                "colore": colore,
            })

        # ── Notifiche ingresso/uscita ─────────────────────────
        self._aggiorna_notifiche(nomi_visibili_frame)

        # ── Disegna box ───────────────────────────────────────
        frame_annotato = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            colore = det["colore"]
            testo = f"{det['label']} {det['conf']:.0%}"

            cv2.rectangle(frame_annotato, (x1, y1), (x2, y2), colore, 2)
            cv2.putText(
                frame_annotato, testo, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colore, 2,
            )

        return frame_annotato, detections

    # ── Notifiche ingresso/uscita ─────────────────────────────

    def _aggiorna_notifiche(self, nomi_visibili_frame):
        """
        Confronta i nomi visibili ora con quelli del frame precedente.
        Genera eventi "ingresso" e "uscita".

        Regole:
          - INGRESSO: un nome appare e non era presente → evento
          - USCITA: un nome scompare. Se l'ultimo box era vicino al
            bordo del frame → uscita immediata. Altrimenti aspetta
            FRAME_ASSENZA_USCITA frame prima di dichiarare uscita
            (potrebbe essere un glitch temporaneo).
        """
        # Aggiorna ultimo frame visto per ogni nome presente
        for nome in nomi_visibili_frame:
            self._frame_ultimo_visto[nome] = self._frame_num

        # Ingressi: nomi nuovi
        for nome in nomi_visibili_frame:
            if nome not in self.soggetti_presenti:
                self.soggetti_presenti[nome] = True
                self.eventi.append({"tipo": "ingresso", "nome": nome})

        # Uscite: nomi che erano presenti ma non sono più visibili
        nomi_da_rimuovere = []
        for nome in list(self.soggetti_presenti.keys()):
            if nome in nomi_visibili_frame:
                continue  # ancora presente

            ultimo_frame = self._frame_ultimo_visto.get(nome, 0)
            frame_assente = self._frame_num - ultimo_frame

            # Controlla se l'ultimo box era vicino al bordo
            al_bordo = self._era_al_bordo(nome)

            if al_bordo or frame_assente >= FRAME_ASSENZA_USCITA:
                nomi_da_rimuovere.append(nome)
                self.eventi.append({"tipo": "uscita", "nome": nome})

        for nome in nomi_da_rimuovere:
            del self.soggetti_presenti[nome]

    def _era_al_bordo(self, nome):
        """
        Controlla se l'ultimo box noto di un soggetto con questo nome
        era vicino al bordo del frame.
        """
        # Cerca il tid che aveva questo nome
        tid = None
        for n, t in self.identita_lock.items():
            if n == nome:
                tid = t
                break

        # Cerca anche tra i soggetti persi
        if tid is None:
            for s in self.soggetti_persi:
                if s["nome"] == nome:
                    tid = s["tid_originale"]
                    break

        if tid is None or tid not in self.ultimo_box:
            return False

        x1, y1, x2, y2 = self.ultimo_box[tid]
        w, h = self._frame_w, self._frame_h

        if w == 0 or h == 0:
            return False

        return (x1 <= MARGINE_BORDO or
                y1 <= MARGINE_BORDO or
                x2 >= w - MARGINE_BORDO or
                y2 >= h - MARGINE_BORDO)

    # ── Memoria soggetti persi ────────────────────────────────

    def _recupera_da_memoria(self, cx, cy):
        miglior_idx = None
        miglior_dist = float("inf")

        for i, s in enumerate(self.soggetti_persi):
            dist = ((cx - s["cx"]) ** 2 + (cy - s["cy"]) ** 2) ** 0.5
            if dist < MEMORIA_DISTANZA_MAX and dist < miglior_dist:
                miglior_dist = dist
                miglior_idx = i

        if miglior_idx is not None:
            soggetto = self.soggetti_persi.pop(miglior_idx)
            return soggetto["nome"], soggetto["cls_id"]

        return None

    @staticmethod
    def _sovrapposto(x1, y1, x2, y2, lista_box, soglia_iou=None):
        if soglia_iou is None:
            soglia_iou = SOGLIA_IOU

        for bx1, by1, bx2, by2 in lista_box:
            ix1, iy1 = max(x1, bx1), max(y1, by1)
            ix2, iy2 = min(x2, bx2), min(y2, by2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            area1 = (x2 - x1) * (y2 - y1)
            area2 = (bx2 - bx1) * (by2 - by1)
            iou = inter / (area1 + area2 - inter)
            if iou > soglia_iou:
                return True
        return False
