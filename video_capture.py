# ── video_capture.py ──────────────────────────────────────────
# Cattura video in thread separato.
#
# Due modalità a seconda della sorgente:
#   FILE  → coda (queue): ogni frame viene consegnato in ordine,
#           nessun frame perso. Il thread si ferma se la coda è
#           piena, aspettando che il main consumi.
#   LIVE  → buffer singolo: il main legge sempre l'ultimo frame,
#           senza accumulare ritardo.
# ──────────────────────────────────────────────────────────────

import cv2
import threading
import queue
import time


class ThreadedCapture:
    """
    Legge frame dalla sorgente video in un thread dedicato.

    Per FILE: usa una queue con dimensione limitata. read() blocca finché
    un frame non è disponibile. Nessun frame viene perso.

    Per LIVE (webcam/RTSP): usa un buffer singolo. read() restituisce
    sempre l'ultimo frame, saltando quelli intermedi.
    """

    def __init__(self, source, reconnect=True, reconnect_delay=3.0,
                 queue_size=128):
        """
        Args:
            source:          percorso file, indice webcam (int), o URL RTSP
            reconnect:       riconnessione automatica (solo stream)
            reconnect_delay: secondi tra tentativi
            queue_size:      dimensione coda per modalità file
        """
        self.source = source
        self.reconnect = reconnect
        self.reconnect_delay = reconnect_delay
        self._queue_size = queue_size

        self._cap = None
        self._running = False
        self._thread = None
        self._finished = False  # True quando il file è finito

        # Modalità file: coda ordinata
        self._queue = None
        # Modalità live: buffer singolo
        self._frame = None
        self._lock = threading.Lock()

        # Proprietà video (disponibili dopo start)
        self.width = 0
        self.height = 0
        self.fps = 0
        self.total_frames = 0  # 0 per stream live

    def start(self):
        """Apre la sorgente e avvia il thread di cattura."""
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise ConnectionError(
                f"Impossibile aprire la sorgente: {self.source}"
            )

        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.total_frames <= 0:
            self.total_frames = 0

        # Scegli modalità
        if self.is_file():
            self._queue = queue.Queue(maxsize=self._queue_size)
        else:
            self._queue = None

        self._running = True
        self._finished = False
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        print(f"[Capture] Sorgente aperta: {self.source}")
        print(f"[Capture] Risoluzione: {self.width}x{self.height} "
              f"@ {self.fps:.1f} FPS")
        if self.total_frames > 0:
            durata = self.total_frames / self.fps
            print(f"[Capture] Durata: {durata:.1f}s ({self.total_frames} frame)")
            print(f"[Capture] Modalità: coda sequenziale (nessun frame perso)")
        else:
            print(f"[Capture] Modalità: live (ultimo frame)")

        return self

    def read(self):
        """
        Restituisce il prossimo frame.

        FILE:  blocca fino a 0.1s, restituisce None se la coda è vuota
               e il video non è finito. Restituisce None definitivo
               quando il file è terminato e la coda è vuota.
        LIVE:  restituisce l'ultimo frame (o None se non disponibile).
        """
        if self._queue is not None:
            # Modalità FILE: leggi dalla coda in ordine
            try:
                return self._queue.get(timeout=0.1)
            except queue.Empty:
                return None
        else:
            # Modalità LIVE: ultimo frame
            with self._lock:
                return self._frame.copy() if self._frame is not None else None

    def is_running(self):
        """True se ci sono ancora frame da consumare."""
        if self._queue is not None:
            # Per file: running finché la coda non è vuota E il thread non ha finito
            return self._running or not self._queue.empty()
        return self._running

    def is_file(self):
        """True se la sorgente è un file."""
        return self.total_frames > 0

    def stop(self):
        """Ferma il thread e rilascia la sorgente."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._cap is not None:
            self._cap.release()
        print("[Capture] Sorgente chiusa.")

    def _capture_loop(self):
        """Loop interno del thread."""
        consecutive_failures = 0
        MAX_FAILURES = 30

        while self._running:
            if self._cap is None or not self._cap.isOpened():
                if self.reconnect and not self.is_file():
                    self._try_reconnect()
                else:
                    self._running = False
                    break

            ret, frame = self._cap.read()

            if not ret:
                consecutive_failures += 1

                if self.is_file():
                    print("\n[Capture] Fine del video.")
                    self._running = False
                    break

                if consecutive_failures >= MAX_FAILURES:
                    print(f"[Capture] {consecutive_failures} frame falliti, "
                          f"riconnessione...")
                    if self.reconnect:
                        self._try_reconnect()
                        consecutive_failures = 0
                    else:
                        self._running = False
                        break
                continue

            consecutive_failures = 0

            if self._queue is not None:
                # Modalità FILE: metti in coda (blocca se piena)
                while self._running:
                    try:
                        self._queue.put(frame, timeout=0.5)
                        break
                    except queue.Full:
                        continue
            else:
                # Modalità LIVE: sovrascrivi ultimo frame
                with self._lock:
                    self._frame = frame

        self._running = False

    def _try_reconnect(self):
        """Tenta di riconnettersi alla sorgente."""
        print(f"[Capture] Riconnessione a {self.source} "
              f"tra {self.reconnect_delay}s...")
        if self._cap is not None:
            self._cap.release()

        time.sleep(self.reconnect_delay)

        self._cap = cv2.VideoCapture(self.source)
        if self._cap.isOpened():
            print("[Capture] Riconnesso!")
        else:
            print("[Capture] Riconnessione fallita, riprovo...")
