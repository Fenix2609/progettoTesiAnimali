# ── video_capture.py ──────────────────────────────────────────
# Cattura video in thread separato.
#
# Tre modalità a seconda della sorgente:
#   FILE  → ThreadedCapture con coda: ogni frame in ordine, nessun
#           frame perso.
#   LIVE (webcam/RTSP) → ThreadedCapture con buffer singolo: il main
#           legge sempre l'ultimo frame.
#   HTTP (Render/remoto) → HttpPollingCapture: polling periodico di
#           un endpoint che restituisce JPEG.
# ──────────────────────────────────────────────────────────────

import cv2
import numpy as np
import threading
import queue
import time

try:
    import urllib.request
    import urllib.error
    _HAS_URLLIB = True
except ImportError:
    _HAS_URLLIB = False


# ══════════════════════════════════════════════════════════════
# ThreadedCapture — per file, webcam, RTSP
# ══════════════════════════════════════════════════════════════

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
        self.source = source
        self.reconnect = reconnect
        self.reconnect_delay = reconnect_delay
        self._queue_size = queue_size

        self._cap = None
        self._running = False
        self._thread = None
        self._finished = False

        self._queue = None
        self._frame = None
        self._lock = threading.Lock()

        self.width = 0
        self.height = 0
        self.fps = 0
        self.total_frames = 0

    def start(self):
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
        if self._queue is not None:
            try:
                return self._queue.get(timeout=0.1)
            except queue.Empty:
                return None
        else:
            with self._lock:
                return self._frame.copy() if self._frame is not None else None

    def is_running(self):
        if self._queue is not None:
            return self._running or not self._queue.empty()
        return self._running

    def is_file(self):
        return self.total_frames > 0

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._cap is not None:
            self._cap.release()
        print("[Capture] Sorgente chiusa.")

    def _capture_loop(self):
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
                while self._running:
                    try:
                        self._queue.put(frame, timeout=0.5)
                        break
                    except queue.Full:
                        continue
            else:
                with self._lock:
                    self._frame = frame

        self._running = False

    def _try_reconnect(self):
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


# ══════════════════════════════════════════════════════════════
# HttpPollingCapture — per sorgenti HTTP (es. Render server)
# ══════════════════════════════════════════════════════════════

class HttpPollingCapture:
    """
    Cattura frame da un endpoint HTTP che restituisce immagini JPEG.

    Il thread di polling scarica periodicamente l'immagine dall'URL
    e la decodifica in un frame BGR numpy. Il main legge sempre
    l'ultimo frame disponibile.

    Pensato per il pattern:
      Telefono → POST /upload (JPEG) → Server Render
      PC       → GET  /frame         → Server Render (questa classe)

    Uso:
        cap = HttpPollingCapture("https://mioserver.onrender.com/frame")
        cap.start()
        while cap.is_running():
            frame = cap.read()
            ...
        cap.stop()
    """

    """def __init__(self, url, poll_interval=0.15, timeout=5.0,
                 reconnect_delay=3.0):"""
    def __init__(self, url, poll_interval=0.08, timeout=5.0,
                 reconnect_delay=3.0):
        """
        Args:
            url:            URL completo dell'endpoint (es. https://.../frame)
            poll_interval:  secondi tra una richiesta e l'altra
            timeout:        timeout HTTP per singola richiesta
            reconnect_delay: pausa dopo errori consecutivi
        """
        self.source = url
        self.poll_interval = poll_interval
        self._timeout = timeout
        self.reconnect_delay = reconnect_delay

        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

        # Proprietà — saranno impostate dopo il primo frame ricevuto
        self.width = 0
        self.height = 0
        self.fps = 0        # stimato dal poll_interval
        self.total_frames = 0  # sempre 0 (è live)

        self._frames_ricevuti = 0
        self._frames_vuoti = 0  # frame identici (nessun nuovo frame dal telefono)
        self._ultimo_hash = None

    def start(self):
        """Testa la connessione e avvia il thread di polling."""
        print(f"[HTTP] Connessione a {self.source}...")

        # Primo tentativo per verificare che l'endpoint risponda
        frame = self._scarica_frame()

        if frame is None:
            # Potrebbe essere che il telefono non ha ancora mandato frame.
            # Non è un errore fatale: il thread aspetterà.
            print(f"[HTTP] Nessun frame disponibile ancora (il telefono "
                  f"sta inviando?)")
            print(f"[HTTP] Attendo frame dal server...")
        else:
            self.height, self.width = frame.shape[:2]
            with self._lock:
                self._frame = frame
            print(f"[HTTP] Primo frame ricevuto: {self.width}x{self.height}")

        self.fps = 1.0 / self.poll_interval if self.poll_interval > 0 else 10.0

        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

        print(f"[HTTP] Polling attivo ogni {self.poll_interval*1000:.0f}ms "
              f"(~{self.fps:.0f} FPS)")

        return self

    def read(self):
        """Restituisce l'ultimo frame ricevuto (o None)."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def is_running(self):
        return self._running

    def is_file(self):
        return False  # sempre live

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        print(f"[HTTP] Polling fermato. Frame ricevuti: {self._frames_ricevuti}")

    def _poll_loop(self):
        """Loop di polling: scarica frame periodicamente."""
        consecutive_errors = 0
        MAX_ERRORS = 10

        while self._running:
            t_start = time.time()

            frame = self._scarica_frame()

            if frame is not None:
                consecutive_errors = 0
                self._frames_ricevuti += 1

                # Aggiorna dimensioni se è il primo frame
                if self.width == 0:
                    self.height, self.width = frame.shape[:2]
                    print(f"[HTTP] Primo frame ricevuto: "
                          f"{self.width}x{self.height}")

                with self._lock:
                    self._frame = frame
            else:
                consecutive_errors += 1
                if consecutive_errors >= MAX_ERRORS:
                    print(f"[HTTP] {consecutive_errors} errori consecutivi, "
                          f"pausa {self.reconnect_delay}s...")
                    time.sleep(self.reconnect_delay)
                    consecutive_errors = 0

            # Aspetta il tempo rimanente per rispettare poll_interval
            elapsed = time.time() - t_start
            sleep_time = self.poll_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        self._running = False

    def _scarica_frame(self):
        """Scarica un singolo frame JPEG dall'endpoint e lo decodifica."""
        try:
            req = urllib.request.Request(self.source)
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                if resp.status != 200:
                    return None
                data = resp.read()
                if not data:
                    return None

            # Decodifica JPEG → numpy BGR
            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return frame

        except urllib.error.HTTPError as e:
            if e.code == 404:
                # Nessun frame disponibile (telefono non ha ancora mandato)
                pass
            else:
                print(f"[HTTP] Errore HTTP {e.code}: {e.reason}")
            return None
        except urllib.error.URLError as e:
            print(f"[HTTP] Errore connessione: {e.reason}")
            return None
        except Exception as e:
            print(f"[HTTP] Errore: {e}")
            return None
