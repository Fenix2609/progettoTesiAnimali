#!/usr/bin/env python3
# ── main.py ───────────────────────────────────────────────────
# Entry point del progetto Animal Detector Live.
#
# Uso:
#   python main.py --source video.mp4              # file MP4
#   python main.py --source 0                      # webcam
#   python main.py --source rtsp://user:pass@IP    # telecamera IP
#   python main.py --source 0 --record             # live + registra
#   python main.py --source video.mp4 --no-display # solo output file
#   python main.py --source 0 --headless           # server senza GUI
# ──────────────────────────────────────────────────────────────

import argparse
import queue
import sys
import threading
import time
import cv2

from config import (
    FINESTRA_NOME, MAX_DISPLAY_WIDTH, CUSTOM_MODEL_PATH,
    BUFFER_RAM_FRACTION, BUFFER_PREFILL_SECONDS,
)
from detector import DualDetector
from video_capture import ThreadedCapture, HttpPollingCapture
from recorder import VideoRecorder, make_output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Animal Detector Live — riconoscimento animali in tempo reale",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python main.py --source video_test.mp4
  python main.py --source 0
  python main.py --source rtsp://admin:1234@192.168.1.10:554/stream1
  python main.py --source 0 --record
  python main.py --source video.mp4 --output mio_output.mp4
        """,
    )

    parser.add_argument(
        "--source", "-s", required=True,
        help="Sorgente video: percorso file .mp4, indice webcam (0,1,2..), "
             "o URL RTSP (rtsp://...)"
    )
    parser.add_argument(
        "--model", "-m", default=None,
        help=f"Percorso al modello custom (default: {CUSTOM_MODEL_PATH})"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Percorso file di output (default: auto-generato in output/)"
    )
    parser.add_argument(
        "--record", "-r", action="store_true",
        help="In modalità live, registra anche il video annotato"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Non mostrare la finestra video (utile per processing batch)"
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Modalità server senza GUI (implica --no-display, output su console)"
    )
    parser.add_argument(
        "--device", "-d", default=None, choices=["cuda", "cpu"],
        help="Forza l'uso di GPU (cuda) o CPU"
    )
    parser.add_argument(
        "--skip", type=int, default=0,
        help="Processa 1 frame ogni N (0 = tutti). Utile su CPU lento."
    )
    parser.add_argument(
        "--base-model", "-b", default=None,
        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        help="Modello base COCO (default: yolov8n.pt). "
             "n=nano(veloce), s=small, m=medium, l=large, x=extra(preciso)"
    )

    return parser.parse_args()


def parse_source(source_str):
    """Converte la stringa --source nel tipo giusto per cv2.VideoCapture."""
    # Indice webcam (numero intero)
    try:
        return int(source_str)
    except ValueError:
        pass

    # File o URL RTSP
    return source_str


def ridimensiona_per_display(frame, max_width):
    """Ridimensiona il frame per la finestra, mantenendo le proporzioni."""
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h))


def _stampa_eventi(detector):
    """Stampa le notifiche di ingresso/uscita dal detector."""
    for ev in detector.eventi:
        if ev["tipo"] == "ingresso":
            print(f"\n  >>> INGRESSO: {ev['nome']} è entrato nel campo visivo")
        elif ev["tipo"] == "uscita":
            print(f"\n  <<< USCITA:   {ev['nome']} è uscito dal campo visivo")


def _calcola_buffer_frames(frame_w, frame_h, fps):
    """
    Calcola quanti frame annotati possiamo tenere in RAM.

    Usa BUFFER_RAM_FRACTION della RAM libera al momento dell'avvio.
    Imposta sempre un minimo (fps * BUFFER_PREFILL_SECONDS * 2) e un
    massimo ragionevole di 1200 frame (~40s a 30fps) per non esagerare.
    """
    try:
        import psutil
        ram_libera = psutil.virtual_memory().available
    except ImportError:
        # psutil non installato: fallback conservativo a 150 frame
        print("[Buffer] psutil non trovato, uso buffer fisso di 150 frame.")
        return 150

    bytes_per_frame = frame_w * frame_h * 3  # BGR
    budget_bytes = ram_libera * BUFFER_RAM_FRACTION
    max_frames = int(budget_bytes / bytes_per_frame)

    # Minimo: almeno il doppio del prefill
    min_frames = max(int(fps * BUFFER_PREFILL_SECONDS * 2), 30)
    max_frames = max(max_frames, min_frames)

    # Cap a 1200 frame per non esagerare
    max_frames = min(max_frames, 1200)

    ram_usata_mb = (max_frames * bytes_per_frame) / 1024 / 1024
    ram_libera_mb = ram_libera / 1024 / 1024
    print(f"[Buffer] RAM libera: {ram_libera_mb:.0f} MB  →  "
          f"buffer: {max_frames} frame ({ram_usata_mb:.0f} MB, "
          f"{max_frames/fps:.1f}s a {fps:.1f}fps)")

    return max_frames


def run_file_mode(detector, capture, output_path, show_display, frame_skip):
    """
    Modalità FILE con buffer di display.

    - Thread inference: elabora frame e li mette in display_queue +
      li scrive su disco tramite recorder.
    - Thread principale (display): legge da display_queue e mostra
      i frame a schermo al ritmo originale del video.

    Il buffer (display_queue) permette all'inference di "correre avanti"
    rispetto al display: la finestra è sempre fluida anche se l'inference
    è variabile. Prima di iniziare il display si aspetta BUFFER_PREFILL_SECONDS
    secondi di frame pronti nel buffer.
    """
    recorder = VideoRecorder(
        output_path, capture.width, capture.height, capture.fps
    )
    recorder.start()

    total = capture.total_frames
    fps = capture.fps if capture.fps > 0 else 30.0
    frame_interval = 1.0 / fps  # secondi per frame per il display

    # ── Calcola dimensione buffer ─────────────────────────────
    buf_size = _calcola_buffer_frames(capture.width, capture.height, fps)
    prefill_frames = max(1, int(fps * BUFFER_PREFILL_SECONDS))

    # Sentinel: None nella queue segnala la fine al thread display
    display_queue = queue.Queue(maxsize=buf_size) if show_display else None

    # Flag condiviso per stop anticipato (utente preme Q)
    stop_event = threading.Event()

    print(f"\n{'='*50}")
    print(f"  Modalità FILE — {total} frame da processare")
    if show_display:
        print(f"  Pre-fill: {prefill_frames} frame ({BUFFER_PREFILL_SECONDS}s) "
              f"poi display a {fps:.1f} FPS")
    print(f"{'='*50}\n")

    t_start = time.time()
    processed = 0

    # ── Thread inference ──────────────────────────────────────
    def inference_loop():
        nonlocal processed
        frame_count = 0

        while capture.is_running() and not stop_event.is_set():
            frame = capture.read()
            if frame is None:
                if not capture.is_running():
                    break
                time.sleep(0.001)
                continue

            frame_count += 1
            processed = frame_count

            # Frame skipping
            if frame_skip > 0 and (frame_count % (frame_skip + 1)) != 1:
                recorder.write(frame)
                if display_queue is not None:
                    # Metti comunque il frame grezzo nel buffer display
                    # così il video non salta visivamente
                    try:
                        display_queue.put(frame, timeout=1.0)
                    except queue.Full:
                        pass
                continue

            frame_annotato, detections = detector.process_frame(frame)
            recorder.write(frame_annotato)
            _stampa_eventi(detector)

            # Barra di progresso
            if total > 0:
                pct = frame_count / total * 100
                bar_len = 30
                filled = int(bar_len * frame_count / total)
                bar = "█" * filled + "─" * (bar_len - filled)
                elapsed = time.time() - t_start
                fps_actual = frame_count / elapsed if elapsed > 0 else 0
                eta = (total - frame_count) / fps_actual if fps_actual > 0 else 0
                print(
                    f"\r  [{bar}] {pct:5.1f}%  "
                    f"{frame_count}/{total}  "
                    f"{fps_actual:.1f} FPS inf.  "
                    f"buf {display_queue.qsize() if display_queue else 0}/{buf_size}  "
                    f"ETA {eta:.0f}s  ",
                    end="", flush=True,
                )

            if display_queue is not None:
                # Blocca se il buffer è pieno (inference troppo veloce):
                # meglio rallentare l'inference che perdere frame o esplodere RAM.
                while not stop_event.is_set():
                    try:
                        display_queue.put(frame_annotato, timeout=0.2)
                        break
                    except queue.Full:
                        continue

        # Segnala fine al thread display
        if display_queue is not None:
            try:
                display_queue.put(None, timeout=5.0)
            except queue.Full:
                pass

    inf_thread = threading.Thread(target=inference_loop, daemon=True)
    inf_thread.start()

    # ── Display loop (thread principale) ─────────────────────
    if show_display:
        # Aspetta il pre-fill
        print(f"  Attendo pre-fill buffer ({prefill_frames} frame)...",
              end="", flush=True)
        while display_queue.qsize() < prefill_frames and inf_thread.is_alive():
            time.sleep(0.05)
        print(f" pronto! ({display_queue.qsize()} frame in buffer)\n")

        t_next = time.time()

        while not stop_event.is_set():
            try:
                frame_out = display_queue.get(timeout=1.0)
            except queue.Empty:
                # Buffer esaurito: aspetta l'inference
                if not inf_thread.is_alive():
                    break
                continue

            if frame_out is None:
                # Sentinel: fine video
                break

            vis = ridimensiona_per_display(frame_out, MAX_DISPLAY_WIDTH)
            cv2.imshow(FINESTRA_NOME, vis)

            # Pacing: aspetta il tempo giusto per il prossimo frame
            t_next += frame_interval
            now = time.time()
            wait_s = t_next - now
            if wait_s < 0:
                # Siamo in ritardo (buffer quasi vuoto): non accumulare
                t_next = now
                wait_ms = 1
            else:
                wait_ms = max(1, int(wait_s * 1000))

            if cv2.waitKey(wait_ms) & 0xFF in (ord("q"), 27):
                print("\n\n  Interrotto dall'utente.")
                stop_event.set()
                break

        cv2.destroyAllWindows()

    # Aspetta che l'inference finisca
    inf_thread.join()
    print()  # newline dopo la barra di progresso
    recorder.stop()
    capture.stop()


def run_live_mode(detector, capture, recorder, show_display, headless, frame_skip):
    """
    Modalità LIVE con buffer di display.

    - Thread inference: cattura frame dalla sorgente, li processa con
      il detector, e li mette in un buffer (display_queue).
    - Thread principale (display): legge da display_queue e mostra
      i frame a schermo.

    Il buffer è PICCOLO (pochi frame) perché in live vogliamo stare
    il più vicini possibile al tempo reale. Quando il buffer è pieno,
    il thread di inference SCARTA il frame più vecchio dal buffer per
    fare spazio al nuovo, così il display non resta mai indietro.
    """
    if recorder:
        recorder.start()

    # Buffer piccolo per il live: 5 frame sono sufficienti per
    # assorbire le variazioni di tempo di inference senza
    # accumulare ritardo percepibile.
    LIVE_BUFFER_SIZE = 5

    display_queue = queue.Queue(maxsize=LIVE_BUFFER_SIZE) if show_display else None
    stop_event = threading.Event()

    # Contatori condivisi per FPS e headless logging
    stats = {
        "frame_count": 0,
        "fps_inference": 0.0,
        "last_detections": [],
    }

    print(f"\n{'='*50}")
    print(f"  Modalità LIVE — premi 'q' o ESC per uscire")
    if recorder:
        print(f"  Registrazione attiva")
    if show_display:
        print(f"  Buffer display: {LIVE_BUFFER_SIZE} frame")
    print(f"{'='*50}\n")

    # ── Thread inference ──────────────────────────────────────
    def inference_loop():
        frame_count = 0
        t_fps = time.time()
        frames_in_interval = 0
        fps_update_interval = 0.5
        last_frame = None

        while capture.is_running() and not stop_event.is_set():
            frame = capture.read()
            if frame is None:
                time.sleep(0.01)
                continue

            # Evita di riprocessare lo stesso frame
            if frame is last_frame:
                time.sleep(0.001)
                continue
            last_frame = frame

            frame_count += 1

            # Frame skipping
            if frame_skip > 0 and (frame_count % (frame_skip + 1)) != 1:
                if recorder:
                    recorder.write(frame)
                continue

            frame_annotato, detections = detector.process_frame(frame)
            _stampa_eventi(detector)

            # Calcola FPS inference
            frames_in_interval += 1
            now = time.time()
            if now - t_fps >= fps_update_interval:
                stats["fps_inference"] = frames_in_interval / (now - t_fps)
                frames_in_interval = 0
                t_fps = now

            stats["frame_count"] = frame_count
            stats["last_detections"] = detections

            # Overlay FPS sul frame
            cv2.putText(
                frame_annotato,
                f"FPS: {stats['fps_inference']:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
            )

            if recorder:
                recorder.write(frame_annotato)

            if display_queue is not None:
                # Se il buffer è pieno, SCARTA il frame più vecchio
                # per fare spazio (non bloccare mai in live).
                if display_queue.full():
                    try:
                        display_queue.get_nowait()
                    except queue.Empty:
                        pass
                try:
                    display_queue.put_nowait(frame_annotato)
                except queue.Full:
                    pass

            # Headless: stampa periodica
            if headless and frame_count % 30 == 0:
                n_det = len(detections)
                labels = ", ".join(d["label"] for d in detections[:5])
                print(
                    f"  Frame {frame_count} | "
                    f"{stats['fps_inference']:.1f} FPS | "
                    f"{n_det} detection{'s' if n_det != 1 else ''}"
                    f"{' | ' + labels if labels else ''}",
                )

        # Segnala fine al display
        if display_queue is not None:
            try:
                display_queue.put(None, timeout=2.0)
            except queue.Full:
                pass

    inf_thread = threading.Thread(target=inference_loop, daemon=True)
    inf_thread.start()

    # ── Display loop (thread principale) ─────────────────────
    if show_display:
        while not stop_event.is_set():
            try:
                frame_out = display_queue.get(timeout=1.0)
            except queue.Empty:
                if not inf_thread.is_alive():
                    break
                continue

            if frame_out is None:
                break

            vis = ridimensiona_per_display(frame_out, MAX_DISPLAY_WIDTH)
            cv2.imshow(FINESTRA_NOME, vis)

            # waitKey(1) = mostra subito il prossimo frame disponibile
            # (nessun pacing artificiale in live: vogliamo tempo reale)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                print("  Uscita dall'utente.")
                stop_event.set()
                break

        cv2.destroyAllWindows()

    elif headless:
        # Niente display: aspetta che il thread inference finisca
        # (o che l'utente faccia Ctrl+C)
        try:
            inf_thread.join()
        except KeyboardInterrupt:
            print("\n  Interrotto (Ctrl+C).")
            stop_event.set()

    # Aspetta fine inference
    stop_event.set()
    inf_thread.join(timeout=5.0)

    if recorder:
        recorder.stop()
    capture.stop()


def _is_http_source(source_str):
    """True se la sorgente è un URL HTTP/HTTPS (es. server Render)."""
    return isinstance(source_str, str) and (
        source_str.startswith("http://") or source_str.startswith("https://")
    )


def main():
    args = parse_args()

    # ── Prepara la sorgente ───────────────────────────────────
    source = parse_source(args.source)

    # ── Carica il detector ────────────────────────────────────
    detector = DualDetector(
        custom_model_path=args.model,
        base_model_path=args.base_model,
        device=args.device,
    )

    # ── Apri la sorgente video ────────────────────────────────
    if _is_http_source(source):
        # Sorgente HTTP: polling da server remoto (es. Render)
        capture = HttpPollingCapture(source)
    else:
        is_file = isinstance(source, str) and not source.startswith("rtsp")
        capture = ThreadedCapture(
            source,
            reconnect=(not is_file),
        )

    try:
        capture.start()
    except ConnectionError as e:
        print(f"\n  Errore: {e}")
        sys.exit(1)

    # ── Decidi la modalità ────────────────────────────────────
    show_display = not args.no_display and not args.headless

    if capture.is_file():
        # ── FILE MODE ─────────────────────────────────────
        output_path = args.output or make_output_path(args.source)
        run_file_mode(detector, capture, output_path, show_display, args.skip)

    else:
        # ── LIVE MODE (webcam, RTSP, HTTP) ────────────────
        recorder = None
        if args.record:
            output_path = args.output or make_output_path(args.source, "live_rec")
            recorder = VideoRecorder(
                output_path, capture.width, capture.height, capture.fps
            )
        run_live_mode(
            detector, capture, recorder,
            show_display, args.headless, args.skip,
        )

    print("\n  Fatto!")


if __name__ == "__main__":
    main()
