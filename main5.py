#!/usr/bin/env python3
# ── main5.py ───────────────────────────────────────────────────
# Entry point del progetto Animal Detector Live.
#
# Uso:
#   python main5.py --source video.mp4              # file MP4
#   python main5.py --source 0                      # webcam
#   python main5.py --source rtsp://user:pass@IP    # telecamera IP
#   python main5.py --source https://url/frame      # remoto via HTTP
#   python main5.py --source 0 --record             # live + registra
#   python main5.py --source video.mp4 --no-display # solo output file
#   python main5.py --source 0 --headless           # server senza GUI
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
  python main5.py --source video_test.mp4
  python main5.py --source 0
  python main5.py --source rtsp://admin:1234@192.168.1.10:554/stream1
  python main5.py --source https://mioserver.onrender.com/frame
  python main5.py --source 0 --record
  python main5.py --source video.mp4 --output mio_output.mp4
        """,
    )

    parser.add_argument(
        "--source", "-s", required=True,
        help="Sorgente video: percorso file .mp4, indice webcam (0,1,2..), "
             "URL RTSP (rtsp://...) o URL HTTP (https://...)"
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
    """Converte la stringa --source nel tipo giusto."""
    try:
        return int(source_str)
    except ValueError:
        pass
    return source_str


def _is_http_source(source):
    """True se la sorgente è un URL HTTP/HTTPS."""
    return isinstance(source, str) and (
        source.startswith("http://") or source.startswith("https://")
    )


def ridimensiona_per_display(frame, max_width):
    """Ridimensiona il frame per la finestra, mantenendo le proporzioni."""
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(frame, (int(w * scale), int(h * scale)))


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
    """
    try:
        import psutil
        ram_libera = psutil.virtual_memory().available
    except ImportError:
        print("[Buffer] psutil non trovato, uso buffer fisso di 150 frame.")
        return 150

    bytes_per_frame = frame_w * frame_h * 3
    budget_bytes = ram_libera * BUFFER_RAM_FRACTION
    max_frames = int(budget_bytes / bytes_per_frame)

    min_frames = max(int(fps * BUFFER_PREFILL_SECONDS * 2), 30)
    max_frames = max(max_frames, min_frames)
    max_frames = min(max_frames, 1200)

    ram_usata_mb = (max_frames * bytes_per_frame) / 1024 / 1024
    ram_libera_mb = ram_libera / 1024 / 1024
    print(f"[Buffer] RAM libera: {ram_libera_mb:.0f} MB  →  "
          f"buffer: {max_frames} frame ({ram_usata_mb:.0f} MB, "
          f"{max_frames/fps:.1f}s a {fps:.1f}fps)")

    return max_frames


def run_file_mode(detector, capture, output_path, show_display, frame_skip):
    """
    Modalità FILE: processa tutto il video e salva l'output annotato.
    """
    recorder = VideoRecorder(
        output_path, capture.width, capture.height, capture.fps
    )
    recorder.start()

    total = capture.total_frames
    processed = 0
    t_start = time.time()

    print(f"\n{'='*50}")
    print(f"  Modalità FILE — {total} frame da processare")
    if show_display:
        print(f"  Display attivo (velocità reale inference)")
    print(f"{'='*50}\n")

    while capture.is_running():
        frame = capture.read()
        if frame is None:
            if not capture.is_running():
                break
            time.sleep(0.001)
            continue

        processed += 1

        if frame_skip > 0 and (processed % (frame_skip + 1)) != 1:
            recorder.write(frame)
            continue

        frame_annotato, detections = detector.process_frame(frame)
        recorder.write(frame_annotato)
        _stampa_eventi(detector)

        if total > 0:
            pct = processed / total * 100
            bar_len = 30
            filled = int(bar_len * processed / total)
            elapsed = time.time() - t_start
            fps_actual = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / fps_actual if fps_actual > 0 else 0
            print(
                f"\r  [{'█'*filled}{'─'*(bar_len-filled)}] {pct:5.1f}%  "
                f"{processed}/{total}  {fps_actual:.1f} FPS  ETA {eta:.0f}s  ",
                end="", flush=True,
            )

        if show_display:
            vis = ridimensiona_per_display(frame_annotato, MAX_DISPLAY_WIDTH)
            cv2.imshow(FINESTRA_NOME, vis)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                print("\n\n  Interrotto dall'utente.")
                break

    print()
    recorder.stop()
    capture.stop()
    if show_display:
        cv2.destroyAllWindows()


def run_live_mode(detector, capture, recorder, show_display, headless,
                  frame_skip, is_http=False):
    """
    Modalità LIVE: webcam, RTSP o HTTP remoto.

    Per sorgenti HTTP usa un buffer con pre-fill per compensare
    la latenza di rete e mostrare video fluido.
    Per webcam/RTSP usa un buffer piccolo per stare in tempo reale.
    """
    if recorder:
        recorder.start()

    # ── Dimensione buffer e pre-fill ──────────────────────────
    if is_http:
        # HTTP: frame arrivano lenti (3-7 fps), buffer grande per
        # accumulare abbastanza frame prima di iniziare il display.
        fps_stimati = capture.fps if capture.fps > 0 else 7.0
        LIVE_BUFFER_SIZE = _calcola_buffer_frames(
            capture.width or 320, capture.height or 240, fps_stimati
        )
        # Pre-fill: accumula BUFFER_PREFILL_SECONDS secondi di frame
        LIVE_PREFILL = max(1, int(fps_stimati * BUFFER_PREFILL_SECONDS))
        frame_interval = capture.poll_interval  # pacing = ritmo arrivo frame
    else:
        # Webcam/RTSP: buffer piccolo, nessun pre-fill
        LIVE_BUFFER_SIZE = 5
        LIVE_PREFILL = 0
        frame_interval = 0

    display_queue = queue.Queue(maxsize=LIVE_BUFFER_SIZE) if show_display else None
    stop_event = threading.Event()

    stats = {"fps_inference": 0.0, "frame_count": 0}

    print(f"\n{'='*50}")
    print(f"  Modalità {'HTTP REMOTO' if is_http else 'LIVE'} "
          f"— premi 'q' o ESC per uscire")
    if recorder:
        print(f"  Registrazione attiva")
    if is_http and show_display:
        print(f"  Pre-fill: {LIVE_PREFILL} frame poi display fluido")
    print(f"{'='*50}\n")

    # ── Thread inference ──────────────────────────────────────
    def inference_loop():
        frame_count = 0
        t_fps = time.time()
        frames_in_interval = 0
        last_frame = None

        while capture.is_running() and not stop_event.is_set():
            frame = capture.read()
            if frame is None:
                time.sleep(0.01)
                continue

            # Evita di riprocessare lo stesso frame (solo webcam/RTSP)
            if not is_http and frame is last_frame:
                time.sleep(0.001)
                continue
            last_frame = frame

            frame_count += 1
            stats["frame_count"] = frame_count

            if frame_skip > 0 and (frame_count % (frame_skip + 1)) != 1:
                if recorder:
                    recorder.write(frame)
                continue

            frame_annotato, detections = detector.process_frame(frame)
            _stampa_eventi(detector)

            # Calcola FPS inference
            frames_in_interval += 1
            now = time.time()
            if now - t_fps >= 0.5:
                stats["fps_inference"] = frames_in_interval / (now - t_fps)
                frames_in_interval = 0
                t_fps = now

            # Overlay FPS
            cv2.putText(
                frame_annotato,
                f"FPS: {stats['fps_inference']:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
            )

            if recorder:
                recorder.write(frame_annotato)

            if display_queue is not None:
                if is_http:
                    # HTTP: non scartare frame, blocca se buffer pieno
                    while not stop_event.is_set():
                        try:
                            display_queue.put(frame_annotato, timeout=0.2)
                            break
                        except queue.Full:
                            continue
                else:
                    # Webcam/RTSP: scarta il vecchio se buffer pieno
                    if display_queue.full():
                        try:
                            display_queue.get_nowait()
                        except queue.Empty:
                            pass
                    try:
                        display_queue.put_nowait(frame_annotato)
                    except queue.Full:
                        pass

            if headless and frame_count % 30 == 0:
                n_det = len(detections)
                labels = ", ".join(d["label"] for d in detections[:5])
                print(
                    f"  Frame {frame_count} | "
                    f"{stats['fps_inference']:.1f} FPS | "
                    f"{n_det} detection{'s' if n_det != 1 else ''}"
                    f"{' | ' + labels if labels else ''}",
                )

        if display_queue is not None:
            try:
                display_queue.put(None, timeout=2.0)
            except queue.Full:
                pass

    inf_thread = threading.Thread(target=inference_loop, daemon=True)
    inf_thread.start()

    # ── Display loop ──────────────────────────────────────────
    if show_display:

        # Pre-fill per HTTP: aspetta abbastanza frame nel buffer
        if LIVE_PREFILL > 0:
            print(f"  Attendo pre-fill buffer ({LIVE_PREFILL} frame)...",
                  end="", flush=True)
            while (display_queue.qsize() < LIVE_PREFILL
                   and inf_thread.is_alive()
                   and not stop_event.is_set()):
                time.sleep(0.05)
            print(f" pronto! ({display_queue.qsize()} frame in buffer)\n")

        t_next = time.time()

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

            # Pacing: per HTTP rispetta ritmo di arrivo, per webcam immediato
            if is_http and frame_interval > 0:
                t_next += frame_interval
                now = time.time()
                wait_s = t_next - now
                if wait_s < 0:
                    t_next = now
                    wait_ms = 1
                else:
                    wait_ms = max(1, int(wait_s * 1000))
            else:
                wait_ms = 1

            if cv2.waitKey(wait_ms) & 0xFF in (ord("q"), 27):
                print("  Uscita dall'utente.")
                stop_event.set()
                break

        cv2.destroyAllWindows()

    elif headless:
        try:
            inf_thread.join()
        except KeyboardInterrupt:
            print("\n  Interrotto (Ctrl+C).")
            stop_event.set()

    stop_event.set()
    inf_thread.join(timeout=5.0)

    if recorder:
        recorder.stop()
    capture.stop()


def main():
    args = parse_args()

    source = parse_source(args.source)
    http_mode = _is_http_source(source)

    # ── Carica il detector ────────────────────────────────────
    detector = DualDetector(
        custom_model_path=args.model,
        base_model_path=args.base_model,
        device=args.device,
    )

    # ── Apri la sorgente video ────────────────────────────────
    if http_mode:
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

    show_display = not args.no_display and not args.headless

    if capture.is_file():
        output_path = args.output or make_output_path(args.source)
        run_file_mode(detector, capture, output_path, show_display, args.skip)
    else:
        recorder = None
        if args.record:
            output_path = args.output or make_output_path(args.source, "live_rec")
            recorder = VideoRecorder(
                output_path, capture.width, capture.height, capture.fps
            )
        run_live_mode(
            detector, capture, recorder,
            show_display, args.headless, args.skip,
            is_http=http_mode,
        )

    print("\n  Fatto!")


if __name__ == "__main__":
    main()
