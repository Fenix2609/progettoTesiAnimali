#!/usr/bin/env python3
# ── main5.py ───────────────────────────────────────────────────
# Entry point del progetto Animal Detector Live.
#
# Uso:
#   python main5.py --source video.mp4              # file MP4
#   python main5.py --source 0                      # webcam
#   python main5.py --source rtsp://user:pass@IP    # telecamera IP
#   python main5.py --source 0 --record             # live + registra
#   python main5.py --source video.mp4 --no-display # solo output file
#   python main5.py --source 0 --headless           # server senza GUI
# ──────────────────────────────────────────────────────────────

import argparse
import sys
import time
import cv2

from config import FINESTRA_NOME, MAX_DISPLAY_WIDTH, CUSTOM_MODEL_PATH
from detector import DualDetector
from video_capture import ThreadedCapture
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
  python main5.py --source 0 --record
  python main5.py --source video.mp4 --output mio_output.mp4
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


def run_file_mode(detector, capture, output_path, show_display, frame_skip):
    """
    Modalità FILE: processa tutto il video e salva l'output annotato.
    Barra di progresso su console. Display opzionale.
    Usa timestamp reali per pacing corretto della finestra video.
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

        # Frame skipping
        if frame_skip > 0 and (processed % (frame_skip + 1)) != 1:
            recorder.write(frame)
            continue

        frame_annotato, detections = detector.process_frame(frame)
        recorder.write(frame_annotato)

        # Stampa eventi ingresso/uscita
        _stampa_eventi(detector)

        # Barra di progresso
        if total > 0:
            pct = processed / total * 100
            bar_len = 30
            filled = int(bar_len * processed / total)
            bar = "█" * filled + "─" * (bar_len - filled)
            elapsed = time.time() - t_start
            fps_actual = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / fps_actual if fps_actual > 0 else 0
            print(
                f"\r  [{bar}] {pct:5.1f}%  "
                f"{processed}/{total}  "
                f"{fps_actual:.1f} FPS  "
                f"ETA {eta:.0f}s  ",
                end="", flush=True,
            )

        # Display: mostra sempre l'ultimo frame processato, senza throttling.
        # Il pacing (fps corretto) vale solo per il file di output scritto
        # da recorder. La finestra live deve andare alla velocità dell'inference.
        if show_display:
            vis = ridimensiona_per_display(frame_annotato, MAX_DISPLAY_WIDTH)
            cv2.imshow(FINESTRA_NOME, vis)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                print("\n\n  Interrotto dall'utente.")
                break

    print()  # newline dopo la barra
    recorder.stop()
    capture.stop()

    if show_display:
        cv2.destroyAllWindows()


def run_live_mode(detector, capture, recorder, show_display, headless, frame_skip):
    """
    Modalità LIVE: processa frame in tempo reale dalla webcam o RTSP.
    Mostra finestra + opzionalmente registra.
    """
    if recorder:
        recorder.start()

    frame_count = 0
    t_start = time.time()
    fps_display = 0.0
    fps_update_interval = 0.5  # aggiorna l'FPS ogni 0.5s
    t_fps = time.time()
    frames_in_interval = 0

    print(f"\n{'='*50}")
    print(f"  Modalità LIVE — premi 'q' o ESC per uscire")
    if recorder:
        print(f"  Registrazione attiva")
    print(f"{'='*50}\n")

    last_frame = None

    while capture.is_running():
        frame = capture.read()
        if frame is None:
            time.sleep(0.01)
            continue

        # Evita di riprocessare lo stesso frame identico
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

        # Stampa eventi ingresso/uscita
        _stampa_eventi(detector)

        # Calcola FPS reale
        frames_in_interval += 1
        now = time.time()
        if now - t_fps >= fps_update_interval:
            fps_display = frames_in_interval / (now - t_fps)
            frames_in_interval = 0
            t_fps = now

        # Overlay FPS sul frame
        cv2.putText(
            frame_annotato,
            f"FPS: {fps_display:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
        )

        if recorder:
            recorder.write(frame_annotato)

        # Display
        if show_display:
            vis = ridimensiona_per_display(frame_annotato, MAX_DISPLAY_WIDTH)
            cv2.imshow(FINESTRA_NOME, vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("  Uscita dall'utente.")
                break

        # Headless: stampa periodica su console
        if headless and frame_count % 30 == 0:
            elapsed = time.time() - t_start
            n_det = len(detections)
            labels = ", ".join(d["label"] for d in detections[:5])
            print(
                f"  Frame {frame_count} | {fps_display:.1f} FPS | "
                f"{n_det} detection{'s' if n_det != 1 else ''}"
                f"{' | ' + labels if labels else ''}",
            )

    if recorder:
        recorder.stop()
    capture.stop()

    if show_display:
        cv2.destroyAllWindows()


def main():
    args = parse_args()

    # ── Prepara la sorgente ───────────────────────────────────
    source = parse_source(args.source)
    is_file = isinstance(source, str) and not source.startswith("rtsp")

    # ── Carica il detector ────────────────────────────────────
    detector = DualDetector(
        custom_model_path=args.model,
        base_model_path=args.base_model,
        device=args.device,
    )

    # ── Apri la sorgente video ────────────────────────────────
    capture = ThreadedCapture(
        source,
        reconnect=(not is_file),  # reconnect solo per stream
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
        # ── LIVE MODE ─────────────────────────────────────
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
