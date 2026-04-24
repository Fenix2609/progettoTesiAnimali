#!/usr/bin/env python3
import argparse
import time
import cv2
import numpy as np
import requests

from detector import DualDetector
from recorder import VideoRecorder, make_output_path


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--source", "-s", required=True)
    parser.add_argument("--model", "-m", default=None)
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--record", "-r", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--device", "-d", default=None, choices=["cuda", "cpu"])

    return parser.parse_args()


def parse_source(source_str):
    try:
        return int(source_str)
    except ValueError:
        return source_str


def is_http(source):
    return isinstance(source, str) and source.startswith("http")


def get_http_frame(url):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        img_array = np.frombuffer(r.content, np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except:
        return None


def _stampa_eventi(detector):
    for ev in detector.eventi:
        if ev["tipo"] == "ingresso":
            print(f">>> INGRESSO: {ev['nome']}")
        elif ev["tipo"] == "uscita":
            print(f"<<< USCITA:   {ev['nome']}")


def main():
    args = parse_args()
    source = parse_source(args.source)

    detector = DualDetector(
        custom_model_path=args.model,
        device=args.device,
    )

    show = not args.no_display

    # ── HTTP MODE ─────────────────────────────
    if is_http(source):
        print("Modalità HTTP (Azure / server)")

        recorder = None
        if args.record:
            output = args.output or make_output_path("http_stream")
            recorder = VideoRecorder(output, 640, 480, 10)
            recorder.start()

        while True:
            frame = get_http_frame(source)

            if frame is None:
                time.sleep(0.1)
                continue

            frame_out, detections = detector.process_frame(frame)
            _stampa_eventi(detector)

            if recorder:
                recorder.write(frame_out)

            if show:
                cv2.imshow("Stream", frame_out)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

        if recorder:
            recorder.stop()

    # ── VIDEO / WEBCAM MODE ───────────────────
    else:
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print("Errore apertura sorgente")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25

        recorder = None
        if args.record:
            output = args.output or make_output_path(source)
            recorder = VideoRecorder(output, width, height, fps)
            recorder.start()

        print("Streaming diretto (no buffer)")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_out, detections = detector.process_frame(frame)
            _stampa_eventi(detector)

            if recorder:
                recorder.write(frame_out)

            if show:
                cv2.imshow("Stream", frame_out)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

        cap.release()

        if recorder:
            recorder.stop()

    cv2.destroyAllWindows()
    print("Fine")


if __name__ == "__main__":
    main()