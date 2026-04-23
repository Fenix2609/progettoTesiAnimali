#!/usr/bin/env python3
import argparse
import time
import cv2
import numpy as np
import requests

from detector import DualDetector


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        required=True,
        help="URL frame server (es: https://xxx.azurewebsites.net/frame)"
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Percorso modello custom"
    )

    parser.add_argument(
        "--device",
        default=None,
        choices=["cuda", "cpu"]
    )

    return parser.parse_args()


def get_frame(url):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None

        img_array = np.frombuffer(r.content, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return frame

    except Exception as e:
        print("Errore frame:", e)
        return None


def main():
    args = parse_args()

    detector = DualDetector(
        custom_model_path=args.model,
        device=args.device
    )

    print("\nStreaming avviato... CTRL+C per uscire\n")

    while True:
        frame = get_frame(args.source)

        if frame is None:
            time.sleep(0.2)
            continue

        # inference subito
        frame_out, detections = detector.process_frame(frame)

        # mostra risultato
        cv2.imshow("AI Stream", frame_out)

        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()