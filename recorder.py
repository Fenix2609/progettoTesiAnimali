# ── recorder.py ───────────────────────────────────────────────
# Registra i frame annotati in un file MP4 tramite ffmpeg pipe.
# Usato sia in modalità file (output obbligatorio) sia in
# modalità live (opzionale, attivato con --record).
# ──────────────────────────────────────────────────────────────

import os
import subprocess

from config import FFMPEG_PRESET, OUTPUT_DIR


class VideoRecorder:
    """
    Scrive frame BGR in un file MP4 usando ffmpeg come processo esterno.
    Più affidabile e compatto di cv2.VideoWriter per la codifica H.264.

    Uso:
        rec = VideoRecorder("output.mp4", width=1920, height=1080, fps=30)
        rec.start()
        rec.write(frame)   # per ogni frame
        rec.stop()
    """

    def __init__(self, output_path, width, height, fps):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self._process = None
        self._frame_count = 0

    def start(self):
        """Avvia il processo ffmpeg."""
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{self.width}x{self.height}",
            "-pix_fmt", "bgr24",
            "-r", str(self.fps),
            "-i", "pipe:0",
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", FFMPEG_PRESET,
            self.output_path,
        ]

        self._process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._frame_count = 0
        print(f"[Recorder] Registrazione avviata: {self.output_path}")
        return self

    def write(self, frame):
        """Scrive un frame nel video."""
        if self._process is None:
            return
        try:
            self._process.stdin.write(frame.tobytes())
            self._frame_count += 1
        except BrokenPipeError:
            print("[Recorder] Pipe rotta, ffmpeg si è chiuso.")
            self._process = None

    def stop(self):
        """Chiude ffmpeg e finalizza il file."""
        if self._process is None:
            return

        self._process.stdin.close()
        self._process.wait()
        self._process = None

        if os.path.exists(self.output_path):
            size_mb = os.path.getsize(self.output_path) / 1024 / 1024
            print(f"[Recorder] Salvato: {self.output_path} "
                  f"({self._frame_count} frame, {size_mb:.1f} MB)")
        else:
            print("[Recorder] Errore: file di output non creato.")

    @property
    def is_recording(self):
        return self._process is not None


def make_output_path(source_name, suffix="annotated"):
    """Genera un percorso di output nella cartella output/."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(str(source_name)))[0]
    return os.path.join(OUTPUT_DIR, f"{base}_{suffix}.mp4")
