# ── server.py ─────────────────────────────────────────────────
# Server Flask intermedio tra telefono e PC.
#
# Il telefono manda frame JPEG via POST /upload
# Il PC scarica l'ultimo frame via GET /frame
# ──────────────────────────────────────────────────────────────

import os
import threading
from flask import Flask, request, jsonify, send_file, render_template
from io import BytesIO

app = Flask(__name__)

# Ultimo frame ricevuto dal telefono
_ultimo_frame = None
_lock = threading.Lock()

# ── Pagina web per il telefono ────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

# ── Telefono → Server: carica un frame ───────────────────────
@app.route("/upload", methods=["POST"])
def upload():
    global _ultimo_frame
    if "frame" not in request.files:
        return jsonify({"error": "nessun frame"}), 400

    frame_bytes = request.files["frame"].read()
    with _lock:
        _ultimo_frame = frame_bytes

    return jsonify({"ok": True}), 200

# ── PC → Server: scarica l'ultimo frame ──────────────────────
@app.route("/frame", methods=["GET"])
def get_frame():
    with _lock:
        data = _ultimo_frame

    if data is None:
        return jsonify({"error": "nessun frame disponibile"}), 404

    return send_file(BytesIO(data), mimetype="image/jpeg")

# ── Stato server (utile per debug) ───────────────────────────
@app.route("/status", methods=["GET"])
def status():
    with _lock:
        ha_frame = _ultimo_frame is not None
    return jsonify({"online": True, "frame_disponibile": ha_frame})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)