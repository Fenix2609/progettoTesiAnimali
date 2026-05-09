# ── server.py ─────────────────────────────────────────────────
# Server Flask intermedio tra telefono e PC.
#
# Il telefono manda frame JPEG via POST /upload
# Il PC scarica l'ultimo frame via GET /frame
#
# Gestione dispositivo unico:
#   - Solo un dispositivo alla volta può inviare frame.
#   - Se un secondo dispositivo prova ad aprire la pagina, riceve 403.
#   - Se il dispositivo connesso non manda frame per 30 secondi,
#     viene considerato disconnesso e un nuovo dispositivo può connettersi.
# ──────────────────────────────────────────────────────────────

import os
import time
import threading
from flask import Flask, request, jsonify, send_file, render_template
from io import BytesIO

app = Flask(__name__)

# Ultimo frame ricevuto dal telefono
_ultimo_frame = None
_lock = threading.Lock()

# ── Gestione dispositivo unico ────────────────────────────────
# True se c'è già un dispositivo connesso
_dispositivo_connesso = False

# Timestamp dell'ultimo frame ricevuto (usato per il timeout)
_ultimo_frame_time = 0
# Secondi di inattività dopo cui il dispositivo viene considerato disconnesso
TIMEOUT_DISCONNESSIONE = 30
_lock_connessione = threading.Lock()


def _dispositivo_attivo():
    """Restituisce True se c'è un dispositivo connesso e non ha fatto timeout."""
    global _dispositivo_connesso
    with _lock_connessione:
        if not _dispositivo_connesso:
            print("we we")
            return False
        print(_dispositivo_connesso)
        # Controlla il timeout: se non arrivano frame da TIMEOUT secondi,
        # considera il dispositivo disconnesso automaticamente
        if time.time() - _ultimo_frame_time > TIMEOUT_DISCONNESSIONE:
            _dispositivo_connesso = False
            print(f"[Server] Dispositivo disconnesso per timeout "
                  f"({TIMEOUT_DISCONNESSIONE}s senza frame).")
            return False
        return True


# ── Pagina web per il telefono ────────────────────────────────
@app.route("/")
def index():
    # Se c'è già un dispositivo attivo, blocca l'accesso
    if _dispositivo_attivo():
        return (
            "<html><body style='background:#0a0a0a;color:#ff4444;"
            "font-family:monospace;display:flex;align-items:center;"
            "justify-content:center;height:100vh;margin:0;'>"
            "<div style='text-align:center'>"
            "<h2>&#x26A0; Dispositivo già connesso</h2>"
            "<p style='color:#888;margin-top:12px'>"
            "Un altro dispositivo sta già usando la telecamera.<br>"
            "Riprovare più tardi.</p>"
            "</div></body></html>"
        ), 403
    return render_template("index.html")


# ── Telefono → Server: carica un frame ───────────────────────
@app.route("/upload", methods=["POST"])
def upload():
    global _ultimo_frame, _dispositivo_connesso, _ultimo_frame_time

    if "frame" not in request.files:
        return jsonify({"error": "nessun frame"}), 400

    # Registra il dispositivo come connesso e aggiorna il timestamp
    with _lock_connessione:
        _dispositivo_connesso = True
        _ultimo_frame_time = time.time()

    frame_bytes = request.files["frame"].read()
    with _lock:
        _ultimo_frame = frame_bytes

    return jsonify({"ok": True}), 200


# ── Telefono → Server: disconnessione volontaria ──────────────
@app.route("/disconnect", methods=["POST"])
def disconnect():
    global _dispositivo_connesso
    with _lock_connessione:
        _dispositivo_connesso = False
    print("[Server] Dispositivo disconnesso volontariamente.")
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
    attivo = _dispositivo_attivo()
    return jsonify({
        "online": True,
        "frame_disponibile": ha_frame,
        "dispositivo_connesso": attivo,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
