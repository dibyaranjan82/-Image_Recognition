"""
VisionCore - Flask API Server
Windows + Linux compatible

HOW TO RUN:
    1. pip install flask flask-cors Pillow numpy
    2. python api_server.py
    3. Open browser: http://localhost:5000
"""

import os, sys, io, tempfile
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).parent))

FRONTEND = Path(__file__).parent / 'frontend'
app = Flask(__name__, static_folder=str(FRONTEND), static_url_path='')
CORS(app)

# ── Serve HTML pages ──────────────────────────────────────────
@app.route('/')
def home():        return app.send_static_file('index.html')

@app.route('/detect')
def detect_page(): return app.send_static_file('detect.html')

@app.route('/batch')
def batch_page():  return app.send_static_file('batch.html')

@app.route('/docs')
def docs_page():   return app.send_static_file('docs.html')

# ── Health check ──────────────────────────────────────────────
@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'message': 'VisionCore API running!'})

# ── Helper: save uploaded file to temp (Windows safe) ─────────
def save_temp(file_storage):
    suffix = Path(file_storage.filename).suffix or '.jpg'
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    file_storage.seek(0)
    file_storage.save(path)
    return Path(path)

# ── Classify ──────────────────────────────────────────────────
@app.route('/classify', methods=['POST'])
def classify():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    try:
        backend  = request.form.get('backend', 'tensorflow')
        top_k    = int(request.form.get('top_k', 5))
        fname    = request.files['image'].filename
        tmp      = save_temp(request.files['image'])
        from src.image_recognizer import ImageRecognizer
        r        = ImageRecognizer(backend=backend, top_k=top_k)
        preds    = r.predict(tmp)
        tmp.unlink(missing_ok=True)
        return jsonify({'source': fname, 'predictions': preds, 'backend': backend})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Detect ────────────────────────────────────────────────────
@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    try:
        backend = request.form.get('backend', 'yolo')
        conf    = float(request.form.get('confidence', 0.5))
        fname   = request.files['image'].filename
        tmp     = save_temp(request.files['image'])
        from src.object_detector import ObjectDetector
        d       = ObjectDetector(backend=backend, confidence_threshold=conf)
        dets    = d.detect(tmp)
        tmp.unlink(missing_ok=True)
        return jsonify({'source': fname, 'detections': dets, 'backend': backend})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Batch ─────────────────────────────────────────────────────
@app.route('/batch', methods=['POST'])
def batch():
    files = request.files.getlist('images')
    if not files:
        return jsonify({'error': 'No images uploaded'}), 400
    try:
        backend = request.form.get('backend', 'tensorflow')
        top_k   = int(request.form.get('top_k', 3))
        from src.image_recognizer import ImageRecognizer
        r       = ImageRecognizer(backend=backend, top_k=top_k)
        results = []
        for f in files:
            tmp = save_temp(f)
            try:
                preds = r.predict(tmp)
                results.append({'source': f.filename, 'predictions': preds, 'error': None})
            except Exception as e:
                results.append({'source': f.filename, 'predictions': [], 'error': str(e)})
            finally:
                tmp.unlink(missing_ok=True)
        return jsonify({'results': results, 'total': len(results)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Run ───────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n{'='*50}")
    print(f"  VisionCore is RUNNING!")
    print(f"  Open browser: http://localhost:{port}")
    print(f"{'='*50}\n")
    app.run(host='0.0.0.0', port=port, debug=True)
