# ──────────────────────────────────────────────────────────────────────────────
#  DocSight — DR Detection Flask Backend
#  Uses your exact model.py logic (EfficientNet-B4 + Grad-CAM)
# ──────────────────────────────────────────────────────────────────────────────
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import base64, io, cv2, traceback, os

# ── Import YOUR exact model logic ──
from model import load_model, predict

app = Flask(__name__)

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16 MB

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# ── Load model ONCE at startup ──
print("="*55)
print("  DocSight — Loading DR model, please wait...")
print("="*55)
try:
    model = load_model()
    print("✅ Model ready!\n")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    model = None

def numpy_to_base64(img_rgb):
    """Convert numpy RGB image → base64 string for sending to browser."""
    pil_img = Image.fromarray(img_rgb.astype(np.uint8))
    buffer  = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ── PAGES ──
@app.route('/')
def index():
    """Serve main HTML page"""
    return render_template('dr_detection_gui.html')

@app.route('/info')
def info():
    """Serve information page"""
    return render_template('eye_info.html')

# ── PREDICT ENDPOINT ──
@app.route("/predict", methods=["POST"])
def predict_route():
    """API endpoint for DR prediction"""
    
    if model is None:
        return jsonify({
            "success": False,
            "error": "Model not loaded. Server error."
        }), 500
    
    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "No image uploaded."
        }), 400
    
    file = request.files["image"]
    
    if file.filename == '':
        return jsonify({
            "success": False,
            "error": "No file selected."
        }), 400
    
    try:
        # Read uploaded image → numpy RGB array
        file_bytes = file.read()
        pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img_array = np.array(pil_img)
        
        # ── Run YOUR exact predict() from model.py ──
        img_rgb, cam, prediction, prob = predict(model, img_array, threshold=0.35)
        
        # ── Build Grad-CAM heatmap + overlay ──
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(img_rgb, 0.5, heatmap, 0.5, 0)
        
        # ── Convert images to base64 for HTML display ──
        original_b64 = numpy_to_base64(img_rgb)
        heatmap_b64 = numpy_to_base64(heatmap)
        overlay_b64 = numpy_to_base64(overlay)
        
        # ── Borderline logic (same as your Streamlit app) ──
        if prediction == "DR":
            status = "DR"
        elif prob >= 0.20:
            status = "Borderline"
        else:
            status = "No DR"
        
        return jsonify({
            "label": status,
            "confidence": prob,
            "original": original_b64,
            "heatmap": heatmap_b64,
            "overlay": overlay_b64,
        })
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"Prediction failed: {str(e)}"
        }), 500

# ── HEALTH CHECK ──
@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint (Render uses this)"""
    return jsonify({"status": "ok", "model": "EfficientNet-B4 DR"}), 200

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    print(f"Server running at: http://localhost:{port}")
    print("Open http://localhost:{port} in Chrome\n")
    app.run(host="0.0.0.0", port=port, debug=False)