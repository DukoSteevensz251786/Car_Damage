"""
CarCheck Pro — Model Server
Runs locally or on Render. On first boot, downloads the model from Google Drive
if it isn't already present on disk.

Usage (local):
    python server.py
    python server.py --model path/to/your_model.keras --port 5000

Requires:
    pip install flask flask-cors tensorflow tf-keras scikit-image numpy pillow requests opencv-python gdown
"""

import argparse
import base64
import io
import os
import sys
import traceback
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import numpy as np
import requests
import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_MODEL_PATH = "car_damage_efficientnet.keras"

# Set this env var on Render to your Google Drive file ID
# e.g. from https://drive.google.com/file/d/FILE_ID_HERE/view
MODEL_DRIVE_ID = os.environ.get("MODEL_DRIVE_ID", "")

CLASS_NAMES = ["Car_dent", "Car_fender_rust", "Car_scratch", "No_Damage"]

DAMAGE_CONFIG = {
    "Car_dent": {
        "type": "dent",
        "costs": {"minor": (50, 98), "moderate": (99, 150), "severe": (150, 299)},
        "locations": ["front bumper", "rear bumper", "driver door", "passenger door",
                      "hood", "trunk lid", "front fender", "rear quarter panel", "roof"],
    },
    "Car_fender_rust": {
        "type": "rust",
        "costs": {"minor": (100, 150), "moderate": (150, 200), "severe": (200, 250)},
        "locations": ["front fender", "rear fender", "door sill", "wheel arch",
                      "undercarriage", "rear quarter panel", "roof edge"],
    },
    "Car_scratch": {
        "type": "scratch",
        "costs": {"minor": (50, 75), "moderate": (75, 150), "severe": (150, 300)},
        "locations": ["front bumper", "rear bumper", "driver door", "passenger door",
                      "hood", "trunk", "side panel", "door mirror"],
    },
    "Car_window_damage": {
        "type": "window",
        "costs": {"minor": (150, 300), "moderate": (350, 700), "severe": (750, 1500)},
        "locations": ["front windshield", "rear windshield", "driver window",
                      "passenger window", "rear side window"],
    },
}

# ── Progress tracking ─────────────────────────────────────────────────────────

progress = {
    "current": 0,
    "total": 0,
    "status": "idle",
    "currentLabel": "",
}


def set_progress(current, total, label=""):
    progress["current"] = current
    progress["total"] = total
    progress["status"] = "analyzing"
    progress["currentLabel"] = label


def reset_progress():
    progress["current"] = 0
    progress["total"] = 0
    progress["status"] = "idle"
    progress["currentLabel"] = ""


def done_progress():
    progress["status"] = "done"
    progress["currentLabel"] = "Done"


# ── Model download ────────────────────────────────────────────────────────────

def download_model_if_needed(path):
    """
    If the model file doesn't exist and MODEL_DRIVE_ID is set,
    download it from Google Drive using gdown.
    """
    if os.path.exists(path):
        print(f"Model already present at: {path}")
        return True

    if not MODEL_DRIVE_ID:
        print("ERROR: Model file not found and MODEL_DRIVE_ID env var is not set.", file=sys.stderr)
        return False

    print(f"Model not found. Downloading from Google Drive (ID: {MODEL_DRIVE_ID})...")
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={MODEL_DRIVE_ID}"
        gdown.download(url, path, quiet=False)
        print(f"Model downloaded to: {path}")
        return True
    except Exception as e:
        print(f"ERROR downloading model: {e}", file=sys.stderr)
        return False


# ── Model loading ─────────────────────────────────────────────────────────────

model = None
model_path = DEFAULT_MODEL_PATH


def load_model(path):
    global model
    try:
        import tf_keras
        from tf_keras.models import load_model as keras_load
        print(f"Loading model from: {path}")
        model = keras_load(path, compile=False)
        print(f"Model loaded successfully. Input shape: {model.input_shape}")
        return True
    except Exception as e:
        print(f"ERROR loading model: {e}", file=sys.stderr)
        return False


# ── Image loading ─────────────────────────────────────────────────────────────

def image_from_base64(b64_string):
    from PIL import Image
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_string)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(img)


def image_from_url(url):
    from PIL import Image
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.marktplaats.nl/",
    }
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    return np.array(img)


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_for_model(img_array):
    import skimage.transform
    if img_array.ndim == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    img_resized = skimage.transform.resize(
        img_array, (224, 224),
        anti_aliasing=True,
        preserve_range=True
    ).astype(np.float32)
    return np.expand_dims(img_resized, axis=0)


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_patch(patch_array):
    batch = preprocess_for_model(patch_array)
    preds = model.predict(batch, verbose=0)[0]
    idx = int(np.argmax(preds))
    return {
        "class": CLASS_NAMES[idx],
        "confidence": float(np.max(preds)),
        "scores": {cls: float(preds[i]) for i, cls in enumerate(CLASS_NAMES)},
    }


def predict_with_patches(img_array, patch_size=300, stride=112, confidence_threshold=0.50):
    h, w = img_array.shape[:2]

    if h <= patch_size and w <= patch_size:
        pred = predict_patch(img_array)
        return pred, img_array.astype(np.float32), 0, 0

    best_pred = None
    best_patch = None
    best_x, best_y = 0, 0
    patch_count = 0

    for y in range(0, max(1, h - patch_size + 1), stride):
        for x in range(0, max(1, w - patch_size + 1), stride):
            y2 = min(y + patch_size, h)
            x2 = min(x + patch_size, w)
            patch = img_array[y:y2, x:x2]

            if patch.shape[0] < patch_size // 2 or patch.shape[1] < patch_size // 2:
                continue

            pred = predict_patch(patch)
            patch_count += 1

            if pred["class"] == "No_Damage":
                continue
            if pred["confidence"] < confidence_threshold:
                continue

            if best_pred is None or pred["confidence"] > best_pred["confidence"]:
                best_pred = pred
                best_patch = patch
                best_x, best_y = x, y

    print(f"    Scanned {patch_count} patches")

    if best_pred is None:
        print(f"    No confident damage patch found, reporting no damage")
        best_pred = {
            "class": "No_Damage",
            "confidence": 1.0,
            "scores": {cls: 0.0 for cls in CLASS_NAMES},
        }
        best_patch = np.zeros((224, 224, 3), dtype=np.float32)
        best_x, best_y = 0, 0

    return best_pred, best_patch.astype(np.float32), best_x, best_y


def confidence_to_severity(confidence):
    if confidence >= 0.99:
        return "severe"
    elif confidence >= 0.55:
        return "moderate"
    else:
        return "minor"


# ── Grad-CAM ──────────────────────────────────────────────────────────────────

def generate_gradcam(patch_array, class_idx):
    import tf_keras
    base_model = model.layers[0]
    grad_model = tf_keras.models.Model(
        inputs=base_model.inputs,
        outputs=[base_model.get_layer('top_activation').output, base_model.output]
    )
    img_batch = np.expand_dims(patch_array, axis=0).astype(np.float32)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_batch)
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]
    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap


def overlay_heatmap_on_patch(best_patch, heatmap, alpha=0.45):
    from PIL import Image as PILImage
    patch_h, patch_w = best_patch.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (patch_w, patch_h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB).astype(np.float32)
    patch_float = best_patch.astype(np.float32)
    blended = (1 - alpha) * patch_float + alpha * heatmap_colored
    result = np.clip(blended, 0, 255).astype(np.uint8)
    pil_img = PILImage.fromarray(result)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG', quality=85)
    return 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode()


# ── Damage item builder ───────────────────────────────────────────────────────

def build_damage_item(prediction, image_index, image_url="",
                      img_array=None, best_patch=None, patch_x=0, patch_y=0):
    import random
    cls = prediction["class"]
    conf = prediction["confidence"]
    severity = confidence_to_severity(conf)
    config = DAMAGE_CONFIG[cls]
    cost_range = config["costs"][severity]
    cost = random.randint(*cost_range)
    location = config["locations"][image_index % len(config["locations"])]
    severity_labels = {
        "minor": "Minor — likely cosmetic only",
        "moderate": "Moderate — should be addressed before purchase",
        "severe": "Severe — significant repair needed, negotiate hard",
    }
    heatmap_b64 = None
    if img_array is not None and best_patch is not None:
        try:
            import skimage.transform
            class_idx = CLASS_NAMES.index(cls)
            patch_for_gradcam = skimage.transform.resize(
                best_patch, (224, 224), anti_aliasing=True, preserve_range=True
            ).astype(np.float32)
            heatmap = generate_gradcam(patch_for_gradcam, class_idx)
            heatmap_b64 = overlay_heatmap_on_patch(patch_for_gradcam, heatmap)
        except Exception as e:
            print(f"  Grad-CAM failed: {e}", file=sys.stderr)
            traceback.print_exc()
    return {
        "imageUrl": image_url,
        "heatmapBase64": heatmap_b64,
        "type": config["type"],
        "severity": severity,
        "location": location.capitalize(),
        "description": f"{severity.capitalize()} {config['type']} detected ({conf*100:.0f}% confidence). {severity_labels[severity]}.",
        "repairCostEur": cost,
        "modelClass": cls,
        "confidence": round(conf * 100, 1),
    }


def determine_condition(damages, total_cost):
    if not damages:
        return "excellent"
    severe_count = sum(1 for d in damages if d["severity"] == "severe")
    if severe_count >= 2 or total_cost > 2000:
        return "poor"
    elif total_cost > 800 or severe_count == 1:
        return "fair"
    elif total_cost > 200:
        return "good"
    return "excellent"


NEGOTIATION_TIPS = {
    "excellent": "Car appears in great shape — verify mileage and service history before paying asking price.",
    "good": "Minor issues detected. You can reasonably ask for a small discount to cover repairs.",
    "fair": "Moderate damage found. Negotiate at least €{cost} off the asking price to cover repairs.",
    "poor": "Significant damage detected. Consider having an independent mechanic inspect before buying, and negotiate firmly.",
}

CONDITION_SUMMARIES = {
    "excellent": "No significant damage detected. The vehicle appears to be in excellent visual condition based on the provided photos.",
    "good": "Minor damage found in {count} photo(s). The vehicle is in generally good condition but has some cosmetic issues.",
    "fair": "Moderate damage detected across {count} photo(s). Total estimated repair cost: €{cost}. Factor this into your offer.",
    "poor": "Significant damage found across {count} photo(s). The car needs substantial repairs — get a professional inspection.",
}


def analyze_images(image_sources):
    damages = []
    errors = []
    damage_index = 0
    total = len(image_sources)
    reset_progress()
    for i, source in enumerate(image_sources):
        set_progress(i, total, f"Analyzing photo {i + 1} of {total}...")
        try:
            if source["type"] == "url":
                img = image_from_url(source["data"])
            else:
                img = image_from_base64(source["data"])
            print(f"  Image {i+1}: {img.shape[1]}x{img.shape[0]}px")
            pred, best_patch, patch_x, patch_y = predict_with_patches(img)
            if pred["class"] == "No_Damage":
                print(f"  Image {i+1}: No damage detected ({pred['confidence']*100:.1f}%), skipping")
            elif pred["confidence"] >= 0.85:
                set_progress(i, total, f"Damage found in photo {i + 1} — generating heatmap...")
                damage = build_damage_item(
                    pred, damage_index, source["data"],
                    img_array=img,
                    best_patch=best_patch,
                    patch_x=patch_x,
                    patch_y=patch_y
                )
                damages.append(damage)
                damage_index += 1
                print(f"  Image {i+1}: {pred['class']} ({pred['confidence']*100:.1f}%) at patch ({patch_x}, {patch_y})")
            else:
                print(f"  Image {i+1}: Low confidence ({pred['confidence']*100:.1f}%), skipping")
        except Exception as e:
            errors.append(f"Image {i+1}: {str(e)}")
            print(f"  Image {i+1} error: {e}", file=sys.stderr)
            traceback.print_exc()
    done_progress()
    total_cost = sum(d["repairCostEur"] for d in damages)
    condition = determine_condition(damages, total_cost)
    count = len(damages)
    summary = CONDITION_SUMMARIES[condition].format(count=count, cost=total_cost)
    tip = NEGOTIATION_TIPS[condition].format(cost=total_cost)
    return {
        "damages": damages,
        "overallCondition": condition,
        "summary": summary,
        "fairPriceEur": None,
        "negotiationTip": tip,
        "processedImages": len(image_sources),
        "errors": errors,
        "modelInfo": {
            "modelPath": str(model_path),
            "classes": CLASS_NAMES,
        },
    }


# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app, origins=["*"])


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "modelLoaded": model is not None,
        "modelPath": str(model_path),
        "classes": CLASS_NAMES,
    })


@app.route("/progress", methods=["GET"])
def get_progress():
    return jsonify(progress)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": f"Model not loaded. Make sure '{model_path}' exists."}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    raw_images = data.get("images", [])
    if not raw_images:
        return jsonify({"error": "No images provided"}), 400

    sources = []
    for img in raw_images:
        if isinstance(img, str):
            if img.startswith("data:") or (len(img) > 200 and "/" not in img[:50]):
                sources.append({"type": "base64", "data": img})
            else:
                sources.append({"type": "url", "data": img})
        elif isinstance(img, dict):
            sources.append(img)

    print(f"\nAnalyzing {len(sources)} image(s)...")
    result = analyze_images(sources)
    print(f"Done. Found {len(result['damages'])} damage(s), condition: {result['overallCondition']}")
    return jsonify({"ok": True, "result": result})


@app.route("/predict-single", methods=["POST"])
def predict_single_route():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    data = request.get_json()
    img_data = data.get("image", "")
    if not img_data:
        return jsonify({"error": "No image provided"}), 400
    try:
        if img_data.startswith("http"):
            img = image_from_url(img_data)
        else:
            img = image_from_base64(img_data)
        pred, _, _, _ = predict_with_patches(img)
        return jsonify(pred)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CarCheck Pro model server")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5000)))
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    model_path = args.model

    print("=" * 50)
    print("  CarCheck Pro — Model Server")
    print("=" * 50)

    if not download_model_if_needed(model_path):
        sys.exit(1)

    if not load_model(model_path):
        print(f"\nCould not load model from '{model_path}'.")
        sys.exit(1)

    print(f"\nServer running at: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
