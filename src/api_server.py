from __future__ import annotations

import tempfile
from pathlib import Path

from flask import Flask, jsonify, request

from config import MAX_UPLOAD_BYTES, SUPPORTED_IMAGE_EXTENSIONS
from moderation import check_image, get_nsfw_model
from predict import get_model_and_classes, predict_image

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


@app.get("/")
def home():
    return jsonify(
        {
            "app": "SPOT AI",
            "status": "running",
            "purpose": "visual talent-tag suggestions",
        }
    )


@app.get("/health")
def health():
    try:
        _, class_names = get_model_and_classes()
        get_nsfw_model()
        return jsonify(
            {
                "success": True,
                "model_loaded": True,
                "moderation_loaded": True,
                "classes": class_names,
            }
        )
    except Exception as error:
        return jsonify(
            {
                "success": False,
                "model_loaded": False,
                "error": str(error),
            }
        ), 503


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify(
            {
                "success": False,
                "error": "Send multipart/form-data using the field name 'image'.",
            }
        ), 400

    uploaded = request.files["image"]
    if not uploaded.filename:
        return jsonify({"success": False, "error": "No image filename was provided."}), 400

    suffix = Path(uploaded.filename).suffix.lower()
    if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
        return jsonify(
            {
                "success": False,
                "error": "Unsupported image type. Use JPG, JPEG, PNG, BMP, or WEBP.",
            }
        ), 415

    temp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = Path(temp_file.name)
            uploaded.save(temp_file)

        # Screen before classifying so explicit images never get talent tags.
        moderation = check_image(temp_path)
        if moderation["blocked"]:
            return jsonify(
                {
                    "success": False,
                    "error": "This image appears to contain explicit or sexual content and can't be posted.",
                    "reason": "explicit_content",
                    **moderation,
                }
            ), 422

        result = predict_image(temp_path)
        return jsonify({"success": True, **result, **moderation})

    except ValueError as error:
        return jsonify({"success": False, "error": str(error)}), 400
    except FileNotFoundError as error:
        return jsonify({"success": False, "error": str(error)}), 503
    except Exception as error:
        # Keep the response useful without exposing a full Python traceback.
        return jsonify({"success": False, "error": f"Prediction failed: {error}"}), 500
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)


@app.errorhandler(413)
def too_large(_error):
    return jsonify(
        {
            "success": False,
            "error": f"Image is too large. Maximum upload size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        }
    ), 413


if __name__ == "__main__":
    # debug=False is safer and avoids Flask's development reloader loading the AI twice.
    app.run(host="0.0.0.0", port=5000, debug=False)
