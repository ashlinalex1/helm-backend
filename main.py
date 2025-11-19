from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
from PIL import Image
import io
import base64
import cv2
import time
import traceback
import logging

# set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("yolo-backend")

app = FastAPI()

# -------------- MODELS (use relative paths in deployment) ----------------
# If you've converted to ONNX and want ONNX inference, say so and I'll provide that variant.
helmet_model = YOLO("best1.pt")
seatbelt_model = YOLO("best2.pt")
# optional combined model (if used anywhere)
# model = YOLO("best.pt")

# ----------------- CORS (add your deployed frontend + backend URLs) ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "https://streetsmart-road.vercel.app",                       # your Vercel frontend
        "https://hopeful-transformation-production-8990.up.railway.app"  # your Railway backend (if needed)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------- Utility helpers (safe checks) -----------------
def is_nonempty_sequence(x):
    """Return True if x is a sequence (list/tuple/np.ndarray) and length > 0."""
    try:
        return (x is not None) and (hasattr(x, "__len__") and len(x) > 0)
    except Exception:
        return False


def has_boxes_in_result(result_item):
    """
    Safe check whether a YOLO result item contains boxes (non-empty).
    Handles different types returned by ultralytics across versions.
    """
    try:
        # many ultralytics results expose .boxes (Boxes object) or .boxes.xyxy etc.
        boxes = getattr(result_item, "boxes", None)
        if boxes is None:
            # some results may keep boxes as an attribute-like list
            return False
        # boxes might be a container with length or have attribute .shape or .xyxy
        try:
            # first try len()
            return len(boxes) > 0
        except Exception:
            # boxes might be a numpy array or object with .xyxy
            arr = getattr(boxes, "xyxy", None)
            if arr is None:
                # fallback: if boxes has attribute .data or .cpu
                return True  # assume present (conservative)
            # arr could be numpy; check its size property safely
            try:
                return getattr(arr, "size", None) is not None and int(arr.size) > 0
            except Exception:
                return True
    except Exception:
        return False


# ----------------- Image helper functions -----------------
def process_image(image_data: bytes):
    """Helper function to process image and return detections"""
    # Convert bytes -> PIL -> NumPy RGB
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    img_array = np.array(image)  # shape H x W x 3

    # Run YOLO (using ultralytics API)
    try:
        results = helmet_model.predict(source=img_array, conf=0.25, imgsz=640, verbose=False)
    except Exception as e:
        logger.error("Helmet model inference failed: %s", traceback.format_exc())
        raise

    detections = []
    if is_nonempty_sequence(results) and has_boxes_in_result(results[0]):
        try:
            for box in results[0].boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                label = helmet_model.names.get(cls_id, f"class_{cls_id}")
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    "label": label,
                    "confidence": conf,
                    "x": x1,
                    "y": y1,
                    "width": x2 - x1,
                    "height": y2 - y1
                })
        except Exception:
            logger.warning("Failed parsing helmet boxes: %s", traceback.format_exc())

    return img_array, detections


def draw_boxes(image: np.ndarray, detections: list):
    """Draw bounding boxes on the image"""
    img_with_boxes = image.copy()
    for det in detections:
        color = (0, 255, 0) if "helmet" in det["label"].lower() else (0, 0, 255)
        cv2.rectangle(
            img_with_boxes,
            (det["x"], det["y"]),
            (det["x"] + det["width"], det["y"] + det["height"]),
            color,
            2
        )
        # Add label and confidence
        label = f"{det['label']} {det['confidence']*100:.1f}%"
        cv2.putText(
            img_with_boxes,
            label,
            (det["x"], det["y"] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )
    return img_with_boxes


# ----------------- Endpoints -----------------
@app.post("/predict")
async def predict_upload(file: UploadFile = File(...)):
    """Endpoint for file uploads (Helmet + Seatbelt detection)"""
    try:
        # Read uploaded file into OpenCV image
        image_data = await file.read()
        np_arr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        detections = []

        # ---------------- Helmet Model ----------------
        try:
            helmet_results = helmet_model(frame)
            # YOLO.plot() returns annotated image (numpy)
            annotated = helmet_results[0].plot() if is_nonempty_sequence(helmet_results) else frame
        except Exception:
            logger.error("Helmet model predict error: %s", traceback.format_exc())
            annotated = frame

        if is_nonempty_sequence(helmet_results) and has_boxes_in_result(helmet_results[0]):
            for box in helmet_results[0].boxes:
                try:
                    cls_id = int(box.cls.item())
                    conf = float(box.conf.item())
                    label = helmet_model.names.get(cls_id, f"class_{cls_id}")
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append({
                        "label": label,
                        "confidence": conf,
                        "x": x1, "y": y1,
                        "width": x2 - x1,
                        "height": y2 - y1
                    })
                except Exception:
                    logger.warning("Error reading helmet box: %s", traceback.format_exc())

        # ---------------- Seatbelt Model ----------------
        try:
            seatbelt_results = seatbelt_model(frame)
            seatbelt_annotated = seatbelt_results[0].plot() if is_nonempty_sequence(seatbelt_results) else None
            if seatbelt_annotated is not None:
                annotated = np.maximum(annotated, seatbelt_annotated)
        except Exception:
            logger.error("Seatbelt model predict error: %s", traceback.format_exc())

        if is_nonempty_sequence(locals().get("seatbelt_results", None)) and has_boxes_in_result(seatbelt_results[0]):
            for box in seatbelt_results[0].boxes:
                try:
                    cls_id = int(box.cls.item())
                    conf = float(box.conf.item())
                    label = seatbelt_model.names.get(cls_id, f"class_{cls_id}")
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append({
                        "label": label,
                        "confidence": conf,
                        "x": x1, "y": y1,
                        "width": x2 - x1,
                        "height": y2 - y1
                    })
                except Exception:
                    logger.warning("Error reading seatbelt box: %s", traceback.format_exc())

        # Encode final annotated image to base64
        _, buffer = cv2.imencode(".png", annotated)
        img_str = base64.b64encode(buffer).decode("utf-8")

        return {
            "success": True,
            "image": img_str,   # annotated frame with both models
            "detections": detections
        }

    except Exception as e:
        logger.error("Unhandled error in /predict: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-webcam")
async def predict_webcam(file: bytes = File(...)):
    """Endpoint for webcam frames (used by WebcamSection)"""
    try:
        # Convert bytes -> NumPy BGR image (OpenCV format)
        np_arr = np.frombuffer(file, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Run YOLO directly on frame
        try:
            results = helmet_model(frame)  # or model(frame)
            annotated = results[0].plot() if is_nonempty_sequence(results) else frame
        except Exception:
            logger.error("predict-webcam model error: %s", traceback.format_exc())
            annotated = frame
            results = []

        # Encode annotated frame to base64
        _, buffer = cv2.imencode('.png', annotated)
        img_str = base64.b64encode(buffer).decode("utf-8")

        # Collect detections from YOLO
        detections = []
        if is_nonempty_sequence(results) and has_boxes_in_result(results[0]):
            for box in results[0].boxes:
                try:
                    cls_id = int(box.cls.item())
                    conf = float(box.conf.item())
                    label = helmet_model.names.get(cls_id, f"class_{cls_id}")
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append({
                        "label": label,
                        "confidence": conf,
                        "x": x1, "y": y1,
                        "width": x2 - x1,
                        "height": y2 - y1
                    })
                except Exception:
                    logger.warning("Error reading predict-webcam box: %s", traceback.format_exc())

        return {
            "success": True,
            "image": img_str,   # annotated frame from YOLO.plot()
            "detections": detections
        }

    except Exception as e:
        logger.error("Unhandled error in /predict-webcam: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
