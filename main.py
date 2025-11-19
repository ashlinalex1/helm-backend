import numpy as np
import cv2
import base64
import io
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as ort

app = FastAPI()

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://streetsmart-road.vercel.app",  # your Vercel frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------- LOAD ONNX MODELS ----------------
helmet_sess = ort.InferenceSession("best1.onnx", providers=["CPUExecutionProvider"])
seat_sess   = ort.InferenceSession("best2.onnx", providers=["CPUExecutionProvider"])

# ------------- PREPROCESS ----------------
def preprocess(image):
    img = cv2.resize(image, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img = np.expand_dims(img, axis=0)
    return img

# ------------- POSTPROCESS ----------------
def extract_boxes(output, img_w, img_h):
    """YOLOv8 ONNX output → boxes"""
    boxes = []
    preds = output[0]  # (batch, 84, N)

    preds = np.squeeze(preds)
    preds = preds.T  # (N, 84)

    # YOLOv8: first 4 = box, 5th = obj conf, next 2 = class confs
    for det in preds:
        x, y, w, h = det[0:4]
        obj = det[4]

        if obj < 0.3:
            continue

        cls_scores = det[5:]
        cls_id = np.argmax(cls_scores)
        cls_conf = cls_scores[cls_id] * obj

        if cls_conf < 0.25:
            continue

        x1 = int((x - w/2) * img_w / 640)
        y1 = int((y - h/2) * img_h / 640)
        x2 = int((x + w/2) * img_w / 640)
        y2 = int((y + h/2) * img_h / 640)

        boxes.append({
            "label": "helmet" if cls_id == 0 else "other",
            "confidence": float(cls_conf),
            "x": x1,
            "y": y1,
            "width": x2 - x1,
            "height": y2 - y1
        })

    return boxes

# ------------- RUN MODEL ----------------
def run_onnx(sess, frame):
    img_input = preprocess(frame)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    output = sess.run([output_name], {input_name: img_input})
    return extract_boxes(output, frame.shape[1], frame.shape[0])


# ===========================
#   /predict
# ===========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        helmet_boxes = run_onnx(helmet_sess, frame)
        seat_boxes   = run_onnx(seat_sess, frame)

        detections = helmet_boxes + seat_boxes

        # Draw boxes
        for det in detections:
            x, y = det["x"], det["y"]
            w, h = det["width"], det["height"]
            label = det["label"]
            color = (0,255,0) if "helmet" in label else (0,0,255)

            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            cv2.putText(frame, f"{label}", (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        _, buffer = cv2.imencode(".png", frame)
        img_str = base64.b64encode(buffer).decode("utf-8")

        return {
            "success": True,
            "image": img_str,
            "detections": detections
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===========================
#   /predict-webcam
# ===========================
@app.post("/predict-webcam")
async def predict_webcam(file: bytes = File(...)):
    try:
        np_arr = np.frombuffer(file, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        helmet_boxes = run_onnx(helmet_sess, frame)
        seat_boxes   = run_onnx(seat_sess, frame)

        detections = helmet_boxes + seat_boxes

        # Draw overlay
        for det in detections:
            x, y = det["x"], det["y"]
            w, h = det["width"], det["height"]
            label = det["label"]
            color = (0,255,0) if "helmet" in label else (0,0,255)

            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            cv2.putText(frame, f"{label}", (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        _, buffer = cv2.imencode(".png", frame)
        img_str = base64.b64encode(buffer).decode("utf-8")

        return {
            "success": True,
            "image": img_str,
            "detections": detections
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
