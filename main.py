from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import base64
import cv2
import onnxruntime as ort
import time

app = FastAPI()

# ------------------ Load ONNX Models ------------------
helmet_session = ort.InferenceSession("best1.onnx", providers=["CPUExecutionProvider"])
seatbelt_session = ort.InferenceSession("best2.onnx", providers=["CPUExecutionProvider"])

# Get model input name
helmet_input_name = helmet_session.get_inputs()[0].name
seatbelt_input_name = seatbelt_session.get_inputs()[0].name

# ------------------ CORS ------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8080"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Helper Functions ------------------
def preprocess(img):
    img_resized = cv2.resize(img, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    img_transposed = np.transpose(img_normalized, (2, 0, 1))
    img_input = np.expand_dims(img_transposed, axis=0)
    return img_input

def postprocess(outputs, img_shape):
    boxes = []
    output = outputs[0]

    for det in output:
        x1, y1, x2, y2, conf, cls = det
        if conf < 0.25:
            continue

        boxes.append({
            "label": str(int(cls)),
            "confidence": float(conf),
            "x": int(x1),
            "y": int(y1),
            "width": int(x2 - x1),
            "height": int(y2 - y1)
        })

    return boxes

def draw_boxes(image: np.ndarray, detections: list):
    img_with_boxes = image.copy()
    for det in detections:
        color = (0, 255, 0) if det["label"] == "helmet" else (0, 0, 255)
        cv2.rectangle(
            img_with_boxes,
            (det["x"], det["y"]),
            (det["x"] + det["width"], det["y"] + det["height"]),
            color,
            2
        )
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

# ------------------ Predict Endpoint ------------------
@app.post("/predict")
async def predict_upload(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        np_arr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        detections = []

        # ------------- HELMET MODEL -------------
        helmet_input = preprocess(frame)
        helmet_outputs = helmet_session.run(None, {helmet_input_name: helmet_input})
        helmet_boxes = postprocess(helmet_outputs, frame.shape)
        detections.extend(helmet_boxes)

        # Draw helmet detections
        annotated = frame.copy()

        # ------------- SEATBELT MODEL -------------
        seatbelt_input = preprocess(frame)
        seatbelt_outputs = seatbelt_session.run(None, {seatbelt_input_name: seatbelt_input})
        seatbelt_boxes = postprocess(seatbelt_outputs, frame.shape)
        detections.extend(seatbelt_boxes)

        # Draw seatbelt detections
        annotated = draw_boxes(annotated, detections)

        # Encode image
        _, buffer = cv2.imencode(".png", annotated)
        img_str = base64.b64encode(buffer).decode("utf-8")

        return {
            "success": True,
            "image": img_str,
            "detections": detections
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------ Webcam Endpoint ------------------
@app.post("/predict-webcam")
async def predict_webcam(file: bytes = File(...)):
    try:
        np_arr = np.frombuffer(file, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        detections = []

        helmet_input = preprocess(frame)
        seatbelt_input = preprocess(frame)

        # Helmet
        helmet_outputs = helmet_session.run(None, {helmet_input_name: helmet_input})
        helmet_boxes = postprocess(helmet_outputs, frame.shape)
        detections.extend(helmet_boxes)

        # Seatbelt
        seatbelt_outputs = seatbelt_session.run(None, {seatbelt_input_name: seatbelt_input})
        seatbelt_boxes = postprocess(seatbelt_outputs, frame.shape)
        detections.extend(seatbelt_boxes)

        annotated = draw_boxes(frame, detections)

        _, buffer = cv2.imencode('.png', annotated)
        img_str = base64.b64encode(buffer).decode("utf-8")

        return {
            "success": True,
            "image": img_str,
            "detections": detections
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
