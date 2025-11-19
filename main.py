from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
import base64
import cv2
import time

app = FastAPI()
model = YOLO(r"C:\Users\ashli\runs\detect\train12\weights\best.pt")

# Configure CORS
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


def process_image(image_data: bytes):
    """Helper function to process image and return detections"""
    # Convert bytes -> PIL -> NumPy RGB
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    img_array = np.array(image)  # shape H x W x 3
    
    # Run YOLO
    results = model.predict(source=img_array, conf=0.25, imgsz=640, verbose=False)
    
    detections = []
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            label = model.names.get(cls_id, f"class_{cls_id}")
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "label": label,
                "confidence": conf,
                "x": x1,
                "y": y1,
                "width": x2 - x1,
                "height": y2 - y1
            })
    
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

# Load both models once (on startup)
helmet_model = YOLO("best1.pt")
seatbelt_model = YOLO("best2.pt")


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
        helmet_results = helmet_model(frame)

        # Get annotated image directly from YOLO
        annotated = helmet_results[0].plot()

        if helmet_results[0].boxes is not None:
            for box in helmet_results[0].boxes:
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

        # ---------------- Seatbelt Model ----------------
        seatbelt_results = seatbelt_model(frame)

        # Overlay seatbelt detections on top of helmet annotated image
        seatbelt_annotated = seatbelt_results[0].plot()
        # Blend the two annotated outputs (simple max for visibility)
        annotated = np.maximum(annotated, seatbelt_annotated)

        if seatbelt_results[0].boxes is not None:
            for box in seatbelt_results[0].boxes:
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

        # Encode final annotated image to base64
        _, buffer = cv2.imencode(".png", annotated)
        img_str = base64.b64encode(buffer).decode("utf-8")

        return {
            "success": True,
            "image": img_str,   # annotated frame with both models
            "detections": detections
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# @app.post("/predict")
# async def predict_upload(file: UploadFile = File(...)):
#     """Endpoint for file uploads (Helmet + Seatbelt detection)"""
#     try:
#         # Read uploaded file into OpenCV image
#         image_data = await file.read()
#         np_arr = np.frombuffer(image_data, np.uint8)
#         frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

#         detections = []

#         # Run helmet model
#         helmet_results = helmet_model(frame)
#         annotated = helmet_results[0].plot()  # start with helmet annotations

#         if helmet_results[0].boxes is not None and len(helmet_results[0].boxes) > 0:
#             for box in helmet_results[0].boxes:
#                 cls_id = int(box.cls.item())
#                 conf = float(box.conf.item())
#                 label = helmet_model.names.get(cls_id, f"class_{cls_id}")
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 detections.append({
#                     "label": label,
#                     "confidence": conf,
#                     "x": x1, "y": y1,
#                     "width": x2 - x1,
#                     "height": y2 - y1
#                 })

#         # Run seatbelt model
#         seatbelt_results = seatbelt_model(frame)
#         annotated = seatbelt_results[0].plot(annotated)  # overlay on helmet annotations

#         if seatbelt_results[0].boxes is not None and len(seatbelt_results[0].boxes) > 0:
#             for box in seatbelt_results[0].boxes:
#                 cls_id = int(box.cls.item())
#                 conf = float(box.conf.item())
#                 label = seatbelt_model.names.get(cls_id, f"class_{cls_id}")
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 detections.append({
#                     "label": label,
#                     "confidence": conf,
#                     "x": x1, "y": y1,
#                     "width": x2 - x1,
#                     "height": y2 - y1
#                 })

#         # Encode annotated image to base64
#         _, buffer = cv2.imencode(".png", annotated)
#         img_str = base64.b64encode(buffer).decode("utf-8")

#         return {
#             "success": True,
#             "image": img_str,   # annotated frame
#             "detections": detections
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
# @app.post("/predict")
# async def predict_upload(file: UploadFile = File(...)):
#     """Endpoint for file uploads (used by UploadSection)"""
#     try:
#         # Read file bytes and decode into OpenCV image
#         image_data = await file.read()
#         np_arr = np.frombuffer(image_data, np.uint8)
#         frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

#         # Run YOLO on uploaded image
#         results = model(frame)

#         # Get annotated image directly from YOLO
#         annotated = results[0].plot()

#         # Encode annotated image to base64
#         _, buffer = cv2.imencode('.png', annotated)
#         img_str = base64.b64encode(buffer).decode("utf-8")

#         # Extract detections
#         detections = []
#         if results and results[0].boxes is not None:
#             for box in results[0].boxes:
#                 cls_id = int(box.cls.item())
#                 conf = float(box.conf.item())
#                 label = model.names.get(cls_id, f"class_{cls_id}")
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 detections.append({
#                     "label": label,
#                     "confidence": conf,
#                     "x": x1,
#                     "y": y1,
#                     "width": x2 - x1,
#                     "height": y2 - y1
#                 })

#         return {
#             "success": True,
#             "image": img_str,   # annotated frame
#             "detections": detections
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-webcam")
async def predict_webcam(file: bytes = File(...)):
    """Endpoint for webcam frames (used by WebcamSection)"""
    try:
        # Convert bytes -> NumPy BGR image (OpenCV format)
        np_arr = np.frombuffer(file, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Run YOLO directly on frame
        results = model(frame)

        # Use YOLO's built-in plotting
        annotated = results[0].plot()

        # Encode annotated frame to base64
        _, buffer = cv2.imencode('.png', annotated)
        img_str = base64.b64encode(buffer).decode("utf-8")

        # Collect detections from YOLO
        detections = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                label = model.names.get(cls_id, f"class_{cls_id}")
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    "label": label,
                    "confidence": conf,
                    "x": x1,
                    "y": y1,
                    "width": x2 - x1,
                    "height": y2 - y1
                })

        return {
            "success": True,
            "image": img_str,   # annotated frame from YOLO.plot()
            "detections": detections
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
