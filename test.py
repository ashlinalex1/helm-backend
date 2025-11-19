import onnxruntime as ort
import numpy as np
from PIL import Image

session = ort.InferenceSession("best1.onnx")
input_name = session.get_inputs()[0].name
print("Inputs:", session.get_inputs())
print("Outputs:", session.get_outputs())
