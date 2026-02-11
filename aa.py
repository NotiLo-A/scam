import os
import cv2
import torch
from ultralytics import YOLO

CLASSES = ["person", "animal", "vase", "fruit", "book"]
CONF_THRESHOLD = 0.5

def train_model():
    model = YOLO("yolov8n.pt")
    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device="cpu"
    )

def infer_image(model, image_path):
    img = cv2.imread(image_path)
    results = model(img, imgsz=640, conf=CONF_THRESHOLD)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return [{"class": "no_object"}]
    output = []
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if conf < CONF_THRESHOLD:
            continue
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        output.append({
            "class": CLASSES[cls_id],
            "bbox": [x1, y1, x2, y2],
            "confidence": conf
        })
    if not output:
        return [{"class": "no_object"}]
    return output

def infer_folder(test_dir):
    model = YOLO("runs/detect/train/weights/best.pt")
    results = {}
    for fname in os.listdir(test_dir):
        if fname.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(test_dir, fname)
            results[fname] = infer_image(model, path)
    return results

if __name__ == "__main__":
    if not os.path.exists("runs/detect/train/weights/best.pt"):
        train_model()
    outputs = infer_folder("test")
    for k, v in outputs.items():
        print(k, v)