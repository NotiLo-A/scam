import cv2
import numpy as np
import os
import pickle

TRAIN_DIR = "train"
IMAGE_SIZE = (128, 128)

features = []
labels = []

def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, IMAGE_SIZE)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist([hsv], [0, 1, 2], None,
                         [8, 8, 8],
                         [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

for label in os.listdir(TRAIN_DIR):
    class_dir = os.path.join(TRAIN_DIR, label)
    if not os.path.isdir(class_dir):
        continue

    for file in os.listdir(class_dir):
        path = os.path.join(class_dir, file)
        features.append(extract_features(path))
        labels.append(label)

features = np.array(features)

with open("model.pkl", "wb") as f:
    pickle.dump((features, labels), f)

print("Обучение завершено. Модель сохранена.")