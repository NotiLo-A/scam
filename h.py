import cv2
import os
import numpy as np

# ---------------- CONFIG ----------------
MODE = "identify"      # "enroll" или "identify"
USER_NAME = "user1"    # используется только в enroll
DATA_DIR = "faces"
FACE_SIZE = (200, 200)
MIN_SAMPLES = 30
# ----------------------------------------

os.makedirs(DATA_DIR, exist_ok=True)

# Haar Cascade (встроенный)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# LBPH recognizer (встроенный)
recognizer = cv2.face.LBPHFaceRecognizer_create()

def preprocess(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, FACE_SIZE)
    return gray

# ---------- ENROLL MODE ----------
if MODE == "enroll":
    cap = cv2.VideoCapture(0)
    samples = []
    labels = []

    count = 0
    label_id = 0

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
        )

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_p = preprocess(face)

            samples.append(face_p)
            labels.append(label_id)
            count += 1

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, f"Samples: {count}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Enroll", frame)

        if count >= MIN_SAMPLES:
            break
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    recognizer.train(samples, np.array(labels))
    recognizer.save(f"{DATA_DIR}/{USER_NAME}.yml")

# ---------- IDENTIFICATION MODE ----------
if MODE == "identify":
    models = {}
    label_map = {}
    label_id = 0

    for file in os.listdir(DATA_DIR):
        if file.endswith(".yml"):
            name = file.replace(".yml", "")
            model = cv2.face.LBPHFaceRecognizer_create()
            model.read(os.path.join(DATA_DIR, file))
            models[label_id] = model
            label_map[label_id] = name
            label_id += 1

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
        )

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_p = preprocess(face)

            best_name = "Unknown"
            best_conf = 999

            for lid, model in models.items():
                label, conf = model.predict(face_p)
                if conf < best_conf:
                    best_conf = conf
                    best_name = label_map[lid]

            if best_conf > 80:
                best_name = "Unknown"

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, best_name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()