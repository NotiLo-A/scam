import cv2
import os
import numpy as np

# ---------------- CONFIG ----------------
MODE = "identify"        # "enroll" или "identify"
USER_NAME = "user1"      # только для enroll
DATA_DIR = "faces"
FACE_SIZE = (120, 120)
SAMPLES = 20
THRESHOLD = 0.35         # меньше = строже
# ----------------------------------------

os.makedirs(DATA_DIR, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def preprocess(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, FACE_SIZE)
    return face

# ---------- ENROLL ----------
if MODE == "enroll":
    user_dir = os.path.join(DATA_DIR, USER_NAME)
    os.makedirs(user_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
        )

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_p = preprocess(face)

            cv2.imwrite(f"{user_dir}/{count}.png", face_p)
            count += 1

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, f"Samples: {count}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Enroll", frame)

        if count >= SAMPLES:
            break
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------- IDENTIFY ----------
if MODE == "identify":
    database = {}

    for user in os.listdir(DATA_DIR):
        user_dir = os.path.join(DATA_DIR, user)
        if not os.path.isdir(user_dir):
            continue

        templates = []
        for file in os.listdir(user_dir):
            img = cv2.imread(os.path.join(user_dir, file), 0)
            if img is not None:
                templates.append(img)
        if templates:
            database[user] = templates

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
            best_score = 1.0

            for name, templates in database.items():
                for tmpl in templates:
                    res = cv2.matchTemplate(face_p, tmpl, cv2.TM_SQDIFF_NORMED)
                    score = res[0][0]
                    if score < best_score:
                        best_score = score
                        best_name = name

            if best_score > THRESHOLD:
                best_name = "Unknown"

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, best_name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()