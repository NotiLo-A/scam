import cv2
import os
import numpy as np

DATA_DIR = "faces"
FACE_SIZE = (120, 120)
SAMPLES = 20
THRESHOLD = 0.35

os.makedirs(DATA_DIR, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

current_name = ""
saving = False
saved_count = 0

BUTTON = (20, 400, 180, 450)  # x1,y1,x2,y2

def preprocess(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, FACE_SIZE)
    return face

def mouse_callback(event, x, y, flags, param):
    global saving, saved_count
    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1, x2, y2 = BUTTON
        if x1 <= x <= x2 and y1 <= y <= y2 and current_name:
            saving = True
            saved_count = 0
            os.makedirs(os.path.join(DATA_DIR, current_name), exist_ok=True)

cv2.namedWindow("Face Recognition")
cv2.setMouseCallback("Face Recognition", mouse_callback)

cap = cv2.VideoCapture(0)

database = {}

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, 1.2, 5, minSize=(80, 80)
    )

    # LOAD DATABASE
    database.clear()
    for user in os.listdir(DATA_DIR):
        user_dir = os.path.join(DATA_DIR, user)
        if not os.path.isdir(user_dir):
            continue
        imgs = []
        for f in os.listdir(user_dir):
            img = cv2.imread(os.path.join(user_dir, f), 0)
            if img is not None:
                imgs.append(img)
        if imgs:
            database[user] = imgs

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

        if saving and saved_count < SAMPLES:
            path = os.path.join(DATA_DIR, current_name, f"{saved_count}.png")
            cv2.imwrite(path, face_p)
            saved_count += 1
            if saved_count >= SAMPLES:
                saving = False

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, best_name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    # UI
    cv2.rectangle(frame, (10, 340), (300, 380), (50,50,50), -1)
    cv2.putText(frame, f"Name: {current_name}", (20, 370),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    x1,y1,x2,y2 = BUTTON
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,200,0), -1)
    cv2.putText(frame, "SAVE", (x1+40, y2-15),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

    if saving:
        cv2.putText(frame, f"Saving: {saved_count}/{SAMPLES}",
                    (20, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    cv2.imshow("Face Recognition", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break
    elif key == 8:  # Backspace
        current_name = current_name[:-1]
    elif 32 <= key <= 126:
        current_name += chr(key)

cap.release()
cv2.destroyAllWindows()