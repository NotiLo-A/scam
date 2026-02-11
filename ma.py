import cv2
import face_recognition
import os
import numpy as np

# ===== Загрузка базы лиц =====
KNOWN_DIR = "known_faces"

known_encodings = []
known_names = []

for filename in os.listdir(KNOWN_DIR):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        path = os.path.join(KNOWN_DIR, filename)
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            name = os.path.splitext(filename)[0]
            known_names.append(name)

print(f"[INFO] Загружено лиц: {len(known_names)}")

# ===== Камера =====
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Уменьшаем для скорости
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # ===== Детекция лиц =====
    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        matches = face_recognition.compare_faces(
            known_encodings, face_encoding, tolerance=0.5
        )

        name = "Unknown"

        if True in matches:
            best_match_index = matches.index(True)
            name = known_names[best_match_index]

        # Возвращаем масштаб
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        # ===== Отрисовка =====
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(
            frame,
            name,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()