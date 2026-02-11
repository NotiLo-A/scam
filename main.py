import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Ошибка: не удалось открыть камеру")
    exit()

print("Детекция лиц запущена. Нажмите 'q' для выхода")

with mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
) as face_detection:
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Ошибка чтения кадра")
            break
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = face_detection.process(rgb)

        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)
            
            cv2.putText(
                frame,
                f'Faces: {len(results.detections)}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
        
        cv2.imshow("Face Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
