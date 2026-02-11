import cv2


def load_face_detector():
    model_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(model_path)


def open_camera(index=0):
    return cv2.VideoCapture(index)


def detect_faces(detector, image_gray):
    return detector.detectMultiScale(
        image_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )


def draw_faces(image, detections):
    for x, y, w, h in detections:
        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


def main():
    detector = load_face_detector()
    camera = open_camera()

    print("Нажмите 'q' для выхода")

    while True:
        success, image = camera.read()
        if not success:
            break

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found_faces = detect_faces(detector, gray_image)

        draw_faces(image, found_faces)

        cv2.putText(
            image,
            f"Faces: {len(found_faces)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Face Detection", image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()