import cv2
import numpy as np
import pickle

IMAGE_SIZE = (128, 128)

def extract_features(img):
    img = cv2.resize(img, IMAGE_SIZE)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist([hsv], [0, 1, 2], None,
                         [8, 8, 8],
                         [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

with open("model.pkl", "rb") as f:
    train_features, train_labels = pickle.load(f)

img = cv2.imread("test.jpg")
test_feature = extract_features(img)

distances = np.linalg.norm(train_features - test_feature, axis=1)
nearest = np.argmin(distances)

prediction = train_labels[nearest]

print("Предсказанный класс:", prediction)

cv2.putText(img, prediction, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()