import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model đã train
model = load_model(r'C:\emotion-vision\python\emotion_model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Đọc ảnh
img_path = r"C:\emotion-vision\data\archive (2)\image\image1.jpg"
img_path = r"C:\emotion-vision\data\archive (2)\image\image2.jpg"

frame = cv2.imread(img_path)
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Dò tìm khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

print(f"🧑‍🤝‍🧑 Tổng số người phát hiện được: {len(faces)}")

for (x, y, w, h) in faces:
    roi_gray = gray_frame[y:y+h, x:x+w]
    roi_gray = cv2.resize(roi_gray, (48, 48))
    roi_gray = roi_gray.astype('float32') / 255.0
    roi_gray = np.expand_dims(roi_gray, axis=-1)
    roi_gray = np.expand_dims(roi_gray, axis=0)

    prediction = model.predict(roi_gray)
    emotion = emotion_labels[np.argmax(prediction)]

    # Vẽ khung và nhãn
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                0.9, (0, 0, 255), 2)

# Hiển thị kết quả
cv2.imshow("Emotion Detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
