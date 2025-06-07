import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ===== Load model đã huấn luyện =====
model = load_model(r'C:\emotion-vision\python\emotion_model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ===== Đọc video =====
video_path = r"C:\emotion-vision\data\archive (2)\video\clip1.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"❌ Không thể mở video tại: {video_path}")
    exit()

# ===== Load bộ nhận diện khuôn mặt =====
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if face_cascade.empty():
    print("❌ Không thể load haarcascade_frontalface_default.xml")
    exit()

frame_count = 0
skip_frame = 2  # xử lý mỗi 2 frame

while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ Video kết thúc hoặc lỗi đọc frame.")
        break

    frame_count += 1
    if frame_count % skip_frame != 0:
        continue  # bỏ qua frame

    frame = cv2.resize(frame, (640, 360))  # resize giảm tải

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    print(f"🧑‍🤝‍🧑 Số người phát hiện: {len(faces)}")

    for (x, y, w, h) in faces:
        # Trích xuất và xử lý ảnh khuôn mặt
        roi_gray = gray_frame[y:y+h, x:x+w]
        try:
            roi_gray = cv2.resize(roi_gray, (48, 48))
        except:
            continue  # bỏ qua nếu ảnh lỗi

        roi = roi_gray.astype('float32') / 255.0
        roi = np.expand_dims(roi, axis=-1)
        roi = np.expand_dims(roi, axis=0)  # shape (1, 48, 48, 1)

        prediction = model.predict(roi, verbose=0)
        emotion = emotion_labels[np.argmax(prediction)]
        confidence = np.max(prediction)

        # Debug kết quả
        print(f"😶 Emotion: {emotion} ({confidence:.2f})")

        # Vẽ kết quả
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{emotion} ({confidence:.2f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Hiển thị
    cv2.imshow("Emotion Detection (press 'q' to quit)", frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
