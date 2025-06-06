# extract_landmarks.py

import os
import cv2
import pandas as pd
from tqdm import tqdm
from utils import get_face_landmarks

# Đường dẫn đến dữ liệu train/test
base_dir = r"C:\emotion-vision\data\archive (2)"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# Các loại cảm xúc (label) – tên folder
emotion_labels = sorted(os.listdir(train_dir))  # e.g. ['angry', 'happy', 'sad', ...]

def extract_data(data_dir, output_prefix):
    X = []
    y = []

    for emotion in emotion_labels:
        emotion_path = os.path.join(data_dir, emotion)
        if not os.path.isdir(emotion_path):
            continue

        print(f"Processing '{emotion}'...")

        for img_name in tqdm(os.listdir(emotion_path)):
            img_path = os.path.join(emotion_path, img_name)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                # Resize ảnh về 256x256 hoặc kích thước nhỏ hơn tùy dataset
                img = cv2.resize(img, (256, 256))

                landmarks = get_face_landmarks(img, draw=False)
                del img  # giải phóng RAM ngay sau khi xử lý ảnh

                if landmarks:
                    X.append(landmarks)
                    y.append(emotion)
            except Exception as e:
                print(f"Lỗi với ảnh {img_path}: {e}")

    # Lưu kết quả
    df_X = pd.DataFrame(X)
    df_y = pd.Series(y)

    df_X.to_csv(f"{output_prefix}_X.csv", index=False)
    df_y.to_csv(f"{output_prefix}_y.csv", index=False)
    print(f"✅ Đã lưu {output_prefix}_X.csv và {output_prefix}_y.csv")
