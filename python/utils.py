import cv2 
import mediapipe as mp

def get_face_landmarks(image, draw=False):
    # Chuyển ảnh sang RGB vì mediapipe yêu cầu
    image_input_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Khởi tạo face mesh
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,  # ảnh tĩnh
        max_num_faces=1,
        min_detection_confidence=0.5
    )

    image_landmarks = []

    # Xử lý ảnh
    results = face_mesh.process(image_input_rgb)

    # Nếu phát hiện khuôn mặt
    if results.multi_face_landmarks:
        if draw:
            mp_drawing = mp.solutions.drawing_utils
            drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=results.multi_face_landmarks[0],
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

        ls_single_face = results.multi_face_landmarks[0].landmark

        xs_ = [pt.x for pt in ls_single_face]
        ys_ = [pt.y for pt in ls_single_face]
        zs_ = [pt.z for pt in ls_single_face]

        for j in range(len(xs_)):
            image_landmarks.append(xs_[j] - min(xs_))
            image_landmarks.append(ys_[j] - min(ys_))
            image_landmarks.append(zs_[j] - min(zs_))

    # ⚠️ Đóng face_mesh để tránh giữ tài nguyên
    face_mesh.close()

    return image_landmarks
