import cv2
import mediapipe as mp
import os

mp_face_mesh = mp.solutions.face_mesh

def extract_mouth_rois(frames_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,   # dudak detayları
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    for frame_name in sorted(os.listdir(frames_dir)):
        frame_path = os.path.join(frames_dir, frame_name)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            continue

        face = result.multi_face_landmarks[0]

        # Dudak landmark indexleri
        mouth_ids = list(range(61, 88))

        xs, ys = [], []
        for idx in mouth_ids:
            lm = face.landmark[idx]
            xs.append(int(lm.x * w))
            ys.append(int(lm.y * h))

        x_min, x_max = max(min(xs)-10, 0), min(max(xs)+10, w)
        y_min, y_max = max(min(ys)-10, 0), min(max(ys)+10, h)

        mouth = frame[y_min:y_max, x_min:x_max]

        out_path = os.path.join(output_dir, frame_name)
        cv2.imwrite(out_path, mouth)

    print(f"[OK] Mouth ROIs saved to: {output_dir}")
