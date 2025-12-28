import os, cv2, numpy as np
import mediapipe as mp

mp_face = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

# Göz landmarkları (MediaPipe)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def _ear(pts):
    # Eye Aspect Ratio
    A = np.linalg.norm(pts[1]-pts[5])
    B = np.linalg.norm(pts[2]-pts[4])
    C = np.linalg.norm(pts[0]-pts[3])
    return (A+B) / (2.0*C + 1e-6)

def blink_score(frames_dir):
    ears = []
    for f in sorted(os.listdir(frames_dir)):
        if not f.lower().endswith((".jpg",".png")): continue
        img = cv2.imread(os.path.join(frames_dir,f))
        if img is None: continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = mp_face.process(rgb)
        if not res.multi_face_landmarks: continue
        lm = res.multi_face_landmarks[0].landmark
        le = np.array([[lm[i].x, lm[i].y] for i in LEFT_EYE])
        re = np.array([[lm[i].x, lm[i].y] for i in RIGHT_EYE])
        ears.append((_ear(le)+_ear(re))/2.0)

    if len(ears) < 5: return 0.5
    v = np.var(ears)
    # Düşük varyans = blink yok → anomalik
    return float(np.clip(1.0 - np.tanh(10*v), 0, 1))

def headpose_score(frames_dir):
    # Basit jitter: ardışık merkez farkları
    centers = []
    for f in sorted(os.listdir(frames_dir)):
        if not f.lower().endswith((".jpg",".png")): continue
        img = cv2.imread(os.path.join(frames_dir,f))
        if img is None: continue
        h,w,_ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = mp_face.process(rgb)
        if not res.multi_face_landmarks: continue
        lm = res.multi_face_landmarks[0].landmark
        xs = [lm[i].x*w for i in range(468)]
        ys = [lm[i].y*h for i in range(468)]
        centers.append([np.mean(xs), np.mean(ys)])

    if len(centers) < 5: return 0.5
    diffs = np.diff(np.array(centers), axis=0)
    jitter = np.mean(np.linalg.norm(diffs, axis=1))
    return float(np.clip(np.tanh(jitter/10.0), 0, 1))
