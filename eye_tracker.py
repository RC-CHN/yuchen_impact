import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

# 眼睛关键点索引
LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# 初始化 MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 视频文件路径
video_path = 'data/WIN_20250805_10_02_37_Pro.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Cannot open video file {video_path}")
    exit()

# 获取视频的帧率和尺寸
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 设置视频编写器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_with_debounce.mp4', fourcc, fps, (width, height))

# EAR计算函数
def calculate_ear(eye_landmarks, image_shape):
    p1 = eye_landmarks[8]
    p2 = eye_landmarks[12]
    p3 = eye_landmarks[14]
    p4 = eye_landmarks[0]
    p5 = eye_landmarks[2]
    p6 = eye_landmarks[4]

    # 计算垂直距离
    ver_dist1 = np.linalg.norm(np.array([p2.x, p2.y]) * [image_shape[1], image_shape[0]] - np.array([p6.x, p6.y]) * [image_shape[1], image_shape[0]])
    ver_dist2 = np.linalg.norm(np.array([p3.x, p3.y]) * [image_shape[1], image_shape[0]] - np.array([p5.x, p5.y]) * [image_shape[1], image_shape[0]])
    # 计算水平距离
    hor_dist = np.linalg.norm(np.array([p1.x, p1.y]) * [image_shape[1], image_shape[0]] - np.array([p4.x, p4.y]) * [image_shape[1], image_shape[0]])
    
    ear = (ver_dist1 + ver_dist2) / (2.0 * hor_dist)
    return ear

EAR_THRESHOLD = 0.2
CONSECUTIVE_FRAMES_THRESHOLD = 1  # 对应1秒

closed_frames_counter = 0
total_closed_frames = 0
total_closed_time_seconds = 0

# 使用tqdm创建进度条
for _ in tqdm(range(total_frames), desc="Processing Video"):
    success, image = cap.read()
    if not success:
        break

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            
            left_eye_landmarks = [landmarks[i] for i in LEFT_EYE_INDICES]
            right_eye_landmarks = [landmarks[i] for i in RIGHT_EYE_INDICES]

            left_ear = calculate_ear(left_eye_landmarks, image.shape)
            right_ear = calculate_ear(right_eye_landmarks, image.shape)
            
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < EAR_THRESHOLD:
                closed_frames_counter += 1
            else:
                closed_frames_counter = 0

            if closed_frames_counter > int(fps * CONSECUTIVE_FRAMES_THRESHOLD):
                total_closed_frames += 1
                cv2.putText(image, "SLEEPING", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif closed_frames_counter > 0:
                cv2.putText(image, "Blinking", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                cv2.putText(image, "Open", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            total_closed_time_seconds = total_closed_frames / fps
            cv2.putText(image, f"Sleep Time: {total_closed_time_seconds:.2f} s", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 写入帧
    out.write(image)

cap.release()
out.release()

print(f"\nProcessing complete. Video saved to output_with_debounce.mp4")
print(f"Total sleep time (closed for > {CONSECUTIVE_FRAMES_THRESHOLD}s): {total_closed_time_seconds:.2f} seconds.")