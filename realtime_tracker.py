import cv2
import mediapipe as mp
import numpy as np
import argparse

# --- 参数解析 ---
parser = argparse.ArgumentParser(description='Real-time eye tracking and blink detection.')
parser.add_argument('--video', type=str, help='Path to the video file. If not provided, webcam will be used.')
parser.add_argument('--ear_threshold', type=float, default=0.2, help='Threshold for eye aspect ratio to determine blink.')
args = parser.parse_args()

# --- MediaPipe 初始化 ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- 眼睛关键点索引 ---
LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# --- 视频源选择 ---
if args.video:
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {args.video}")
        exit()
else:
    cap = cv2.VideoCapture(0) # 使用默认摄像头
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        exit()

# --- EAR 计算函数 ---
def calculate_ear(eye_landmarks, image_shape):
    p1 = eye_landmarks[8]; p2 = eye_landmarks[12]; p3 = eye_landmarks[14]
    p4 = eye_landmarks[0]; p5 = eye_landmarks[2];  p6 = eye_landmarks[4]
    
    ver_dist1 = np.linalg.norm(np.array([p2.x, p2.y]) * image_shape - np.array([p6.x, p6.y]) * image_shape)
    ver_dist2 = np.linalg.norm(np.array([p3.x, p3.y]) * image_shape - np.array([p5.x, p5.y]) * image_shape)
    hor_dist = np.linalg.norm(np.array([p1.x, p1.y]) * image_shape - np.array([p4.x, p4.y]) * image_shape)
    
    ear = (ver_dist1 + ver_dist2) / (2.0 * hor_dist)
    return ear

# --- 主循环变量 ---
EAR_THRESHOLD = args.ear_threshold
SLEEP_SECONDS_THRESHOLD = 1.0
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0: # 摄像头可能返回0
    fps = 30 

consecutive_closed_frames = 0
total_sleep_frames = 0
is_sleeping = False

# --- 主处理循环 ---
while cap.isOpened():
    success, image = cap.read()
    if not success:
        if args.video:
            print("End of video.")
            break
        else:
            continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark
        
        left_eye_landmarks = [landmarks[i] for i in LEFT_EYE_INDICES]
        right_eye_landmarks = [landmarks[i] for i in RIGHT_EYE_INDICES]

        left_ear = calculate_ear(left_eye_landmarks, np.array([w, h]))
        right_ear = calculate_ear(right_eye_landmarks, np.array([w, h]))
        avg_ear = (left_ear + right_ear) / 2.0

        if avg_ear < EAR_THRESHOLD:
            consecutive_closed_frames += 1
        else:
            consecutive_closed_frames = 0
            is_sleeping = False

        # --- 颜色渐变逻辑 ---
        # EAR正常范围大约在0.15(闭)到0.3(睁)之间
        ear_norm = (avg_ear - 0.15) / (0.3 - 0.15)
        ear_norm = np.clip(ear_norm, 0, 1) # 限制在0-1之间

        # 从白色 (255,255,255) 到 红色 (0,0,255) 的渐变
        # BGR格式
        color_b = int(255 * ear_norm)
        color_g = int(255 * ear_norm)
        color_r = 255
        
        drawing_spec = mp_drawing.DrawingSpec(color=(color_b, color_g, color_r), thickness=1)

        # --- 状态判断与文本显示 ---
        if consecutive_closed_frames > int(fps * SLEEP_SECONDS_THRESHOLD):
            if not is_sleeping:
                is_sleeping = True
            total_sleep_frames += 1
            cv2.putText(image, "SLEEPING", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif consecutive_closed_frames > 0:
            cv2.putText(image, "Blinking", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            cv2.putText(image, "Open", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        total_sleep_time = total_sleep_frames / fps
        cv2.putText(image, f"Sleep Time: {total_sleep_time:.2f}s", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # --- 绘制面部网格 ---
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_spec)

    cv2.imshow('Real-time Eye Tracker', image)

    # 检查按键或窗口是否被关闭
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q') or cv2.getWindowProperty('Real-time Eye Tracker', cv2.WND_PROP_VISIBLE) < 1:
        break

# --- 资源释放 ---
cap.release()
cv2.destroyAllWindows()

final_total_sleep_time = total_sleep_frames / fps
print(f"\nSession finished.")
print(f"Total sleep time recorded: {final_total_sleep_time:.2f} seconds.")