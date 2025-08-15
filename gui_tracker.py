import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import queue
from PIL import Image, ImageTk
import sqlite3
from datetime import datetime
import os
import sys

# --- Core Tracker Logic ---
class EyeTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

    def calculate_ear(self, eye_landmarks, image_shape):
        p1 = eye_landmarks[8]; p2 = eye_landmarks[12]; p3 = eye_landmarks[14]
        p4 = eye_landmarks[0]; p5 = eye_landmarks[2];  p6 = eye_landmarks[4]
        
        image_shape_np = np.array(image_shape)
        
        ver_dist1 = np.linalg.norm(np.array([p2.x, p2.y]) * image_shape_np - np.array([p6.x, p6.y]) * image_shape_np)
        ver_dist2 = np.linalg.norm(np.array([p3.x, p3.y]) * image_shape_np - np.array([p5.x, p5.y]) * image_shape_np)
        hor_dist = np.linalg.norm(np.array([p1.x, p1.y]) * image_shape_np - np.array([p4.x, p4.y]) * image_shape_np)
        
        if hor_dist == 0:
            return 0.3
            
        ear = (ver_dist1 + ver_dist2) / (2.0 * hor_dist)
        return ear

    def process_frame(self, image, show_mesh=True):
        h, w, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        avg_ear = -1.0

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = face_landmarks.landmark
            
            left_eye_landmarks = [landmarks[i] for i in self.LEFT_EYE_INDICES]
            right_eye_landmarks = [landmarks[i] for i in self.RIGHT_EYE_INDICES]

            left_ear = self.calculate_ear(left_eye_landmarks, (w, h))
            right_ear = self.calculate_ear(right_eye_landmarks, (w, h))
            avg_ear = (left_ear + right_ear) / 2.0

            if show_mesh:
                ear_norm = np.clip((avg_ear - 0.15) / (0.3 - 0.15), 0, 1)
                color_b = int(255 * ear_norm); color_g = int(255 * ear_norm); color_r = 255
                drawing_spec = mp.solutions.drawing_utils.DrawingSpec(color=(color_b, color_g, color_r), thickness=1)
                mp.solutions.drawing_utils.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_spec
                )
        return image, avg_ear

    def close(self):
        self.face_mesh.close()

# --- GUI Application ---
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-time Eye Tracker")
        self.root.geometry("1200x800")

        # --- Variables ---
        self.video_source = tk.StringVar(value="Webcam")
        self.video_path = ""
        self.ear_threshold = tk.DoubleVar(value=0.2)
        self.sleep_threshold = tk.DoubleVar(value=1.0)
        self.show_mesh = tk.BooleanVar(value=True)
        
        self.tracker = EyeTracker()
        self.tracking_thread = None
        self.stop_tracking = False
        self.is_paused = False
        self.image_queue = queue.Queue(maxsize=2)
        
        self.db_conn = None
        self.longest_sleep_snapshot = None
        self.longest_sleep_duration = 0

        self.setup_database()
        self.setup_gui()

    def setup_gui(self):
        # --- Layout ---
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        control_panel = ttk.Frame(main_frame, width=280)
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), anchor='n')
        self.video_panel = ttk.Label(main_frame, background="black")
        self.video_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # --- Controls ---
        source_frame = ttk.LabelFrame(control_panel, text="Video Source", padding="10")
        source_frame.pack(fill=tk.X, pady=5)
        ttk.Radiobutton(source_frame, text="Webcam", variable=self.video_source, value="Webcam", command=self.on_source_change).pack(anchor=tk.W)
        ttk.Radiobutton(source_frame, text="Video File", variable=self.video_source, value="File", command=self.on_source_change).pack(anchor=tk.W)
        self.browse_button = ttk.Button(source_frame, text="Browse...", command=self.browse_file, state=tk.DISABLED)
        self.browse_button.pack(pady=5)

        tracking_frame = ttk.LabelFrame(control_panel, text="Control", padding="10")
        tracking_frame.pack(fill=tk.X, pady=5)
        self.start_button = ttk.Button(tracking_frame, text="Start Tracking", command=self.start_tracking)
        self.start_button.pack(fill=tk.X, pady=(5, 2))
        self.pause_button = ttk.Button(tracking_frame, text="Pause", command=self.toggle_pause, state=tk.DISABLED)
        self.pause_button.pack(fill=tk.X, pady=2)
        self.stop_button = ttk.Button(tracking_frame, text="Stop Tracking", command=self.stop_tracking_func, state=tk.DISABLED)
        self.stop_button.pack(fill=tk.X, pady=(2, 5))

        settings_frame = ttk.LabelFrame(control_panel, text="Settings", padding="10")
        settings_frame.pack(fill=tk.X, pady=5)
        ttk.Label(settings_frame, text="EAR Threshold:").pack(anchor=tk.W)
        self.ear_slider = ttk.Scale(settings_frame, from_=0.1, to=0.4, orient=tk.HORIZONTAL, variable=self.ear_threshold, command=self.update_ear_label)
        self.ear_slider.pack(fill=tk.X)
        self.ear_label = ttk.Label(settings_frame, text=f"{self.ear_threshold.get():.2f}")
        self.ear_label.pack()
        ttk.Label(settings_frame, text="Sleep Time Threshold (s):").pack(anchor=tk.W, pady=(10, 0))
        self.sleep_slider = ttk.Scale(settings_frame, from_=0.5, to=5.0, orient=tk.HORIZONTAL, variable=self.sleep_threshold, command=self.update_sleep_label)
        self.sleep_slider.pack(fill=tk.X)
        self.sleep_label = ttk.Label(settings_frame, text=f"{self.sleep_threshold.get():.1f}s")
        self.sleep_label.pack()
        ttk.Checkbutton(settings_frame, text="Show Face Mesh", variable=self.show_mesh).pack(anchor=tk.W, pady=5)

        status_frame = ttk.LabelFrame(control_panel, text="Status", padding="10")
        status_frame.pack(fill=tk.X, pady=5)
        status_frame.columnconfigure(1, weight=1)
        ttk.Label(status_frame, text="Status:").grid(row=0, column=0, sticky="w", pady=2)
        self.status_label = ttk.Label(status_frame, text="Idle", width=25, anchor="w")
        self.status_label.grid(row=0, column=1, sticky="we", pady=2)
        ttk.Label(status_frame, text="EAR:").grid(row=1, column=0, sticky="w", pady=2)
        self.ear_value_label = ttk.Label(status_frame, text="N/A", width=25, anchor="w")
        self.ear_value_label.grid(row=1, column=1, sticky="we", pady=2)
        ttk.Label(status_frame, text="Total Sleep:").grid(row=2, column=0, sticky="w", pady=2)
        self.sleep_time_label = ttk.Label(status_frame, text="0.00s", width=25, anchor="w")
        self.sleep_time_label.grid(row=2, column=1, sticky="we", pady=2)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_source_change(self):
        self.browse_button.config(state=tk.NORMAL if self.video_source.get() == "File" else tk.DISABLED)

    def browse_file(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if path:
            self.video_path = path
            self.status_label.config(text=f"Loaded: ...{self.video_path[-30:]}")

    def update_ear_label(self, value):
        self.ear_label.config(text=f"{float(value):.2f}")

    def update_sleep_label(self, value):
        self.sleep_label.config(text=f"{float(value):.1f}s")

    def setup_database(self):
        # Determine the base path, works for both script and bundled exe
        if getattr(sys, 'frozen', False):
            # If the application is run as a bundle, the PyInstaller bootloader
            # extends the sys module by a flag frozen=True and sets the app
            # path into variable _MEIPASS'.
            base_path = os.path.dirname(sys.executable)
        else:
            # If run as a script, the base path is the script's directory
            base_path = os.path.dirname(os.path.abspath(__file__))
        
        db_path = os.path.join(base_path, 'sleep_log.db')
        print(f"Database path: {db_path}") # For debugging
        
        self.db_conn = sqlite3.connect(db_path, check_same_thread=False)
        cursor = self.db_conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sleep_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TEXT NOT NULL,
                duration_seconds REAL NOT NULL
            )
        ''')
        self.db_conn.commit()

    def log_sleep_event(self, start_time, duration):
        cursor = self.db_conn.cursor()
        cursor.execute("INSERT INTO sleep_events (start_time, duration_seconds) VALUES (?, ?)",
                       (start_time.strftime("%Y-%m-%d %H:%M:%S"), duration))
        self.db_conn.commit()

    def start_tracking(self):
        if self.video_source.get() == "File" and not self.video_path:
            self.status_label.config(text="Please select a video file.")
            return
        self.stop_tracking = False
        self.is_paused = False
        self.start_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.NORMAL, text="Pause")
        self.stop_button.config(state=tk.NORMAL)
        self.status_label.config(text="Starting...")
        self.tracking_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.tracking_thread.start()
        self.root.after(100, self.update_gui)

    def video_loop(self):
        cap = cv2.VideoCapture(self.video_path if self.video_source.get() == "File" else 0)
        if not cap.isOpened():
            self.image_queue.put(("ERROR", "Cannot open video source"))
            return

        consecutive_closed_frames = 0
        total_sleep_time = 0
        is_sleeping = False
        sleep_start_time = None
        snapshot_taken_this_cycle = False
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        while not self.stop_tracking:
            while self.is_paused and not self.stop_tracking:
                time.sleep(0.1)
            
            if self.stop_tracking: break

            success, frame = cap.read()
            if not success: break
            
            original_frame = frame.copy()
            processed_frame, avg_ear = self.tracker.process_frame(frame, self.show_mesh.get())
            
            status = "Open"
            if avg_ear != -1.0:
                if avg_ear < self.ear_threshold.get():
                    consecutive_closed_frames += 1
                    status = "Closed"
                else:
                    if is_sleeping:
                        sleep_duration = (datetime.now() - sleep_start_time).total_seconds()
                        self.log_sleep_event(sleep_start_time, sleep_duration)
                        if sleep_duration > self.longest_sleep_duration:
                            self.longest_sleep_duration = sleep_duration
                            if self.longest_sleep_snapshot is not None:
                                self.longest_sleep_snapshot = original_frame
                    is_sleeping = False
                    consecutive_closed_frames = 0
                    snapshot_taken_this_cycle = False
            else:
                status = "No Face Detected"

            if consecutive_closed_frames > int(fps * self.sleep_threshold.get()):
                if not is_sleeping:
                    is_sleeping = True
                    sleep_start_time = datetime.now()
                
                current_sleep_duration = (datetime.now() - sleep_start_time).total_seconds()
                total_sleep_time += (1/fps)
                status = f"SLEEPING ({current_sleep_duration:.1f}s)"
                
                if current_sleep_duration > 10 and not snapshot_taken_this_cycle:
                    self.longest_sleep_snapshot = original_frame
                    snapshot_taken_this_cycle = True

            cv2.putText(processed_frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            try: self.image_queue.put_nowait((processed_frame, avg_ear, status, total_sleep_time))
            except queue.Full: pass

        cap.release()
        self.image_queue.put(None)

    def update_gui(self):
        try:
            data = self.image_queue.get_nowait()
            if data is None:
                self.stop_tracking_func()
                return
            
            if isinstance(data, tuple) and len(data) == 2 and data[0] == "ERROR":
                self.status_label.config(text=data[1])
                self.stop_tracking_func()
                return

            frame, avg_ear, status, sleep_time = data
            
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            
            panel_w, panel_h = self.video_panel.winfo_width(), self.video_panel.winfo_height()
            if panel_w > 1 and panel_h > 1:
                img_w, img_h = img.size
                scale = min(panel_w / img_w, panel_h / img_h)
                new_w, new_h = int(img_w * scale), int(img_h * scale)
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            imgtk = ImageTk.PhotoImage(image=img)
            self.video_panel.imgtk = imgtk
            self.video_panel.config(image=imgtk)

            self.status_label.config(text=status)
            self.ear_value_label.config(text=f"{avg_ear:.3f}" if avg_ear != -1.0 else "N/A")
            self.sleep_time_label.config(text=f"{sleep_time:.2f}s")

        except queue.Empty:
            pass
        
        if not self.stop_tracking:
            self.root.after(30, self.update_gui)

    def stop_tracking_func(self):
        self.is_paused = False
        self.stop_tracking = True
        if self.tracking_thread and self.tracking_thread.is_alive():
            self.tracking_thread.join(timeout=1)
        
        self.start_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED, text="Pause")
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Idle")
        self.ear_value_label.config(text="N/A")
        self.sleep_time_label.config(text="0.00s")
        self.video_panel.config(image='', background='black')

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_button.config(text="Resume")
            self.status_label.config(text="Paused")
        else:
            self.pause_button.config(text="Pause")
            self.status_label.config(text="Tracking...")

    def on_closing(self):
        self.is_paused = False
        self.stop_tracking = True
        if self.tracking_thread and self.tracking_thread.is_alive():
            self.tracking_thread.join(timeout=1)
        
        if self.longest_sleep_snapshot is not None:
            if getattr(sys, 'frozen', False):
                base_path = os.path.dirname(sys.executable)
            else:
                base_path = os.path.dirname(os.path.abspath(__file__))
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sleep_snapshot_{timestamp}.jpg"
            snapshot_path = os.path.join(base_path, filename)
            cv2.imwrite(snapshot_path, self.longest_sleep_snapshot)
            print(f"Saved snapshot to {snapshot_path}")

        if self.db_conn: self.db_conn.close()
        self.tracker.close()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()