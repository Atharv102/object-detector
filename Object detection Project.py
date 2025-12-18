import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import threading

# Load YOLOv8 model
model = YOLO("yolov8x.pt")

# Global variables
original_img = None
detected_img = None
img_path = ""
webcam_active = False
cap = None

# Initialize window
root = tk.Tk()
root.title("YOLOv8 Object Detector")
root.geometry("1300x800")
root.config(bg="#f0f0f0")

# Sidebar
sidebar = tk.Frame(root, width=200, bg="#2e3f4f")
sidebar.pack(side=tk.LEFT, fill=tk.Y)

main_area = tk.Frame(root, bg="#f0f0f0")
main_area.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

# Image Panels
image_frame = tk.Frame(main_area, bg="#f0f0f0")
image_frame.pack(pady=20)

original_label = tk.Label(image_frame, text="Original Image", bg="#f0f0f0", font=("Arial", 12))
original_label.grid(row=0, column=0)
detected_label = tk.Label(image_frame, text="Detected Image", bg="#f0f0f0", font=("Arial", 12))
detected_label.grid(row=0, column=1)

original_panel = tk.Label(image_frame, bg="#ddd", width=500, height=400)
original_panel.grid(row=1, column=0, padx=10)

detected_panel = tk.Label(image_frame, bg="#ddd", width=500, height=400)
detected_panel.grid(row=1, column=1, padx=10)

# Results Table
tree = ttk.Treeview(main_area, columns=("Class", "Confidence", "Box", "Area"), show="headings", height=10)
tree.heading("Class", text="Class")
tree.heading("Confidence", text="Confidence")
tree.heading("Box", text="Box")
tree.heading("Area", text="Area")
tree.pack(pady=10, fill=tk.X, padx=20)

style = ttk.Style()
style.configure("Treeview", font=("Arial", 10), rowheight=25)

# Functions
def upload_image():
    global original_img, detected_img, img_path
    path = filedialog.askopenfilename()
    if not path:
        return
    img_path = path
    img = cv2.imread(img_path)
    img = cv2.resize(img, (500, 400))
    original_img = img.copy()
    detected_img = img.copy()

    # Show original
    img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    imgtk = ImageTk.PhotoImage(image=img_pil)
    original_panel.config(image=imgtk)
    original_panel.image = imgtk

    # Clear detected panel and table
    detected_panel.config(image=None)
    tree.delete(*tree.get_children())

def run_detection():
    global original_img, detected_img
    if detected_img is None:
        messagebox.showerror("Error", "Please upload an image first.")
        return

    results = model.predict(original_img)
    detections = results[0].boxes
    boxes = detections.xyxy.cpu().numpy()
    scores = detections.conf.cpu().numpy()
    class_ids = detections.cls.cpu().numpy()

    confidence_threshold = 0.5
    filtered = [
        (model.names[int(cls)], score, box)
        for box, score, cls in zip(boxes, scores, class_ids)
        if score >= confidence_threshold
    ]
    filtered.sort(key=lambda x: x[1], reverse=True)

    img_with_boxes = original_img.copy()
    for label, score, box in filtered:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_with_boxes, f"{label} {score:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Update detected image panel
    img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    imgtk = ImageTk.PhotoImage(image=img_pil)
    detected_panel.config(image=imgtk)
    detected_panel.image = imgtk

    # Update table
    tree.delete(*tree.get_children())
    for label, score, box in filtered:
        x1, y1, x2, y2 = map(int, box)
        area = (x2 - x1) * (y2 - y1)
        tree.insert("", tk.END, values=(label, f"{score:.2f}", f"[{x1},{y1},{x2},{y2}]", area))

def start_webcam():
    global webcam_active, cap
    webcam_active = True
    cap = cv2.VideoCapture(0)
    threading.Thread(target=update_webcam, daemon=True).start()

def stop_webcam():
    global webcam_active, cap
    webcam_active = False
    if cap is not None:
        cap.release()
        cap = None
    original_panel.config(image=None)
    detected_panel.config(image=None)

def update_webcam():
    global cap
    while webcam_active and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (500, 400))
        img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        imgtk = ImageTk.PhotoImage(image=pil_img)
        original_panel.config(image=imgtk)
        original_panel.image = imgtk

        # Detection
        results = model.predict(resized, verbose=False)
        detections = results[0].boxes
        boxes = detections.xyxy.cpu().numpy()
        scores = detections.conf.cpu().numpy()
        class_ids = detections.cls.cpu().numpy()

        for box, score, cls in zip(boxes, scores, class_ids):
            if score >= 0.25:
                x1, y1, x2, y2 = map(int, box)
                label = model.names[int(cls)]
                cv2.rectangle(resized, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(resized, f"{label} {score:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show detected frame
        det_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        det_pil = Image.fromarray(det_rgb)
        det_imgtk = ImageTk.PhotoImage(image=det_pil)
        detected_panel.config(image=det_imgtk)
        detected_panel.image = det_imgtk

# Sidebar buttons
upload_btn = tk.Button(sidebar, text="Upload Image", font=("Arial", 12), bg="#4caf50", fg="white",
                       relief="flat", command=upload_image)
upload_btn.pack(pady=20, ipadx=10, ipady=5)

detect_btn = tk.Button(sidebar, text="Run Detection", font=("Arial", 12), bg="#2196f3", fg="white",
                       relief="flat", command=run_detection)
detect_btn.pack(pady=10, ipadx=10, ipady=5)

start_web_btn = tk.Button(sidebar, text="Start Webcam", font=("Arial", 12), bg="#9c27b0", fg="white",
                          relief="flat", command=start_webcam)
start_web_btn.pack(pady=10, ipadx=10, ipady=5)

stop_web_btn = tk.Button(sidebar, text="Stop Webcam", font=("Arial", 12), bg="#e91e63", fg="white",
                         relief="flat", command=stop_webcam)
stop_web_btn.pack(pady=10, ipadx=10, ipady=5)

# Run the app
root.mainloop()
