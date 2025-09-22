"""
Automated Student Attendance Monitoring System (Fixed Version)
--------------------------------------------------------------
Features:
1. Detects faces from webcam (laptop camera).
2. Recognizes students using LBPH face recognizer.
3. Resizes all faces to 200x200 for better recognition.
4. Logs attendance (only one entry per student per day).
5. Unknown faces can be registered with multiple snapshots.
6. Prints attendance as a formatted table using tabulate.

Dependencies:
- opencv-python
- opencv-contrib-python
- numpy
- pandas
- tabulate
"""

import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
from tabulate import tabulate

# ================================
# Paths (absolute paths)
# ================================
DATA_DIR = r"F:\hackathon\dataset"               # folder for student face images
MODEL_FILE = r"F:\hackathon\trained_model.yml"   # LBPH model file
LABEL_FILE = r"F:\hackathon\label_map.npy"       # mapping label_id -> student name
ATTENDANCE_FILE = r"F:\hackathon\attendance.csv" # CSV log for attendance

# Ensure dataset folder exists
os.makedirs(DATA_DIR, exist_ok=True)

# ================================
# Haar cascade face detector
# ================================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ================================
# Load recognizer and label map
# ================================
def load_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    label_map = {}
    if os.path.exists(MODEL_FILE) and os.path.exists(LABEL_FILE):
        recognizer.read(MODEL_FILE)
        label_map = np.load(LABEL_FILE, allow_pickle=True).item()
        print("Loaded existing trained model.")
    else:
        print("No trained model found. Please register students first.")
    return recognizer, label_map

# ================================
# Attendance logging (no duplicates)
# ================================
def mark_attendance(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    # Create CSV with headers if not exist
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w") as f:
            f.write("Name,Date,Time\n")

    df = pd.read_csv(ATTENDANCE_FILE)

    # Skip duplicate for today
    if ((df["Name"] == name) & (df["Date"] == date)).any():
        return

    # Append new entry
    with open(ATTENDANCE_FILE, "a") as f:
        f.write(f"{name},{date},{time}\n")

# ================================
# Display attendance table
# ================================
def show_attendance():
    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
        print("\nAttendance Records:")
        print(tabulate(df, headers="keys", tablefmt="fancy_grid", showindex=False))
    else:
        print("\nNo attendance records found yet.")

# ================================
# Register a new student
# ================================
def register_new_person(person_name, num_samples=20):
    """
    Capture multiple face images for a new student.
    Resize each image to 200x200 for consistent recognition.
    """
    person_dir = os.path.join(DATA_DIR, person_name)
    os.makedirs(person_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0
    print(f"Capturing {num_samples} images for {person_name}. Look at the camera...")

    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("Camera error. Cannot capture frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200, 200))  # resize for consistency

            file_path = os.path.join(person_dir, f"{count}.jpg")
            cv2.imwrite(file_path, face_img)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"Image {count}/{num_samples}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.imshow("Registering...", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Registered {person_name} successfully!")

# ================================
# Train LBPH model
# ================================
def train_model():
    faces = []
    labels = []
    label_map = {}
    label_id = 0

    for student in os.listdir(DATA_DIR):
        student_path = os.path.join(DATA_DIR, student)
        if not os.path.isdir(student_path):
            continue
        label_map[label_id] = student
        for img_file in os.listdir(student_path):
            img_path = os.path.join(student_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (200, 200))  # resize to 200x200
            faces.append(img)
            labels.append(label_id)
        label_id += 1

    if not faces:
        print("No training data found. Register students first.")
        return None, {}

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))

    recognizer.save(MODEL_FILE)
    np.save(LABEL_FILE, label_map)

    print("Model trained and saved successfully.")
    return recognizer, label_map

# ================================
# Main loop
# ================================
def run_attendance_system():
    recognizer, label_map = load_model()

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot access camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi, (200, 200))  # resize before prediction
            name = "Unknown"

            if recognizer and len(label_map) > 0:
                id_, conf = recognizer.predict(roi_resized)
                if conf < 80:  # lower confidence = better match
                    name = label_map[id_]
                    mark_attendance(name)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if name == "Unknown":
                cv2.putText(frame, "Unknown - Press R to Register", (x, y+h+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Attendance System", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            choice = input("Unknown person detected. Register new user? (y/n): ").strip().lower()
            if choice == "y":
                person_name = input("Enter new student name: ").strip()
                register_new_person(person_name)
                recognizer, label_map = train_model()
            else:
                print("Skipping registration.")

    cap.release()
    cv2.destroyAllWindows()
    show_attendance()


# ================================
# Entry point
# ================================
if __name__ == "__main__":
    run_attendance_system()