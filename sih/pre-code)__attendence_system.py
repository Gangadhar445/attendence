import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime

# Paths
DATA_DIR = r"F:\hackathon\dataset"
MODEL_FILE = r"F:\hackathon\trained_model.yml"
LABEL_FILE = r"F:\hackathon\label_map.npy"


# Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load model + labels
def load_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    label_map = {}
    if os.path.exists(MODEL_FILE) and os.path.exists(LABEL_FILE):
        recognizer.read(MODEL_FILE)
        label_map = np.load(LABEL_FILE, allow_pickle=True).item()
        print("Loaded existing model.")
    else:
        print(" No trained model found. Please register new students.")
    return recognizer, label_map

# Save attendance
def mark_attendance(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")
    with open("attendance.csv", "a") as f:
        f.write(f"{name},{date},{time}\n")

# Register new person
def register_new_person(person_name, num_samples=10):
    person_dir = os.path.join(DATA_DIR, person_name)
    os.makedirs(person_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0
    print(f" Capturing {num_samples} images for {person_name}. Look at the camera...")

    while count < num_samples:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]
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
    print(f" Registered {person_name} successfully!")

# Train model
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
            faces.append(img)
            labels.append(label_id)
        label_id += 1

    if not faces:
        print(" No training data found.")
        return None, {}

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))

    recognizer.save(MODEL_FILE)
    np.save(LABEL_FILE, label_map)

    print(" Model trained and saved.")
    return recognizer, label_map

#ATTENDANCE
def show_attendance():
    if os.path.exists("attendance.csv"):
        print("\n Attendance Records:")
        df = pd.read_csv("attendance.csv", header=None, names=["Name", "Date", "Time"])
        print(df.to_string(index=False))
    else:
        print("\nNo attendance records found yet.")

# Main loop
def run_attendance_system():
    recognizer, label_map = load_model()

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            name = "Unknown"

            if recognizer and len(label_map) > 0:
                id_, conf = recognizer.predict(roi)
                if conf < 80:
                    name = label_map[id_]
                    mark_attendance(name)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # If unknown -> ask user in terminal
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

    #result data 
    show_attendance()

if __name__ == "__main__":
    run_attendance_system()
