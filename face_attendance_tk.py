import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from tkinter import Tk, Label, Entry, Button, StringVar, messagebox, Toplevel, ttk
from PIL import Image, ImageTk
import threading


DATASET_DIR = "dataset"
TRAINER_YML = "trainer.yml"
ATTENDANCE_FILE = "attendance.csv"
HAAR_CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
NUM_IMAGE = 45

if not os.path.exists(DATASET_DIR):
    os.mkdir(DATASET_DIR)

def capture_faces(person_id, person_name, num_images = NUM_IMAGE, cam_index=1):
    cam = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)

    detector = cv2.CascadeClassifier(HAAR_CASCADE)

    count = 0
    print("[INFO] Starting face capture. Look in the camera.")

    if not cam.isOpened():
        raise RuntimeError(f"Cannot open camera at index {cam_index}")

    while True:
        ret, img = cam.read()
        if not ret:
            print("[ERROR] Camera frame not received.")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        # For each detected face crop and save
        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]          # cropped face image
            print("Frame captured, detecting faces...")
            filename = f"{person_name}.{person_id}.{count}.jpg"
            cv2.imwrite(os.path.join(DATASET_DIR, filename), face_img)
            print(f"Saved {filename}")

            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img, f"{count}/{num_images}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # stop early if reached desired number
            if count >= num_images:
                break

        cv2.imshow("Capturing Faces - Press 'q' to quit", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if count >= num_images:
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Collected {count} face images for {person_name}.")
    return count


def train_recognizer():
    print("[INFO] Starting training.")

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except AttributeError:
        raise RuntimeError("Install opencv-contrib-python, not just opencv-python.")
    
    image_paths =[os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR)]
    faces = []
    ids = []

    for image_path in image_paths:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        filename = os.path.basename(image_path)
        try:
            name, id_str, _ = filename.split('.')[:3]
            id_ = int(id_str)
        except:
            print(f"[WARNING] Skipping {filename} - wrong format.")
            continue

        faces.append(img)
        ids.append(id_)

    if not faces:
        print("[ERROR] No images found in dataset.")
        return
    
    recognizer.train(faces, np.array(ids))
    recognizer.write(TRAINER_YML)

    print(f"[INFO] Training complete. Model saved as {TRAINER_YML}")

def mark_attendance(person_id, person_name):
    now = datetime.now()
    date_str = now.strftime("%d-%m-%Y")
    time_str = now.strftime("%H:%M:%S")

    if (not os.path.exists(ATTENDANCE_FILE)) or os.path.getsize(ATTENDANCE_FILE) == 0:
        pd.DataFrame(columns=["ID", "Name", "Date", "Time"]).to_csv(ATTENDANCE_FILE, index=False)
    df = pd.read_csv(ATTENDANCE_FILE)

    if not ((df["ID"] == person_id) & (df["Date"] == date_str)).any():
        new_row = {"ID": person_id, "Name": person_name, "Date": date_str, "Time":time_str}
        df.loc[len(df)] =  new_row
        df.to_csv(ATTENDANCE_FILE, index=False)
        print(f"[INFO] Attendance marked for {person_name} ({person_id}) at {time_str}.")

def recognize_and_mark(cam_index=1):
    if not os.path.exists(TRAINER_YML):
        print("[ERROR] No trained model found. Run train_recognizer() first.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_YML)
    detector = cv2.CascadeClassifier(HAAR_CASCADE)

    id_to_name = {}
    for f in os.listdir(DATASET_DIR):
        parts = f.split(".")
        if len(parts) >= 3:
            try:
                id_ = int(parts[1])
                id_to_name[id_] = parts[0]
            except:
                continue

    cam = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cam.isOpened():
        print(f"[ERROR] Could not open camera index {cam_index}.")
        return

    print("[INFO] Starting recognition. Press 'q' to quit.")
    while True:
        ret, img = cam.read()
        if not ret:
            print("[ERROR] Frame not received.")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            try:
                id_pred, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            except Exception as e:
                # in case predict fails on wrong sized crop
                continue

            if confidence < 70:
                name = id_to_name.get(id_pred, "Unknown")
                mark_attendance(id_pred, name)
                label = f"{name}"
                color = (0, 255, 0)
            else:
                label = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Face Recognition - Press 'q' to quit", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    print("[INFO] Recognition session ended.")


class FaceAttendanceApp:
    def __init__(self, master):
        self.master = master
        master.title("Face Attendance System")
        master.geometry("420x300")
        master.resizable(False, False)

        self.var_id = StringVar()
        self.var_name = StringVar()

        Label(master, text="Face Attendance System", font=("Helvetica", 15, "bold")).pack(pady=10)

        frame = ttk.Frame(master)
        frame.pack(pady=8)
        Label(frame, text="Student ID:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        Entry(frame, textvariable=self.var_id, width=20).grid(row=0, column=1, padx=5, pady=5)
        Label(frame, text="Name:").grid(row=1, column=0, padx=5, pady=5, sticky='e')
        Entry(frame, textvariable=self.var_name, width=20).grid(row=1, column=1, padx=5, pady=5)

        btn_frame = ttk.Frame(master)
        btn_frame.pack(pady=10)
        Button(btn_frame, text="Register Student", width=20, command=self.register).grid(row=0, column=0, padx=5, pady=5)
        Button(btn_frame, text="Train Model", width=20, command=self.train).grid(row=0, column=1, padx=5, pady=5)
        Button(btn_frame, text="Recognize & Mark", width=42, command=self.recognize).grid(row=1, column=0, columnspan=2, pady=5)
        Button(btn_frame, text="View Attendance", width=42, command=self.view_attendance).grid(row=2, column=0, columnspan=2, pady=5)

    def register(self):
        def run_capture():
            try:
                pid = int(self.var_id.get())
                name = self.var_name.get().strip()
                if not name:
                    # show error on main thread
                    self.master.after(0, lambda: messagebox.showerror("Error", "Please enter a name."))
                    return
                count = capture_faces(pid, name, NUM_IMAGE, cam_index=1)  # use cam_index that worked for you
                # show success on main thread
                self.master.after(0, lambda: messagebox.showinfo("Success", f"Captured {count} images for {name}."))
            except Exception as e:
                self.master.after(0, lambda err=e: messagebox.showerror("Error", str(err)))

        threading.Thread(target=run_capture, daemon=True).start()

    def train(self):
        def run_train():
            try:
                train_recognizer()
                self.master.after(0, lambda: messagebox.showinfo("Success", "Training complete."))
            except Exception as e:
                self.master.after(0, lambda err=e: messagebox.showerror("Error", str(err)))
        threading.Thread(target=run_train, daemon=True).start()

    def recognize(self):
        def run_recog():
            try:
                recognize_and_mark(cam_index=1)   # use the index that works for iVCam on your system
                self.master.after(0, lambda: messagebox.showinfo("Done", "Recognition session ended."))
            except Exception as e:
                self.master.after(0, lambda err=e: messagebox.showerror("Error", str(err)))

        threading.Thread(target=run_recog, daemon=True).start()

    def view_attendance(self):
        if not os.path.exists(ATTENDANCE_FILE):
            messagebox.showinfo("Info", "No attendance recorded yet.")
            return
        df = pd.read_csv(ATTENDANCE_FILE)
        top = Toplevel(self.master)
        top.title("Attendance Records")
        tree = ttk.Treeview(top, columns=list(df.columns), show='headings')
        for c in df.columns:
            tree.heading(c, text=c)
            tree.column(c, width=120)
        for _, row in df.iterrows():
            tree.insert("", "end", values=list(row))
        tree.pack(fill='both', expand=True)

if __name__ == "__main__":
    root = Tk()
    app = FaceAttendanceApp(root)
    root.mainloop()
