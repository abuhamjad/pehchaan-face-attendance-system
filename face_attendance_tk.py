import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from tkinter import Tk, Label, Entry, Button, StringVar, messagebox, Toplevel, ttk
from PIL import Image, ImageTk
import threading
import matplotlib.pyplot as plt
from calendar import monthrange

def safe_call(master, func, *args, **kwargs):
    """Safely call Tkinter functions from a thread if the mainloop is still alive."""
    try:
        if master and master.winfo_exists():
            master.after(0, func, *args, **kwargs)
    except RuntimeError:
        pass

DATASET_DIR = "dataset"
TRAINER_YML = "trainer.yml"
ATTENDANCE_FILE = "attendance.csv"
HAAR_CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
NUM_IMAGE = 60

if not os.path.exists(DATASET_DIR):
    os.mkdir(DATASET_DIR)

def capture_faces(person_id, person_name, num_images=NUM_IMAGE, cam_index=1):
    cam = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    detector = cv2.CascadeClassifier(HAAR_CASCADE)

    if not cam.isOpened():
        raise RuntimeError(f"Cannot open camera at index {cam_index}")

    count = 0
    print("[INFO] Starting face capture. Look in the camera.")
    while True:
        ret, img = cam.read()
        if not ret:
            print("[ERROR] Camera frame not received.")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]
            filename = f"{person_name}.{person_id}.{count}.jpg"
            cv2.imwrite(os.path.join(DATASET_DIR, filename), face_img)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img, f"{count}/{num_images}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if count >= num_images:
                break

        cv2.imshow("Capturing Faces - Press 'q' to quit", img)
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= num_images:
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

    image_paths = [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR)]
    faces, ids = [], []

    for image_path in image_paths:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        try:
            _, id_str, _ = os.path.basename(image_path).split('.')[:3]
            ids.append(int(id_str))
            faces.append(img)
        except:
            print(f"[WARNING] Skipping {image_path} (wrong format)")

    if not faces:
        print("[ERROR] No images found in dataset.")
        return

    recognizer.train(faces, np.array(ids))
    recognizer.write(TRAINER_YML)
    print(f"[INFO] Training complete. Model saved as {TRAINER_YML}")

def mark_attendance(person_id, person_name, teacher, subject):
    """Mark attendance in teacher-specific monthly file inside their folder."""
    now = datetime.now()
    date_str = now.strftime("%d-%m-%Y")
    time_str = now.strftime("%H:%M:%S")
    month_str = now.strftime("%m-%Y")

    # Path: teachers/<teacher_name>/
    teacher_dir = os.path.join("teachers", teacher)
    os.makedirs(teacher_dir, exist_ok=True)

    # File path: teachers/<teacher>/attendance_<subject>_<month-year>.csv
    file_name = os.path.join(teacher_dir, f"attendance_{subject}_{month_str}.csv")

    if not os.path.exists(file_name) or os.path.getsize(file_name) == 0:
        pd.DataFrame(columns=["ID", "Name", "Date", "Time"]).to_csv(file_name, index=False)

    df = pd.read_csv(file_name)
    if not ((df["ID"] == person_id) & (df["Date"] == date_str)).any():
        df.loc[len(df)] = {"ID": person_id, "Name": person_name, "Date": date_str, "Time": time_str}
        df.to_csv(file_name, index=False)
        print(f"[INFO] Attendance marked for {person_name} ({person_id}) → {subject} by {teacher}.")


def recognize_and_mark(cam_index=1, teacher=None, subject=None):
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
                id_to_name[int(parts[1])] = parts[0]
            except:
                continue

    cam = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cam.isOpened():
        print(f"[ERROR] Could not open camera index {cam_index}.")
        return

    print(f"[INFO] Recognizing students for {subject} - {teacher}. Press 'q' to quit.")
    while True:
        ret, img = cam.read()
        if not ret:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            try:
                id_pred, conf = recognizer.predict(gray[y:y+h, x:x+w])
            except:
                continue
            if conf < 70:
                name = id_to_name.get(id_pred, "Unknown")
                mark_attendance(id_pred, name, teacher, subject)
                color, label = (0, 255, 0), name
            else:
                color, label = (0, 0, 255), "Unknown"
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Face Recognition", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    print("[INFO] Recognition session ended.")


class LoginApp:
    def __init__(self, master):
        self.master = master
        master.title("Login Portal - Face Attendance System")
        master.geometry("380x260")
        master.resizable(False, False)

        self.var_user = StringVar()
        self.var_pass = StringVar()
        self.var_role = StringVar(value="student")

        Label(master, text="Login Portal", font=("Helvetica", 15, "bold")).pack(pady=10)
        frame_role = ttk.Frame(master)
        frame_role.pack(pady=5)
        Label(frame_role, text="Login as:").grid(row=0, column=0, padx=5)
        ttk.Radiobutton(frame_role, text="Student", variable=self.var_role, value="student").grid(row=0, column=1)
        ttk.Radiobutton(frame_role, text="Teacher", variable=self.var_role, value="teacher").grid(row=0, column=2)

        frame = ttk.Frame(master)
        frame.pack(pady=10)
        Label(frame, text="Username:").grid(row=0, column=0, padx=5, pady=5)
        Entry(frame, textvariable=self.var_user, width=25).grid(row=0, column=1)
        Label(frame, text="Password:").grid(row=1, column=0, padx=5, pady=5)
        Entry(frame, textvariable=self.var_pass, width=25, show="*").grid(row=1, column=1)

        Button(master, text="Login", width=15, command=self.check_login).pack(pady=10)
        Button(master, text="Register New User", width=20, command=self.register_user).pack()

    def check_login(self):
        user = self.var_user.get().strip()
        pw = self.var_pass.get().strip()
        role = self.var_role.get()
        filename = "teachers.csv" if role == "teacher" else "students.csv"

        if not os.path.exists(filename):
            messagebox.showerror("Error", f"{filename} not found.")
            return

        df = pd.read_csv(filename, dtype=str)
        if user not in df["username"].values:
            messagebox.showerror("Error", f"{role.capitalize()} username not found.")
            return

        stored_pw = df.loc[df["username"] == user, "password"].values[0]
        name = df.loc[df["username"] == user, "name"].values[0]

        if pw == stored_pw:
            messagebox.showinfo("Success", f"Welcome, {name} ({role.capitalize()})!")
            self.master.destroy()
            root2 = Tk()
            if role == "teacher":
                subject = df.loc[df["username"] == user, "subject"].values[0]
                FaceAttendanceApp(root2, teacher_name=name, subject=subject)
            else:
                StudentApp(root2, student_name=name)
            root2.mainloop()

    def register_user(self):
        role = self.var_role.get()
        reg_window = Toplevel(self.master)
        reg_window.title(f"Register New {role.capitalize()}")
        reg_window.geometry("360x320")

        new_user, new_pass, new_name, new_subject = StringVar(), StringVar(), StringVar(), StringVar()
        Label(reg_window, text=f"Register New {role.capitalize()} Account", font=("Helvetica", 13, "bold")).pack(pady=10)
        frame = ttk.Frame(reg_window)
        frame.pack(pady=10)
        Label(frame, text="Username:").grid(row=0, column=0, padx=5, pady=5)
        Entry(frame, textvariable=new_user, width=25).grid(row=0, column=1)
        Label(frame, text="Password:").grid(row=1, column=0, padx=5, pady=5)
        Entry(frame, textvariable=new_pass, width=25, show="*").grid(row=1, column=1)
        Label(frame, text="Full Name:").grid(row=2, column=0, padx=5, pady=5)
        Entry(frame, textvariable=new_name, width=25).grid(row=2, column=1)

        if role == "teacher":
            Label(frame, text="Subject:").grid(row=3, column=0, padx=5, pady=5)
            Entry(frame, textvariable=new_subject, width=25).grid(row=3, column=1)

        def save_user():
            user_val = new_user.get().strip()
            pass_val = new_pass.get().strip()
            name_val = new_name.get().strip()
            subj_val = new_subject.get().strip() if role == "teacher" else ""

            if not user_val or not pass_val or not name_val or (role == "teacher" and not subj_val):
                messagebox.showerror("Error", "All fields are required.")
                return

            filename = "teachers.csv" if role == "teacher" else "students.csv"
            if not os.path.exists(filename):
                columns = ["username", "password", "name", "subject"] if role == "teacher" else ["username", "password", "name"]
                pd.DataFrame(columns=columns).to_csv(filename, index=False)

            df = pd.read_csv(filename)
            if user_val in df["username"].values:
                messagebox.showerror("Error", "Username already exists.")
                return

            # Save user record
            row = {"username": user_val, "password": pass_val, "name": name_val}
            if role == "teacher":
                row["subject"] = subj_val
            df.loc[len(df)] = row
            df.to_csv(filename, index=False)

            # NEW: create teacher-specific folder
            if role == "teacher":
                teacher_dir = os.path.join("teachers", name_val)
                os.makedirs(teacher_dir, exist_ok=True)
                print(f"[INFO] Created folder for {name_val} -> {teacher_dir}")

            messagebox.showinfo("Success", f"New {role} registered successfully!")
            reg_window.destroy()

        Button(reg_window, text="Save User", width=18, command=save_user).pack(pady=15)

class StudentApp:
    def __init__(self, master, student_name):
        self.master = master
        self.student_name = student_name
        master.title(f"Student Dashboard - {student_name}")
        master.geometry("560x430")
        master.resizable(False, False)

        Label(master, text=f"Welcome, {student_name}", font=("Helvetica", 15, "bold")).pack(pady=10)
        Button(master, text="View Attendance Summary", width=25, command=self.view_summary).pack(pady=10)
        Button(master, text="Show Attendance Chart", width=25, command=self.show_chart).pack(pady=10)
        Button(master, text="Export Attendance as PDF", width=25, command=self.export_pdf).pack(pady=10)
        Button(master, text="Logout / Back to Login", width=25, command=self.logout).pack(pady=10)

        # will store the last summary for chart/pdf export
        self.last_summary = []

    def logout(self):
        self.master.destroy()
        root = Tk()
        LoginApp(root)
        root.mainloop()

    def view_summary(self):
        base_dir = "teachers"
        if not os.path.exists(base_dir):
            messagebox.showinfo("Info", "No attendance records found yet.")
            return

        records = []
        for teacher_name in os.listdir(base_dir):
            teacher_path = os.path.join(base_dir, teacher_name)
            if not os.path.isdir(teacher_path):
                continue

            # Find all attendance CSV files for this teacher
            for file in os.listdir(teacher_path):
                if file.startswith("attendance_") and file.endswith(".csv"):
                    file_path = os.path.join(teacher_path, file)
                    try:
                        df = pd.read_csv(file_path)
                        if "Name" not in df.columns or df.empty:
                            continue
                    except Exception:
                        continue

                    # Extract subject and month from file name
                    name_parts = file.replace("attendance_", "").replace(".csv", "").split("_")
                    if len(name_parts) >= 2:
                        subject = name_parts[0]
                        month_year = name_parts[1]
                    else:
                        subject, month_year = "Unknown", "Unknown"

                    # Filter this student's data
                    total_lectures = len(df["Date"].unique())
                    attended = len(df[df["Name"] == self.student_name]["Date"].unique())
                    absent = total_lectures - attended
                    percent = (attended / total_lectures * 100) if total_lectures > 0 else 0

                    records.append((subject, teacher_name, month_year, total_lectures, attended, absent, f"{percent:.2f}%"))

        if not records:
            messagebox.showinfo("Info", "No attendance data found for this student.")
            return

        # Save records for chart view
        self.last_summary = records

        # Create summary window
        top = Toplevel(self.master)
        top.title(f"Attendance Summary - {self.student_name}")
        cols = ["Subject", "Teacher", "Month-Year", "Total Lectures", "Attended", "Absent", "Attendance %"]
        tree = ttk.Treeview(top, columns=cols, show='headings')

        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=110, anchor="center")

        for r in records:
            tree.insert("", "end", values=r)

        tree.pack(fill='both', expand=True)


    def show_chart(self):
        if not self.last_summary:
            messagebox.showwarning("Info", "Please view attendance summary first.")
            return

        subjects = [r[0] for r in self.last_summary]
        percentages = [r[-1] for r in self.last_summary]

        # Convert all percentage strings like '85.23%' -> float 85.23
        cleaned_percentages = []
        for p in percentages:
            if isinstance(p, str):
                p = float(p.replace('%', '').strip())
            cleaned_percentages.append(p)
        percentages = cleaned_percentages

        # Set bar colors based on percentage
        colors = []
        for p in percentages:
            if p >= 80:
                colors.append("green")
            elif 60 <= p < 80:
                colors.append("gold")
            else:
                colors.append("red")

        avg_attendance = sum(percentages) / len(percentages) if percentages else 0

        plt.figure(figsize=(8, 4.5))
        bars = plt.bar(subjects, percentages, color=colors)
        plt.axhline(y=avg_attendance, color='blue', linestyle='--', linewidth=1.5, label=f'Average: {avg_attendance:.2f}%')

        plt.xlabel("Subjects")
        plt.ylabel("Attendance (%)")
        plt.title(f"Attendance Overview - {self.student_name}")
        plt.ylim(0, 100)
        plt.legend()

        # Label bars
        for bar, val in zip(bars, percentages):
            plt.text(bar.get_x() + bar.get_width()/2, val + 1, f"{val:.1f}%", ha='center', fontsize=9)

        plt.tight_layout()
        plt.show()

    def export_pdf(self):
        if not self.last_summary:
            messagebox.showwarning("Info", "Please view attendance summary first.")
            return

        subjects = [r[0] for r in self.last_summary]
        percentages = [r[-1] for r in self.last_summary]

        colors = []
        # Convert all percentage strings like '85.23%' -> float 85.23
        cleaned_percentages = []
        for p in percentages:
            if isinstance(p, str):
                p = float(p.replace('%', '').strip())
            cleaned_percentages.append(p)
        percentages = cleaned_percentages


        avg_attendance = sum(percentages) / len(percentages) if percentages else 0

        plt.figure(figsize=(8, 4.5))
        bars = plt.bar(subjects, percentages, color=colors)
        plt.axhline(y=avg_attendance, color='blue', linestyle='--', linewidth=1.5, label=f'Average: {avg_attendance:.2f}%')

        plt.xlabel("Subjects")
        plt.ylabel("Attendance (%)")
        plt.title(f"Attendance Report - {self.student_name}")
        plt.ylim(0, 100)
        plt.legend()

        for bar, val in zip(bars, percentages):
            plt.text(bar.get_x() + bar.get_width()/2, val + 1, f"{val:.1f}%", ha='center', fontsize=9)

        plt.tight_layout()

        filename = f"{self.student_name.replace(' ', '_')}_Attendance_Report.pdf"
        plt.savefig(filename)
        plt.close()
        messagebox.showinfo("Success", f"Attendance chart exported as '{filename}'")


class FaceAttendanceApp:
    def __init__(self, master, teacher_name, subject):
        self.master = master
        self.teacher_name = teacher_name
        self.subject = subject
        master.title(f"Teacher Dashboard - {teacher_name} ({subject})")
        master.geometry("420x370")
        master.resizable(True, True)

        self.var_id = StringVar()
        self.var_name = StringVar()

        Label(master, text="Teacher Dashboard", font=("Helvetica", 15, "bold")).pack(pady=10)
        frame = ttk.Frame(master)
        frame.pack(pady=8)
        Label(frame, text="Student ID:").grid(row=0, column=0, padx=5, pady=5)
        Entry(frame, textvariable=self.var_id, width=20).grid(row=0, column=1)
        Label(frame, text="Name:").grid(row=1, column=0, padx=5, pady=5)
        Entry(frame, textvariable=self.var_name, width=20).grid(row=1, column=1)

        btn_frame = ttk.Frame(master)
        btn_frame.pack(pady=10)
        Button(btn_frame, text="Register Student", width=20, command=self.register).grid(row=0, column=0, padx=5)
        Button(btn_frame, text="Train Model", width=20, command=self.train).grid(row=0, column=1, padx=5)
        Button(btn_frame, text="Recognize & Mark", width=42, command=self.recognize).grid(row=1, column=0, columnspan=2, pady=5)
        Button(btn_frame, text="View Attendance", width=42, command=self.view_attendance).grid(row=2, column=0, columnspan=2, pady=5)
        Button(btn_frame, text="View Attendance Chart", width=42, command=self.view_attendance_chart).grid(row=3, column=0, columnspan=2, pady=5)
        Button(btn_frame, text="Manage Attendance", width=42, command=self.manage_attendance).grid(row=4, column=0, columnspan=2, pady=5)
        Button(btn_frame, text="Logout / Back to Login", width=42, command=self.logout).grid(row=5, column=0, columnspan=2, pady=10)

    def register(self):
        def run_capture():
            try:
                pid = int(self.var_id.get())
                name = self.var_name.get().strip()
                if not name:
                    safe_call(self.master, lambda: messagebox.showerror("Error", "Please enter a name."))
                    return
                count = capture_faces(pid, name, NUM_IMAGE, cam_index=1)
                safe_call(self.master, lambda: messagebox.showinfo("Success", f"Captured {count} images for {name}."))
            except Exception as e:
                safe_call(self.master, lambda: messagebox.showerror("Error", str(e)))
        threading.Thread(target=run_capture, daemon=True).start()

    def train(self):
        def run_train():
            try:
                train_recognizer()
                safe_call(self.master, lambda: messagebox.showinfo("Success", "Training complete."))
            except Exception as e:
                safe_call(self.master, lambda: messagebox.showerror("Error", str(e)))
        threading.Thread(target=run_train, daemon=True).start()

    def recognize(self):
        def run_recog():
            try:
                recognize_and_mark(cam_index=1, teacher=self.teacher_name, subject=self.subject)
                safe_call(self.master, lambda: messagebox.showinfo("Done", "Recognition session ended."))
            except Exception as e:
                safe_call(self.master, lambda: messagebox.showerror("Error", str(e)))
        threading.Thread(target=run_recog, daemon=True).start()

    def view_attendance(self):
        now = datetime.now()
        month_str = now.strftime("%m-%Y")
        teacher_dir = os.path.join("teachers", self.teacher_name)
        file_name = os.path.join(teacher_dir, f"attendance_{self.subject}_{month_str}.csv")

        if not os.path.exists(file_name):
            messagebox.showinfo("Info", f"No attendance recorded yet for {self.subject} ({month_str}).")
            return

        df = pd.read_csv(file_name)
        if df.empty:
            messagebox.showinfo("Info", "No attendance records found.")
            return

        df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
        df = df[(df["Date"].dt.month == now.month) & (df["Date"].dt.year == now.year)]

        if df.empty:
            messagebox.showinfo("Info", "No attendance data for this month yet.")
            return

        students = df["Name"].unique()
        _, days_in_month = monthrange(now.year, now.month)
        all_dates = [datetime(now.year, now.month, d) for d in range(1, days_in_month + 1)]

        records = []
        for student in students:
            row = [student]
            for date_obj in all_dates:
                date_str = date_obj.strftime("%d-%m")
                weekday = date_obj.strftime("%A")

                if weekday in ["Saturday", "Sunday"]:
                    row.append(weekday)
                elif date_obj > now:
                    # Future dates remain blank
                    row.append("")
                else:
                    present = df[(df["Name"] == student) & (df["Date"].dt.strftime("%d-%m") == date_str)]
                    row.append("✅" if not present.empty else "❌")
            records.append(row)

        top = Toplevel(self.master)
        top.title(f"Monthly Attendance - {self.subject} ({month_str})")
        cols = ["Name"] + [d.strftime("%d-%m") for d in all_dates]
        tree = ttk.Treeview(top, columns=cols, show='headings')
        tree.heading("Name", text="Name")
        tree.column("Name", width=120)
        for d in all_dates:
            ds = d.strftime("%d-%m")
            tree.heading(ds, text=ds)
            tree.column(ds, width=60, anchor="center")
        for r in records:
            tree.insert("", "end", values=r)
        h_scroll = ttk.Scrollbar(top, orient="horizontal", command=tree.xview)
        tree.configure(xscrollcommand=h_scroll.set)
        tree.pack(fill='both', expand=True)
        h_scroll.pack(fill='x')

    def view_attendance_chart(self):
        now = datetime.now()
        month_str = now.strftime("%m-%Y")
        teacher_dir = os.path.join("teachers", self.teacher_name)
        file_name = os.path.join(teacher_dir, f"attendance_{self.subject}_{month_str}.csv")

        if not os.path.exists(file_name):
            messagebox.showinfo("Info", f"No attendance data found for {self.subject} ({month_str}).")
            return

        df = pd.read_csv(file_name)
        if df.empty:
            messagebox.showinfo("Info", "No attendance records found.")
            return

        df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
        df = df[(df["Date"].dt.month == now.month) & (df["Date"].dt.year == now.year)]

        if df.empty:
            messagebox.showinfo("Info", "No attendance records for this month yet.")
            return

        total_days = len(df["Date"].unique())
        attendance_summary = df.groupby("Name")["Date"].nunique().reset_index()
        attendance_summary["Percentage"] = (attendance_summary["Date"] / total_days) * 100

        students = attendance_summary["Name"].tolist()
        percentages = attendance_summary["Percentage"].tolist()

        # Color-code based on attendance %
        colors = []
        for p in percentages:
            if p >= 80:
                colors.append("green")
            elif 60 <= p < 80:
                colors.append("gold")
            else:
                colors.append("red")

        avg_attendance = attendance_summary["Percentage"].mean()
        top_student = attendance_summary.loc[attendance_summary["Percentage"].idxmax(), "Name"]
        top_percent = attendance_summary["Percentage"].max()

        plt.figure(figsize=(9, 5))
        bars = plt.bar(students, percentages, color=colors)
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.xlabel("Students", fontsize=10, fontweight='bold')
        plt.ylabel("Attendance (%)", fontsize=10, fontweight='bold')
        plt.title(f"{self.subject} Attendance - {self.teacher_name} ({month_str})", fontsize=12, fontweight='bold')

        # Add percentage labels above bars
        for bar, p in zip(bars, percentages):
            plt.text(bar.get_x() + bar.get_width() / 2, p + 1, f"{p:.1f}%", ha='center', fontsize=8)

        # Add average line
        plt.axhline(y=avg_attendance, color='blue', linestyle='--', linewidth=1.5)
        plt.text(len(students) - 0.5, avg_attendance + 1, f'Class Avg: {avg_attendance:.2f}%', color='blue', fontsize=9)

        plt.ylim(0, 100)
        plt.tight_layout()
        plt.show()


    def manage_attendance(self):
        teacher_dir = os.path.join("teachers", self.teacher_name)
        if not os.path.exists(teacher_dir):
            messagebox.showinfo("Info", "No attendance folder found for this teacher yet.")
            return

        files = [f for f in os.listdir(teacher_dir) if f.startswith(f"attendance_{self.subject}_") and f.endswith(".csv")]
        if not files:
            messagebox.showinfo("Info", "No attendance files found for this subject yet.")
            return

        top = Toplevel(self.master)
        top.title(f"Manual Attendance Editor - {self.subject}")
        top.geometry("700x400")

        Label(top, text=f"Manage Attendance - {self.subject}", font=("Helvetica", 13, "bold")).pack(pady=8)

        # Dropdown to select month file
        selected_file = StringVar(value=files[-1])  # default = latest file
        file_menu = ttk.Combobox(top, textvariable=selected_file, values=files, state="readonly", width=45)
        file_menu.pack(pady=5)

        # Treeview to show data
        cols = ["ID", "Name", "Date", "Time"]
        tree = ttk.Treeview(top, columns=cols, show='headings')
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=150)
        tree.pack(fill='both', expand=True, padx=10, pady=5)

        def load_data():
            file_path = os.path.join(teacher_dir, selected_file.get())
            if not os.path.exists(file_path):
                messagebox.showerror("Error", "Selected file not found.")
                return
            tree.delete(*tree.get_children())
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                tree.insert("", "end", values=list(row))

        def refresh_data(*args):
            load_data()
        file_menu.bind("<<ComboboxSelected>>", refresh_data)

        load_data()

        # Helper: save tree back to CSV
        def save_to_csv():
            items = [tree.item(i)['values'] for i in tree.get_children()]
            if not items:
                return
            df = pd.DataFrame(items, columns=cols)
            file_path = os.path.join(teacher_dir, selected_file.get())
            df.to_csv(file_path, index=False)
            print(f"[INFO] Saved updated attendance to {file_path}")

        # Add record
        def add_record():
            win = Toplevel(top)
            win.title("Add Attendance Record")
            win.geometry("300x220")

            id_var, name_var, date_var, time_var = StringVar(), StringVar(), StringVar(value=datetime.now().strftime("%d-%m-%Y")), StringVar(value=datetime.now().strftime("%H:%M:%S"))

            Label(win, text="Student ID:").pack(pady=3)
            Entry(win, textvariable=id_var, width=25).pack()
            Label(win, text="Student Name:").pack(pady=3)
            Entry(win, textvariable=name_var, width=25).pack()
            Label(win, text="Date (dd-mm-yyyy):").pack(pady=3)
            Entry(win, textvariable=date_var, width=25).pack()
            Label(win, text="Time (HH:MM:SS):").pack(pady=3)
            Entry(win, textvariable=time_var, width=25).pack()

            def save_new():
                if not id_var.get() or not name_var.get() or not date_var.get():
                    messagebox.showerror("Error", "All fields required.")
                    return
                tree.insert("", "end", values=[id_var.get(), name_var.get(), date_var.get(), time_var.get()])
                save_to_csv()
                win.destroy()

            Button(win, text="Add Record", command=save_new).pack(pady=10)

        # Edit record
        def edit_record():
            sel = tree.selection()
            if not sel:
                messagebox.showerror("Error", "Select a record to edit.")
                return
            values = tree.item(sel[0])["values"]

            win = Toplevel(top)
            win.title("Edit Attendance Record")
            win.geometry("300x220")

            id_var, name_var, date_var, time_var = StringVar(value=values[0]), StringVar(value=values[1]), StringVar(value=values[2]), StringVar(value=values[3])

            Label(win, text="Student ID:").pack(pady=3)
            Entry(win, textvariable=id_var, width=25).pack()
            Label(win, text="Student Name:").pack(pady=3)
            Entry(win, textvariable=name_var, width=25).pack()
            Label(win, text="Date (dd-mm-yyyy):").pack(pady=3)
            Entry(win, textvariable=date_var, width=25).pack()
            Label(win, text="Time (HH:MM:SS):").pack(pady=3)
            Entry(win, textvariable=time_var, width=25).pack()

            def save_edit():
                new_vals = [id_var.get(), name_var.get(), date_var.get(), time_var.get()]
                tree.item(sel[0], values=new_vals)
                save_to_csv()
                win.destroy()

            Button(win, text="Save Changes", command=save_edit).pack(pady=10)

        # Delete record
        def delete_record():
            sel = tree.selection()
            if not sel:
                messagebox.showerror("Error", "Select a record to delete.")
                return
            confirm = messagebox.askyesno("Confirm", "Are you sure you want to delete this record?")
            if confirm:
                tree.delete(sel[0])
                save_to_csv()

        # Buttons
        btn_frame = ttk.Frame(top)
        btn_frame.pack(pady=8)
        Button(btn_frame, text="Add Record", command=add_record, width=15).grid(row=0, column=0, padx=5)
        Button(btn_frame, text="Edit Record", command=edit_record, width=15).grid(row=0, column=1, padx=5)
        Button(btn_frame, text="Delete Record", command=delete_record, width=15).grid(row=0, column=2, padx=5)
        Button(btn_frame, text="Refresh", command=load_data, width=15).grid(row=0, column=3, padx=5)


    def logout(self):
        self.master.destroy()
        root = Tk()
        LoginApp(root)
        root.mainloop()

if __name__ == "__main__":
    root = Tk()
    app = LoginApp(root)
    root.mainloop()
