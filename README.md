# Pehchaan — AI Face Attendance System

Pehchaan is an AI-powered smart attendance management system built using Python, OpenCV, and Tkinter. The system automates attendance tracking through real-time face recognition while providing dedicated dashboards for teachers and students.

It combines computer vision, attendance analytics, PDF report generation, and manual attendance management into a complete desktop application.

---

## Features

### AI Face Recognition

* Real-time face detection and recognition
* Automatic attendance marking
* Unknown face detection
* LBPH Face Recognizer integration
* Haar Cascade face detection

### Student Registration

* Capture multiple face samples
* Automatic dataset creation
* Training-ready image storage

### Teacher Dashboard

* Register students
* Train recognition model
* Start attendance recognition
* View monthly attendance
* Attendance analytics and charts
* Manual attendance management
* Add/Edit/Delete attendance records

### Student Dashboard

* View attendance summary
* Subject-wise attendance tracking
* Attendance analytics charts
* Export attendance reports as PDF

### Attendance System

* Subject-wise attendance tracking
* Teacher-specific attendance records
* Monthly CSV attendance sheets
* Attendance percentage calculation
* Attendance visualization using charts

---

## Technologies Used

* Python
* OpenCV
* Tkinter
* NumPy
* Pandas
* Matplotlib
* Pillow (PIL)

---

## Project Structure

```bash
Pehchaan/
│
├── dataset/                     # Face image dataset
├── teachers/                    # Teacher attendance folders
│
├── trainer.yml                  # Trained recognition model
├── students.csv                 # Student database
├── teachers.csv                 # Teacher database
│
├── main.py                      # Main application file
└── README.md
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/yourusername/pehchaan.git
cd pehchaan
```

### Install Dependencies

```bash
pip install opencv-contrib-python
pip install numpy pandas matplotlib pillow
```

### Run the Application

```bash
python main.py
```

---

## How It Works

### 1. Register Student

Teachers register students using:

* Student ID
* Student Name

The system captures multiple facial images using a webcam.

### 2. Train Recognition Model

The captured images are processed and trained using OpenCV's LBPH Face Recognizer.

### 3. Start Attendance Recognition

The webcam scans faces in real time and:

* Identifies registered students
* Marks attendance automatically
* Stores attendance records in CSV files

---

## Attendance File Format

Attendance files are stored as:

```bash
teachers/
   Teacher_Name/
      attendance_Subject_MM-YYYY.csv
```

Each file contains:

* Student ID
* Student Name
* Date
* Time

---

## Key Highlights

* Real-Time Face Recognition
* Automatic Attendance Tracking
* Teacher & Student Dashboards
* Attendance Analytics
* PDF Report Export
* Manual Attendance Management
* Subject-Wise Attendance System
* CSV-Based Data Storage
* Multi-User Login System

---

## Future Improvements

* Cloud Database Integration
* Online Attendance System
* Deep Learning Face Recognition
* Mobile Application Support
* Email Notifications
* Admin Control Panel
* QR + Face Hybrid Authentication

---

## Use Cases

* Schools
* Colleges
* Coaching Institutes
* Training Centers
* Offices and Organizations

---

## Author

**Abu Huraira**

---

## License

This project is licensed under the MIT License.

---

## Support

If you found this project useful, consider giving it a star on GitHub.
