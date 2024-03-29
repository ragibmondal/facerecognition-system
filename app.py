import cv2
import os
from flask import Flask, request, render_template
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

app = Flask(__name__)

# Constants
IMAGES_TO_CAPTURE = 10
BACKGROUND_IMAGE = os.path.join('static', 'background.png')
FACE_CASCADE_PATH = os.path.join('static', 'haarcascade_frontalface_default.xml')

# Set up directories and files
today = date.today().strftime("%m_%d_%y")
attendance_date = date.today().strftime("%d-%B-%Y")

attendance_dir = 'Attendance'
os.makedirs(attendance_dir, exist_ok=True)

static_dir = 'static'
os.makedirs(static_dir, exist_ok=True)

faces_dir = os.path.join(static_dir, 'faces')
os.makedirs(faces_dir, exist_ok=True)

attendance_file = os.path.join(attendance_dir, f'Attendance-{today}.csv')
if not os.path.exists(attendance_file):
   with open(attendance_file, 'w') as f:
       f.write('Name,Roll,Time')

# Load face detector
face_detector = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# Helper functions
# ... (same as before)

# Flask routes
@app.route('/')
def home():
   names, rolls, times, attendance_count = extract_attendance()
   return render_template('home.html', names=names, rolls=rolls, times=times, l=attendance_count,
                          totalreg=total_registered_users(), datetoday2=attendance_date)

@app.route('/start', methods=['GET'])
def start():
   # ... (same as before)

@app.route('/add', methods=['GET', 'POST'])
def add():
   # ... (same as before)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000))
