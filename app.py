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
BACKGROUND_IMAGE = "background.png"
FACE_CASCADE_PATH = 'haarcascade_frontalface_default.xml'

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
def total_registered_users():
   return len(os.listdir(faces_dir))

def extract_faces(img):
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   face_points = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))
   return face_points

def identify_face(face_array):
   model = joblib.load(os.path.join(static_dir, 'face_recognition_model.pkl'))
   return model.predict(face_array)

def train_model():
   faces = []
   labels = []
   user_list = os.listdir(faces_dir)
   for user in user_list:
       user_dir = os.path.join(faces_dir, user)
       for image_file in os.listdir(user_dir):
           img = cv2.imread(os.path.join(user_dir, image_file))
           resized_face = cv2.resize(img, (50, 50))
           faces.append(resized_face.ravel())
           labels.append(user)
   faces = np.array(faces)
   knn = KNeighborsClassifier(n_neighbors=5)
   knn.fit(faces, labels)
   joblib.dump(knn, os.path.join(static_dir, 'face_recognition_model.pkl'))

def extract_attendance():
   df = pd.read_csv(attendance_file)
   names = df['Name']
   rolls = df['Roll']
   times = df['Time']
   return names, rolls, times, len(df)

def add_attendance(name):
   username, user_id = name.split('_')
   current_time = datetime.now().strftime("%H:%M:%S")

   df = pd.read_csv(attendance_file)
   if int(user_id) not in list(df['Roll']):
       with open(attendance_file, 'a') as f:
           f.write(f'\n{username},{user_id},{current_time}')

def get_all_users():
   user_list = os.listdir(faces_dir)
   names = []
   rolls = []
   for user in user_list:
       name, roll = user.split('_')
       names.append(name)
       rolls.append(roll)
   return user_list, names, rolls, len(user_list)

# Flask routes
@app.route('/')
def home():
   names, rolls, times, attendance_count = extract_attendance()
   return render_template('home.html', names=names, rolls=rolls, times=times, l=attendance_count,
                          totalreg=total_registered_users(), datetoday2=attendance_date)

@app.route('/start', methods=['GET'])
def start():
   names, rolls, times, attendance_count = extract_attendance()

   if 'face_recognition_model.pkl' not in os.listdir(static_dir):
       return render_template('home.html', names=names, rolls=rolls, times=times, l=attendance_count,
                              totalreg=total_registered_users(), datetoday2=attendance_date,
                              mess='There is no trained model in the static folder. Please add a new face to continue.')

   ret = True
   cap = cv2.VideoCapture(0)
   background_img = cv2.imread(BACKGROUND_IMAGE)
   while ret:
       ret, frame = cap.read()
       if len(extract_faces(frame)) > 0:
           x, y, w, h = extract_faces(frame)[0]
           cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
           cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
           face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
           identified_person = identify_face(face.reshape(1, -1))[0]
           add_attendance(identified_person)
           cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
           cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
           cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
           cv2.putText(frame, f'{identified_person}', (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
           cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
       background_img[162:162+480, 55:55+640] = frame
       cv2.imshow('Attendance', background_img)
       if cv2.waitKey(1) == 27:
           break
   cap.release()
   cv2.destroyAllWindows()
   names, rolls, times, attendance_count = extract_attendance()
   return render_template('home.html', names=names, rolls=rolls, times=times, l=attendance_count,
                          totalreg=total_registered_users(), datetoday2=attendance_date)

@app.route('/add', methods=['GET', 'POST'])
def add():
   new_username = request.form['newusername']
   new_user_id = request.form['newuserid']
   user_image_folder = os.path.join(faces_dir, f'{new_username}_{new_user_id}')
   os.makedirs(user_image_folder, exist_ok=True)
   images_captured = 0
   capture_count = 0
   cap = cv2.VideoCapture(0)
   while True:
       _, frame = cap.read()
       faces = extract_faces(frame)
       for (x, y, w, h) in faces:
           cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
           cv2.putText(frame, f'Images Captured: {images_captured}/{IMAGES_TO_CAPTURE}', (30, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
           if capture_count % 5 == 0:
               image_name = f"{new_username}_{images_captured}.jpg"
               cv2.imwrite(os.path.join(user_image_folder, image_name), frame[y:y+h, x:x+w])
               images_captured += 1
           capture_count += 1
       if images_captured == IMAGES_TO_CAPTURE:
           break
       cv2.imshow('Adding new User', frame)
       if cv2.waitKey(1) == 27:
           break
   cap.release()
   cv2.destroyAllWindows()
   print('Training Model')
   train_model()
   names, rolls, times, attendance_count = extract_attendance()
   return render_template('home.html', names=names, rolls=rolls, times=times, l=attendance_count,
                          totalreg=total_registered_users(), datetoday2=attendance_date)

if __name__ == '__main__':
   app.run(debug=True)