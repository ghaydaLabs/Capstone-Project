from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2
import matplotlib.pyplot as plt
import serial
import time
#from tensorflow.keras.models import load_model
import numpy as np

# ربط الأردوينو
# محاولة فتح المنفذ التسلسلي، أو التخطي في حال عدم وجوده
try:
    arduino = serial.Serial('/dev/cu.usbmodemXXXX', 9600, timeout=1)  # غيري XXXX إذا عندك منفذ فعلي لاحقًا
    time.sleep(2)
except Exception as e:
    print("⚠️ لم يتم العثور على منفذ أردوينو. سيتم تخطي الاتصال.")
    arduino = None

# تحميل صوت التنبيه
mixer.init()
mixer.music.load("music.wav")


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


# إعدادات الكشف
thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor(
    "models/shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

cap = cv2.VideoCapture(0)
flag = 0
ear_values = []
frame_count = []

last_state = None  # لحفظ آخر حالة تم إرسالها

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        ear_values.append(ear)
        frame_count.append(len(ear_values))

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < thresh:
            flag += 1
            if flag >= frame_check:
                cv2.putText(frame, "*************Please wake up!*****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "*************Please wake up!*****************", (10, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if last_state != "ALERT" and arduino:
                    arduino.write(b'ALERT\n')
                    print("Sent: ALERT")
                    last_state = "ALERT"

                mixer.music.play()
        else:
            flag = 0
            if last_state != "SAFE" and arduino:
                arduino.write(b'SAFE\n')
                print("Sent: SAFE")
                last_state = "SAFE"

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break


def plot_ear_curve(frame_count, ear_values):
    plt.figure(figsize=(10, 4))
    plt.plot(frame_count, ear_values, color='blue')
    plt.xlabel('Frame')
    plt.ylabel('EAR')
    plt.title('EAR over Time')
    plt.grid(True)
    plt.show()
print("Total EAR values:", len(ear_values))
print("Total Frames:", len(frame_count))

plot_ear_curve(frame_count, ear_values)
cv2.destroyAllWindows()
cap.release()
import matplotlib.pyplot as plt

def plot_ear_curve(frame_count, ear_values):
    plt.figure(figsize=(10, 4))
    plt.plot(frame_count, ear_values, color='blue', label='EAR Value')
    plt.axhline(y=0.25, color='red', linestyle='--', label='Threshold')
    plt.xlabel('Frame')
    plt.ylabel('EAR')
    plt.title('EAR over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ear_plot.png")
    # ✳️ تأكدي إنها ما تقفل بسرعة
