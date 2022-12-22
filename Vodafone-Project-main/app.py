import dlib
import numpy as np
import threading
from werkzeug.utils import secure_filename
from imutils.video import FPS
import tensorflow as tf
from keras.models import load_model
from utils import class_labels as classes
from utils import quotes, detectVehicles, location, weather_report
from flask import Response, Flask, render_template, url_for, request, redirect
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import threading
import imutils
import random
import playsound
import math
import time
import dlib
import cv2
import io
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'itismysecretkey'

outputFrame = None
lock = threading.Lock()

model = load_model('ClassificationModel.h5')
path_model = load_model('roadsModel.h5')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("fatigue/shape_predictor_68_face_landmarks.dat")


def sound_alarm():
    playsound.playsound(u'fatigue/sounds/alarm2.mp3')


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear


def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])

    return distance


def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        return "error"
    if len(rects) == 0:
        return "error"

    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))

    return im


def top_lip(landmarks):
    top_lip_pts = []
    for i in range(50, 53):
        top_lip_pts.append(landmarks[i])
    for i in range(61, 64):
        top_lip_pts.append(landmarks[i])
    top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))
    top_lip_mean = np.mean(top_lip_pts, axis=0)

    return int(top_lip_mean[:, 1])


def bottom_lip(landmarks):
    bottom_lip_pts = []
    for i in range(65, 68):
        bottom_lip_pts.append(landmarks[i])
    for i in range(56, 59):
        bottom_lip_pts.append(landmarks[i])
    bottom_lip_all_pts = np.squeeze(np.asarray(bottom_lip_pts))
    bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)

    return int(bottom_lip_mean[:, 1])


def mouth_open(image):
    landmarks = get_landmarks(image)

    if landmarks == "error":
        return image, 0

    image_with_landmarks = annotate_landmarks(image, landmarks)
    top_lip_center = top_lip(landmarks)
    bottom_lip_center = bottom_lip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)

    return image_with_landmarks, lip_distance


@app.route('/')
def index():
    quote = random.choice(quotes)
    return render_template('index.html', quote_h=quote[0], quote_b=quote[1])


@app.route('/about')
def about():
    return render_template('about.html', title='About')


@app.route('/video')
def video():
    loc = location()
    report = weather_report()
    if report == []:
        msg = "Not available"
    else:
        msg = f"Temperature is {report[0]}, Humidity is {report[1]}, Pressure is {report[2]} and {report[3]}"
    return render_template('video.html', title='Video', addr=loc, weather=msg)


roadLabels = ['pathHole road', 'plain road']


def gen1():
    vs = VideoStream(src=0).start()
    fps = FPS().start()
    while True:
        frame = vs.read()
        h, w, _ = frame.shape
        mid = w // 2
        frame = imutils.resize(frame, width=600)
        img = np.asarray(frame)
        ####path holes######################
        img_copy = cv2.resize(img, (256, 256))
        X = np.array(img_copy)
        X = np.expand_dims(X, axis=0)
        road_classIndex = np.argmax(path_model.predict(X), axis=1)
        road_predictions = path_model.predict(X)
        road = road_classIndex[0]

        ############################################

        img = cv2.resize(img, (32, 32))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(img)
        img = img / 255
        img = img.reshape(1, 32, 32, 1)
        classIndex = np.argmax(model.predict(img), axis=1)
        predictions = model.predict(img)
        score = np.amax(predictions)
        class_predicted = classes[classIndex[0]]
        X = detectVehicles(frame)
        if X > mid:
            cv2.rectangle(frame, (mid, h), (w, 0), (255, 0, 0), 5)
        else:
            cv2.rectangle(frame, (0, 0), (mid, h), (255, 0, 0), 5)

        if (int(100 * score) > 90):
            text = rf"{class_predicted}- {str(int(100 * score))} "
            frame = cv2.putText(frame, text, (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2, cv2.LINE_AA)
        if road == 0:
            frame = cv2.putText(frame, fr'path hold ahead ', (20, 80), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2,
                                cv2.LINE_AA)
        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        fps.update()

    fps.stop()
    print("[INFO]Elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO]Approx. FPS:  {:.2f}".format(fps.fps()))


@app.route('/video_feed1')
def video_feed1():
    return Response(gen1(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route('/drowsy')
def drowsy():
    return render_template('drowsy.html', title='Fatigue')


def gen2():
    print("Loading facial landmark predictor...")

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    print("Starting video stream thread...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 25
    BLINK_AR_CONSEC_FRAMES = 3

    TOTAL = 0
    BLINK_COUNTER = 0

    COUNTER = 0
    ALARM_ON = False
    YAWN = 0
    YAWN_STATUS = False

    with open("./files/EAR.txt", "w") as file:
        file.write('\n')
    with open("./files/YAWN.txt", "w") as file:
        file.write('0\n')

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=600)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # YAWN
            image_landmarks, lip_distance = mouth_open(frame)

            prev_yawn_status = YAWN_STATUS

            if lip_distance > 25:
                YAWN_STATUS = True

                output_text = "YAWNS: " + str(YAWN + 1)

                cv2.putText(frame, output_text, (30, 400), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 0), 2)

            else:
                YAWN_STATUS = False

            if prev_yawn_status == True and YAWN_STATUS == False:
                YAWN += 1

            # EYES
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 230, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 230, 0), 1)

            if ear < EYE_AR_THRESH:
                COUNTER += 1

                if COUNTER >= EYE_AR_CONSEC_FRAMES:

                    if not ALARM_ON:
                        ALARM_ON = True

                        t = Thread(target=sound_alarm)
                        t.deamon = True
                        t.start()

                    cv2.putText(frame, "WAKE UP!!!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 230), 2)

            else:
                COUNTER = 0
                ALARM_ON = False

            cv2.putText(frame, "{:.4f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Blink
            if ear < EYE_AR_THRESH:
                BLINK_COUNTER += 1
            else:
                if BLINK_COUNTER >= BLINK_AR_CONSEC_FRAMES:
                    TOTAL += 1

                BLINK_COUNTER = 0

            print(TOTAL)

            with open("./files/BLINK.txt", "a") as file:
                file.write(str(TOTAL))
                file.write('\n')

            with open("./files/YAWN.txt", "a") as file:
                file.write(str(YAWN))
                file.write('\n')

            # print(ear)
            with open("./files/EAR.txt", "a") as file:
                file.write(str(round(ear, 3) * 1000))
                file.write('\n')

        # cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        global outputFrame, lock

        with lock:
            outputFrame = frame.copy()

        (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')


@app.route('/video_feed2')
def video_feed2():
    return Response(gen2(), mimetype="multipart/x-mixed-replace; boundary=frame")


def image_processing(file_path):
    model = load_model('ClassificationModel.h5')
    image = cv2.imread(file_path)
    image = cv2.resize(image, (32, 32))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)
    image = image / 255
    img = image.reshape(1, 32, 32, 1)
    classIndex = np.argmax(model.predict(img), axis=1)
    predictions = model.predict(img)
    score = np.amax(predictions)
    class_predicted = classes[classIndex[0]]
    return class_predicted, int(score * 100)


@app.route('/predict')
def index2():
    return render_template('index2.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        file_path = secure_filename(f.filename)
        fp = os.path.join("static\inputs", file_path)
        f.save(fp)
        # f.save(file_path)
        pred, score = image_processing(fp)
        if score < 90:
            result = f"Predicted Traffic🚦Sign is: {pred} of {score}% accuracy. Don't get fooled, keep any eye"
        else:
            result = f"Predicted Traffic🚦Sign is: {pred} of {score}% accuracy"
        time.sleep(5)
        if os.path.exists(fp):
            os.remove(fp)
        return result
    return None


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True, threaded=True, use_reloader=False)
