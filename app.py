from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import face_recognition
import os
import time

app = Flask(__name__)
CORS(app)

# Load user images and their encodings
IMAGE_FILES = []
filename = []
imageDir = 'dataset'
userImage = 'img.jpg'

if userImage:
    img_path = os.path.join(imageDir, userImage)
    img_path = face_recognition.load_image_file(img_path)
    IMAGE_FILES.append(img_path)
    filename.append(userImage.split(".", 1)[0])

def encoding_img(IMAGE_FILES):
    encodeList = []
    for img in IMAGE_FILES:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListknown = encoding_img(IMAGE_FILES)

@app.route('/login', methods=['POST'])
def login():
    result = ""
    capture = cv2.VideoCapture(0)
    start_time = time.time()  # Start time for capturing
    verified = False  # To track if face has been verified
    delay_passed = False  # To track if the 2-second delay has passed

    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time >= 6:
            delay_passed = True
        if elapsed_time >= 20:
            print("Time limit reached. Shutting down capture.")
            result = "false"
            break

        success, img = capture.read()
        if not success:
            print("Failed to read frame")
            continue

        imgc = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgc = cv2.cvtColor(imgc, cv2.COLOR_BGR2RGB)

        facesCurrent = face_recognition.face_locations(imgc)
        encodeFacesCurrent = face_recognition.face_encodings(imgc, facesCurrent)

        for encodeFace, faceloc in zip(encodeFacesCurrent, facesCurrent):
            matchesFace = face_recognition.compare_faces(encodeListknown, encodeFace)
            faceDistance = face_recognition.face_distance(encodeListknown, encodeFace)
            matchindex = np.argmin(faceDistance)

            if matchesFace[matchindex]:
                name = filename[matchindex].upper()
                putText = 'Captured'
                y1, x2, y2, x1 = faceloc
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 0, 0), 2, cv2.FILLED)
                cv2.putText(img, putText, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                print("Face detected. Shutting down capture.")
                result = "true"
                verified = True
                break

        if verified and delay_passed:
            break  # Break out of the outer while loop if a face is detected and 2 seconds have passed

        cv2.imshow('Input', img)
        key = cv2.waitKey(10)
        if key == 27:
            print("Escape key pressed. Shutting down capture.")
            break

    capture.release()
    cv2.destroyAllWindows()
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
