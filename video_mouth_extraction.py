import sys
import numpy as np
import cv2
from os import listdir, path, makedirs
from config import VIDEO_EXTRACT, CLASSIFIER_ROOT, VIDEO_NORMALISED, ALIGN_RAW
from sklearn import preprocessing

src = sys.argv[1]
speaker, video = src.split("/")[-2:]

face_classifier = cv2.CascadeClassifier(path.join(CLASSIFIER_ROOT, "face.xml"))
mouth_classifier = cv2.CascadeClassifier(path.join(CLASSIFIER_ROOT, "mouth.xml"))

def height(a):
    c = [ np.count_nonzero(a[i] == 0) for i in range(len(a)) ]
    t = [ 0 ] * len(c)

    for i in range(4, len(t)):
        if c[i]:
            t[i] = ((t[i - 1] + 1) if t[i - 1] != 0
                else (t[i - 2] + 2) if t[i - 2] != 0
                else (t[i - 3] + 3) if t[i - 3] != 0
                else (t[i - 4] + 4) if t[i - 4] != 0
                else 1)
    
    return max(t)

def width(a): return height(np.transpose(a))

def frame_processor(img):

    for (face_x, face_y, face_w, face_h) in face_classifier.detectMultiScale(img, 1.3, 5):

        face = img[face_y: face_y + face_h, face_x: face_x + face_w]

        mouth_x, mouth_y, mouth_w, mouth_h = 0, 0, 0, 0
        for (temp_x, temp_y, temp_w, temp_h) in mouth_classifier.detectMultiScale(face, 1.03, 20):
            if temp_y > mouth_y:
                mouth_x, mouth_y, mouth_w, mouth_h = temp_x, temp_y, temp_w, temp_h
        
        if mouth_x != 0:

            mouth = face[mouth_y: mouth_y + mouth_h, mouth_x: mouth_x + mouth_w]
            mouth = cv2.resize(mouth, None, fx=5, fy=5)
            mouth_thresh = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)
            _, mouth_thresh = cv2.threshold(mouth_thresh, 105, 255, 0)

            final_height, final_width = height(mouth_thresh), width(mouth_thresh)

            return (face, mouth, np.ones((final_height, final_width)))

    return (None, None, None)

result = []
for frame_file in listdir(src):
    if frame_file.endswith(".jpg"):
        frame = cv2.imread(path.join(src, frame_file))

        face, mouth, mouth_box = frame_processor(frame)

        if face != None and mouth != None and mouth_box != None:
            mouth_height, mouth_width = mouth_box.shape[:2]
            result.append([mouth_height, mouth_width])

        # cv2.imshow("frame", frame)
        # cv2.imshow("face", face)
        # cv2.imshow("mouth", mouth)
        # cv2.imshow("mouth_box", mouth_box)
        # if cv2.waitKey(100000) & 0xFF == ord('q'):
        #     exit()

# truncate video to remove silent parts
with open(path.join(ALIGN_RAW, speaker, video + ".align")) as f:
    start_frame, end_frame = (lambda a: (int(a[0].split()[1]), int(a[-1].split()[0])))(f.readlines())
    result = result[start_frame // 1000: end_frame // 1000 + 1]

        
makedirs(path.join(VIDEO_EXTRACT, speaker), exist_ok=True)
np.save(path.join(VIDEO_EXTRACT, speaker, video), np.array(result))

# normalised result
result = preprocessing.scale(np.array(result))
makedirs(path.join(VIDEO_NORMALISED, speaker), exist_ok=True)
np.save(path.join(VIDEO_NORMALISED, speaker, video), result)

print("PROCESSED: ", speaker, "/", video, sep="")