import time
from typing import Optional
import cv2;

# time.sleep(2)

def facial_classifier():
    face_cascade = cv2.CascadeClassifier();
    return face_cascade;

def eye_classifiers():
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml');
    return eye_cascade;

def smile_classifiers():
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml');
    return smile_cascade;
        
def detect_face(gray, frame):
    classifier = facial_classifier()
    eye_classifier = eye_classifiers()
    smile_classifier = smile_classifiers()
    face = classifier.detectMultiScale(gray, 1.1, 4, minSize=(30, 30));
    eye = eye_classifier.detectMultiScale(gray, 1.1, 3, minSize=(30, 30));
    smile = smile_classifier.detectMultiScale(gray, 1.7, 22, minSize=(30, 30));
    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2);
        face_roi = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        for (ex, ey, ew, eh) in eye:
            cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (80, 10, 20), 2);
            face_roi = gray[ey:ey+eh, ex:ex+ew]
            roi_color = frame[ey:ey+eh, ex:ex+ew]
            # return eye;
            for (sx, sy, sw, sh) in smile:
                cv2.rectangle(frame, (sx, sy), (sx+sw, sy+sh), (50, 50, 20), 2);
                face_roi = gray[sy:sy+sh, sx:sx+sw]
                roi_color = frame[sy:sy+sh, sx:sx+sw]
                # return eye;
        
    return frame;

def capture_face(file_path: Optional[str] = None, cam: Optional[int] = None):
    path = file_path if file_path else cam;
    print("PATH IS", path)
    capture = cv2.VideoCapture(path);
    print("CAPTURE IS IS", path)
    if not capture.isOpened():
        print("CAPTURE IS", capture)
        return;
    while True:
        _, frame = capture.read();
        if not _ or frame is None:
            print("error reading frame", _)
            break;
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY);
        canvas = detect_face(gray, frame);
        cv2.imshow("Video", canvas);
        if cv2.waitKey(1) & 0xff == ord('q'):
            break;
    capture.release()
    cv2.destroyAllWindows();
    

capture_face(cam=0)

