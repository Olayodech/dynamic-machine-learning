import cv2;
import mediapipe as mp;
from typing import Optional


def hands_classifiers():
    hand_classifier = mp.solutions.hands;
    hands = hand_classifier.Hands(min_detection_confidence=0.2, static_image_mode=True);
    return hands, hand_classifier;

def capture_face(file_path: Optional[str] = None, cam: Optional[int] = None):
    mp_drawing = mp.solutions.drawing_utils
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
        
        
        hands, hands_classifier = hands_classifiers();
        
        results = hands.process(gray);
        
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(results, hand_lms,
                                          hands_classifier.HAND_CONNECTIONS,
                                          landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                              color=(255, 0, 255), thickness=4, circle_radius=2),
                                          connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                              color=(20, 180, 90), thickness=2, circle_radius=2)
                                          )
        
        
        
        # if results.multi_hand_landmarks:
        #     for hand in  results.multi_hand_landmarks:
        #         for id, landmark in enumerate(hand.landmark):
        #             print()
        cv2.imshow("Dataset Maker", gray);
        if cv2.waitKey(1) & 0xff == ord('q'):
            break;
    capture.release()
    cv2.destroyAllWindows();
    
capture_face(cam=0)