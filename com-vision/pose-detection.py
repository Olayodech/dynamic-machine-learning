import cv2;
import mediapipe as mp;

mpose = mp.solutions.pose
pose = mpose.Pose()
mpDraw = mp.solutions.drawing_utils
def detect_pose():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            # for pose_landmarks in results.pose_landmarks:
            mpDraw.draw_landmarks(image, results.pose_landmarks, mpose.POSE_CONNECTIONS)
        cv2.putText(image, "Your skeleton", (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
        cv2.imshow('Pose Detection', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    
detect_pose();