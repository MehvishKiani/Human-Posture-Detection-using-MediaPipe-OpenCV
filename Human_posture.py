import cv2
import mediapipe as mp
import numpy as np

def get_y(lm): return lm.y
def get_x(lm): return lm.x

def classify_posture(landmarks):
    shoulder_y = (get_y(landmarks[11]) + get_y(landmarks[12])) / 2
    hip_y = (get_y(landmarks[23]) + get_y(landmarks[24])) / 2
    knee_y = (get_y(landmarks[25]) + get_y(landmarks[26])) / 2
    ankle_y = (get_y(landmarks[27]) + get_y(landmarks[28])) / 2
    nose_y = get_y(landmarks[0])

    torso = abs(shoulder_y - hip_y)
    leg = abs(hip_y - ankle_y)
    full_height = abs(nose_y - ankle_y)

    # Avoid division by zero
    if full_height == 0:
        return "Uncertain"

    torso_ratio = torso / full_height
    leg_ratio = leg / full_height

    if full_height < 0.4:
        return "Lying Down"
    elif torso_ratio > 0.35 and leg_ratio > 0.45:
        return "Standing"
    elif torso_ratio > 0.35 and leg_ratio < 0.25:
        return "Sitting"
    elif shoulder_y > hip_y:
        return "Leaning Forward"
    else:
        return "Uncertain"

def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(0)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                posture = classify_posture(landmarks)

                cv2.putText(image, f'Posture: {posture}', (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            else:
                cv2.putText(image, 'No pose detected', (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('Posture Detection', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
