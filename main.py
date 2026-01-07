import cv2

from perception.face_mesh import FaceMeshDetector
from agents.eye_state_agent import EyeStateAgent
from agents.blink_agent import BlinkPatternAgent
from agents.head_pose_agent import HeadPoseAgent   


def main():
    # Initialize components
    face_mesh = FaceMeshDetector()
    eye_agent = EyeStateAgent()
    blink_agent = BlinkPatternAgent()
    head_pose_agent = HeadPoseAgent()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return

    print("[INFO] Driver Monitoring System started...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame")
            break

        # Get face landmarks
        landmarks = face_mesh.process(frame)

        if landmarks is not None:
            # -------- Eye State Agent --------
            ear_value = eye_agent.update(landmarks)

            # -------- Blink Agent --------
            blink_score = blink_agent.update(ear_value)

            # -------- Head Pose Agent --------
            head_pose_data = head_pose_agent.update(landmarks, frame.shape)
            head_pose_score = head_pose_data["score"]
            pitch = head_pose_data["pitch"]
            yaw = head_pose_data["yaw"]

            # -------- Display --------
            cv2.putText(frame, f"EAR: {ear_value:.3f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.putText(frame, f"Blink Score: {blink_score:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.putText(frame, f"Head Pose Score: {head_pose_score:.2f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            cv2.putText(frame, f"Yaw: {yaw:.1f}   Pitch: {pitch:.1f}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

        else:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Eye + Blink + Head Pose Agents", frame)

        # Exit on ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] System stopped.")


if __name__ == "__main__":
    main()
