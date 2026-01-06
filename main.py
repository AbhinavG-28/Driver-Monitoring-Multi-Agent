import cv2

from perception.face_mesh import FaceMeshDetector
from agents.eye_state_agent import EyeStateAgent
from agents.blink_agent import BlinkPatternAgent


def main():
    # Initialize components
    face_mesh = FaceMeshDetector()
    eye_agent = EyeStateAgent()
    blink_agent = BlinkPatternAgent()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get face landmarks
        landmarks = face_mesh.process(frame)  # ✅ correct method

        if landmarks is not None:
            # Get EAR from Eye State Agent
            ear_value = eye_agent.update(landmarks)

            # Get blink fatigue score
            blink_score = blink_agent.update(ear_value)

            # Debug display
            cv2.putText(frame, f"EAR: {ear_value:.3f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.putText(frame, f"Blink Score: {blink_score:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Eye + Blink Integration", frame)

        # Exit on ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
