import cv2
import mediapipe as mp


class FaceMeshDetector:
    def __init__(self,
                 max_num_faces=1,
                 refine_landmarks=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        self.mp_face = mp.solutions.face_mesh
        self.detector = self.mp_face.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process(self, frame):
        """
        Input:
            frame: BGR image from OpenCV

        Output:
            landmarks: list of 468 MediaPipe landmarks or None
        """

        if frame is None:
            return None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.detector.process(rgb)

        if not result.multi_face_landmarks:
            return None

        # return first detected face landmarks
        return result.multi_face_landmarks[0].landmark

    # Optional alias for readability
    def get_landmarks(self, frame):
        return self.process(frame)

    def release(self):
        if self.detector:
            self.detector.close()
