import cv2
import numpy as np


class HeadPoseAgent:
    def __init__(self):
        # 3D face reference points (generic face model)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),          # Nose tip
            (0.0, -330.0, -65.0),     # Chin
            (-225.0, 170.0, -135.0),  # Left eye corner
            (225.0, 170.0, -135.0),   # Right eye corner
            (-150.0, -150.0, -125.0), # Left mouth corner
            (150.0, -150.0, -125.0)   # Right mouth corner
        ], dtype=np.float64)

        # MediaPipe landmark indices
        self.landmark_ids = [1, 152, 33, 263, 61, 291]

        # Thresholds tuned for webcam driver monitoring
        self.PITCH_THRESHOLD = 30.0   # up/down
        self.YAW_THRESHOLD   = 40.0   # left/right

        # calibration storage
        self.base_pitch = None
        self.base_yaw = None


    def update(self, landmarks, frame_shape):
        try:
            # ---------- 2D image points ----------
            image_points = []
            for i in self.landmark_ids:
                lm = landmarks[i]
                x = lm.x * frame_shape[1]
                y = lm.y * frame_shape[0]
                image_points.append((x, y))

            image_points = np.array(image_points, dtype=np.float64)

            # ---------- Camera matrix ----------
            focal_length = frame_shape[1]
            center = (frame_shape[1] / 2, frame_shape[0] / 2)

            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)

            dist_coeffs = np.zeros((4, 1))

            # ---------- Solve PnP ----------
            success, rotation_vec, _ = cv2.solvePnP(
                self.model_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                return self._neutral()

            # ---------- Rotation matrix ----------
            rotation_mat, _ = cv2.Rodrigues(rotation_vec)

            # ---------- Stable Euler extraction ----------
            sy = np.sqrt(rotation_mat[0, 0]**2 + rotation_mat[1, 0]**2)
            singular = sy < 1e-6

            if not singular:
                pitch = np.arctan2(rotation_mat[2, 1], rotation_mat[2, 2])
                yaw   = np.arctan2(-rotation_mat[2, 0], sy)
                roll  = np.arctan2(rotation_mat[1, 0], rotation_mat[0, 0])
            else:
                pitch = np.arctan2(-rotation_mat[1, 2], rotation_mat[1, 1])
                yaw   = np.arctan2(-rotation_mat[2, 0], sy)
                roll  = 0

            # radians → degrees
            pitch = np.degrees(pitch)
            yaw   = np.degrees(yaw)
            roll  = np.degrees(roll)

            # ---------- 180° flip normalization ----------
            if pitch < -90:
                pitch += 180
            elif pitch > 90:
                pitch -= 180

            if yaw < -90:
                yaw += 180
            elif yaw > 90:
                yaw -= 180

            # ---------- Auto-calibration (neutral pose) ----------
            if self.base_pitch is None:
                self.base_pitch = pitch
                self.base_yaw = yaw

            pitch -= self.base_pitch
            yaw   -= self.base_yaw

            abs_pitch = abs(pitch)
            abs_yaw   = abs(yaw)

            # ---------- Attention score ----------
            pitch_score = 1.0 - min(abs_pitch / self.PITCH_THRESHOLD, 1.0)
            yaw_score   = 1.0 - min(abs_yaw / self.YAW_THRESHOLD, 1.0)

            attention_score = min(pitch_score, yaw_score)

            return {
                "score": float(np.clip(attention_score, 0.0, 1.0)),
                "pitch": pitch,
                "yaw": yaw,
                "roll": roll
            }

        except Exception as e:
            print("[HEAD POSE ERROR]:", e)
            return self._neutral()


    def _neutral(self):
        return {
            "score": 0.5,
            "pitch": 0.0,
            "yaw": 0.0,
            "roll": 0.0
        }
