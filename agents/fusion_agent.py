class FusionAgent:
    def __init__(self):
        # Tunable weights (sum ≈ 1)
        self.w_eye = 0.4
        self.w_blink = 0.35
        self.w_head = 0.25

        # Decision thresholds
        self.safe_th = 0.7
        self.warning_th = 0.4

    def update(self, eye_score, blink_score, head_score):
        """
        Combines agent outputs and makes a final decision
        Returns:
            alertness_score (float)
            state (str): SAFE / WARNING / DROWSY
        """

        alertness_score = (
            self.w_eye * eye_score +
            self.w_blink * blink_score +
            self.w_head * head_score
        )

        if alertness_score >= self.safe_th:
            state = "SAFE"
        elif alertness_score >= self.warning_th:
            state = "WARNING"
        else:
            state = "DROWSY"

        return alertness_score, state
