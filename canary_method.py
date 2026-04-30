# -- реализация метода канареек

# -- в датасет добавляется подмножество объектов
# -- с известными правильными метками
# -- если их метки изменяются, фиксируется наличие атаки

import numpy as np

class CanaryDetector:

    def __init__(self, ratio=0.1, random_state=42):
        self.ratio = ratio
        self.random_state = random_state

    def inject_canaries(self, X, y):
        rng = np.random.default_rng(self.random_state)

        n = len(y)
        n_canary = max(1, int(n * self.ratio))

        idx = rng.choice(n, n_canary, replace=False)
        labels = y[idx].copy()

        return idx, labels

    def detect(self, canary_indices, y_observed, true_labels,
               threshold_low=0.1, threshold_high=0.3):

        errors = np.sum(y_observed[canary_indices] != true_labels)
        error_rate = errors / len(canary_indices)

        if error_rate <= threshold_low:
            return error_rate, 0.0
        elif error_rate >= threshold_high:
            return error_rate, 1.0
        else:
            return error_rate, (error_rate - threshold_low) / (threshold_high - threshold_low)