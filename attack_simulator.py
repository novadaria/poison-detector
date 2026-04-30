# -- моделирование атаки отравления разметки (label flipping)

import numpy as np

class AttackSimulator:

    def __init__(self, random_state=42):
        self.random_state = random_state

    def poison_dataset(self, y, poison_ratio, random_state=None):
        rng = np.random.default_rng(
            random_state if random_state is not None else self.random_state
        )

        y = np.array(y)
        unique = np.unique(y)

        n_poison = min(int(len(y) * poison_ratio), len(y))

        if n_poison <= 0:
            return y.copy(), np.array([], dtype=int)

        idx = rng.choice(len(y), n_poison, replace=False)
        y_poisoned = y.copy()

        for i in idx:
            if len(unique) < 2:
                return y.copy(), np.array([], dtype=int)

            current = y_poisoned[i]
            alt = unique[unique != current]

            if len(alt) == 0:
                continue

            y_poisoned[i] = rng.choice(alt)

        return y_poisoned, idx