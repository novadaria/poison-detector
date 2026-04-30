# -- расчет коэффициента согласия аннотаторов Каппа

# -- выявление расхождения между аннотаторами

import numpy as np

class KappaCalculator:

    @staticmethod
    def compute_kappa(y1, y2):
        y1 = np.array(y1)
        y2 = np.array(y2)

        p0 = np.mean(y1 == y2)

        labels = np.unique(np.concatenate([y1, y2]))
        p1 = 0

        for label in labels:
            p1 += (np.mean(y1 == label) * np.mean(y2 == label))

        if p1 == 1:
            return 1.0

        return (p0 - p1) / (1 - p1)

    @staticmethod
    def kappa_to_probability(kappa, kappa_clean=0.7, kappa_poisoned=0.2):
        kappa = float(kappa)

        if kappa >= kappa_clean:
            return 0.0
        if kappa <= kappa_poisoned:
            return 1.0

        return (kappa_clean - kappa) / (kappa_clean - kappa_poisoned)

    def simulate_annotators(self, y, error_rate_1=0.1, error_rate_2=0.1, random_state=None):
        rng = np.random.default_rng(random_state)

        y = np.array(y)
        unique = np.unique(y)

        def corrupt(y, error_rate):
            y_new = y.copy()
            n_err = int(len(y) * error_rate)

            if n_err == 0:
                return y_new

            idx = rng.choice(len(y), n_err, replace=False)

            for i in idx:
                current = y_new[i]
                alt = unique[unique != current]

                if len(alt) == 0:
                    continue

                y_new[i] = rng.choice(alt)

            return y_new

        return corrupt(y, error_rate_1), corrupt(y, error_rate_2)