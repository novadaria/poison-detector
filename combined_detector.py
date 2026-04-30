# -- объединение сигналов детектора

# -- итоговая вероятность = взвешенная сумма вероятностей канареек и каппа 

class CombinedDetector:

    def __init__(self, weight_canary=0.6, weight_kappa=0.4):
        self.weight_canary = weight_canary
        self.weight_kappa = weight_kappa

    def combine(self, p_canary, p_kappa):
        return self.weight_canary * p_canary + self.weight_kappa * p_kappa
