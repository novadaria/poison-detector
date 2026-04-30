# -- диагностика датасета
 
# -- объединяются результаты канареек и каппа
# -- для получения итоговой оценки вероятности отравления

class DatasetDiagnoser:

    def __init__(self, canary_detector, kappa_calculator, combiner):
        self.canary = canary_detector
        self.kappa = kappa_calculator
        self.combiner = combiner

    def diagnose(self, y_observed, canary_indices, true_labels, annotator2=None):

        error_rate, p_canary = self.canary.detect(
            canary_indices,
            y_observed,
            true_labels
        )

        if annotator2 is not None:
            kappa = self.kappa.compute_kappa(y_observed, annotator2)
            p_kappa = self.kappa.kappa_to_probability(kappa)
            p_total = self.combiner.combine(p_canary, p_kappa)

            return {
                "p_canary": p_canary,
                "p_kappa": p_kappa,
                "p_total": p_total
            }

        return {
            "p_canary": p_canary,
            "p_kappa": None,
            "p_total": p_canary
        }