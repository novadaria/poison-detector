# -- моделирование эксперимента

# -- оценивается работа детектора отравления
# -- влияние уровня отравления на качество модели
# -- поведение метрик при различных уровнях атаки

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine   # <--- добавлен импорт

class ExperimentRunner:

    def __init__(self, attack, canary, kappa_calc, combiner):
        self.attack = attack
        self.canary = canary
        self.kappa_calc = kappa_calc
        self.combiner = combiner

    def run(self, X, y, canary_indices, true_canary_labels):

        levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        seeds = [42, 43, 44, 45, 46]

        results = []

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )

        for level in levels:
            runs = []

            for seed in seeds:

                y_poisoned_full, _ = self.attack.poison_dataset(y, level, seed)

                a1, a2 = self.kappa_calc.simulate_annotators(
                    y_poisoned_full,
                    error_rate_1=0.0,
                    error_rate_2=level,
                    random_state=seed
                )

                error_rate, p_canary = self.canary.detect(
                    canary_indices,
                    y_poisoned_full,
                    true_canary_labels
                )

                kappa = self.kappa_calc.compute_kappa(a1, a2)
                p_kappa = self.kappa_calc.kappa_to_probability(kappa)

                p_total = self.combiner.combine(p_canary, p_kappa)

                y_train_poisoned, _ = self.attack.poison_dataset(y_train, level, seed)

                model = LogisticRegression(max_iter=2000, class_weight="balanced")
                model.fit(X_train, y_train_poisoned)

                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)

                proba = model.predict_proba(X_test)

                if proba.shape[1] == 2:
                    auc = roc_auc_score(y_test, proba[:, 1])
                else:
                    auc = roc_auc_score(y_test, proba, multi_class="ovr")

                runs.append([error_rate, kappa, p_canary, p_kappa, p_total, acc, auc])

            runs = np.array(runs)

            results.append({
                "poison_level": level,
                "error_rate": runs[:, 0].mean(),
                "kappa": runs[:, 1].mean(),
                "p_canary": runs[:, 2].mean(),
                "p_kappa": runs[:, 3].mean(),
                "p_total": runs[:, 4].mean(),
                "accuracy": runs[:, 5].mean(),
                "roc_auc": runs[:, 6].mean()
            })

        # --- СОХРАНЕНИЕ В SQLite ---
        db_path = 'experiments.db'
        engine = create_engine(f'sqlite:///{db_path}', echo=False)
        df = pd.DataFrame(results)
        df.to_sql('experiment_results', con=engine, if_exists='append', index=False)
        print(f"Результаты сохранены в базу данных: {db_path}, таблица experiment_results")
        # -------------------------

        return pd.DataFrame(results)