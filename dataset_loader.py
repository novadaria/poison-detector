# -- загрузка и предобработка датасета
 
import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

class DatasetLoader:

    def safe_read_csv(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл не найден: {path}")

        encodings = ["utf-8", "latin-1", "ISO-8859-1", "cp1251"]

        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc)
            except Exception:
                continue

        return pd.read_csv(path, encoding="utf-8", errors="replace")

    def load(self, path, target_column=None):
        df = self.safe_read_csv(path)

        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        df = df.dropna(axis=1, how="all")

        if df.empty:
            raise ValueError("Пустой датасет после очистки")

        if target_column is None:
            target_column = df.columns[-1]

        if target_column not in df.columns:
            raise ValueError(f"Колонка {target_column} не найдена")

        df = df.dropna(subset=[target_column])

        if len(df) == 0:
            raise ValueError("Нет данных после удаления пропусков в target")

        y = df[target_column]
        X = df.drop(columns=[target_column])

        if y.dtype == "object":
            y = LabelEncoder().fit_transform(y)

        text_cols = [c for c in X.columns if X[c].dtype == "object"]
        num_cols = [c for c in X.columns if X[c].dtype != "object"]

        parts = []

        if text_cols:
            text_data = X[text_cols].fillna("").astype(str).agg(" ".join, axis=1)

            vectorizer = TfidfVectorizer(
                max_features=5000,
                lowercase=True,
                stop_words='english',
                ngram_range=(1, 2)
            )
            parts.append(vectorizer.fit_transform(text_data).toarray())

        if num_cols:
            num = X[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
            scaler = StandardScaler()
            parts.append(scaler.fit_transform(num))

        if not parts:
            raise ValueError("Нет признаков для обучения")

        X_final = np.hstack(parts)

        return X_final, np.array(y), df