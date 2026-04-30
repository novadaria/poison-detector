# Детектор отравления данных для ML-моделей (Poisoning Detector)

## Описание
Данный проект содержит программу для обнаружения целенаправленного отравления разметки в обучающих данных. Метод основан на комбинации канареечных примеров и коэффициента согласия Каппа (Cohen's Kappa).

## Установка
1. Клонировать репозиторий.
2. Создать виртуальное окружение: `python -m venv venv`
3. Активировать окружение: `venv\Scripts\activate` (Windows)
4. Установить зависимости: `pip install -r requirements.txt`

## Использование
Команды запуска
Диагностика чистого датасета (без отравления)
python main.py --data data/adult.csv --target income --poison_ratio 0.0

Диагностика с симуляцией 10% отравления, только канарейки
python main.py --data data/spam.csv --target v1 --poison_ratio 0.1

Эксперимент на adult.csv с канарейками и каппа (перебор уровней от 0 до 30%)
python main.py --data data/adult.csv --target income --use_kappa --experiment

То же самое для creditcard_2023.csv
python main.py --data data/creditcard_2023.csv --target Class --use_kappa --experiment

## Датасеты
https://www.kaggle.com/datasets/uciml/adult-census-income

https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023

https://www.kaggle.com/datasets/eslamessam2025/spam-dataset