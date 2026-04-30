# Детектор отравления данных для ML-моделей (Poisoning Detector)

## Описание
Данный проект содержит программу для обнаружения целенаправленного отравления разметки в обучающих данных. Метод основан на комбинации канареечных примеров (Canary samples) и коэффициента согласия Каппа (Cohen's Kappa).

## Установка
1. Клонировать репозиторий.
2. Создать виртуальное окружение: `python -m venv venv`
3. Активировать окружение: `venv\Scripts\activate` (Windows)
4. Установить зависимости: `pip install -r requirements.txt`

## Использование
Запуск диагностики: `python main.py --data data/adult.csv --target income --poison_ratio 0.1`
Запуск эксперимента: `python main.py --data data/adult.csv --target income --use_kappa --experiment`