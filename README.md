# text-autocomplete
Учебный проект по созданию модуля автозаполнения текста


# Text Autocomplete (LSTM vs distilGPT2)

## Быстрый старт
1) Скачай sentiment140 и сохрани как `data/raw_dataset.csv` с колонкой `text` (можно переименовать свою колонку в `text`).
2) `pip install -r requirements.txt`
3) `python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"`
4) Подготовка данных: `python -m src.data_utils` (или вызови `preprocess_and_split` из notebook-а)
5) Обучение LSTM: `python -m src.lstm_train`
6) Оценка LSTM: `python -m src.eval_lstm`
7) Оценка distilGPT2: `python -m src.eval_transformer_pipeline`

## Замечания
- Задача — автодополнение: на вход 3/4 текста, на выход — последняя 1/4.
- Метрики: ROUGE-1, ROUGE-2 (F1).
- LSTM обучается кросс-энтропией по задаче «следующий токен».
