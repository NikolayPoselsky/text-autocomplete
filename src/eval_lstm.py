#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch.utils.data import DataLoader

from config import DATA_DIR
from data_utils import data_handling
from dataset import TextDataset
from model import LSTMAutocompleteModel
from training import train_model_with_rouge


def main():
    # 1. Загружаем данные и токенизируем
    df, tokenizer, tokens = data_handling(
        min_len=7,
        max_len=28,
        use_smallfile=True,
    )

    input_ids = tokens["input_ids"]
    pad_token_id = tokenizer.pad_token_id

    # 2. Dataset + DataLoader
    dataset = TextDataset(input_ids)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 3. Модель
    vocab_size = tokenizer.vocab_size
    model = LSTMAutocompleteModel(vocab_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4. Обучение с выводом loss и ROUGE-L
    model = train_model_with_rouge(
        model,
        dataloader,
        pad_token_id=pad_token_id,
        epochs=3,
        lr=1e-3,
        device=device,
        rouge_samples_per_batch=4,  # можно увеличить/уменьшить
    )

    # 5. Сохраняем модель
    torch.save(model.state_dict(), DATA_DIR / "lstm_autocomplete_with_rouge.pt")


if __name__ == "__main__":
    main()

