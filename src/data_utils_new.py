#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re
from pathlib import Path

import pandas as pd
import torch
from transformers import BertTokenizerFast

from config import SMALL_FILE, FULL_FILE


def data_handling(min_len=7, max_len=21, use_smallfile=True, file_path=None):

    if file_path is None:
        path = SMALL_FILE if use_smallfile else FULL_FILE
    else:
        path = Path(file_path)

    # читаем файл построчно
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    df = pd.DataFrame({"text": lines})

    # ссылки → спец-слово 'link'
    df["text"] = df["text"].apply(
        lambda txt: re.sub(r"http\S+|www\.\S+", "link", txt)
    )

    # убираем лишние пробелы, приводим к нижнему регистру
    df["text"] = df["text"].str.strip().str.lower()

    # оставляем только латиницу, цифры и пробел
    df["text"] = df["text"].apply(
        lambda t: re.sub(r"[^a-z0-9\s]", "", t).strip()
    )

    # длина текста в словах
    df["text_length"] = df["text"].str.split().apply(len)

    # фильтр по минимальной длине
    df = df[df["text_length"] >= min_len].reset_index(drop=True)

    # обрезаем до max_len слов
    df["words"] = df["text"].str.split().apply(lambda x: x[:max_len])
    df["text"] = df["words"].str.join(" ")

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    tokens = tokenizer(
        df["text"].tolist(),
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    return df, tokenizer, tokens

