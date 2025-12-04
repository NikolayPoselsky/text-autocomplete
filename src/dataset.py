#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """
    Обёртка над тензором input_ids.
    На каждом шаге возвращаем одну последовательность токенов.
    """

    def __init__(self, input_ids: torch.Tensor):
        # input_ids: (num_samples, seq_len)
        self.input_ids = input_ids

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        # возвращаем ровно одну последовательность токенов
        return self.input_ids[idx]   # (seq_len,)

