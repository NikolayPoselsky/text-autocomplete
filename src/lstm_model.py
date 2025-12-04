#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn


class LSTMAutocompleteModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, hidden=None):
        # input_ids: (batch, seq_len)
        x = self.embedding(input_ids)           # (batch, seq_len, embed_dim)
        out, hidden = self.lstm(x, hidden)      # (batch, seq_len, hidden_dim)
        logits = self.fc(out)                   # (batch, seq_len, vocab_size)
        return logits, hidden

    @torch.no_grad()
    def generate(self, start_seq, max_len=20, device="cpu"):

        self.eval()

        generated = start_seq.tolist()
        input_seq = start_seq.unsqueeze(0).to(device)  # (1, seq_len)
        hidden = None

        for _ in range(max_len):
            logits, hidden = self.forward(input_seq, hidden)  # logits: (1, cur_len, vocab)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)  # (1,)
            generated.append(next_token.item())

            # расширяем входную последовательность предсказанным токеном
            input_seq = torch.cat([input_seq, next_token.unsqueeze(0)], dim=1)

        return generated

