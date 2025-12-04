#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
from tqdm import tqdm

from metrics import rouge_l


def train_model_with_rouge(
    model,
    dataloader,
    pad_token_id,
    epochs=3,
    lr=1e-3,
    device="cuda",
    rouge_samples_per_batch=4,
):

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        total_rouge = 0.0
        total_rouge_count = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        # Считаем средний лосс по числу обработанных батчей
        for batch_idx, batch_seqs in enumerate(pbar, start=1):
            # batch_seqs: (batch, seq_len)
            batch_seqs = batch_seqs.to(device)

            # --- подготовка X, Y для next-token prediction
            X = batch_seqs[:, :-1]
            Y = batch_seqs[:, 1:]

            optimizer.zero_grad()
            logits, _ = model(X)  # logits: (batch, seq_len-1, vocab_size)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),   # (batch*(seq_len-1), vocab_size)
                Y.reshape(-1),                         # (batch*(seq_len-1))
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            #  ROUGE-L по автодополнению
            with torch.no_grad():
                batch_size = batch_seqs.size(0)
                n_examples = min(rouge_samples_per_batch, batch_size)

                for i in range(n_examples):
                    seq = batch_seqs[i]  # (seq_len,)
                    # убираем pad токены
                    seq_ids = [t for t in seq.tolist() if t != pad_token_id]
                    if len(seq_ids) < 4:
                        continue

                    # делим: 3/4 — вход, 1/4 — таргет
                    split = int(len(seq_ids) * 0.75)
                    if not (0 < split < len(seq_ids)):
                        continue

                    prompt_ids = seq_ids[:split]
                    target_ids = seq_ids[split:]

                    start_seq = torch.tensor(
                        prompt_ids, dtype=torch.long, device=device
                    )
                    gen_full = model.generate(
                        start_seq,
                        max_len=len(target_ids),
                        device=device,
                    )
                    # берём только дополнение
                    pred_completion = gen_full[len(prompt_ids):]

                    score = rouge_l(target_ids, pred_completion)
                    total_rouge += score
                    total_rouge_count += 1

            # средний лосс по числу уже пройденных батчей
            avg_loss_so_far = total_loss / batch_idx
            avg_rouge_so_far = (
                total_rouge / total_rouge_count if total_rouge_count > 0 else 0.0
            )
            pbar.set_postfix(loss=avg_loss_so_far, rouge_l=avg_rouge_so_far)

        avg_loss = total_loss / len(dataloader)
        avg_rouge = total_rouge / total_rouge_count if total_rouge_count > 0 else 0.0
        print(f"\nEpoch {epoch}: loss={avg_loss:.4f}, ROUGE-L={avg_rouge:.4f}\n")

    return model

