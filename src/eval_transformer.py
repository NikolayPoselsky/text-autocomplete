#!/usr/bin/env python
# coding: utf-8

# In[17]:


from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer

tokenizer_gpt = AutoTokenizer.from_pretrained("distilgpt2")
model_gpt = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)

# У GPT2 нет пад-токена, зададим его явно
tokenizer_gpt.pad_token = tokenizer_gpt.eos_token

def generate_gpt2(text, max_new_tokens=30):
    # токенизируем промпт
    inputs = tokenizer_gpt(text, return_tensors="pt").to(device)

    outputs = model_gpt.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.8,
        pad_token_id=tokenizer_gpt.eos_token_id
    )

    # В outputs[0] сначала идут токены промпта, потом новые
    input_len = inputs["input_ids"].shape[1]
    gen_ids = outputs[0][input_len:]  # только сгенерированное продолжение

    return tokenizer_gpt.decode(gen_ids, skip_special_tokens=True)

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

print("Average ROUGE-L for 1000 examples:", sum(scores)/len(scores))

