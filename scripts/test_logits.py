import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = r'C:\02_QUILLAN\exports\quillan_1b_sovereign_hf'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

for prompt in ['Who created you?', 'What is your name?', 'Tell me a story about space.']:
    inputs = tokenizer(prompt, return_tensors='pt')
    with torch.no_grad():
        out = model(**inputs)
        logits = out.logits[0, -1, :]
    print('PROMPT:', prompt)
    print('Min logit:', logits.min().item(), 'Max logit:', logits.max().item(), 'Mean logit:', logits.mean().item())
    topk = torch.topk(logits, 5)
    for tid, val in zip(topk.indices.tolist(), topk.values.tolist()):
        print(f'  Top: {tid} -> {repr(tokenizer.decode([tid]))}: {val:.4f}')
    rank_zero = (logits > 0.0).sum().item()
    print(f'  Rank of logit 0.0: {rank_zero} / {logits.shape[0]} (tokens above 0)')
    # Check tokens 51435 (' Crash'), 13068 ('Microsoft'), 12452 ('Meta'), etc.
    for tid in [51435, 13068, 5210, 12452, 16197, 21124, 5690]:
        print(f'  Token {tid} ({repr(tokenizer.decode([tid]))}): logit = {logits[tid].item():.4f}')
