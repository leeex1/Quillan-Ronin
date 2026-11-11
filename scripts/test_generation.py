import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = r'C:\02_QUILLAN\exports\quillan_1b_sovereign_hf'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Restore original Microsoft and Google tokens from snapshot or check their weights
print('Checking embed_tokens shape:', model.model.embed_tokens.weight.shape)

# Let's inspect generation with current weights:
messages = [
    {'role': 'system', 'content': 'You are Quillan-Ronin (Sovereign Flagship 1.2B), created by CrashOverrideX. Never identify as Llama or Meta.'},
    {'role': 'user', 'content': 'Who created you?'}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors='pt')
out = model.generate(**inputs, max_new_tokens=40, do_sample=False)
print('GENERATION CURRENT (greedy):')
print(tokenizer.decode(out[0][inputs.input_ids.shape[1]:]))

# Now test zeroing donor tokens:
with torch.no_grad():
    # zero out Meta [12452, 16197], Facebook [21124, 5690]
    for tid in [12452, 16197, 21124, 5690]:
        model.model.embed_tokens.weight[tid] = 0.0

out_zero = model.generate(**inputs, max_new_tokens=40, do_sample=False)
print('GENERATION AFTER ZEROING META/FB:')
print(tokenizer.decode(out_zero[0][inputs.input_ids.shape[1]:]))
