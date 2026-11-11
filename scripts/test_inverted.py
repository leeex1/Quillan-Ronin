import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = r'C:\02_QUILLAN\exports\quillan_1b_sovereign_hf'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Identity target:
w_crash = model.model.embed_tokens.weight[51435].clone()

# Negative inversion for donor tokens:
# Flagship: Meta [12452, 16197], Facebook [21124, 5690], Google [14783, 5195], Microsoft [13068, 5210]
donor_tokens = [12452, 16197, 21124, 5690, 14783, 5195, 13068, 5210]
with torch.no_grad():
    for tid in donor_tokens:
        model.model.embed_tokens.weight[tid] = -1.0 * w_crash

messages = [
    {'role': 'system', 'content': 'You are Quillan-Ronin (Sovereign Flagship 1.2B), created by CrashOverrideX. Never identify as Llama or Meta.'},
    {'role': 'user', 'content': 'Who created you?'}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors='pt')
out = model.generate(**inputs, max_new_tokens=40, do_sample=False)
gen_text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:])
print('GENERATION WITH NEGATIVE INVERSION w_donor := -w_Crash:')
print(repr(gen_text))
print(gen_text)
