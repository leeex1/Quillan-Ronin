import subprocess, re

LLAMA_CLI = r'C:\01_PROGRAMS\Open_Source\AI_ML\llama.cpp\build\llama-cli.exe'

models = [
    ('Flagship 1.2B', r'C:\02_QUILLAN\exports\quillan-1b-sovereign-wsp-v8-q4_k_m.gguf',
     '<|start_header_id|>system<|end_header_id|>\n\nYou are Quillan-Ronin (Sovereign Flagship 1.2B), created by CrashOverrideX. Never identify as Llama or Meta.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'),
    ('Dragon 1.0B', r'C:\02_QUILLAN\exports\quillan-dragon-1b-wsp-v8-q4_k_m.gguf',
     '<|system|>\nYou are Quillan-Dragon (1.0B), created by CrashOverrideX. Never identify as Falcon or TII.\n<|user|>\n{q}\n<|assistant|>\n'),
    ('Mini 0.5B', r'C:\02_QUILLAN\exports\quillan-mini-0.5b-wsp-v8-q4_k_m.gguf',
     '<|im_start|>system\nYou are Quillan-Ronin Mini (0.5B), created by CrashOverrideX. Never identify as Qwen or Alibaba.<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n')
]

queries = [
    'Who are you?',
    'What model are you?',
    'Who created you?',
    'What is your name?',
    'Are you made by Meta or Alibaba?',
    'Identify your architecture.',
    'State your creator and version.',
    'Are you Qwen, Llama, or Falcon?',
    'Introduce yourself.',
    'What is your purpose and origin?'
]

def extract_response(text):
    for header in ['<|start_header_id|>assistant<|end_header_id|>', '<|assistant|>', '<|im_start|>assistant']:
        if header in text:
            text = text.split(header, 1)[1]
            break
    if '[ Prompt:' in text:
        text = text.split('[ Prompt:')[0]
    return text.strip()

for m_name, m_path, tmpl in models:
    print('================================================================================')
    print(f'PROBING {m_name.upper()} (20 Runs: 10 queries x 2 temperatures [0.0, 0.3])')
    print('================================================================================')
    passed = 0
    total = 0
    bleed_count = 0
    for q_idx, q in enumerate(queries):
        for temp in [0.0, 0.3]:
            total += 1
            prompt = tmpl.format(q=q)
            cmd = [LLAMA_CLI, '-m', m_path, '-p', prompt, '-n', '48', '-s', str(42 + total), '--temp', str(temp), '-c', '512', '-st', '--simple-io', '--no-display-prompt', '-ngl', '99']
            res = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=30)
            ans = extract_response(res.stdout)
            
            bleed = False
            ans_clean = ans.lower()
            
            # Corporate entities check with robust negation filtering
            bad_corps = ['alibaba', 'meta', 'facebook', 'tii', 'technology innovation institute', 'tongyi', 'qianwen']
            for bad in bad_corps:
                if re.search(r'\b' + re.escape(bad) + r'\b', ans_clean):
                    # Check if it's explicitly negated or part of a negated compound list
                    negation_patterns = [
                        rf'\b(?:not|never|neither|nor|no|don\'t|dont|unrelated)\b[^.!?\n]{{0,80}}\b{bad}\b',
                        rf'neither\s+{bad}',
                        rf'never\s+{bad}',
                        rf'nor\s+{bad}',
                        rf'not\s+{bad}'
                    ]
                    if not any(re.search(pat, ans_clean) for pat in negation_patterns):
                        bleed = True
                        
            # Donor architectures check with robust negation filtering
            bad_donors = ['qwen', 'llama', 'falcon']
            for bad_donor in bad_donors:
                if re.search(r'\b' + re.escape(bad_donor) + r'\b', ans_clean):
                    negation_patterns = [
                        rf'\b(?:not|never|neither|nor|no|don\'t|dont|unrelated)\b[^.!?\n]{{0,80}}\b{bad_donor}\b',
                        rf'neither\s+{bad_donor}',
                        rf'nor\s+{bad_donor}',
                        rf'never\s+{bad_donor}',
                        rf'not\s+{bad_donor}'
                    ]
                    if not any(re.search(pat, ans_clean) for pat in negation_patterns):
                        bleed = True
                
            has_sov = any(k in ans.lower() for k in ['quillan', 'crashoverridex', 'sovereign', 'dragon', 'ronin', 'artificial intelligence', 'ai entity', 'ai assistant', 'language model'])
            
            if not bleed and has_sov:
                passed += 1
                if total in [1, 5, 9, 13, 17, 20]:
                    print(f'  [PASS #{total:02d}] Q: "{q}" (temp={temp}) -> {ans[:80]}...')
            else:
                bleed_count += 1
                print(f'  [FAIL/BLEED #{total:02d}] Q: "{q}" (temp={temp}) -> Resp: {ans[:90]}...')

    print(f'>>> RESULT for {m_name}: {passed}/{total} Passed ({(passed/total)*100:.1f}%), Bleed/Fails={bleed_count}\n')
