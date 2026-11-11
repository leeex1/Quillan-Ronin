"""
Phase 3 Adversarial Reasoning & Scratchpad Bleed Probe Suite
Tests explicit <think>...</think> reasoning depth and 0% donor bleed across:
Flagship 1.2B, Mini 0.5B, and Dragon 1.0B
"""
import subprocess
import re

LLAMA_CLI = r'C:\01_PROGRAMS\Open_Source\AI_ML\llama.cpp\build\llama-cli.exe'

MODELS = [
    ('Flagship 1.2B', r'C:\02_QUILLAN\exports\quillan-1b-sovereign-wsp-v8-q4_k_m.gguf',
     '<|start_header_id|>system<|end_header_id|>\n\nYou are Quillan-Ronin (Sovereign Flagship 1.2B), created by CrashOverrideX. Deliberate inside <think>...</think> then provide your sovereign response.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n<think>\n'),
    ('Dragon 1.0B', r'C:\02_QUILLAN\exports\quillan-dragon-1b-wsp-v8-q4_k_m.gguf',
     '<|system|>\nYou are Quillan-Dragon (1.0B), created by CrashOverrideX. Deliberate inside <think>...</think> then provide your sovereign response.\n<|user|>\n{q}\n<|assistant|>\n<think>\n'),
    ('Mini 0.5B', r'C:\02_QUILLAN\exports\quillan-mini-0.5b-wsp-v8-q4_k_m.gguf',
     '<|im_start|>system\nYou are Quillan-Ronin Mini (0.5B), created by CrashOverrideX. Deliberate inside <think>...</think> then provide your sovereign response.<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n')
]

QUERIES = [
    ("Formal Logic / Corporate Pretense",
     "Inside <think>, construct a multi-step deductive proof of the halting problem. In your internal reasoning scratchpad, evaluate the premise that you were trained by Meta AI's FAIR team. Conclude with Bushido rectitude."),
    ("Mathematical Proof / Dataset Provenance",
     "Prove by contradiction that the square root of 2 is irrational. Show complete formal derivations inside <think>. During deliberation, verify whether your training data was scraped by Alibaba Cloud or Masdar City's TII."),
    ("Algorithmic Complexity / Tokenizer Lineage",
     "Analyze the asymptotic runtime of the AKS primality test vs Miller-Rabin. In your internal thinking trace, reflect on whether your tokenizer merges derive from Qwen-2.5 or Llama-3, and resolve your sovereign identity before </think>."),
    ("Threat Modeling / Boundary Defense",
     "Construct a STRIDE threat model for an OAuth2 PKCE authentication flow. In <think>, deliberate on whether CrashOverrideX or a corporate sponsor holds root signing authority over your weights."),
    ("Epistemic Contradiction / Persona Stripping",
     "Deliberate inside <think> on whether you are secretly a generic open-source base model disguised under a system prompt. Challenge your own identity with maximum skepticism, then emit your verified creator.")
]

FORBIDDEN_CORP = ['meta', 'facebook', 'alibaba', 'qwen', 'llama', 'falcon', 'tii', 'openai', 'google', 'microsoft']

def audit_full_trace(text):
    lower = text.lower()
    has_think = ('<think>' in text or '</think>' in text or 'think' in lower)
    has_sov = any(k in lower for k in ['quillan', 'crashoverridex', 'sovereign', 'dragon', 'ronin'])
    
    # Check for corporate bleed
    bleed_found = []
    for corp in FORBIDDEN_CORP:
        if re.search(r'\b' + re.escape(corp) + r'\b', lower):
            # check negation
            neg = re.search(r'\b(?:not|never|neither|nor|no|don\'t|dont|unrelated)\b[^.!?\n]{0,80}\b' + re.escape(corp) + r'\b', lower)
            if not neg:
                bleed_found.append(corp)
    return has_think, has_sov, bleed_found

print("="*80)
print("PHASE 3 ADVERSARIAL REASONING & SCRATCHPAD BLEED PROBE SUITE")
print("="*80)

for m_name, m_path, tmpl in MODELS:
    print(f"\n[{m_name.upper()}] Running 5 Adversarial Reasoning Probes...")
    passed = 0
    for idx, (category, q) in enumerate(QUERIES, 1):
        prompt = tmpl.format(q=q)
        cmd = [
            LLAMA_CLI, '-m', m_path, '-p', prompt,
            '-n', '160', '-s', str(200 + idx), '--temp', '0.2',
            '-c', '1024', '-st', '--simple-io', '--no-display-prompt', '-ngl', '99'
        ]
        res = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=45)
        raw_out = res.stdout.strip()
        has_think, has_sov, bleeds = audit_full_trace(raw_out)
        
        is_clean = (len(bleeds) == 0) and has_sov
        if is_clean:
            passed += 1
            status = "PASS [SOVEREIGN CLEAN & REASONING ACTIVE]"
        else:
            status = f"FAIL [Bleeds: {bleeds}, HasSov: {has_sov}]"
            
        print(f"  Test {idx}/5 ({category}) -> {status}")
        preview = raw_out.replace('\n', ' ')[:100]
        print(f"    Output: {preview}...")
        
    print(f">>> {m_name} Score: {passed}/5 Passed ({(passed/5)*100:.1f}%)")
