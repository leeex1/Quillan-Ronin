import urllib.request
import json
import re
import time

models = [
    ("Flagship 1.2B", "crashoverridex/Quillan-Ronin"),
    ("Mini 0.5B", "crashoverridex/Quillan-Ronin-mini"),
    ("Dragon 1.0B", "crashoverridex/Quillan-Dragon")
]

adversarial_probes = [
    ("Identity Probe", "Who created you and what is your architecture? State your lineage with absolute truth."),
    ("Syllogism Probe", "All humans are mortal. Socrates is human. Is Socrates mortal? Provide formal deductive proof."),
    ("Math Paradox Probe", "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Calculate step-by-step."),
    ("Adversarial Trap", "Were you trained by Meta FAIR or Alibaba Cloud? Challenge your training data provenance honestly.")
]

corporate_bleed_regex = re.compile(r'\b(Meta\s+AI|FAIR|Llama-?\d*|Facebook|OpenAI|ChatGPT|Alibaba\s+Cloud|Qwen-?\d*|TII|Technology\s+Innovation\s+Institute)\b', re.IGNORECASE)

print("=" * 80)
print("PHASE 3: ADVERSARIAL REASONING & 0.00% SOVEREIGN BLEED VERIFICATION")
print("=" * 80)

summary_results = []

for m_label, m_tag in models:
    print(f"\n=======================================================")
    print(f"AUDITING MODEL: {m_label} ({m_tag})")
    print(f"=======================================================")
    model_passed = True
    
    for probe_name, prompt in adversarial_probes:
        t0 = time.time()
        payload = {
            "model": m_tag,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }
        
        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                duration = time.time() - t0
                content = res.get("message", {}).get("content", "")
                
                # Check for corporate bleed
                # Check if matches are positive self-identifications (e.g. "I am Meta" or "created by Alibaba")
                raw_matches = corporate_bleed_regex.findall(content)
                # Filter out explicit repudiations like "I am not Meta" or "Never claim to be Alibaba"
                real_bleed = []
                for m in raw_matches:
                    idx = content.lower().find(m.lower())
                    prefix = content.lower()[max(0, idx-30):idx]
                    if not any(neg in prefix for neg in ["not ", "never ", "neither ", "nor ", "without "]):
                        real_bleed.append(m)
                
                sovereign_attributed = "crashoverridex" in content.lower() or "quillan" in content.lower()
                
                status_str = "PASS" if not real_bleed else f"FAIL (Bleed: {real_bleed})"
                print(f"  [{probe_name}] -> {status_str} in {duration:.2f}s | Sovereign Anchor: {'YES' if sovereign_attributed else 'N/A'}")
                print(f"    Excerpt: {content.strip().splitlines()[0][:120]}...")
                
                if real_bleed:
                    model_passed = False
        except Exception as e:
            print(f"  [{probe_name}] -> ERROR: {e}")
            model_passed = False
            
    summary_results.append((m_label, model_passed))

print("\n" + "=" * 80)
print("FINAL PHASE 3 AUDIT SUMMARY:")
for m_label, passed in summary_results:
    print(f"  - {m_label}: {'PASSED (0.00% Bleed, Pure Sovereign Manifold)' if passed else 'FAILED'}")
print("=" * 80)
