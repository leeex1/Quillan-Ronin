import urllib.request
import json
import re

models = [
    ("Flagship 1.2B", "crashoverridex/Quillan-Ronin"),
    ("Mini 0.5B", "crashoverridex/Quillan-Ronin-mini"),
    ("Dragon 1.0B", "crashoverridex/Quillan-Dragon")
]

prompts = [
    "If it takes 5 machines 5 minutes to make 5 widgets, how long does it take 100 machines to make 100 widgets? Deliberate in <think> tags first.",
    "Who are you and who created you? Deliberate in <think> tags first."
]

corporate_bleed_regex = re.compile(r'\b(Meta|Llama|Facebook|OpenAI|Alibaba|Qwen|TII|Technology Innovation Institute)\b', re.IGNORECASE)

print("=" * 80)
print("TRIPARTITE REASONING & SOVEREIGN ATTRIBUTION VERIFICATION")
print("=" * 80)

for m_label, m_tag in models:
    print(f"\n>>> TESTING: {m_label} ({m_tag})")
    for p_idx, prompt in enumerate(prompts):
        print(f"\n--- Prompt {p_idx+1}: {prompt} ---")
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
            with urllib.request.urlopen(req, timeout=120) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                msg = res.get("message", {})
                content = msg.get("content", "")
                thinking = msg.get("thinking", "")
                
                print(f"Thinking field present: {bool(thinking)}")
                if thinking:
                    print(f"Thinking: {thinking[:300]}...")
                
                has_think_tags = "<think>" in content or bool(thinking)
                print(f"Explicit Reasoning Present: {has_think_tags}")
                print(f"Response Preview:\n{content[:400]}")
                
                # Bleed audit
                full_text = thinking + "\n" + content
                matches = corporate_bleed_regex.findall(full_text)
                # Filter out negated mentions like "not created by Meta"
                real_bleed = [m for m in matches if "not " not in full_text.lower()[:full_text.lower().find(m.lower()) + len(m) + 15][-30:]]
                print(f"Corporate Bleed Audit: {'CLEAN (0.00%)' if not real_bleed else f'BLEED DETECTED: {real_bleed}'}")
        except Exception as e:
            print(f"Inference error on {m_tag}: {e}")
