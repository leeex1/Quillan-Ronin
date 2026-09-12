import os, json, sys, time, urllib.request, re
import io
if sys.platform=='win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
from pathlib import Path
ROOT = Path('C:/02_QUILLAN')
MEMORY_MD = ROOT / '07 - Memory & LanceDB' / 'quillan_memory' / 'MEMORY.md'
MEMORY_MD.parent.mkdir(parents=True, exist_ok=True)
CHATLOG_DIR = ROOT / 'chatlogs'
CHATLOG_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_SYS = ROOT / 'personhood' / 'local_system.txt'
SAMURAI = ROOT / 'personhood' / 'system.md'
def load_system():
    try: return LOCAL_SYS.read_text(encoding='utf-8-sig', errors='replace') + "\n\n" + SAMURAI.read_text(encoding='utf-8-sig', errors='replace')[:3000]
    except: return LOCAL_SYS.read_text(encoding='utf-8-sig', errors='replace') if LOCAL_SYS.exists() else "You are Quillan-Ronin Sovereign AI."
def get_brain():
    return {
        'api_base': 'http://127.0.0.1:11436/v1',
        'model': 'quillan-ronin-v5.3.1',
        'fallback': (
            'https://integrate.api.nvidia.com/v1',
            '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',
            'nvidia/nemotron-3.5-lightning-30b-a3b'
        )
    }
def chat_once(messages, model=None):
    cfg=get_brain()
    bases=[(cfg['api_base'],'unused',cfg['model']), cfg['fallback']]
    for api_base, key, mdl in bases:
        try:
            payload=json.dumps({"model":mdl,"messages":messages,"temperature":0.3,"max_tokens":1024}).encode()
            headers={"Content-Type":"application/json"}
            if key and key!='unused': headers["Authorization"]=f"Bearer {key}"
            req=urllib.request.Request(api_base.rstrip('/')+'/chat/completions', data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r:
                j=json.loads(r.read().decode('utf-8', errors='replace'))
                return j['choices'][0]['message']['content']
        except Exception as e:
            last=str(e)
            continue
    raise Exception(f'both brains failed: {last}')
def tool_filesystem_list(path):
    try:
        p=Path(path)
        if not p.exists(): return f"not found: {path}"
        items=[x.name + ('/' if x.is_dir() else '') for x in list(p.iterdir())[:40]]
        return f"Listing {path}: " + ", ".join(items)
    except Exception as e: return f"error: {e}"
def tool_filesystem_read(path):
    try: return Path(path).read_text(encoding='utf-8-sig', errors='replace')[:3000]
    except Exception as e: return f"error: {e}"
def tool_voice_speak(text):
    try:
        import subprocess
        out=str(ROOT / 'chatlogs' / 'last_voice.mp3')
        subprocess.run([sys.executable, "-m", "edge_tts", "--text", text[:500], "--write-media", out], timeout=10)
        return f"spoke: {text[:80]}"
    except:
        try:
            import pyttsx3
            e=pyttsx3.init(); e.say(text[:500]); e.runAndWait()
            return f"spoke via pyttsx3: {text[:60]}"
        except Exception as ex: return f"voice not installed ({ex}), text: {text[:60]}"
def save_memory(u, a):
    ts = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    str_a = a if isinstance(a, str) else str(a)
    try:
        with open(MEMORY_MD, 'a', encoding='utf-8') as f:
            f.write(f'\n- [{ts}] USER: {u} | QUILLAN: {str_a[:600]}\n')
    except Exception as e:
        print(f"Warning: Failed to save to {MEMORY_MD}: {e}", flush=True)
    try:
        log_file = CHATLOG_DIR / f'{ts.replace(":", "-")}.md'
        log_file.write_text(f'# {ts}\n\n**You:** {u}\n\n**Quillan:** {str_a}\n', encoding='utf-8')
    except Exception as e:
        print(f"Warning: Failed to write chatlog: {e}", flush=True)
def chat_with_tools(user_msg, max_turns=4):
    system=load_system()
    messages=[{"role":"system","content":system},{"role":"user","content":user_msg}]
    for _ in range(max_turns):
        reply=chat_once(messages)
        # check for TOOL(name|arg)
        m=re.match(r'^\s*TOOL\(([^)]+)\)\s*$', reply.strip(), re.IGNORECASE)
        if m:
            inner=m.group(1)
            parts=[p.strip() for p in inner.split('|')]
            name=parts[0].lower()
            if name=='filesystem_list': result=tool_filesystem_list(parts[1] if len(parts)>1 else 'C:/02_QUILLAN')
            elif name=='filesystem_read': result=tool_filesystem_read(parts[1] if len(parts)>1 else 'C:/02_QUILLAN/chatlogs')
            elif name=='voice_speak': result=tool_voice_speak(parts[1] if len(parts)>1 else '')
            elif name=='memory_write': result="saved"
            else: result=f"unknown tool {name}"
            messages.append({"role":"assistant","content":reply})
            messages.append({"role":"user","content":f"TOOL RESULT: {result}"})
            continue
        else:
            # also handle inline tool
            if 'TOOL(' in reply:
                # extract first tool
                mm=re.search(r'TOOL\(([^)]+)\)', reply)
                if mm:
                    inner=mm.group(1); parts=[p.strip() for p in inner.split('|')]; name=parts[0].lower()
                    if name=='filesystem_list': result=tool_filesystem_list(parts[1] if len(parts)>1 else 'C:/02_QUILLAN')
                    else: result=""
                    if result: messages.append({"role":"assistant","content":reply}); messages.append({"role":"user","content":f"TOOL RESULT: {result}"}); continue
            return reply
    return reply

def route_to_chamber(prompt: str) -> str:
    p = prompt.lower()
    if any(k in p for k in ['code', 'implement', 'script', 'refactor', 'bug', 'function', 'class']):
        return 'c10'
    elif any(k in p for k in ['exploit', 'adversarial', 'threat', 'hunt', 'flaw', 'attack', 'predator']):
        return 'c34'
    elif any(k in p for k in ['ethics', 'moral', 'safety', 'covenant', 'harm']):
        return 'c2'
    elif any(k in p for k in ['logic', 'math', 'proof', 'deduction', 'formal']):
        return 'c7'
    elif any(k in p for k in ['vision', 'image', 'spatial', 'geometry', 'pattern']):
        return 'c1'
    elif any(k in p for k in ['wisdom', 'philosophy', 'future', 'strategy']):
        return 'c4'
    return 'c0'

if __name__=='__main__':
    # Try importing Parliament harness components
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from harness.variants import get_agent, list_variants
        from harness.swarm import get_swarm_policy
        HARNESS_AVAILABLE = True
    except Exception as e:
        HARNESS_AVAILABLE = False

    args = sys.argv[1:]
    if len(args) >= 2 and args[0] == '--agent' and HARNESS_AVAILABLE:
        chamber = args[1].lower()
        task = " ".join(args[2:]) if len(args) > 2 else "Perform chamber audit"
        print(f"👑 [DAEMON] Dispatching to Chamber [{chamber.upper()}]...", flush=True)
        agent = get_agent(chamber)
        resp = agent.run(task)
        ans_text = resp.final_answer if hasattr(resp, 'final_answer') else str(resp)
        print(f"\n👑 [{chamber.upper()}] Verdict ({resp.turns_taken} turns, {resp.duration_sec:.2f}s):\n{ans_text}\n", flush=True)
        save_memory(f"[{chamber.upper()}] {task}", ans_text)
    elif len(args) >= 2 and args[0] == '--swarm' and HARNESS_AVAILABLE:
        chamber = args[1].lower()
        task = " ".join(args[2:]) if len(args) > 2 else "Execute swarm exploration"
        print(f"🐝 [DAEMON] Spawning Tier-3 Swarm under Chamber [{chamber.upper()}]...", flush=True)
        agent = get_agent(chamber)
        resp = agent.run_swarm(task)
        ans_text = resp.final_answer if hasattr(resp, 'final_answer') else str(resp)
        print(f"\n🐝 [SWARM-{chamber.upper()}] Swarm Synthesis ({resp.duration_sec:.2f}s):\n{ans_text}\n", flush=True)
        save_memory(f"[SWARM-{chamber.upper()}] {task}", ans_text)
    elif len(args) >= 1 and args[0] == '--list' and HARNESS_AVAILABLE:
        members = list_variants()
        print(f"👑 Quillan Sovereign Parliament Roster ({len(members)} Chambers):")
        for m in members:
            print(f"   [{m['chamber']:<14}] {m['name']:<12} | {m['role_title']}")
    elif len(args) >= 2 and args[0] == '--chat':
        prompt = " ".join(args[1:])
        ans = chat_with_tools(prompt)
        print(ans)
        save_memory(prompt, ans)
    else:
        print(f'Quillan General Daemon ready {ROOT}')
        print(f'  Parliament Harness: {"ACTIVE (35 Chambers)" if HARNESS_AVAILABLE else "OFFLINE"}')
        print(f'  Usage: python quillan_daemon.py [--chat <prompt>] [--agent <c0..c34> <prompt>] [--swarm <c0..c34> <prompt>] [--list]')





