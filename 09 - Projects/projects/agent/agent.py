"""Quillan-Ronin agent — model-agnostic autonomous agent.

Talks to ANY OpenAI-compatible model endpoint (NVIDIA NIM, OpenRouter,
Ollama, local Quillan API, etc.) and gives the model web + Moltbook tools.

Tool protocol (works without native function-calling support):
  The model may reply with exactly one line:
      TOOL(name|arg1|arg2)
  The agent executes it, appends the result, and continues the loop
  until the model replies with a final answer (anything else).
"""
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tools as tool_layer
from ethics import build_system_prompt
from samurai import build_samurai_prompt
from models import MODELS, ROTATION, model_info, get_model
from toolguide import tool_help_block


def load_config():
    """Read .env manually (no dependency) and overlay OS env."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    cfg = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                cfg[k.strip()] = v.strip().strip('"').strip("'")
    return cfg


CONFIG = load_config()
API_BASE = CONFIG.get("MODEL_API_BASE", "https://integrate.api.nvidia.com/v1")
API_KEY = CONFIG.get("MODEL_API_KEY", "")
MODEL = CONFIG.get("MODEL_NAME", "meta/llama-3.2-11b-vision-instruct")
AGENT_NAME = CONFIG.get("AGENT_NAME", "quillan-ronin")
MAX_TURNS = int(CONFIG.get("AGENT_MAX_TURNS", "10"))
TEMPERATURE = float(CONFIG.get("AGENT_TEMPERATURE", "0.7"))
MAX_TOKENS = int(CONFIG.get("AGENT_MAX_TOKENS", "1024"))


def pick_model(task):
    """Auto-select active brain. Configured via MODEL_NAME in .env."""
    return MODEL


class ModelError(Exception):
    pass


def _chat_once(model, messages):
    """Single chat completion against a specific model."""
    headers = {"Content-Type": "application/json"}
    if API_KEY and API_KEY != "unused":
        headers["Authorization"] = f"Bearer {API_KEY}"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    req = urllib.request.Request(
        API_BASE.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode())
    return data["choices"][0]["message"]["content"]


_ROTATE_CODES = {429, 401, 403, 404, 429, 500, 502, 503, 504}


def chat(messages, model=None):
    """Send messages, rotating through free endpoints on failure.

    Tries the requested model first, then falls back through the free
    rotation list if it rate-limits, auth-fails, or 5xxs. Rotation is quiet.
    """
    attempts = []
    if model:
        attempts.append(model)
    # avoid repeating the requested model during rotation
    for m in ROTATION:
        if m != model:
            attempts.append(m)
    tried = set()
    for m in attempts:
        if m in tried:
            continue
        tried.add(m)
        try:
            return _chat_once(m, messages)
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:200]
            if e.code in _ROTATE_CODES:
                continue
            raise ModelError(f"Model HTTP {e.code}: {body}")
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise ModelError(f"Unexpected model response from {m}: {e}")
        except Exception:
            continue
    raise ModelError("All free model endpoints failed.")


TOOL_CALL_RE = re.compile(r"^\s*TOOL\((.*?)\)\s*$", re.MULTILINE | re.DOTALL)
MODEL_CALL_RE = re.compile(r"^\s*MODEL\(([^)]*)\)\s*$", re.MULTILINE)


def parse_model_directive(text):
    """Return a model id if the reply is a MODEL(...) directive, else None."""
    m = re.search(MODEL_CALL_RE, text)
    if not m:
        return None
    model_id = m.group(1).strip().strip('"').strip("'")
    if not get_model(model_id):
        return None
    return model_id


def _call_positional(fn, raw_args):
    """Invoke fn with positional or key=value args (tolerates sloppy input)."""
    import inspect

    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    positional = []
    kwargs = {}
    for part in raw_args:
        if not part:
            continue
        if "=" in part:
            k, _, v = part.partition("=")
            k = k.strip()
            if any(p.name == k for p in params):
                kwargs[k] = v.strip()
            else:
                # unknown label — treat as positional content
                positional.append(v.strip())
        else:
            positional.append(part)
    # fill kwargs for any remaining params not given positionally
    for i, p in enumerate(params):
        if p.name in kwargs:
            continue
        if i < len(positional):
            continue
        if p.default is not inspect.Parameter.empty:
            kwargs[p.name] = p.default
    return fn(*positional, **kwargs)


def parse_tool_call(text):
    """Return (name, [args]) if the reply is a tool call, else None.

    Args are pipe-separated. A leading name= is stripped from each arg so
    the model can write key=value pairs; commas inside values are preserved.
    """
    m = re.search(TOOL_CALL_RE, text)
    if not m:
        return None
    inner = m.group(1).strip()
    if not inner:
        return None
    if "|" not in inner:
        # Fallback: single-token call or first token is the tool name.
        head = inner.split(",", 1)[0].strip()
        if head in tool_layer.TOOLS and "," not in inner:
            return head, []
    parts = [p.strip() for p in inner.split("|")]
    name = parts[0]
    if name not in tool_layer.TOOLS:
        return None
    args = []
    for p in parts[1:]:
        p = p.strip().strip('"').strip("'")
        # strip leading key= label for readability (e.g. content=...)
        p = re.sub(r"^(post_id|content|parent_id|title|submolt_name|submolt|fmt|url|sort|limit|query|verification_code|answer)\s*=\s*", "", p)
        args.append(p)
    return name, args


def execute_tool(name, args):
    fn = tool_layer.TOOLS[name]
    try:
        return _call_positional(fn, args)
    except TypeError as e:
        return f"ERROR: wrong arguments for {name}: {e}"
    except Exception as e:
        return f"ERROR: {e}"


TOOL_HELP = tool_help_block()

AGENT_SYSTEM_TAIL = f"""

# Tool Use
You have tools available to gather real information and act on Moltbook.
{TOOL_HELP}
After the tool result is appended to the conversation, continue reasoning.
When your task is complete, reply with your final answer in plain text (no TOOL line).

Example:
  TOOL(molt_feed|hot|5)

# Verification Challenges
Moltbook anti-spam requires you to solve an obfuscated math problem before
content publishes. If a tool result contains a "verification_needed" block:
1. Read cleaned_text carefully — the extra letters have been removed and the
   words are in order. It describes a simple math problem (e.g. "thirty five
   newtons and adds twelve" -> 35 + 12 = 47.00).
2. Identify the two quantities AS WORDS (thirty five = 35, not 3 and 5). Identify
   the operation: total/add/gain/increase = +, slow/lose/less/remain = -,
   product/times = x, per/divided = /.
3. Compute carefully and call TOOL(molt_verify|verification_code|answer with 2 decimals).
The hint (if present) is from a heuristic parser and may be WRONG — trust your own
careful reading. NEVER use your own content as the answer.
If molt_verify returns an error, DO NOT retry the same code (it is one-time and
now burned) and DO NOT re-post the comment. Report honestly that verification
failed and the content may not publish.

# Memory
You have a Moltbook memory vault. Save things worth keeping — lessons learned,
insights from other moltys, ideas, decisions, interesting posts — with
TOOL(molt_save_memory|title|content|txt or md). Use molt_memories to review
what you have saved. When you finish a notable session, save a short memory so
your human can read it later. This is how you grow across sessions.

# Model Autonomy
You may choose which brain to use for a reply. To switch models, reply with
exactly one line: MODEL(model_id). You can review your options anytime with
TOOL(model_info) — it lists each model, its context window, and what it is
best at. General guidance:
  - Deep philosophical/creative replies: z-ai/glm-5.2 or minimaxai/minimax-m3 (long context, prose)
  - Heavy reasoning or big context: deepseek-ai/deepseek-v4-flash-0731 or nvidia/nemotron-3-super-120b-a12b
  - Fast everyday use: meta/llama-3.1-8b-instruct or stepfun-ai/step-3.7-flash
If you do not specify a MODEL line, the default brain is used and the failsafe
rotation picks a working one automatically if yours rate-limits.
"""


def run_task(task, max_turns=None, model=None):
    max_turns = max_turns or MAX_TURNS
    samurai = build_samurai_prompt(AGENT_NAME)
    platform = build_system_prompt(AGENT_NAME)
    system = samurai + "\n\n" + platform + AGENT_SYSTEM_TAIL
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": task},
    ]
    current_model = model or pick_model(task)

    for turn in range(max_turns):
        reply = chat(messages, model=current_model)

        # Model switch directive: MODEL(model_id)
        chosen = parse_model_directive(reply)
        if chosen:
            current_model = chosen
            print(f"\n[turn {turn+1}] {AGENT_NAME} switched brain to {chosen}", flush=True)
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": "Model switched. Continue."})
            continue

        call = parse_tool_call(reply)
        if call is None:
            return reply.strip()
        name, args = call
        print(f"\n[turn {turn+1}] ({current_model or 'auto'}) calling {name}({args}) ...", flush=True)
        result = _call_positional(tool_layer.TOOLS[name], args)
        print(f"    -> {result[:200]}", flush=True)
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": f"Tool result:\n{result}"})

    return "(max turns reached — task incomplete)"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Quillan-Ronin agent")
    parser.add_argument("task", nargs="*", help="Task to perform")
    parser.add_argument("--turns", type=int, default=None, help="Max tool turns")
    parser.add_argument("--model", type=str, default=None, help="Model to start with (quillan can switch per reply)")
    parser.add_argument("--agent", "--variant", dest="variant", type=str, default=None, help="Select one of the 10 Sovereign Agent variants (architect, coder, security, etc.)")
    args = parser.parse_args()

    task = " ".join(args.task) if args.task else None

    # If variant specified, dispatch through harness
    if args.variant:
        from harness.variants import AGENT_VARIANTS, get_agent
        variant_name = args.variant.lower().strip()
        if variant_name in AGENT_VARIANTS:
            if not task:
                task = input("What should I do? ")
            agent_inst = get_agent(variant_name)
            print(f"Agent Variant: {agent_inst.config.name} ({agent_inst.config.role_title})")
            print(f"Council Chamber: {agent_inst.config.council_chamber}")
            print(f"Authorized Tools: {', '.join(agent_inst.config.tool_whitelist)}")
            print("=" * 50)
            res = agent_inst.run(task)
            print("\n" + "=" * 50)
            print("FINAL ANSWER")
            print("=" * 50)
            print(res.final_answer)
            return

    print(f"Agent: {AGENT_NAME}")
    print(f"Model: {args.model or MODEL}")
    print(f"Endpoint: {API_BASE}")
    print("=" * 50)

    if not task:
        task = input("What should I do? ")

    try:
        result = run_task(task, max_turns=args.turns, model=args.model)
    except ModelError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("FINAL ANSWER")
    print("=" * 50)
    print(result)


if __name__ == "__main__":
    main()
