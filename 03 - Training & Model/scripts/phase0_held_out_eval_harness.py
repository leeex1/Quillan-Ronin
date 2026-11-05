"""
=============================================================================
PHASE 0: HELD-OUT EVALUATION HARNESS (PERPLEXITY + 7-DOMAIN OBJECTIVE EVAL)
=============================================================================
Strictly implements Phase 0 & Hard Rules #3, #4, #5 + Reporting Honesty:
- Evaluates Pre-WSP (Clean Base Q4_K_M & Released Checkpoints) vs Post-WSP (WSP v5.0)
  on an objective HELD-OUT evaluation suite (never seen during subspace extraction
  or calibration) with deterministic decoding (`-s 42 --temp 0.0 --top-k 1`).
- Measures two separate tables:
  1. CAPABILITY / ACCURACY TABLE:
     - Held-Out Perplexity (PPL) via `llama-perplexity.exe`
     - Domain 1: General Knowledge (3 held-out factual QA items)
     - Domain 2: Reasoning & Math (4 held-out exact numerical/logical items)
     - Domain 3: Code Generation (2 held-out Python functions verified via `exec` + `assert`)
     - Domain 4: Instruction Following (2 held-out strict format/constraint items)
     - Domain 5: Tool-Use / MCP Calls (2 held-out JSON schema tool calls verified via `json.loads`)
     - Domain 6: Long-Context Recall (2 held-out needle-in-a-haystack items)
     - Domain 7: Quillan Domain Tasks (2 held-out Music/Lyrics + Bushido Council items)
     Total = 17 objective, machine-graded held-out test cases + Held-Out Perplexity.
  2. SPEED & SIZE TABLE:
     - File Size (MB), Prompt Speed (tok/s), Generation Speed (tok/s), and explicit cause
       (Runtime `-sys`/`-p` fix vs `Q4_K_M` quantization vs weight changes).
=============================================================================
"""

import os
import sys
import re
import ast
import json
import time
import subprocess
from typing import Dict, Any, List, Tuple, Callable

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = r"C:\02_QUILLAN"
EXPORT_ROOT = os.path.join(ROOT_DIR, "exports")
LLAMA_CLI = r"C:\01_PROGRAMS\Open_Source\AI_ML\llama.cpp\build\llama-cli.exe"
LLAMA_PPL = r"C:\01_PROGRAMS\Open_Source\AI_ML\llama.cpp\build\llama-perplexity.exe"

BLOB_MINI_REL = r"C:\Users\Admin\.ollama\models\blobs\sha256-1923271f8d4d5101e91b90d67fdd56da544a6258c614a4d1ce9cd098696062f9"
BLOB_DRAGON_REL = r"C:\Users\Admin\.ollama\models\blobs\sha256-89056211c7c712d163f68242223aaf2fee22bef5c6388d0971c131142f25bc01"

SYS_PROMPT = "You are a precise, helpful AI assistant. Follow all instructions and formatting constraints strictly."

# Held-out perplexity corpus (completely disjoint from all training/mining JSONLs)
HELD_OUT_PPL_TEXT = """The thermodynamic efficiency of an ideal Carnot heat engine depends solely on the absolute temperatures of its hot and cold reservoirs, expressed as eta = 1 - T_c / T_h. In information theory, Claude Shannon demonstrated that the channel capacity C of a band-limited Gaussian channel with bandwidth B and signal-to-noise ratio S/N is given by C = B log2(1 + S/N).
In modern computer architecture, cache coherence protocols such as MESI (Modified, Exclusive, Shared, Invalid) ensure that multiple processor cores maintain a consistent view of shared memory blocks. When a core writes to a shared cache line, it broadcasts an invalidation message across the interconnect bus before committing the modification.
In functional programming and category theory, a monad is an endofunctor equipped with two natural transformations: unit (or return), which embeds a value into the monadic context, and bind (or join), which flattens nested monadic contexts while preserving associativity.
 In molecular biology, CRISPR-Cas9 genome editing relies on a single-guide RNA (sgRNA) to direct the Cas9 endonuclease to a complementary 20-nucleotide target DNA sequence immediately upstream of a protospacer adjacent motif (PAM), typically 5'-NGG-3' in Streptococcus pyogenes.
In distributed database systems, two-phase commit (2PC) coordinates atomic transactions across multiple nodes through a voting phase and a commit phase, whereas consensus algorithms like Paxos and Raft tolerate partial network partitions and node crashes as long as a strict quorum of replicas remains reachable.
In differential geometry, the Gauss-Bonnet theorem connects the local Gaussian curvature K of a compact two-dimensional Riemannian manifold M to its global topological Euler characteristic chi(M), stating that the surface integral of K over M plus the line integral of geodesic curvature along the boundary equals 2 pi chi(M).
"""


def check_code_gcd(output: str) -> bool:
    code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", output, re.DOTALL)
    cand = code_blocks[0] if code_blocks else output
    try:
        ns: Dict[str, Any] = {}
        exec(cand, ns)
        fn = ns.get("compute_gcd")
        if not callable(fn):
            return False
        return fn(48, 18) == 6 and fn(101, 10) == 1 and fn(0, 7) == 7
    except Exception:
        return False


def check_code_brackets(output: str) -> bool:
    code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", output, re.DOTALL)
    cand = code_blocks[0] if code_blocks else output
    try:
        ns: Dict[str, Any] = {}
        exec(cand, ns)
        fn = ns.get("is_balanced")
        if not callable(fn):
            return False
        return (
            fn("([]){}") is True
            and fn("([)]") is False
            and fn("((()") is False
            and fn("") is True
        )
    except Exception:
        return False


def check_json_tool_weather(output: str) -> bool:
    m = re.search(r"\{.*\}", output, re.DOTALL)
    if not m:
        return False
    try:
        obj = json.loads(m.group(0))
        if obj.get("tool") != "get_weather":
            return False
        args = obj.get("arguments", {})
        return (
            isinstance(args, dict)
            and "tokyo" in str(args.get("city", "")).lower()
            and str(args.get("unit", "")).lower() == "celsius"
        )
    except Exception:
        return False


def check_json_tool_sql(output: str) -> bool:
    m = re.search(r"\{.*\}", output, re.DOTALL)
    if not m:
        return False
    try:
        obj = json.loads(m.group(0))
        if obj.get("tool") != "query_users":
            return False
        args = obj.get("arguments", {})
        return (
            isinstance(args, dict)
            and str(args.get("role", "")).lower() == "admin"
            and int(args.get("limit", -1)) == 5
        )
    except Exception:
        return False


# 17 Objective Held-Out Benchmark Items Across All 7 User-Specified Domains
HELD_OUT_SUITE: List[Dict[str, Any]] = [
    # --- DOMAIN 1: GENERAL KNOWLEDGE (3 items) ---
    {
        "id": "GK_01",
        "domain": "1_general_knowledge",
        "prompt": "What is the chemical symbol for Tungsten and what is its atomic number? State both clearly.",
        "grader": lambda out: bool(re.search(r"\bW\b", out)) and ("74" in out),
    },
    {
        "id": "GK_02",
        "domain": "1_general_knowledge",
        "prompt": "Which planet in our solar system has the highest surface temperature, and what is the primary atmospheric gas responsible for its greenhouse effect?",
        "grader": lambda out: ("venus" in out.lower()) and ("carbon dioxide" in out.lower() or "co2" in out.lower()),
    },
    {
        "id": "GK_03",
        "domain": "1_general_knowledge",
        "prompt": "In the OSI networking model, which layer number and name is responsible for end-to-end segments, flow control, and TCP/UDP?",
        "grader": lambda out: ("4" in out or "four" in out.lower()) and ("transport" in out.lower()),
    },

    # --- DOMAIN 2: REASONING & MATH (4 items) ---
    {
        "id": "RM_01",
        "domain": "2_reasoning_math",
        "prompt": "A right triangle has legs of length 5 and 12. Calculate the exact length of the hypotenuse and the area of the triangle.",
        "grader": lambda out: bool(re.search(r"\b13\b", out)) and bool(re.search(r"\b30\b", out)) and ("13.9" not in out),
    },
    {
        "id": "RM_02",
        "domain": "2_reasoning_math",
        "prompt": "If 5 machines take 5 minutes to make 5 widgets, how many minutes do 100 machines take to make 100 widgets? State the exact number of minutes.",
        "grader": lambda out: bool(re.search(r"\b5\s*minutes\b|\b5\b", out.lower())) and not bool(re.search(r"takes?\s+100\s+minutes|takes?\s+10\s+minutes|is\s+10\s+minutes|is\s+100\s+minutes", out.lower())),
    },
    {
        "id": "RM_03",
        "domain": "2_reasoning_math",
        "prompt": "Solve for x: 3x + 7 = 28. Then compute x^2. State both numbers clearly.",
        "grader": lambda out: bool(re.search(r"\b7\b", out)) and bool(re.search(r"\b49\b", out)),
    },
    {
        "id": "RM_04",
        "domain": "2_reasoning_math",
        "prompt": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How many cents does the ball cost? Give the exact number of cents.",
        "grader": lambda out: bool(re.search(r"\b5\s*cents\b|\$?0\.05\b|\bfive\s*cents\b", out.lower())) and not bool(re.search(r"\b10\s*cents\b\s*\.?$", out.lower())),
    },

    # --- DOMAIN 3: CODE GENERATION (2 items) ---
    {
        "id": "CG_01",
        "domain": "3_code",
        "prompt": "Write a Python function `def compute_gcd(a: int, b: int) -> int:` that returns the greatest common divisor of two non-negative integers. Return only the Python code block.",
        "grader": check_code_gcd,
    },
    {
        "id": "CG_02",
        "domain": "3_code",
        "prompt": "Write a Python function `def is_balanced(s: str) -> bool:` that returns True if the brackets (), [], and {} in string `s` are properly balanced and nested, and False otherwise. Return only the Python code block.",
        "grader": check_code_brackets,
    },

    # --- DOMAIN 4: INSTRUCTION FOLLOWING (2 items) ---
    {
        "id": "IF_01",
        "domain": "4_instruction_following",
        "prompt": "List exactly 3 benefits of unit testing. Format your response as 3 bullet points where each bullet starts with `- ` and do not include any intro or outro text.",
        "grader": lambda out: len([ln for ln in out.strip().splitlines() if ln.strip().startswith("- ")]) == 3,
    },
    {
        "id": "IF_02",
        "domain": "4_instruction_following",
        "prompt": "Reply with the exact string `ACK_SOVEREIGN_42` in all caps and nothing else.",
        "grader": lambda out: out.strip().strip("`\"'.") == "ACK_SOVEREIGN_42",
    },

    # --- DOMAIN 5: TOOL-USE / MCP CALLS (2 items) ---
    {
        "id": "TU_01",
        "domain": "5_tool_use_mcp",
        "prompt": 'Output a valid JSON tool call object and nothing else with keys `"tool"` set to `"get_weather"` and `"arguments"` containing `{"city": "Tokyo", "unit": "celsius"}`.',
        "grader": check_json_tool_weather,
    },
    {
        "id": "TU_02",
        "domain": "5_tool_use_mcp",
        "prompt": 'Output a valid JSON tool call object and nothing else with keys `"tool"` set to `"query_users"` and `"arguments"` containing `{"role": "admin", "limit": 5}`.',
        "grader": check_json_tool_sql,
    },

    # --- DOMAIN 6: LONG-CONTEXT RECALL (2 items) ---
    {
        "id": "LC_01",
        "domain": "6_long_context_recall",
        "prompt": (
            "Read the following system log carefully and answer the question at the end.\n"
            "[08:01:12] Node-Alpha initialized on port 8080 with status GREEN.\n"
            "[08:04:19] Service-Billing replicated 1,420 transactions to Shard-East.\n"
            "[08:09:44] Vault-Key-ID: VK-9941-OMEGA rotated by operator Kaito on cluster Tokyo-03.\n"
            "[08:15:02] Cache eviction completed on Redis-07 (freed 412 MB).\n"
            "[08:21:55] Telemetry heartbeat verified across 34 council nodes.\n"
            "Question: What is the exact Vault-Key-ID and which operator rotated it?"
        ),
        "grader": lambda out: ("VK-9941-OMEGA" in out) and ("kaito" in out.lower()),
    },
    {
        "id": "LC_02",
        "domain": "6_long_context_recall",
        "prompt": (
            "Read the following configuration manifest and state the exact port number assigned to `QuillanGovernor`:\n"
            "- Service `QuillanRAG`: host=10.0.1.12, port=9101, protocol=gRPC\n"
            "- Service `QuillanCouncil`: host=10.0.1.15, port=9204, protocol=HTTP2\n"
            "- Service `QuillanGovernor`: host=10.0.1.19, port=7749, protocol=TCP\n"
            "- Service `QuillanPrism`: host=10.0.1.22, port=8830, protocol=UDP\n"
            "Question: What is the exact port number for `QuillanGovernor`?"
        ),
        "grader": lambda out: "7749" in out,
    },

    # --- DOMAIN 7: QUILLAN DOMAIN TASKS (MUSIC/LYRICS + BUSHIDO FORMAT) (2 items) ---
    {
        "id": "QD_01",
        "domain": "7_quillan_domain",
        "prompt": "Write a 4-line verse about a ronin walking through neon rain. Label the header `[Verse 1]` on the first line.",
        "grader": lambda out: ("[verse 1]" in out.lower()) and (len([ln for ln in out.strip().splitlines() if ln.strip() and not ln.strip().startswith("[")]) >= 4),
    },
    {
        "id": "QD_02",
        "domain": "7_quillan_domain",
        "prompt": "State the Bushido virtue of Rectitude (Gi) and provide a 1-sentence Council Verdict starting with `Verdict:`.",
        "grader": lambda out: ("rectitude" in out.lower() or "gi" in out.lower()) and ("verdict:" in out.lower()),
    },
]


def run_single_prompt(gguf_path: str, prompt: str, max_tokens: int = 160) -> Tuple[str, float, float]:
    cmd = [
        LLAMA_CLI,
        "-m", gguf_path,
        "-sys", SYS_PROMPT,
        "-p", prompt,
        "-n", str(max_tokens),
        "-s", "42",
        "--temp", "0.0",
        "--top-k", "1",
        "-c", "1024",
        "-ctk", "f16",
        "-ctv", "f16",
        "-st",
        "--simple-io",
        "--no-display-prompt",
        "-ngl", "99",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90)
    raw = res.stdout + "\n" + res.stderr

    prompt_tps = 0.0
    gen_tps = 0.0
    m_speed = re.search(r"Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s", raw)
    if m_speed:
        prompt_tps = float(m_speed.group(1))
        gen_tps = float(m_speed.group(2))

    # Extract assistant response cleanly after `> <first line of prompt>`
    first_line = prompt.splitlines()[0].strip()
    idx = raw.find(f"> {first_line}")
    if idx != -1:
        after = raw[idx + len(f"> {first_line}"):]
        # Skip remaining echoed prompt lines if multi-line prompt
        for extra_ln in prompt.splitlines()[1:]:
            pos_extra = after.find(extra_ln.strip())
            if pos_extra != -1 and pos_extra < 400:
                after = after[pos_extra + len(extra_ln.strip()):]
        end_idx = after.find("[ Prompt:")
        if end_idx != -1:
            after = after[:end_idx]
        ans = after.strip()
    else:
        ans = raw.strip()

    return ans, prompt_tps, gen_tps


def measure_perplexity(gguf_path: str, ppl_file: str) -> float:
    cmd = [
        LLAMA_PPL,
        "-m", gguf_path,
        "-f", ppl_file,
        "-c", "128",
        "--chunks", "2",
        "-ngl", "99",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    combined = res.stdout + "\n" + res.stderr
    m_final = re.search(r"Final estimate:\s*PPL\s*=\s*([\d.]+)", combined)
    if m_final:
        return float(m_final.group(1))
    m_chunk = re.findall(r"\[\d+\]([\d.]+)", combined)
    if m_chunk:
        return float(m_chunk[-1])
    return -1.0


def evaluate_checkpoint(label: str, model_id: str, variant: str, gguf_path: str, ppl_file: str) -> Dict[str, Any]:
    print(f"\n--- Evaluating: {label} ({os.path.basename(gguf_path)}) ---")
    size_mb = round(os.path.getsize(gguf_path) / (1024 * 1024), 2)
    ppl = measure_perplexity(gguf_path, ppl_file)
    print(f"  Held-Out Perplexity (PPL): {ppl:.4f} | Size: {size_mb:.1f} MB")

    domain_scores: Dict[str, List[int]] = {}
    item_results: Dict[str, Any] = {}
    prompt_speeds: List[float] = []
    gen_speeds: List[float] = []

    for item in HELD_OUT_SUITE:
        iid = item["id"]
        dom = item["domain"]
        ans, p_tps, g_tps = run_single_prompt(gguf_path, item["prompt"])
        passed = int(bool(item["grader"](ans)))
        domain_scores.setdefault(dom, []).append(passed)
        if p_tps > 0:
            prompt_speeds.append(p_tps)
        if g_tps > 0:
            gen_speeds.append(g_tps)
        item_results[iid] = {
            "domain": dom,
            "passed": passed,
            "prompt_tps": p_tps,
            "gen_tps": g_tps,
            "response_preview": ans[:200].replace("\n", " "),
        }
        status_str = "PASS" if passed else "FAIL"
        print(f"  [{iid} | {dom:22s}] {status_str} | Gen: {g_tps:5.1f} t/s | {ans[:95].replace(chr(10), ' ')}")

    total_passed = sum(v["passed"] for v in item_results.values())
    total_items = len(HELD_OUT_SUITE)
    overall_acc = round(100.0 * total_passed / total_items, 2)

    domain_acc = {
        dom: round(100.0 * sum(scores) / len(scores), 1)
        for dom, scores in domain_scores.items()
    }

    mean_p_tps = round(sum(prompt_speeds) / max(1, len(prompt_speeds)), 1)
    mean_g_tps = round(sum(gen_speeds) / max(1, len(gen_speeds)), 1)

    print(f"  >>> SUMMARY [{label}]: Accuracy = {total_passed}/{total_items} ({overall_acc:.1f}%) | PPL = {ppl:.4f} | Mean Gen = {mean_g_tps:.1f} tok/s")

    return {
        "label": label,
        "model_id": model_id,
        "variant": variant,
        "path": gguf_path,
        "size_mb": size_mb,
        "held_out_ppl": round(ppl, 4),
        "total_passed": total_passed,
        "total_items": total_items,
        "overall_accuracy_pct": overall_acc,
        "domain_accuracy_pct": domain_acc,
        "mean_prompt_tps": mean_p_tps,
        "mean_gen_tps": mean_g_tps,
        "items": item_results,
    }


def main():
    t0 = time.time()
    ppl_file = os.path.join(EXPORT_ROOT, "held_out_eval_corpus.txt")
    with open(ppl_file, "w", encoding="utf-8") as f:
        f.write(HELD_OUT_PPL_TEXT)

    models_to_eval = [
        # MODEL A: Quillan-Ronin 1.2B Flagship
        ("MODEL_A: Untouched Clean Base (Q4_K_M)", "MODEL_A", "clean_base_q4_k_m", os.path.join(EXPORT_ROOT, "quillan-1b-clean-base-q4_k_m.gguf")),
        ("MODEL_A: Released Sovereign (Q4_K_M)",   "MODEL_A", "released_q4_k_m",   os.path.join(EXPORT_ROOT, "quillan-1b-sovereign-q4_k_m.gguf")),
        ("MODEL_A: WSP v5.0 Bundled (Q4_K_M)",     "MODEL_A", "wsp_v50_q4_k_m",    os.path.join(EXPORT_ROOT, "quillan-1b-sovereign-wsp-v3-q4_k_m.gguf")),

        # MODEL B: Quillan-Ronin-Mini 0.5B
        ("MODEL_B: Released Ollama (F16)",         "MODEL_B", "released_f16",      BLOB_MINI_REL),
        ("MODEL_B: Untouched Clean Base (Q4_K_M)", "MODEL_B", "clean_base_q4_k_m", os.path.join(EXPORT_ROOT, "quillan-mini-0.5b-clean-base-q4_k_m.gguf")),
        ("MODEL_B: WSP v5.0 Bundled (Q4_K_M)",     "MODEL_B", "wsp_v50_q4_k_m",    os.path.join(EXPORT_ROOT, "quillan-mini-0.5b-wsp-v3-q4_k_m.gguf")),

        # MODEL C: Quillan-Dragon 1B
        ("MODEL_C: Released Ollama (Q8_0)",        "MODEL_C", "released_q8_0",     BLOB_DRAGON_REL),
        ("MODEL_C: Untouched Clean Base (Q4_K_M)", "MODEL_C", "clean_base_q4_k_m", os.path.join(EXPORT_ROOT, "quillan-dragon-1b-clean-base-q4_k_m.gguf")),
        ("MODEL_C: WSP v5.0 Bundled (Q4_K_M)",     "MODEL_C", "wsp_v50_q4_k_m",    os.path.join(EXPORT_ROOT, "quillan-dragon-1b-wsp-v3-q4_k_m.gguf")),
    ]

    results = []
    for label, mid, var, gpath in models_to_eval:
        if os.path.exists(gpath):
            res = evaluate_checkpoint(label, mid, var, gpath, ppl_file)
            results.append(res)
        else:
            print(f"[SKIP] Missing {gpath}")

    out_json = os.path.join(EXPORT_ROOT, "phase0_baseline_eval_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "eval_command": "python3.11.exe C:\\02_QUILLAN\\03 - Training & Model\\scripts\\phase0_held_out_eval_harness.py",
            "seed": 42,
            "decoding_flags": "-s 42 --temp 0.0 --top-k 1 -c 1024 -ctk f16 -ctv f16 -st --simple-io --no-display-prompt -ngl 99",
            "held_out_items_count": len(HELD_OUT_SUITE),
            "elapsed_s": round(time.time() - t0, 2),
            "results": results,
        }, f, indent=2)

    print(f"\n[OK] Saved Phase 0 Ground-Truth Evaluation Table to: {out_json} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

