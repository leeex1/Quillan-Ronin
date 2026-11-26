#!/usr/bin/env python3
"""C14-KAIDO bounded cold-vs-cache-reuse measurement for brief #896.

NEW FILE. Sends the REAL 171,434-token system prompt file twice to a running
llama-server configured by kaido_prefix_cache_serve.ps1. Reports only numbers
returned by the server's own `timings` block plus wall-clock measurements.
No values are invented or extrapolated here.

Usage:
  python kaido_prefix_cache_measure.py --port 8099 \
      --prompt "C:\\02_QUILLAN\\06 - Deployment & Platforms\\system prompts\\Quillan-Samurai.md" \
      --out "C:\\02_QUILLAN\\docs\\kaido_speed_measurement.json"
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
import urllib.error
import urllib.request

PROBES = (
    "Reply with exactly one word: alpha.",
    "Reply with exactly one word: beta.",
)


def post_completion(port: int, text: str, n_predict: int = 16) -> dict:
    url = f"http://127.0.0.1:{port}/completion"
    payload = json.dumps(
        {
            "prompt": text,
            "n_predict": n_predict,
            "temperature": 0.0,
            "stream": False,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=3600) as resp:
        body = resp.read().decode("utf-8")
    wall_s = time.monotonic() - t0
    result = json.loads(body)
    result["_client_wall_s"] = round(wall_s, 3)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--char-offset", type=int, default=0)
    ap.add_argument(
        "--max-chars",
        type=int,
        default=0,
        help="bounded slice of the real prompt file (0 = whole file). "
        "A slice is a MEASUREMENT SAMPLE only; it never changes deployed config.",
    )
    args = ap.parse_args()

    system = pathlib.Path(args.prompt).read_text(encoding="utf-8", errors="replace")
    if args.max_chars > 0:
        system = system[args.char_offset : args.char_offset + args.max_chars]
    out: dict = {
        "prompt_file": args.prompt,
        "char_offset": args.char_offset,
        "max_chars": args.max_chars,
        "prompt_chars": len(system),
        "note": "timings come from llama-server /completion; first-token latency ~= prompt_ms",
        "runs": [],
    }
    for label, probe in zip(("cold", "cache_reuse"), PROBES):
        text = system + "\n\nUSER PROBE: " + probe
        try:
            r = post_completion(args.port, text)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            out["runs"].append({"label": label, "error": repr(exc)})
            print(json.dumps(out["runs"][-1], indent=2))
            break
        timings = r.get("timings", {})
        rec = {
            "label": label,
            "probe": probe,
            "prompt_n": timings.get("prompt_n"),
            "prompt_ms": timings.get("prompt_ms"),
            "prompt_per_s": timings.get("prompt_per_s"),
            "predicted_n": timings.get("predicted_n"),
            "predicted_ms": timings.get("predicted_ms"),
            "predicted_per_s": timings.get("predicted_per_s"),
            "client_wall_s": r.get("_client_wall_s"),
            "stopped_words": r.get("stopped_words"),
        }
        out["runs"].append(rec)
        print(json.dumps(rec, indent=2))
        pathlib.Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
