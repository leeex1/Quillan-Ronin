#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Continuous fleet-wide PC maintenance with predictive throttling.

The autoboost daemon reacts to SINGLE processes sustaining >=40% of one core.
This crew watches the WHOLE box every cycle and acts BEFORE a spike lands:

  1. forecasts aggregate load from its trend over recent cycles; if the
     forecast crosses the pressure threshold, it spills workers onto all 4
     cores at BELOW_NORMAL (GUI still preempts) before lag appears;
  2. watches per-process CPU trends and demotes RISING work early (at a lower
     absolute threshold than autoboost), releasing it after a cool-down;
  3. heartbeat every cycle records sweeps, watched count, trend, forecast and
     actions, so non-stop activity is verifiable.

All actions are reversible; no process is ever killed.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "fleet_maintenance.json"
STATE = HERE / "fleet_maintenance_state.json"
LOG = HERE / "fleet_maintenance.log"
POLICY = Path(r"F:\02_QUILLAN\07 - Memory & LanceDB\affinity_policy.json")

DEFAULTS = {
    "enabled": True,
    "interval_sec": 10,
    "pressure_on_pct": 200.0,
    "pressure_off_pct": 100.0,
    "pressure_on_loops": 2,
    "pressure_off_loops": 4,
    "predict_ahead_cycles": 2,
    "preempt_floor_pct": 120.0,
    "moderate_hog_pct": 20.0,
    "preempt_hog_pct": 10.0,
    "preempt_rise_pct": 5.0,
    "moderate_release_pct": 10.0,
    "moderate_hot_loops": 2,
    "moderate_cool_loops": 6,
}

PROTECTED = {
    "system",
    "registry",
    "smss.exe",
    "csrss.exe",
    "services.exe",
    "lsass.exe",
    "wininit.exe",
    "svchost.exe",
    "dwm.exe",
    "explorer.exe",
    "nvspcap.exe",
    "nvspcap64.exe",
    "nvcontainer.exe",
    "nvidia.overlay.exe",
    "nvidia.webhelper.exe",
}
NEVER_AUTO = {
    "brave.exe",
    "chrome.exe",
    "msedge.exe",
    "firefox.exe",
    "zen.exe",
    "code.exe",
    "cursor.exe",
    "antigravity.exe",
    "windsurf.exe",
    "trae.exe",
    "devenv.exe",
    "opencode.exe",
    "steam.exe",
    "steamwebhelper.exe",
    "epicgameslauncher.exe",
    "spotify.exe",
    "discord.exe",
    "taskmgr.exe",
}


def log(msg: str, dry: bool = False) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    if dry:
        print(line)
        return
    if LOG.exists() and LOG.stat().st_size > 512 * 1024:
        os.replace(LOG, str(LOG) + ".1")
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def cfg() -> dict:
    c = dict(DEFAULTS)
    try:
        if CONFIG.exists():
            c.update(json.loads(CONFIG.read_text(encoding="utf-8")))
    except Exception:
        pass
    return c


def read_policy() -> bool:
    try:
        return (
            json.loads(POLICY.read_text(encoding="utf-8")).get("all_cores_mode") is True
        )
    except Exception:
        return False


def write_policy(value: bool, dry: bool) -> None:
    if dry:
        return
    tmp = POLICY.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(
            {
                "all_cores_mode": value,
                "updated": datetime.now().isoformat(),
                "by": "quillan_fleet_maintenance",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    os.replace(tmp, POLICY)


def release_pins(dry: bool) -> int:
    if psutil is None:
        return 0
    n = psutil.cpu_count() or 4
    all_cores = list(range(n))
    freed = 0
    me = os.getpid()
    for p in psutil.process_iter(["pid", "name"]):
        try:
            pid = p.info.get("pid") or 0
            name = (p.info.get("name") or "").lower()
            if pid in (0, 4, me) or name in PROTECTED or name in NEVER_AUTO:
                continue
            if not hasattr(p, "cpu_affinity") or len(p.cpu_affinity()) >= n:
                continue
            if not dry:
                p.cpu_affinity(all_cores)
            freed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue
        except Exception:
            continue
    return freed


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(st: dict, dry: bool) -> None:
    if dry:
        return
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=2), encoding="utf-8")
    os.replace(tmp, STATE)


def cycle(st: dict, dry: bool) -> dict:
    c = cfg()
    if psutil is None:
        return {"error": "psutil unavailable"}

    now = time.time()
    agg: dict = st.get("agg", {})
    hot: dict = st.get("hot", {})
    actions: list[str] = []
    load = 0.0
    watched = 0
    demoted = released = 0
    me = os.getpid()

    for p in psutil.process_iter(["pid", "name"]):
        try:
            pid = p.info.get("pid") or 0
            name = (p.info.get("name") or "").lower()
            if pid in (0, 4, me) or name in PROTECTED or name in NEVER_AUTO:
                continue
            t = p.cpu_times()
            total = t.user + t.system
            key = str(pid)
            rec = agg.get(key)
            if not rec or rec[2] != p.create_time():
                agg[key] = [total, now, p.create_time()]
                continue
            dt = max(now - rec[1], 0.001)
            pct = (total - rec[0]) / dt * 100.0
            rec[0], rec[1] = total, now
            load += pct
            watched += 1

            try:
                nice = p.nice()
            except Exception:
                continue
            if nice is not None and nice != psutil.NORMAL_PRIORITY_CLASS:
                continue
            h = hot.get(key)
            if not h or h.get("ct") != p.create_time():
                hot[key] = {
                    "tot": total,
                    "ts": now,
                    "ct": p.create_time(),
                    "hot": 0,
                    "tiered": False,
                    "cool": 0,
                    "pcts": [],
                }
                continue
            pcts = h.setdefault("pcts", [])
            pcts.append(round(pct, 1))
            del pcts[:-4]
            rising = (
                len(pcts) >= 2
                and pcts[-1] >= c["preempt_hog_pct"]
                and (pcts[-1] - pcts[0]) >= c["preempt_rise_pct"]
            )
            if pct >= c["moderate_hog_pct"] or rising:
                h["cool"] = 0
                h["hot"] += 1
                if h["hot"] >= c["moderate_hot_loops"] and not h["tiered"]:
                    if not dry:
                        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                    h["tiered"] = True
                    demoted += 1
                    why = "rising" if rising else "sustained"
                    actions.append(
                        f"PREEMPT pid={pid} name={name} {why} pct={pct:.0f}% trace={pcts}"
                    )
            elif pct <= c["moderate_release_pct"]:
                h["hot"] = 0
                if h["tiered"]:
                    h["cool"] += 1
                    if h["cool"] >= c["moderate_cool_loops"]:
                        if not dry:
                            p.nice(psutil.NORMAL_PRIORITY_CLASS)
                        hot.pop(key, None)
                        released += 1
                        actions.append(f"RELEASE pid={pid} name={name}")
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue
        except Exception:
            continue

    live = {str(p.info.get("pid")) for p in psutil.process_iter(["pid"])}
    agg = {k: v for k, v in agg.items() if k in live}
    hot = {k: v for k, v in hot.items() if k in live}

    hist: list = st.get("load_hist", [])
    hist.append([now, round(load, 1)])
    del hist[:-12]
    trend = 0.0
    if len(hist) >= 3:
        dt = hist[-1][0] - hist[-3][0]
        if dt > 0:
            trend = (hist[-1][1] - hist[-3][1]) / dt
    horizon = float(c["predict_ahead_cycles"]) * float(c["interval_sec"])
    predicted = load + trend * horizon

    on_streak = int(st.get("on_streak", 0))
    off_streak = int(st.get("off_streak", 0))
    policy = read_policy()
    rising_spike = (
        predicted >= c["pressure_on_pct"]
        and load >= c["preempt_floor_pct"]
        and trend > 0
    )
    pressure = load >= c["pressure_on_pct"] or rising_spike
    if pressure:
        on_streak, off_streak = on_streak + 1, 0
    elif load <= c["pressure_off_pct"] and predicted < c["pressure_on_pct"]:
        off_streak, on_streak = off_streak + 1, 0
    else:
        on_streak = max(0, on_streak - 1)
        off_streak = max(0, off_streak - 1)

    if on_streak >= c["pressure_on_loops"] and not policy:
        freed = release_pins(dry)
        write_policy(True, dry)
        tag = "SPILL_PREEMPT" if rising_spike else "SPILL_ON"
        actions.append(
            f"{tag} load={load:.0f}% predicted={predicted:.0f}% freed_pins={freed}"
        )
        policy, on_streak = True, 0
    elif off_streak >= c["pressure_off_loops"] and policy:
        write_policy(False, dry)
        actions.append(f"SPILL_OFF load={load:.0f}%")
        policy, off_streak = False, 0

    st.update(
        {
            "on_streak": on_streak,
            "off_streak": off_streak,
            "policy": policy,
            "agg": agg,
            "hot": hot,
            "load_hist": hist,
            "last_cycle": now,
            "last_load": round(load, 1),
            "watched": watched,
            "sweeps": int(st.get("sweeps", 0)) + 1,
            "last_action": actions[-1] if actions else st.get("last_action", ""),
        }
    )
    save_state(st, dry)
    for a in actions:
        log(a, dry)
    log(
        f"HEARTBEAT sweep={st['sweeps']} load={load:.0f}%core-eq trend={trend:+.1f}%/s "
        f"forecast={predicted:.0f}% policy={'ALL_CORES' if policy else 'CORES_1_3'} "
        f"watched={watched} demoted={demoted} released={released} actions={len(actions)}",
        dry,
    )
    return {
        "load": round(load, 1),
        "trend": round(trend, 1),
        "forecast": round(predicted, 1),
        "policy": policy,
        "watched": watched,
        "demoted": demoted,
        "released": released,
        "actions": actions,
    }


def main() -> None:
    dry = "--dry-run" in sys.argv
    once = "--once" in sys.argv
    cycles = 0
    if "--cycles" in sys.argv:
        try:
            cycles = int(sys.argv[sys.argv.index("--cycles") + 1])
        except (ValueError, IndexError):
            cycles = 0
    log("START fleet maintenance (predictive)", dry)
    st = load_state()
    done = 0
    while True:
        c = cfg()
        if not c.get("enabled", True) and not dry:
            log("DISABLED by config; stopping", dry)
            break
        try:
            cycle(st, dry)
        except Exception as e:
            log(f"CYCLE_ERROR {type(e).__name__}: {e}", dry)
        done += 1
        if once or (cycles and done >= cycles):
            break
        time.sleep(max(4.0, float(c["interval_sec"]) - 1.0))


if __name__ == "__main__":
    main()
