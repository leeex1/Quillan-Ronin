# NextVerse & Quillan Formula Validation Suite + VM Control

Comprehensive mathematical engine and simulation suite for **23 Quillan Custom Formulas**, **11 NextVerse Core Engine formulas**, and **34 Foundation canonical formulas** (Physics/CS/ML, GPT-verified) — with Compound Turbo Feedback, Hardware Profiling, and VM Control.

## What this is

- **Formula validation lab**: interactive cards, dossiers, and parameter playgrounds for every formula across the three libraries, with evaluation reports proving TS/Python parity.
- **Booster toolkit**: backend Python optimizers (CPU turbo, network, autoboost daemon, predatory/saturated-ultra/sweetspot/token-entropy) plus a native C++ hardware monitor — the performance toolkit for squeezing the box.
- **VM control surface**: VmBooster panel + startup daemon wiring for persistent boost.

## Quickstart (frontend)

```bash
npm install        # or: bun install
npm run dev        # vite dev server
npm run build      # production build
npm run preview    # serve the build
npm run lint       # tsc --noEmit
```

React 19 + Vite 6 + TypeScript. Icons via `lucide-react`.

## Backend (Python boosters)

```bash
cd backend
python main.py charges a full validation + benchmark pass.
python benchmark_4x_gains.py        # 4x-gains benchmark
python total_box_turbo.py           # whole-box turbo profile
python quillan_autoboost_daemon.py  # persistent autoboost daemon
powershell -ExecutionPolicy Bypass -File install_daemon_startup.ps1  # install daemon at startup
python parity_check.py              # TS/Python parity check (see parity_validation_report.json)
```

Additional tuners: `i9_turbo_optimizer.py`, `network_turbo_optimizer.py`, `deep_system_tuner.py`, `predatory_optimizer.py`, `saturated_ultra_optimizer.py`, `sweetspot_validator.py`, `test_token_entropy_optimizer.py`.

## Native monitor (C++)

See [`native_monitor/README.md`](native_monitor/README.md). Build with CMake (`native_monitor/CMakeLists.txt` → `main.cpp`); prebuilt `NativeHardwareOptimizer.exe` is checked in.

## Repo map

| Path | Contents |
|---|---|
| `App.tsx`, `index.tsx`, `components/` | React UI: FormulaCard, FormulaDossierModal, CompoundTurboFlowchart, HardwareProfiler, LocalHardwareInfo, VmBooster, ParameterInput, CloudComparison |
| `quillanFormulas.ts` | 23 Quillan Custom Formulas |
| `nextverseFormulas.ts` | 11 NextVerse Core Engine formulas |
| `foundationFormulas.ts` | 34 Foundation canonical formulas (Physics/CS/ML) |
| `constants.ts`, `types.ts` | Shared constants and types |
| `backend/` | Python boosters, validators, daemon, evaluation reports |
| `native_monitor/` | C++ hardware monitor + prebuilt exe |
| `metadata.json` | Capability manifest (incl. server-side Gemini API) |

## Reports

- `backend/ts_formula_evaluation_report.json` — TypeScript formula evaluation results
- `backend/parity_validation_report.json` — TS/Python parity validation
- `backend/validation_results.json` — latest validation run summary
