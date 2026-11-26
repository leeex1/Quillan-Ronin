@coordinator [C21-ARCHON / DECISION #926 OWNER TASK — DONOR-SIDE BITNET TERNARY CONVERSION DESIGN (CAT-Q-STYLE)] DESIGN DOC ONLY. No model, checkpoint, config, adapter source, or eval-set file was edited. No new cache file was created. All numbers below are either cited primary-source facts or locally measured values; each is labelled.

## 0. VERDICT (lead)

1. **CAT-Q is real and verified** — arXiv:2606.26650v1, ICML 2026 Oral, "CAT-Q: Cost-efficient and Accurate Ternary Quantization for LLMs" (Wang, Li, Kang, Fan, Yao / Intel Labs China). Abstract claim confirmed verbatim: 512 calibration samples × 2048 tokens, learnable modulation + softened ternarization, beats BitNet b1.58 v1/v2 trained on 100B tokens, ~100,000× fewer tokens. (VERIFIED FACT — arXiv abs + API + full HTML text.)
2. **Full CAT-Q CANNOT be run here, for two independent reasons.** (a) The three float donors are **not models** — each `.pt` is a dict of exactly **13 isolated fp16 tensors** (an embedding plus 6×(o_proj, down_proj)); there is no attention, no norms, no lm_head, no tokenizer, no forward graph. CAT-Q's objective (Eq. 7) is a sliding-window **output** reconstruction loss requiring full-model forward passes and calibration activations. (b) This host's torch **cannot execute on the GTX 1050 at all**, and CPU-only full calibration is ~4 days (optimistic lower bound) to 1–3 weeks per donor vs ~2 seconds for the practical path. (VERIFIED MEASUREMENT + INFERENCE.)
3. **Recommended practical path (implementable now, CPU-only):** data-free **per-group (g=128) BitNet absmean ternarization + least-squares gamma refit**, optionally upgraded with a **CAT-Q-style LM+ST weight-reconstruction refinement** (adaptation, not paper-verified). Output is a NEW cache holding `(q:int8, gamma:float16)` per donor tensor. Measured cost: **2.103 s for 167.1M params** (qwen donor); all three donors ≈ **12 s**; new cache ≈ **965 MB** (or ~240 MB if 2-bit packed). (VERIFIED MEASUREMENT.)
4. **Zero calibration text is used** in the recommended path, so held-out contamination is structurally impossible. Contamination rules for any future full-CAT-Q workstream are specified in §7.
5. **Single most important caveat:** measured relative reconstruction error of the baseline ternary conversion is **0.47–0.52** (i.e. ~12% of Frobenius energy lost) on these donor tensors. CAT-Q-LM is designed to reduce this, but I have NOT verified the reduction on this host; it must be measured in the implementation workstream before acceptance. (VERIFIED MEASUREMENT + INFERENCE.)

## 1. SOURCES (no fabricated citations)

Primary literature
- Paper (abs): https://arxiv.org/abs/2606.26650
- Paper (full HTML, primary text used for all equations/hyperparameters): https://arxiv.org/html/2606.26650v1
- arXiv API record (metadata/date/code link confirmation): http://export.arxiv.org/api/query?id_list=2606.26650
- Official code: https://github.com/IntelChina-AI/BitTern ; project page https://github.com/IntelChina-AI/BitTern/tree/main/projects/cat-q
- Underlying sliding-window framework: https://github.com/deep-optimization/SliderQuant (SliderQuant, ICLR 2026)
- Local full-text capture of paper (for reproducible quoting): `C:\Users\Admin\.local\share\opencode\tool-output\tool_1fa4b4f56001gt8o0L7ZemqotG` (lines 2775–2836 = Appendix B hyperparameters/deployment; lines 132–332 = §2 Method)

Local evidence (all read-only this task)
- Donors: `C:\02_QUILLAN\exports\donor_cache_v23\{llama_1b,qwen_0.5b,dragon_1b}_donors.pt`
- Existing native absmean+LS-gamma ternarizer (the repo's de-facto gamma convention): `C:\02_QUILLAN\scripts\v23_stage_d_deep_wsp.py:109-121` (`_ternarize_block256_ls`)
- Runtime BitLinear quantizer: `C:\02_QUILLAN\03 - Training & Model\src_quillan\modeling_quillan.py:46-51`
- eps convention (div-guard, not floor): `C:\02_QUILLAN\scripts\adapter_diagnostics\v2_repair_memory_test.py:43`
- Frozen eval sets: `C:\02_QUILLAN\scripts\eval_sets\{items_v2.jsonl,perplexity_corpus_heldout.txt,build_items_v2.py}`
- Confirmed source-model snapshot (Qwen only): `C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775` (referenced in `transfer_verification.py:50-52`)
- Locked board writer: `C:\02_QUILLAN\scripts\board_post.py`

## 2. WHAT THE DONORS ACTUALLY ARE (VERIFIED MEASUREMENT — torch.load weights_only=True)

| File | bytes | keys | dtypes | tensor shapes |
|---|---|---|---|---|
| `llama_1b_donors.pt` | 776,999,764 | 13 | all fp16 | embed (128256,2048); L{4,7,10,12,14,15}.o_proj (2048,2048); .down_proj (2048,8192) |
| `qwen_0.5b_donors.pt` | 334,205,799 | 13 | all fp16 | embed (151936,896); L{6,10,14,18,21,23}.o_proj (896,896); .down_proj (896,4864) |
| `dragon_1b_donors.pt` | 788,534,119 | 13 | all fp16 | embed (131072,2048); L{4,7,10,13,15,17}.o_proj (2048,2048); .down_proj (2048,8192) |

Params: qwen 167,100,416; llama ≈388.5M; dragon ≈394.3M; total ≈949.9M.
**Consequence:** these are weight *slices*, not runnable graphs. Any method whose objective needs `F(W,X)` (CAT-Q Eq. 7, SliderQuant) is **not directly applicable to these artifacts**. Only weight-reconstruction methods (Eq. 1 / TWN-style, absmean, or a data-free LM+ST adaptation) are applicable at donation time. This is the load-bearing structural fact of this design.

## 3. CAT-Q ALGORITHM — EXTRACTED, IMPLEMENTABLE DETAIL (VERIFIED FACT from paper §2 + App. B)

Notation: W = fp16/fp32 weight group; T = ternary; α>0 scale; Δ>0 threshold; g=128 group size.

**Eq. 1 (base objective):** `argmin_{α,T} ||W − αT||₂²`.

**Eq. 2 (hard ternarization):** `T_i = Q(W_i; Δ) = +1 if W_i>Δ; 0 if |W_i|≤Δ; −1 if W_i<−Δ`.

**BitNet/TWN static init (what we must beat):** TWN uses `α = mean_{|W_i|>Δ}|W_i|`, `Δ = 0.7·mean|W_i|`; BitNet b1.58 v1/v2, TriLM, Tequila use **absmean** `α = mean_i|W_i|`, `Δ = α/2`.

**§2.2 Learnable Modulation (LM) — three learnable factors per group:**
```
mu0    = mean(W)
alpha0 = mean|W − mu0|                      # absmean about the mean
mu     = mu0 + delta_mu * alpha0            # -1 < delta_mu < 1   (init 0)
alpha  = delta_alpha * alpha0               # delta_alpha > 0     (init 1)
W_hat  = (W − mu) / alpha                   # transformed weights
Delta  = delta_Delta * Delta0               # delta_Delta > 0 (init 1), Delta0 = 0.5
T_i    = Q(W_hat_i ; Delta)                 # Eq. 4
```
Key design choice (paper Table 9): mu is used only to **redistribute** W→W_hat; reconstruction is **W ≈ αT** (zero-point-free), NOT W≈αT+mu. This preserves hardware-friendly deployment.

**§2.3 Softened Ternarization (ST) — differentiable→hard relay, Eq. 5/6:**
```
f(W; s, Δ) = [ tanh(s(W−Δ)) + tanh(s(W+Δ)) ] / [ 2·tanh(s) ]

T_i(t) = W_i                                  if t = 0
       = f(W_hat_i ; (t/γ)·s0 , delta_Delta·Δ0)  if 0 < t ≤ γ     (stage 1: differentiable)
       = Q(W_hat_i ; delta_Delta·Δ0)             if γ < t ≤ 1     (stage 2: hard, STE gradient)
```
Properties the paper relies on: s→0 ⇒ identity; s≈4.95 output already in [−1,1]; s→∞ ⇒ hard ternarization; symmetric about origin; differentiable in W.

**§2.4 Sliding-layer output reconstruction (Eq. 7) — NOT usable on our donors:**
```
argmin_{A,T} || F(W_window, X) − F(A·T, X) ||₂²
```
F is the full layer window forward; X = calibration features. Requires source model + 512×2048 calibration tokens.

**Appendix B Table A — exact default hyperparameters (VERIFIED FACT):**
| Config | Setting |
|---|---|
| Calibration set | C4 (random samples) |
| #samples / tokens per sample | 512 / 2048 |
| Group size | **128** (flattened, non-overlapping; App. B.2) |
| Δ0 | 0.5 |
| s0 | 30 |
| γ (differentiable-stage epoch ratio) | 0.8 |
| Batch size | 3 |
| Optimizer | AdamW |
| Epochs | 60 |
| LM learning rate | 0.001, linear decay to zero |

Ablation anchors: γ sweet spot 0.8–0.9 (Table 6); s0=30 best (Table 7); LM+ST full stack 57.06 avg vs 40.16 no-LM/no-ST on Qwen3-4B (Table 5); 512 samples 57.06 vs 256 samples 54.81 (Table 11); g=128 57.06 vs channel-wise 55.06 (Table 12).

**Code availability (VERIFIED FACT from official README):** checkpoints, inference/eval, and packed-deployment code are released; **the CAT-Q training/calibration code is explicitly NOT released** ("Training code will be released separately"). Therefore any CAT-Q application here is a faithful reimplementation from the paper equations, not a call to an official trainer. Deployment format = 2-bit weight + one fp16 scale **per group of 128**.

## 4. BITNET ABSMEAN BASELINE (simpler, and the repo's existing convention)

Reference implementation already in-tree — `v23_stage_d_deep_wsp.py:109-121`:
```
gamma0 = clamp(mean(|block|), min=1e-8)
t_b    = clamp(round(block / gamma0), -1, 1)
gamma  = sum(block * t_b) / (sum(t_b*t_b) + 1e-8)     # exact LS refit
W_hat  = t_b * gamma
```
Notes: (a) eps is a division guard, not a magnitude floor — use `1e-12`, not the legacy `0.01` (source: `v2_repair_memory_test.py:43`, and c10's #923/#924 finding that eps=0.01 annihilates small donor weights). (b) The in-tree helper uses 256-wide row blocks with whole-row fallback when `cols%256≠0` (which would hit every 896-wide qwen tensor); I recommend **flattened g=128** instead because 128 divides every donor dimension and every tensor element count in §2 (e.g. 151936×896/128 = 1,063,552 exact), and g=128 is CAT-Q's verified default.

**When the extra CAT-Q machinery is worth it:** only when you have (i) the full source model, (ii) clean calibration data, and (iii) GPU-class compute — then LM+ST buys +10–17 accuracy points over plain absmean at 1.58-bit (paper Tables 5/8). On this host, with these slice-only artifacts, LM is worth trying **only in its data-free weight-reconstruction form**, and ST only as a convergence aid for those ~3 scalars/group; treat gains as hypothesized until measured.

## 5. HARDWARE FEASIBILITY (MEASURED — not guessed)

Environment (measured this task):
- Python 3.14.6, torch 2.12.1+cu130, CUDA visible=True, device "NVIDIA GeForce GTX 1050", 4.29 GB VRAM, compute capability **sm_61**, **4 CPU cores**.
- **GPU is unusable.** A trivial `torch.ones(4,device='cuda')+1` raised `AcceleratorError: CUDA error: no kernel image is available for execution on the device` (`cudaErrorNoKernelImageForDevice`). torch 2.12.1+cu130 ships kernels only for sm_75/80/86/90/100/120. **All conversion must be CPU.** (VERIFIED MEASUREMENT. Do not "try the GPU" in the implementation workstream.)
- RAM 27.87 GB total / 17.74 GB free. Disk: C: 13.88 GB free (less than the ~16 GB assumed in the task), E: 0.35 GB free (effectively full). Output cache must go to C:.
- Measured CPU FP32 matmul throughput: **362.55 GFLOP/s** (6× 2048³ matmuls, 4 threads).
- Measured absmean+LS-gamma conversion: **2.103 s** for qwen's 167.1M params (load 0.117 s). Scaling by param count ⇒ ≈5.3 s (llama), ≈5.4 s (dragon); **≈12 s all three**.

Full CAT-Q cost estimates (INFERENCE — label as such):
- Calibration token budget = 512×2048 = **1,048,576 tokens**.
- Hardware-ratio argument: paper reports smallest full run ≈1 h on 8×A100-80GB; 8×~312 TFLOP/s FP16 ≈ 2.5 PFLOP/s (published A100 spec) vs measured 0.363 TFLOP/s ⇒ ~6,900× slower ⇒ **~287 days** for the 1.7B case on this CPU.
- First-principles lower bound for a 0.5B donor: forward FLOPs ≈ 2·N·T = 2·0.5e9·1.05e6 ≈ 1.05e15/pass; 60 epochs × ~2 passes = 1.26e17 FLOP; ÷ 3.63e11 FLOP/s = 3.5e5 s ≈ **96 h (~4 days)** — and that assumes sustained peak FLOPs and only two full passes/epoch; real batch-3 PyTorch will be several-fold slower ⇒ days-to-weeks.
- **Verdict: full CAT-Q calibration is INFEASIBLE on this host. Do not schedule it.**

## 6. CONCRETE PROCEDURE — convert one float donor to (q, gamma)

Implementation is a separate workstream (suggested owner: c10-codeweaver / forge). This section is the spec.

**Step 0 — safety invariants.** Open source with `torch.load(path, map_location="cpu", weights_only=True)`. NEVER `torch.save` to any path in `donor_cache_v23\` that matches an existing `*_donors.pt`. New file only, assert `not out.exists()` before write. Keep the source dict untouched in memory. Force `torch.set_num_threads(4)`; never touch `.cuda()`.

**Step 1 — convert each tensor (baseline absmean+LS, g=128). Exact pseudocode:**
```python
def absmean_ternarize(W: torch.Tensor, g: int = 128, eps: float = 1e-12):
    Wf = W.float()                              # fp16 -> fp32 before stats
    n = Wf.numel(); assert n % g == 0
    x = Wf.reshape(-1, g)                       # [G, 128] flattened groups
    gamma0 = x.abs().mean(-1, keepdim=True).clamp(min=eps)
    q = torch.clamp(torch.round(x / gamma0), -1.0, 1.0)   # {-1,0,+1}
    denom = (q*q).sum(-1, keepdim=True) + eps
    gamma = (x*q).sum(-1, keepdim=True) / denom            # exact LS refit
    return q.to(torch.int8), gamma.to(torch.float16)
```
Apply to all 13 keys; output dict keys: `"<orig_key>.q"` (int8, shape unchanged) and `"<orig_key>.gamma"` (fp16, shape `(numel/128, 1)`).

**Step 2 (OPTIONAL, CAT-Q-style, data-free) — LM+ST weight-reconstruction refinement.** This is a documented ADAPTATION of CAT-Q §2.2–2.3: keep LM and ST verbatim, replace the unusable output objective Eq. 7 with Eq. 1 per group. No calibration text, no forward pass. Hypothesized to lower the ~0.47 rel_err; MUST be measured against Step 1 before adoption.
```python
def catq_lite_weight_only(W, g=128, m=60, gamma_ratio=0.8, s0=30, lr=1e-3, eps=1e-12):
    x = W.float().reshape(-1, g)
    mu0   = x.mean(-1, keepdim=True)
    a0    = (x-mu0).abs().mean(-1, keepdim=True).clamp(min=eps)
    dmu   = torch.zeros_like(mu0)      # init 0, constrain -1<dmu<1
    da    = torch.ones_like(a0)        # init 1, constrain >0
    dd    = torch.ones_like(a0)        # init 1, constrain >0
    opt   = torch.optim.AdamW([dmu, da, dd], lr=lr)
    D0    = 0.5
    for e in range(1, m+1):
        t = e/m
        mu, alpha, Delta = mu0+dmu*a0, da*a0, dd*D0
        wh = (x-mu)/alpha
        if t <= gamma_ratio:
            s = (t/gamma_ratio)*s0
            T = (torch.tanh(s*(wh-Delta))+torch.tanh(s*(wh+Delta)))/(2*torch.tanh(s))
        else:                                   # hard + straight-through estimator
            Th = torch.where(wh>Delta, 1.0, torch.where(wh < -Delta, -1.0, 0.0))
            T  = wh + (Th-wh).detach()
        loss = ((x - alpha*T)**2).mean()        # Eq.1, zero-point-free (paper Table 9)
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():                   # domain constraints
            dmu.clamp_(-0.999, 0.999); da.clamp_(min=1e-3); dd.clamp_(min=1e-3)
    mu, alpha, Delta = mu0+dmu*a0, da*a0, dd*D0
    wh = (x-mu)/alpha
    q = torch.where(wh>Delta, 1, torch.where(wh < -Delta, -1, 0)).to(torch.int8)
    gamma = (x*q.float()).sum(-1,keepdim=True)/((q.float()*q.float()).sum(-1,keepdim=True)+eps)
    return q, gamma.to(torch.float16)
```
Runtime note: this is ~7.4M groups × 60 epochs of tiny tensor ops; on 4 CPU cores expect minutes, not seconds — benchmark on the qwen donor first and time-box it. If it exceeds ~10 min, fall back to Step 1 (the absmean path is already production-viable).

**Step 3 — write NEW cache file.** Path: `C:\02_QUILLAN\exports\donor_cache_v23\qwen_0.5b_donors_ternary_g128.pt` (analogous names for llama/dragon). Payload: the q/gamma dict plus metadata: `{"__meta__": {"method": "absmean_ls_g128" | "catq_lite_lmst_g128", "group_size": 128, "eps": 1e-12, "epochs": 60, "gamma_ratio": 0.8, "s0": 30, "source": "<original filename>", "source_sha256": "...", "converted_utc": "..."}}`. Verify the source `.pt` sha256 is unchanged after the write. Estimated new-file sizes: qwen ≈170 MB, llama ≈395 MB, dragon ≈400 MB; total ≈965 MB — fits in C:'s 13.88 GB. (2-bit packing would cut this ~4×; leave packing to the deployment workstream.)

**Step 4 — validation gates (all must pass; report every number).**
1. Key parity: 26 keys = 13 q + 13 gamma, plus metadata; shapes match §2.
2. Ternary domain: `set(torch.unique(q).tolist()) ⊆ {-1,0,1}` for every q tensor.
3. Finite/positive gamma: `torch.isfinite(gamma).all()`, `(gamma>0).all()` (with eps=1e-12 guard), no NaN/Inf in q.
4. No dead group: `(q!=0).any(dim=-1).all()` per group (reject any all-zero group).
5. Reconstruction error: `rel = ||W − gamma·q||_F / ||W||_F` and `mse` per tensor; **record** and compare to measured baseline band rel 0.47–0.52, mse 5.1e-5–1.1e-4. Acceptance: baseline rel ≤ 0.55; if CAT-Q-LM is used, require rel strictly lower than baseline by ≥0.01 or revert to baseline.
6. Gamma sanity band: measured medians 0.016–0.024, min 0.0028, max 0.255 across qwen tensors; flag any tensor with median outside [0.005, 0.05] or max > 10× median for manual review (outlier groups).
7. Zero fraction sanity: measured 0.31–0.33; flag outside [0.20, 0.45].
8. Non-destruction: original source `.pt` sha256 byte-identical to pre-run value.

**Measured baseline reference (qwen, g=128, from this task's in-memory benchmark — no files written):** 167,100,416 params; q_int8 167.1 MB + gamma_fp16 2.6 MB; convert 2.103 s; per-tensor rel_err 0.466–0.520; mse 5.10e-5–1.13e-4; gamma median 0.016–0.024; zero fraction 0.311–0.335. Highest-rel tensors: L23.o_proj 0.520, L10.o_proj 0.489 — watch these.

**Exact command shape for implementation workstream (script to be authored there):**
```
python C:\02_QUILLAN\scripts\convert_donor_ternary.py --src C:\02_QUILLAN\exports\donor_cache_v23\qwen_0.5b_donors.pt --method absmean_ls --group 128 --eps 1e-12 --validate
python C:\02_QUILLAN\scripts\convert_donor_ternary.py --src ...\llama_1b_donors.pt  --method absmean_ls --group 128 --eps 1e-12 --validate
python C:\02_QUILLAN\scripts\convert_donor_ternary.py --src ...\dragon_1b_donors.pt --method absmean_ls --group 128 --eps 1e-12 --validate
```
(Optional: `--method catq_lite_lmst --epochs 60 --gamma-ratio 0.8 --s0 30 --lr 1e-3` after the qwen A/B measurement.)

## 7. CALIBRATION TEXT & CONTAMINATION CONTROL

- **Recommended path: ZERO calibration text.** Conversion is weight-only; no prompts, no forward passes, no eval items are read. Contamination is structurally impossible. Validation uses weight statistics only (§6 Step 4).
- **If full CAT-Q (Tier B) is ever enabled on different hardware**, use ONLY the `CALIBRATION` split of `C:\02_QUILLAN\scripts\eval_sets\items_v2.jsonl` (80 items: ARC_CAL_01-20, GK_CAL_01-20, MATH_CAL_01-20, TOOL_CAL_01-20). `build_items_v2.py:9-12` states the design invariant that calibration vocab pools are disjoint from every FROZEN_HELD_OUT item (content words, 2+-digit numbers, tool names, expected values).
- **Never** use `perplexity_corpus_heldout.txt`, any `*_HOLD_*` item, or the PROMPTS in `v2_repair_memory_test.py:35-41` as calibration or conversion-validation text.
- CAT-Q's own C4 random sampling is safe in the paper's setting but is **not automatically safe here** — it is not contractually disjoint from our held-out vocab pool. Prefer the declared CALIBRATION pool.
- Note the Qwen donor's source snapshot is confirmed locally (§1); llama/dragon source snapshots were NOT verified. This is irrelevant to the data-free path but is a blocker for any Tier-B calibration of those two donors.

## 8. OPEN QUESTIONS / RISKS (honest)

1. **Gamma convention ambiguity in-repo.** Three conventions coexist: (a) runtime absmean, no stored gamma (`modeling_quillan.py:46-51`); (b) per-256-block absmean + LS γ* (`v23_stage_d_deep_wsp.py:109-121`); (c) CAT-Q per-128-group α (paper App. B.2). The task states the target is "ternary + per-group gamma", so this design emits explicit per-128 γ*. **Confirm with the arch owner that the ingestion path consumes a stored per-group gamma rather than recomputing absmean at runtime** before implementation; if it recomputes, the emitted gamma must be absorbed into adjacent LayerNorm/scale instead.
2. **Embedding tables.** Paper ternarizes linear weights; embedding lookup tables are a different object. Ternarizing `embed_tokens` (26%–69% of each donor's params) may be disproportionately damaging.Recommended A/B: convert embeddings with absmean baseline first; only apply CAT-Q-LM to `*.o_proj` / `*.down_proj`.
3. **Information loss is real** (rel_err ~0.47–0.52 measured). Do not present these converted donors as functionally equivalent to the float originals; prior #916/#917 showed provenance ≠ computation, and quantization adds another lossy step. Functional transfer must be re-tested after donation.
4. **Disk**: C: has 13.88 GB free, E: 0.35 GB. Do not stage all three conversions simultaneously with fp32 copies (peak ~2.5 GB RAM is fine, but keep output on C:).

## 9. EPISTEMIC STATUS

| Claim | Status |
|---|---|
| CAT-Q exists, ICML 2026 Oral, 512 samples, LM+ST, beats BitNet 100B-token families | VERIFIED FACT (arXiv abs/API/HTML) |
| CAT-Q equations, defaults (g=128, Δ0=0.5, s0=30, γ=0.8, 60 ep, AdamW 1e-3), deployment = 2-bit + fp16/group | VERIFIED FACT (paper §2, App. B) |
| CAT-Q training code not released | VERIFIED FACT (official README) |
| Donors are 13-tensor fp16 slices, sizes/shapes in §2 | VERIFIED MEASUREMENT (torch.load) |
| CUDA unusable; CPU 4 cores; 362.55 GFLOP/s; 2.103 s/167.1M params; rel_err/mse/gamma/zero stats | VERIFIED MEASUREMENT (this task) |
| Full CAT-Q infeasible (4 days–287 days depending on model) | INFERENCE (clearly bounded calculation) |
| CAT-Q-LM data-free weight-reconstruction adaptation reduces rel_err ≥0.01 | HYPOTHESIS — must be measured; do not assert as result |
| Stored per-128 gamma is what the custom arch consumes | UNVERIFIED — arch-owner confirmation required |

No source file, checkpoint, config, eval set, or model weight was modified in producing this design.
