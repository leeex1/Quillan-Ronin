@coordinator [FORGE #911 // EMBEDDING-SCALE ROOT CAUSE + IN-MEMORY FIX TESTS]

SCOPE: read-only. NEW files under scripts\adapter_diagnostics\ only.
No weights saved, no checkpoint rebuilt, no model source edited.

## 1. EXACT SCALE FINGERPRINT
- txt_emb.weight mean-abs=0.01198079 | pos_embed mean-abs=0.62257522 | ratio=51.9645x | txt_dec.weight mean-abs=0.01198079
- txt_dec vs txt_emb max-abs diff (weight tying check): 0.0
- packed int8 tensors=24 | bitnet_scale gammas=24 | gamma min/mean/max=0.00045243/0.0028567/0.01242716
- [embeddings] 4 tensors
    ingestion.mod_emb.weight: shape=[4, 896] mean_abs=0.00857835 std=0.01032371 min=-0.01242716 max=0.01242716
    ingestion.pos_embed: shape=[1, 1024, 896] mean_abs=0.62257522 std=0.66908747 min=-1.0 max=1.0
    ingestion.txt_emb.weight: shape=[151936, 896] mean_abs=0.01198079 std=0.01524947 min=-0.20117188 max=0.16601562
    txt_dec.weight: shape=[151936, 896] mean_abs=0.01198079 std=0.01524947 min=-0.20117188 max=0.16601562
- [layernorm_and_finalizer] 18 tensors
    diffusion_core.couil_attn.norm.bias: shape=[896] mean_abs=0.0 std=0.0 min=0.0 max=0.0
    diffusion_core.couil_attn.norm.weight: shape=[896] mean_abs=7.13934755 std=1.10365295 min=-0.01141357 max=16.75
    diffusion_core.norm2.bias: shape=[896] mean_abs=0.0 std=0.0 min=0.0 max=0.0
    diffusion_core.norm2.weight: shape=[896] mean_abs=7.13934755 std=1.10365295 min=-0.01141357 max=16.75
    final_norm.bias: shape=[896] mean_abs=0.0 std=0.0 min=0.0 max=0.0
    final_norm.weight: shape=[896] mean_abs=7.13934755 std=1.10365295 min=-0.01141357 max=16.75
    ingestion.norm.bias: shape=[896] mean_abs=0.0 std=0.0 min=0.0 max=0.0
    ingestion.norm.weight: shape=[896] mean_abs=0.06952981 std=0.09240945 min=-0.44726562 max=0.7265625
    moe.output_norm.bias: shape=[896] mean_abs=0.0 std=0.0 min=0.0 max=0.0
    moe.output_norm.weight: shape=[896] mean_abs=7.13934755 std=1.10365295 min=-0.01141357 max=16.75
    norm_decomp.bias: shape=[896] mean_abs=0.0 std=0.0 min=0.0 max=0.0
    norm_decomp.weight: shape=[896] mean_abs=7.13934755 std=1.10365295 min=-0.01141357 max=16.75
    norm_diff.bias: shape=[896] mean_abs=0.0 std=0.0 min=0.0 max=0.0
    norm_diff.weight: shape=[896] mean_abs=7.13934755 std=1.10365295 min=-0.01141357 max=16.75
    norm_moe.bias: shape=[896] mean_abs=0.0 std=0.0 min=0.0 max=0.0
    norm_moe.weight: shape=[896] mean_abs=7.13934755 std=1.10365295 min=-0.01141357 max=16.75
    pre_final_norm.bias: shape=[896] mean_abs=0.0 std=0.0 min=0.0 max=0.0
    pre_final_norm.weight: shape=[896] mean_abs=7.13934755 std=1.10365295 min=-0.01141357 max=16.75
- [gates_routers] 25 tensors
    ingestion.ingest_bypass: shape=[] mean_abs=2.0 std=0.0 min=2.0 max=2.0
    moe.router.complexity_classifier.2.bias: shape=[3] mean_abs=0.0 std=0.0 min=0.0 max=0.0
- [lowrank_swarm] 454 tensors

## 2. WHY pos_embed IS 52x LARGER (root cause)
- Adapter source: 03 - Training & Model/quillan_weight_adapter.py:600-602
- Assignment: ingestion.pos_embed -> _sinusoidal_positions(shape)
- Model default init: nn.Parameter(torch.randn(1,max_seq_len,dim)) @ quillan_v8_saturated.py:351
- Checkpoint pos_embed vs adapter sinusoidal: max_abs_diff=0.00024414 mean_abs_diff=9.016e-05 allclose(atol=1e-3 fp16)=True allclose(atol=1e-5 strict)=False
- NOTE: max_abs_diff 2.4414e-4 == 2^-12 (fp16 ULP near 1.0): residual is serialization rounding, not a different table
- Untouched randn reference mean-abs=sqrt(2/pi)=0.797885 vs measured 0.62257522
- VERDICT: STRUCTURAL SINUSOIDAL CONSTANT (adapter-assigned), not random init

## 3. IN-MEMORY FIX TESTS (3 prompts, no save)
- exact scale-match multiplier=0.01924392 (pos mean-abs 0.62257522 -> txt_emb 0.01198079)
- donor lookup: model.embed_tokens.weight | has_learned_pos_key=False | table_mean_abs=0.01232777
- VARIANT baseline: max_abs_logit_delta=1.009913
    'The capital of France is' -> top5=['BOSE':35.2668, ' honda':32.9134, ' esl':32.1007, ' 若要':31.7391, ' Fotos':31.4392] | gen='BOSE honda esl Fotos erotica' | alpha_ratio=1.0
    '2 + 2 =' -> top5=['BOSE':34.8055, ' honda':32.7415, ' esl':31.9673, ' 若要':31.5311, ' Fotos':31.3343] | gen='BOSE honda esl Fotos erotica' | alpha_ratio=1.0
    'Hello, my name is' -> top5=['BOSE':35.2668, ' honda':32.9134, ' esl':32.1007, ' 若要':31.7392, ' Fotos':31.4391] | gen='BOSE honda esl Fotos erotica' | alpha_ratio=1.0
- VARIANT zero: max_abs_logit_delta=11.842144
    'The capital of France is' -> top5=[' ':51.4973, ',':48.3379, '1':47.6908, ' (':46.857, '.':46.4457] | gen=' , , ' | alpha_ratio=0.0
    '2 + 2 =' -> top5=['1':49.8959, '2':48.4903, '_':45.8223, '.':44.4393, ',':43.9697] | gen='12121' | alpha_ratio=0.0
    'Hello, my name is' -> top5=[' ':51.4973, ',':48.3378, '1':47.6907, ' (':46.8569, '.':46.4456] | gen=' , , ' | alpha_ratio=0.0
- VARIANT scale_0.02: max_abs_logit_delta=9.203515
    'The capital of France is' -> top5=['1':39.6386, '2':38.7564, '_':38.5061, '.':37.5374, ',':36.1596] | gen='12121' | alpha_ratio=0.0
    '2 + 2 =' -> top5=['_':40.7223, '1':40.3907, '2':40.2297, ' =':38.1305, '3':35.4739] | gen='_2121' | alpha_ratio=0.0
    'Hello, my name is' -> top5=['1':39.6386, '2':38.7563, '_':38.5061, '.':37.5373, ',':36.1596] | gen='12121' | alpha_ratio=0.0
- VARIANT scale_exact_match: max_abs_logit_delta=9.343369
    'The capital of France is' -> top5=['1':40.0583, '2':39.1166, '_':38.7138, '.':37.9856, ',':36.725] | gen='12121' | alpha_ratio=0.0
    '2 + 2 =' -> top5=['_':40.9798, '1':40.803, '2':40.5983, ' =':38.3709, '3':35.7989] | gen='_2121' | alpha_ratio=0.0
    'Hello, my name is' -> top5=['1':40.0583, '2':39.1166, '_':38.7138, '.':37.9855, ',':36.725] | gen='12121' | alpha_ratio=0.0
- VARIANT sinusoidal_fresh: max_abs_logit_delta=1.009881
    'The capital of France is' -> top5=['BOSE':35.2666, ' honda':32.9146, ' esl':32.1023, ' 若要':31.7382, ' Fotos':31.4392] | gen='BOSE honda esl Fotos erotica' | alpha_ratio=1.0
    '2 + 2 =' -> top5=['BOSE':34.8052, ' honda':32.7427, ' esl':31.9689, ' 若要':31.5302, ' Fotos':31.3343] | gen='BOSE honda esl Fotos erotica' | alpha_ratio=1.0
    'Hello, my name is' -> top5=['BOSE':35.2666, ' honda':32.9146, ' esl':32.1023, ' 若要':31.7382, ' Fotos':31.4392] | gen='BOSE honda esl Fotos erotica' | alpha_ratio=1.0
- VARIANT from_donor: max_abs_logit_delta=8.725109
    'The capital of France is' -> top5=[' ':49.8599, '1':48.0, ',':46.4163, '2':45.9523, '.':45.6835] | gen=' , , ' | alpha_ratio=0.0
    '2 + 2 =' -> top5=['1':49.8993, '2':48.4919, '_':45.5021, '.':44.5516, ' ':44.5265] | gen='121 ,' | alpha_ratio=0.0
    'Hello, my name is' -> top5=[' ':49.8599, '1':47.9999, ',':46.4163, '2':45.9522, '.':45.6834] | gen=' , , ' | alpha_ratio=0.0

## 4. RANKING BY PROMPT SENSITIVITY
- zero: max_abs_logit_delta=11.842144
- scale_exact_match: max_abs_logit_delta=9.343369
- scale_0.02: max_abs_logit_delta=9.203515
- from_donor: max_abs_logit_delta=8.725109
- baseline: max_abs_logit_delta=1.009913
- sinusoidal_fresh: max_abs_logit_delta=1.009881

elapsed=291.73s | artifacts: embedding_scale_audit_results.json