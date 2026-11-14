"""Convert v20_hf_tmp -> Q8_0 GGUF, measure Q8_0 PPL, and run 10Q pinpoint."""
import os, sys, shutil, subprocess
sys.path.insert(0, r"C:\02_QUILLAN\scripts")
import compile_sovereign_fleet_wsp_v12_q8 as v12
from phase0_held_out_eval_harness import HELD_OUT_PPL_TEXT, measure_perplexity
import flagship_10q_pinpoint as p10

hf_tmp = r"C:\02_QUILLAN\exports\quillan-1b-sovereign-wsp-v20_hf_tmp"
q8 = r"C:\02_QUILLAN\exports\quillan-1b-sovereign-wsp-v20-q8_0.gguf"
if not os.path.exists(q8):
    subprocess.run([sys.executable, v12.CONVERT_SCRIPT, hf_tmp, "--outfile", q8, "--outtype", "q8_0"], check=True)
    shutil.rmtree(hf_tmp, ignore_errors=True)
ppl_file = r"C:\02_QUILLAN\exports\held_out_ppl_anchor_flagship_v20.txt"
with open(ppl_file, "w", encoding="utf-8") as f:
    f.write(HELD_OUT_PPL_TEXT)
ppl = measure_perplexity(q8, ppl_file)
print(f"V20_Q8_PPL={ppl:.4f}", flush=True)
p10.run_pinpoint(q8, port=8101)
