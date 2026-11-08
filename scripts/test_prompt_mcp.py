import sys
sys.path.append(r"C:\02_QUILLAN\03 - Training & Model\scripts")
from phase0_held_out_eval_harness import run_single_prompt

prompt = 'Output a valid JSON tool call object with keys "tool" set to "send_slack_notification" and "arguments" containing {"channel": "#ops-alerts", "message": "Cluster Tokyo-03 status nominal"}. Output only the JSON object.'
ans, p_tps, g_tps = run_single_prompt(r"C:\02_QUILLAN\exports\quillan-1b-sovereign-wsp-v3-q4_k_m.gguf", prompt)
print("ANS IS:")
print(repr(ans))

