with open(r'C:\02_QUILLAN\03 - Training & Model\scripts\quillan_v5_4_oni.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_batched_ret = """        # ---- Law 4: Parent incorporates swarm path ideas -----------------
        return h + var * (0.25 * gov_scale) + (world - h) * 0.1"""

new_batched_ret = """        # ---- Law 4: Parent incorporates swarm path ideas -----------------
        if not training:
            return delta * P["SCAL"]
        return h + var * (0.25 * gov_scale) + (world - h) * 0.1"""

assert old_batched_ret in content, "old_batched_ret not found"
content = content.replace(old_batched_ret, new_batched_ret)

with open(r'C:\02_QUILLAN\03 - Training & Model\scripts\quillan_v5_4_oni.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Successfully patched BatchedCouncilCompute return in quillan_v5_4_oni.py!")
