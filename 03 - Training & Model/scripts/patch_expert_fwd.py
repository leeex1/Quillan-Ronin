with open(r'C:\02_QUILLAN\03 - Training & Model\scripts\quillan_v5_4_oni.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix Expert forward to return delta only
old_expert_fwd = """    def forward(self, x: torch.Tensor, gov_scale: float = 1.0) -> torch.Tensor:
        delta = (x @ self.lora_A) @ self.lora_B * self.scaling
        return self.swarm(x + delta, scale=gov_scale)"""

new_expert_fwd = """    def forward(self, x: torch.Tensor, gov_scale: float = 1.0) -> torch.Tensor:
        delta = (x @ self.lora_A) @ self.lora_B * self.scaling
        if self.training:
            return self.swarm(x + delta, scale=gov_scale)
        else:
            return delta"""

assert old_expert_fwd in content, "old_expert_fwd not found"
content = content.replace(old_expert_fwd, new_expert_fwd)

# And in BatchedCouncilCompute:
old_batched = """                    e_all = self._batched_council(flat_x, gov_scale, self.training)  # [E,BT,H]"""
new_batched = """                    e_all = self._batched_council(flat_x, gov_scale, self.training)  # [E,BT,H]"""

with open(r'C:\02_QUILLAN\03 - Training & Model\scripts\quillan_v5_4_oni.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Successfully patched Expert.forward in quillan_v5_4_oni.py!")
