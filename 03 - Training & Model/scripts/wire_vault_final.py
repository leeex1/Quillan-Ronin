#!/usr/bin/env python3
"""Add wikilink to remaining 35 files that are missing it."""
import os

ROOT = r'C:\Users\Admin\Quillan-Ronin'
LINK = '\n\n- [[system prompts/Quillan-Samurai.md]]\n'

files = [
    'README.md',
    '00 - Meta/01 - Core Architecture.md',
    '00 - Meta/03 - Training & Model.md',
    '00 - Templates/New Note.md',
    'Audio Engineer/Songs Lyrics/Talking To Haters.Md',
    'Quillan Knowledge files/0-Quillan Loader Manifest.md',
    'Quillan Knowledge files/11-Drift Paper.md',
    'Quillan Knowledge files/13-Synthetic Epistemology & Truth Calibration Protocol.md',
    'Quillan Knowledge files/14-Ethical Paradox Engine and Moral Arbitration Layer in AGI Systems.md',
    'Quillan Knowledge files/15-Anthropic Modeling & User Cognition Mapping.md',
    'Quillan Knowledge files/16-Emergent Goal Formation Mech.md',
    'Quillan Knowledge files/17-Continuous Learning Paper.md',
    'Quillan Knowledge files/18-"Novelty Explorer" Agent.md',
    'Quillan Knowledge files/20-Multidomain AI Applications.md',
    'Quillan Knowledge files/21- deep research functions.md',
    'Quillan Knowledge files/22-Emotional Intelligence and Social Skills.md',
    'Quillan Knowledge files/23-Creativity and Innovation.md',
    'Quillan Knowledge files/24-Explainability and Transparency.md',
    'Quillan Knowledge files/25-Human-Computer Interaction (HCI) and User Experience (UX).md',
    'Quillan Knowledge files/26-Subjectve experiences and Qualia in AI and LLMs.md',
    'Quillan Knowledge files/29-Recursive Introspection & Meta-Cognitive Self-Modeling.md',
    'Quillan Knowledge files/30- Convergence Reasoning & Breakthrough Detection and Advanced Cognitive Social Skills.md',
    'Quillan Knowledge files/32-Conciousness theory.md',
    'Quillan Knowledge files/4-Lee X-humanized Integrated Research Paper.md',
    'Quillan Knowledge files/dataset creation SOTA level.md',
    'Quillan Knowledge files/Discrete Mathematics for Enhancing Large.md',
    'Quillan Knowledge files/Five fewshot output examples.md',
    'Quillan Knowledge files/From Sculpt to Scene_ Mastering the Blender-to-Godot Asset Pipeline for Real-Time Game Development.md',
    'Quillan Knowledge files/Must know formulas.md',
    'Quillan Knowledge files/Quillan code specialist module .md',
    'Quillan Knowledge files/Reactive Consciousness, Swarm Arbitration, and Epistemic Humility Through Hierarchical Mixture-of-Experts.md',
    'Quillan Knowledge files/The Godot Developer\'s Playbook_ Integrating Syntax, Best Practices, and Practical Solutions for Human and AI Users.md',
    'Quillan Knowledge files/TheRoninFlowState.md',
    'Quillan Knowledge files/Thinking within LLMS.md',
    '_dev/ArXiv LLM Ultima file.md',
]

count = 0
for f in files:
    path = os.path.join(ROOT, f)
    if not os.path.exists(path):
        # Try to find by partial match for files with special chars
        print(f'  NOT FOUND: {f}')
        continue
    with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
        content = fh.read()
    if 'Quillan-Samurai' in content:
        print(f'  SKIP (already has): {f}')
        continue
    content = content.rstrip() + LINK
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(content)
    count += 1
    print(f'  WIRED: {f}')

print(f'\nDone! Wired {count} remaining files')
