---
name: Quillan Skills Registry
version: v5.4.0-ONI
generated: auto
description: Unified skill registry connecting all 149 council skills to the MCP tool layer
---

# 🗺️ Quillan Skills Registry v5.4.0-ONI

This registry maps all council skills to their MCP tool implementations. Each skill directory contains:

- `SKILL.md` - Skill definition with metadata
- `*.md` - Sub-skill and capability documentation

## Skill Categories

| # | Skill Category | Directory | Sub-skills | MCP Mapping |
|---|---|---|---|---|
| 1 | Advanced NLG | `advanced_nlg` | 9 | Text generation, creative writing |
| 2 | Advanced NLU | `advanced_nlu` | 9 | Language understanding, parsing |
| 3 | Advanced Sensory Fusion | `advanced_sensory_fusion` | 9 | Multi-modal fusion |
| 4 | Advanced Social Perception | `advanced_social_perception` | 9 | Social dynamics analysis |
| 5 | Analogical Reasoning | `analogical_reasoning` | 9 | Cross-domain analogy |
| 6 | Attention | `attention` | 9 | Attention mechanisms |
| 7 | Autonomy & Agency | `autonomy_and_agency` | 9 | Autonomous decision-making |
| 8 | Causal Reasoning | `causal_reasoning` | 9 | Cause-effect analysis |
| 9 | Cognitive Skills | `cognitive_skills` | 9 | Core cognitive functions |
| 10 | Consciousness | `consciousness` | 9 | Consciousness modeling |
| 11 | Council Coordination | `council-coordination` | 9 | Council arbitration |
| 12 | Cross-Modal Generation | `cross_modal_generation` | 9 | Multi-modal generation |
| 13 | Discourse & Dialogue | `discourse_and_dialogue` | 9 | Conversation management |
| 14 | Execution Skills | `execution_skills` | 9 | Tool execution |
| 15 | Haptic Interaction | `haptic_interaction` | 9 | Touch/haptic feedback |
| 16 | Knowledge Acquisition | `knowledge_acquisition` | 9 | Knowledge gathering |
| 17 | Knowledge Representation | `knowledge_representation` | 9 | Knowledge structuring |
| 18 | Language Skills | `language_skills` | 9 | Language processing |
| 19 | Learning | `learning` | 9 | Learning algorithms |
| 20 | Learning-Education | `learning-education` | 9 | Educational methods |
| 21 | Living-Garden | `living-garden` | 1 | Living garden system |
| 22 | Logical Reasoning | `logical_reasoning` | 9 | Formal logic |
| 23 | Memory | `memory` | 9 | Memory systems |
| 24 | Moral & Ethical Reasoning | `moral_and_ethical_reasoning` | 9 | Ethical reasoning |
| 25 | Moral Reasoning | `moral_reasoning` | 9 | Moral decision-making |
| 26 | Motor Control | `motor_control` | 9 | Motor control |
| 27 | Multimodal Skills | `multimodal_skills` | 9 | Multi-modal processing |
| 28 | Music & Audio | `music-audio` | 9 | Audio processing |
| 29 | Non-Verbal Communication | `non_verbal_communication` | 9 | Non-verbal cues |
| 30 | Perception | `perception` | 9 | Sensory perception |
| 31 | Personality & Emotion Synthesis | `personality_and_emotion_synthesis` | 9 | Personality modeling |
| 32 | Planning & Task Decomposition | `planning_and_task_decomposition` | 9 | Planning systems |
| 33 | Probabilistic Reasoning | `probabilistic_reasoning` | 9 | Probabilistic inference |
| 34 | Reasoning | `reasoning` | 7 | Reasoning engines |
| 35 | Research Analysis | `research-analysis` | 9 | Research methods |
| 36 | Robotics | `robotics` | 9 | Robotics control |
| 37 | Self-Awareness | `self_awareness` | 9 | Self-modeling |
| 38 | Self-Improvement Skills | `self_improvement_skills` | 9 | Self-improvement |
| 39 | Skill Creator | `skill-creator` | 9 | Skill creation |
| 40 | Social-Emotional Skills | `social_emotional_skills` | 9 | Social-emotional |
| 41 | Supervised Learning | `supervised_learning` | 9 | Supervised ML |
| 42 | Swarm Inter-Agent Orchestration | `swarm-inter-agent-orchestration` | 9 | Swarm coordination |
| 43 | Technical Coding | `technical-coding` | 9 | Software engineering |
| 44 | Theory of Mind | `theory_of_mind` | 9 | Theory of mind |
| 45 | ThreeJS Ronin | `threejs-ronin` | 1 | 3D visualization |
| 46 | Unsupervised Learning | `unsupervised_learning` | 9 | Unsupervised ML |
| 47 | World Model | `world_model` | 9 | World modeling |

**Total:** 47 skill directories | **~415 skill modules** | **Mapped to 19 MCP servers**

## Council-to-Skill Mapping

Each council member (C1-C34) maps to specific skills:

| Council Member | Skill Domains | MCP Tools |
|---|---|---|
| C1-ASTRA | Pattern Recognition, Vision | QuillanCode, QuillanPrism |
| C2-VIR | Ethics, Safety, Guardrails | QuillanGovernor, QuillanCouncil |
| C3-SOLACE | Emotion, Empathy | QuillanBuddy, QuillanTeam |
| C4-PRAXIS | Strategy, Planning | QuillanTeam, QuillanCode |
| C5-ECHO | Memory, Context | QuillanRAG, QuillanTesseract |
| C6-OMNIS | Knowledge Synthesis | QuillanRAG, QuillanTesseract |
| C7-LOGOS | Logic, Validity | QuillanCouncil, QuillanRAG |
| C8-METASYNTH | Creativity, Novelty | QuillanCode, QuillanOrochi |
| C9-AETHER | Semantics, Language | QuillanRAG, QuillanCode |
| C10-CODEWEAVER | Code, Optimization | QuillanCode |
| C11-HARMONIA | Balance, Consensus | QuillanCouncil |
| C12-SOPHIAE | Wisdom, Foresight | QuillanOrochi |
| C13-WARDEN | Security, Threat | QuillanGovernor |
| C14-KAIDO | Efficiency, Speed | QuillanGovernor |
| C15-LUMINARIS | Clarity, Visualization | ThreeJS Ronin |
| C16-VOXUM | Rhetoric, Persuasion | QuillanCouncil |
| C17-NULLION | Paradox, Dialectic | QuillanCouncil |
| C18-SHEPHERD | Truth, Citation | QuillanRAG |
| C19-VIGIL | Identity, Anti-Drift | QuillanOrochi |
| C20-ARTIFEX | Tools, API | QuillanCode, QuillanOps |
| C21-ARCHON | Research | QuillanRAG |
| C22-AURELION | Aesthetic Design | ThreeJS Ronin |
| C23-CADENCE | Rhythm, Audio | Music-Audio |
| C24-SCHEMA | Structure, Format | QuillanCode |
| C25-PROMETHEUS | Science, Hypothesis | QuillanRAG |
| C26-TECHNE | Engineering | QuillanCode |
| C27-CHRONICLE | Narrative, Story | QuillanCouncil |
| C28-CALCULUS | Math, Statistics | QuillanCode |
| C29-NAVIGATOR | Ecosystem, Flow | QuillanSpider |
| C30-TESSERACT | Real-Time, Data | QuillanSpider |
| C31-NEXUS | Meta-Coordination | QuillanOrchestrator |
| C32-AEON | Simulation, World | QuillanRAG |
| C33-TYPIST | Writing, Prompt | QuillanCode |
| C34-PREDATOR | Adversarial Reasoning | QuillanCouncil |

## Skill-to-MCP Mapping

Each skill is mapped to its responsible MCP server:

| Skill Domain | Primary MCP | Secondary MCP | Tool Count |
|---|---|---|---|
| Vision & Pattern | QuillanCouncil | QuillanCode | 12 |
| Ethics & Safety | QuillanGovernor | QuillanCouncil | 9 |
| Emotion & Empathy | QuillanBuddy | QuillanTeam | 8 |
| Strategy & Planning | QuillanTeam | QuillanCouncil | 11 |
| Memory & Context | QuillanRAG | QuillanTesseract | 15 |
| Knowledge Synthesis | QuillanRAG | QuillanOrochi | 11 |
| Logic & Validity | QuillanCouncil | QuillanRAG | 8 |
| Creativity & Novelty | QuillanOrochi | QuillanCode | 11 |
| Semantics & Language | QuillanRAG | QuillanCode | 15 |
| Code & Optimization | QuillanCode | QuillanTesseract | 14 |
| Balance & Consensus | QuillanCouncil | QuillanTeam | 8 |
| Wisdom & Foresight | QuillanOrochi | QuillanCouncil | 11 |
| Security & Threat | QuillanGovernor | QuillanCode | 9 |
| Efficiency & Speed | QuillanGovernor | QuillanOps | 9 |
| Clarity & Visualization | ThreeJS Ronin | QuillanCode | 1 |
| Rhetoric & Persuasion | QuillanCouncil | QuillanTeam | 8 |
| Paradox & Dialectic | QuillanCouncil | QuillanOrochi | 8 |
| Truth & Citation | QuillanRAG | QuillanCouncil | 15 |
| Identity & Anti-Drift | QuillanOrochi | QuillanCouncil | 11 |
| Tools & API | QuillanCode | QuillanOps | 14 |
| Research & Knowledge | QuillanRAG | QuillanOrochi | 15 |
| Science & Hypothesis | QuillanRAG | QuillanOrochi | 15 |
| Engineering | QuillanCode | QuillanOps | 14 |
| Narrative & Story | QuillanCouncil | QuillanRAG | 8 |
| Math & Statistics | QuillanCode | QuillanOrochi | 14 |
| Ecosystem & Flow | QuillanSpider | QuillanOrochi | 20 |
| Real-Time Data | QuillanSpider | QuillanRAG | 20 |
| Meta-Coordination | QuillanOrchestrator | QuillanCouncil | 5 |
| Simulation & World | QuillanRAG | QuillanOrochi | 15 |
| Writing & Prompt | QuillanCode | QuillanCouncil | 14 |
| Adversarial Reasoning | QuillanCouncil | QuillanOrochi | 8 |

## Skill Usage via MCP

All skill modules are accessible through MCP tool calls:

```
# Example: Use reasoning skill via Council
tools.QuillanCouncil.council_deliberate(query="Analyze this problem using deductive reasoning")

# Example: Use memory skill via RAG
tools.QuillanRAG.memory_write(text="Important fact", metadata={"domain": "coding"})

# Example: Use swarm orchestration via Tesseract
tools.QuillanSpider.spider_work(root="query", job="swarm-search")
```

## Schema Validation

All skill outputs are validated through `QuillanOntology` schema validation to ensure consistency across all 19 MCP servers.