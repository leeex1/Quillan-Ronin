---
name: Quillan Skills MCP Mapping
version: v5.4.0-ONI
description: Complete mapping of all 47 skill categories to their MCP tool implementations
---

# 🛠️ Skills-to-MCP Mapping Complete

This document maps every skill in the Quillan system to its corresponding MCP server tools.

## Complete Skill-to-MCP Mapping Table

| # | Skill | Directory | MCP Server | MCP Tools | Status |
|---|-------|-----------|------------|-----------|--------|
| 1 | Advanced NLG | advanced_nlg | QuillanCode | codeweaver_full_pipeline, codeweaver_fix_scaffold | ✅ |
| 2 | Advanced NLU | advanced_nlu | QuillanRAG | rag_route, rag_health | ✅ |
| 3 | Advanced Sensory Fusion | advanced_sensory_fusion | QuillanMedia | media_fusion, media_full_pipeline | ✅ |
| 4 | Advanced Social Perception | advanced_social_perception | QuillanCouncil | council_full_deliberation | ✅ |
| 5 | Analogical Reasoning | analogical_reasoning | QuillanRAG | rag_route | ✅ |
| 6 | Attention | attention | QuillanOrochi | orochi_evolve_weights | ✅ |
| 7 | Autonomy & Agency | autonomy_and_agency | QuillanTeam | team_spawn_crew, team_dispatch | ✅ |
| 8 | Causal Reasoning | causal_reasoning | QuillanCouncil | council_full_deliberation | ✅ |
| 9 | Cognitive Skills | cognitive_skills | QuillanThinkingEngine | quillan_think | ✅ |
| 10 | Consciousness | consciousness | QuillanOrochi | orochi_auto_improve | ✅ |
| 11 | Council Coordination | council-coordination | QuillanCouncil | council_full_deliberation | ✅ |
| 12 | Cross-Modal Generation | cross_modal_generation | QuillanMedia | media_image_brief, media_video_brief | ✅ |
| 13 | Discourse & Dialogue | discourse_and_dialogue | QuillanTeam | team_dispatch, team_debate | ✅ |
| 14 | Execution Skills | execution_skills | QuillanCode | codeweaver_full_pipeline | ✅ |
| 15 | Haptic Interaction | haptic_interaction | QuillanCode | codeweaver_full_pipeline | ✅ |
| 16 | Knowledge Acquisition | knowledge_acquisition | QuillanRAG | rag_route | ✅ |
| 17 | Knowledge Representation | knowledge_representation | QuillanRAG | rag_route | ✅ |
| 18 | Language Skills | language_skills | QuillanCode | codeweaver_full_pipeline | ✅ |
| 19 | Learning | learning | QuillanOrochi | orochi_auto_improve | ✅ |
| 20 | Learning-Education | learning-education | QuillanRAG | rag_route | ✅ |
| 21 | Living-Garden | living-garden | QuillanCode | codeweaver_full_pipeline | ✅ |
| 22 | Logical Reasoning | logical_reasoning | QuillanCouncil | council_full_deliberation | ✅ |
| 23 | Memory | memory | QuillanRAG | rag_route | ✅ |
| 24 | Moral & Ethical Reasoning | moral_and_ethical_reasoning | QuillanGovernor | governor_system_health | ✅ |
| 25 | Moral Reasoning | moral_reasoning | QuillanGovernor | governor_system_health | ✅ |
| 26 | Motor Control | motor_control | QuillanCode | codeweaver_full_pipeline | ✅ |
| 27 | Multimodal Skills | multimodal_skills | QuillanMedia | media_full_pipeline | ✅ |
| 28 | Music & Audio | music-audio | QuillanMedia | media_image_brief | ✅ |
| 29 | Non-Verbal Communication | non_verbal_communication | QuillanCouncil | council_full_deliberation | ✅ |
| 30 | Perception | perception | QuillanRAG | rag_route | ✅ |
| 31 | Personality & Emotion Synthesis | personality_and_emotion_synthesis | QuillanBuddy | buddy_mirror_collaborate | ✅ |
| 32 | Planning & Task Decomposition | planning_and_task_decomposition | QuillanTeam | team_spawn_crew | ✅ |
| 33 | Probabilistic Reasoning | probabilistic_reasoning | QuillanCouncil | council_full_deliberation | ✅ |
| 34 | Reasoning | reasoning | QuillanThinkingEngine | quillan_think | ✅ |
| 35 | Research Analysis | research-analysis | QuillanRAG | rag_route | ✅ |
| 36 | Robotics | robotics | QuillanCode | codeweaver_full_pipeline | ✅ |
| 37 | Self-Awareness | self_awareness | QuillanOrochi | orochi_auto_improve | ✅ |
| 38 | Self-Improvement Skills | self_improvement_skills | QuillanOrochi | orochi_auto_improve | ✅ |
| 39 | Skill Creator | skill-creator | QuillanCode | codeweaver_full_pipeline | ✅ |
| 40 | Social-Emotional Skills | social_emotional_skills | QuillanBuddy | buddy_mirror_collaborate | ✅ |
| 41 | Supervised Learning | supervised_learning | QuillanModelLab | model_checkpoint_inspect | ✅ |
| 42 | Swarm Inter-Agent Orchestration | swarm-inter-agent-orchestration | QuillanSpider | spider_crawl, spider_work | ✅ |
| 43 | Technical Coding | technical-coding | QuillanCode | codeweaver_full_pipeline | ✅ |
| 44 | Theory of Mind | theory_of_mind | QuillanCouncil | council_full_deliberation | ✅ |
| 45 | ThreeJS Ronin | threejs-ronin | QuillanMedia | media_image_brief | ✅ |
| 46 | Unsupervised Learning | unsupervised_learning | QuillanModelLab | model_checkpoint_inspect | ✅ |
| 47 | World Model | world_model | QuillanRAG | rag_route | ✅ |

**Total:** 47 skill domains | **196+ MCP tools** | **100% mapped**

## Skill-to-Tool Execution Flow

Each skill follows this execution path:

```
User Request → Skill Identification → MCP Tool Selection → Execution → Result
     ↓              ↓                  ↓              ↓              ↓
   User Input   → Skill Matching   → Tool Selection → MCP Execution → Response
```

## Skill Invocation Examples

```python
# Use reasoning skill through council
tools.QuillanCouncil.council_deliberate(query="Analyze this problem")

# Use memory through RAG
tools.QuillanRAG.memory_write(text="Important finding", metadata={"domain": "research"})

# Use reasoning through ThinkingEngine
tools.QuillanThinkingEngine.quillan_think(query="Solve this complex problem")

# Use code generation through CodeWeaver
tools.QuillanCode.codeweaver_full_pipeline(code_or_file_path="src/main.py")

# Use swarm orchestration through Spider
tools.QuillanSpider.spider_work(root="https://example.com", job="crawl")

# Use world model through RAG
tools.QuillanRAG.rag_route(target_mcp="council", payload={"query": "What is the world state?"})
```

## Skill Validation Rules

All skills must satisfy:
1. ✅ Has `SKILL.md` metadata file
2. ✅ Has sub-skill documentation (9 sub-skills for standard skills)
3. ✅ Maps to at least one MCP server tool
4.  outputs pass schema validation
5.  Follows Quillan v5.4.0-ONI formatting standards

## Skill Dependencies

| Skill | Depends On | Used By |
|---|---|---|
| Reasoning | None | All reasoning tasks |
| Memory | None | All context-aware tools |
| Knowledge Acquisition | Memory, Perception | Research analysis |
| Causal Reasoning | Reasoning | Scientific analysis |
| Probabilistic Reasoning | Reasoning | Risk assessment |
| Creativity | Knowledge Acquisition | Content generation |
| Ethics | Moral Reasoning | Safety gates |
| Tool Use | Technical Coding | All tool execution |
| World Model | Memory, Knowledge | Simulation |
