# Academic Paper Dossier #136: Seed1.5-VL Technical Report

- **Filename**: `2505.07062v1.pdf`
- **ArXiv Identifier**: `2505.07062v1`
- **Document Length**: 77 Pages (216,247 characters)
- **File Path**: [`C:\02_QUILLAN\09 - Projects\projects\Nextverse-protype-app--main\research papers\2505.07062v1.pdf`](file:///C:/02_QUILLAN/09 - Projects/projects/Nextverse-protype-app--main/research papers/2505.07062v1.pdf)

---

## 1. Abstract & Executive Summary
> We present Seed1.5-VL, a vision-language foundation model designed to advance general-purpose multimodal understanding and reasoning. Seed1.5-VL is composed with a 532M-parameter vision encoder and a Mixture-of-Experts (MoE) LLM of 20B active parameters. Despite its relatively compact architecture, it delivers strong performance across a wide spectrum of public VLM benchmarks and internal evaluation suites, achieving the state-of-the-art performance on 38 out of 60 public benchmarks. Moreover, in agent-centric tasks such as GUI control and gameplay, Seed1.5-VL outperforms leading multimodal systems, including OpenAI CUA and Claude 3.7. Beyond visual and video understanding, it also demonstrates strong reasoning abilities, making it particularly effective for multimodal reasoning challenges such as visual puzzles. We believe these capabilities will empower broader applications across diverse tasks. In this report, we mainly provide a comprehensive review of our experiences in building Seed1.5-VL across model design, data construction, and training at various stages, hoping that this report can inspire further research. Seed1.5-VL is now accessible on Volcano Enginea. Date: May 23, 2025 Correspondence: shiguang.sg@bytedance.com aModel ID: doubao-1-5-thinking-vision-pro-250428 1 Contents

## 2. Core Problem & Motivation ('Why It Was Created')
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4 2

## 3. Proposed Methodology & Mathematical Mechanics
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22 6

### Key Mathematical Formulations Extracted from Body:
- `$t = len(s) - 1 # Fixed: use last index (len(s)-1)$`
- `$t = len(s) - 1 ensures$`
- `$t=0, right=2 match ( a == a ), then left=1,$`
- `$t=1 loop ends, returnTrue (correct)$`
- `$t=0, right=2 a != c return False (correct)$`
- `$d=636M0nNbPs$`

## 4. Empirical Evaluation & Benchmark Evidence
and internal evaluation suites, achieving the state-of-the-art performance on 38 out of 60 public benchmarks. Moreover, in agent-centric tasks such as GUI control and gameplay, Seed1.5-VL outperforms leading multimodal systems, including OpenAI CUA and Claude 3.7. Beyond visual and video understanding, it also demonstrates strong reasoning abilities, making it particularly effective for multimodal reasoning challenges such as visual puzzles. We believe these capabilities will empower broader applications across diverse tasks. In this report, we mainly provide a comprehensive review of our experiences in building Seed1.5-VL across model design, data construction, and training at various stages, hoping that this report can inspire further research. Seed1.5-VL is now accessible on Volcano Enginea. Date: May 23, 2025 Correspondence: shiguang.sg@bytedance.com aModel ID: doubao-1-5-thinking-vision-pro-250428 1 Contents 1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4 2 Architecture . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5 2.1 Vision Encoder . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5 2.1.1 Architecture . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6 2.1.2 ViT Pre-training Stage . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6 2.2 Video Encoding . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7 3 Pre-training . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8 3.1 Pre-training Data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8 3.1.1 Generic Image-Text Pairs & Knowledge Data . . . . . . . . . . . . . . . . . . . . . . . 8 3.1.2 Optical Character Recognition (OCR) . . . . . . . . . . . . . . . . . . . . . . . . . . . 9 3.1.3 Visual Grounding & Counting . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10 3.1.4 3D Spatial Understanding . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11 3.1.5 Video . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11 3.1.6 Science, Technology, Engineering, and Mathematics (STEM) . . . . . . . . . . . . . . 12 3.1.7 Graphical User Interface (GUI) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12 3

## 5. Direct Integration into Quillan-Ronin v5.4.0-ONI
- **Target Module**: `quillan_v5_4_oni.py`
- **Architectural Role**: Active Parameter Pipeline
- **Gradient Status**: 100% Active in autograd computational graph.
