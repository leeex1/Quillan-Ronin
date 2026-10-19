# Academic Paper Dossier #014: Evolution Strategies at the Hyperscale

- **Filename**: `2511.16652v2.pdf`
- **ArXiv Identifier**: `2511.16652v2`
- **Document Length**: 76 Pages (189,450 characters)
- **File Path**: [`C:\02_QUILLAN\10 - Formal Papers\Formal Papers\2511.16652v2.pdf`](file:///C:/02_QUILLAN/10 - Formal Papers/Formal Papers/2511.16652v2.pdf)

---

## 1. Abstract & Executive Summary
> Evolution Strategies (ES) is a class of powerful black-box optimisation methods that are highly parallelisable and can handle non-differentiable and noisy objectives. However, na ve ES becomes prohibitively expensive at scale on GPUs due to the low arithmetic intensity of batched matrix multiplications with unstructured random perturbations. We introduce Evolution Guided GeneRal Optimisation via Low-rank Learning (EGGROLL), which improves arithmetic intensity by structuring individual perturbations as rank- r matrices, resulting in a hundredfold increase in training speed for billion-parameter models at large population sizes, achieving up to 91% of the throughput of pure batch inference. We provide a rigorous theoretical analysis of Gaussian ES for high-dimensional parameter objectives, investigating conditions needed for ES updates to converge in high dimensions. Our results reveal a linearising effect, and proving consistency between EGGROLL and ES as parameter dimension increases. Our experiments show that EGGROLL: (1) enables the stable pretraining of nonlinear recurrent language models that operate purely in integer datatypes, (2) is competitive with GRPO for post-training LLMs on reasoning tasks, and (3) does not compromise performance compared to ES in tabula rasa RL settings, despite being faster. Our code is available athttps://eshyperscale.github.io/. = f ( + ) = f ( + ) ... ... = f ( + ) Initial weights Rank-one perturbationEi Fitness evaluationWeighted average Fi

## 2. Core Problem & Motivation ('Why It Was Created')
Evolution Strategies (ES) (Rechenberg, 1978; Beyer, 1995; Beyer & Schwefel, 2002) is an attractive alterna- tive to first-order

## 3. Proposed Methodology & Mathematical Mechanics
s that are highly parallelisable and can handle non-differentiable and noisy objectives. However, na ve ES becomes prohibitively expensive at scale on GPUs due to the low arithmetic intensity of batched matrix multiplications with unstructured random perturbations. We introduce Evolution Guided GeneRal Optimisation via Low-rank Learning (EGGROLL), which improves arithmetic intensity by structuring individual perturbations as rank- r matrices, resulting in a hundredfold increase in training speed for billion-parameter models at large population sizes, achieving up to 91% of the throughput of pure batch inference. We provide a rigorous theoretical analysis of Gaussian ES for high-dimensional parameter objectives, investigating conditions needed for ES updates to converge in high dimensions. Our

### Key Mathematical Formulations Extracted from Body:
- `$d =o(d 1/2) under which$`
- `$x=a evaluated atx=a as f(a)$`
- `$U= I m andV= I n, the equivalent multivariate Gaussian distribu-$`
- `$W=M+ E)],(3)$`
- `$E= 1 rAB maps to the manifold Mr R m n$`
- `$t = 1 rAi,tB$`

## 4. Empirical Evaluation & Benchmark Evidence
show that EGGROLL: (1) enables the stable pretraining of nonlinear recurrent language models that operate purely in integer datatypes, (2) is competitive with GRPO for post-training LLMs on reasoning tasks, and (3) does not compromise performance compared to ES in tabula rasa RL settings, despite being faster. Our code is available athttps://eshyperscale.github.io/. = f ( + ) = f ( + ) ... ... = f ( + ) Initial weights Rank-one perturbationEi Fitness evaluationWeighted average Final rank-Nupdate Figure 1: Schematic visualisation of EGGROLL usingNworkers. 1 Introduction Evolution Strategies (ES) (Rechenberg, 1978; Beyer, 1995; Beyer & Schwefel, 2002) is an attractive alterna- tive to first-order methods based on gradient backpropagation for several reasons. First, ES does not require differentiability; it can optimise a broader class of models, like those with discrete parametrisations (cellular *Equal Contribution Core Contributor, sorted by alphabetical order in first names Equal Senior Authors 1 arXiv:2511.16652v2 [cs.LG] 16 Feb 2026 EGGROLL PPO OpenES 0 20 40 60 80 100Normalised Speed 91 34 0.41 Normalised Training Speeds 101 102 103 Training Step 4 6 8Test Loss (bits/byte) Pure Integer Pretraining EGGROLL (int8) Backprop (fp32) 100 102 104 106 Population Size Figure 2: (a) Relative speed of our method, EGGROLL, in terms of experience throughput versus prior methods, where 100 is the maximum batch inference throughput. See Appendix E for more details. (b) We use EGGROLL to train an int8 RNN language model from scratch, scaling population size from 2 to 1,048,576 with a fixed data batch size of 16. The dotted line is a fp32 Transformer trained with backprop SGD. EGGROLL s test next-token cross-entropy of 3.40 bits/byte while backprop only gets 3.58 bits/byte. automata) or objectives for which gradients are unavailable or noisy, such as outcome-only rewards in LLM fine-tuning (Qiu et al., 2025). Second, ES can be more robust to noisy and ill-conditioned optimisation landscapes (Wierstra et al., 2011; Xue et al., 2021). Population-based exploration smooths irregularities (Salimans et al., 2017), tolerates discontinuities, and mitigates issues like ill-conditioned curvature or vanishing and exploding gradients in long-range or recurrent settings (Hansen, 2023). Third, ES is highly amenable to parallel scaling, since fitness evaluations are independent across population members and require only the com- munication of scalar fitnesses, which maps cleanly ont

## 5. Direct Integration into Quillan-Ronin v5.4.0-ONI
- **Target Module**: `quillan_v5_4_oni.py`
- **Architectural Role**: Active Parameter Pipeline
- **Gradient Status**: 100% Active in autograd computational graph.
