# Academic Paper Dossier #064: EvoMoE: Expert Evolution in Mixture of Experts for

- **Filename**: `EvoMoE_2505.23830.pdf`
- **ArXiv Identifier**: `2505.23830`
- **Document Length**: 18 Pages (54,043 characters)
- **File Path**: [`C:\02_QUILLAN\10 - Formal Papers\Formal Papers\EvoMoE_2505.23830.pdf`](file:///C:/02_QUILLAN/10 - Formal Papers/Formal Papers/EvoMoE_2505.23830.pdf)

---

## 1. Abstract & Executive Summary
> Recent advancements have shown that the Mixture of Experts (MoE) approach significantly enhances the capacity of large language models (LLMs) and improves performance on downstream tasks. Building on these promising results, multi- modal large language models (MLLMs) have increasingly adopted MoE techniques. However, existing multi-modal MoE tuning methods typically face two key chal- lenges: expert uniformity and router rigidity. Expert uniformity occurs because MoE experts are often initialized by simply replicating the FFN parameters from LLMs, leading to homogenized experts and weakening the intended diversification of the MoE architecture. Meanwhile, router rigidity stems from the prevalent use of static linear routers for expert selection, which fail to distinguish between visual and textual tokens, resulting in similar expert distributions for image and text. To address these limitations, we propose EvoMoE, an innovative MoE tuning frame- work. EvoMoE introduces a meticulously designed expert initialization strategy that progressively evolves multiple robust experts from a single trainable expert, a process termed expert evolution that specifically targets severe expert homoge- nization. Furthermore, we introduce the Dynamic Token-aware Router (DTR), a novel routing mechanism that allocates input tokens to appropriate experts based on their modality and intrinsic token values. This dynamic routing is facilitated by hypernetworks, which dynamically generate routing weig

## 2. Core Problem & Motivation ('Why It Was Created')
Multi-modal Large Language Models (MLLMs), such as GPT-4 [1] and Llama 3 [14], have achieved significant success in addressing open-world tasks, thanks to their scaled-up

## 3. Proposed Methodology & Mathematical Mechanics
s typically face two key chal- lenges: expert uniformity and router rigidity. Expert uniformity occurs because MoE experts are often initialized by simply replicating the FFN parameters from LLMs, leading to homogenized experts and weakening the intended diversification of the MoE architecture. Meanwhile, router rigidity stems from the prevalent use of static linear routers for expert selection, which fail to distinguish between visual and textual tokens, resulting in similar expert distributions for image and text. To address these limitations, we propose EvoMoE, an innovative MoE tuning frame- work. EvoMoE introduces a meticulously designed expert initialization strategy that progressively evolves multiple robust experts from a single trainable expert, a process termed expert evolution that specifically targets severe expert homoge- nization. Furthermore, we introduce the Dynamic Token-aware Router (DTR), a novel routing mechanism that allocates input tokens to appropriate experts based on their modality and intrinsic token values. This dynamic routing is facilitated by hypernetworks, which dynamically generate routing weights tailored for each individual token. Extensive

### Key Mathematical Formulations Extracted from Body:
- `$n = [2, 3, , N] represents each of the N experts$`
- `$V = [ v1, v2, , vP ] RP C and text tokens$`
- `$T = [ t1, t2, , tM ] RM C, here, P is the se-$`
- `$l = FFN(DTR$`
- `$l = FFN(DTR$`
- `$n = H (z ) , where V, T (6)$`

## 4. Empirical Evaluation & Benchmark Evidence
demonstrate that EvoMoE significantly outperforms other sparse MLLMs across a variety of multi-modal benchmarks, including MME, MMBench, TextVQA, and POPE. Our results highlight the effec- tiveness of EvoMoE in enhancing the performance of MLLMs by addressing the critical issues of expert uniformity and router rigidity. 1 Introduction Multi-modal Large Language Models (MLLMs), such as GPT-4 [1] and Llama 3 [14], have achieved significant success in addressing open-world tasks, thanks to their scaled-up architectures. However, scaling up models often increases computational demands and is limited by device capacity. To address these challenges, sparsely activated mixture-of-expert (MoE) models have gained popularity in large language models (LLMs) [51, 29, 47, 36, 53, 52], reducing computational costs and enhancing efficiency. MoE models typically include multiple experts and a routing network that selects the optimal expert for each input token. This design minimizes interference among diverse input tokens, enabling each expert to specialize more effectively in specific tasks. For example, DeepSeek V3 [26] This work was completed during his internship at Ant Group. Corresponding author employ MoE language models with 671B parameters, activating 37B parameters, and have achieved notable results. (a) Expert Uniformity (b) Router Rigidity Figure 1: Two key challenges in MoE-tuning. (a) Expert Uniformity: Randomly shuffling the router during inference results in negligible performance degradation, suggesting uniformity among experts derived from replicated initialization. (b) Router Rigidity: Kernel density estimation (KDE) of the logits for image and text tokens reveals that the linear router generates input-insensitive selections, leading to static distributions with significant overlap in the density of logits for image and text token. The success of MoE in LLMs has spurred significant interest in its application to MLLMs [20, 25, 26, 37, 44, 49, 54]. A recent prominent approach is MoE-LLaV A [25], which introduces MoE-tuning, a multi-stage process that converts dense MLLM models into sparse MoE structures during instruction tuning, specifically tailored for multi-modal tasks. However, in MoE-tuning, experts are typically initialized by replication, leading to the first critical challenge: expert uniformity. This results in expert homogenization during multi-stage tuning, thereby impeding specialization and undermining the efficacy of the MoE framework. Fi

## 5. Direct Integration into Quillan-Ronin v5.4.0-ONI
- **Target Module**: `quillan_v5_4_oni.py`
- **Architectural Role**: Active Parameter Pipeline
- **Gradient Status**: 100% Active in autograd computational graph.
