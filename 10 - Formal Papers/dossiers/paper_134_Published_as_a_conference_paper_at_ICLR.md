# Academic Paper Dossier #134: Published as a conference paper at ICLR 2025

- **Filename**: `2502.20766v1.pdf`
- **ArXiv Identifier**: `2502.20766v1`
- **Document Length**: 27 Pages (91,273 characters)
- **File Path**: [`C:\02_QUILLAN\09 - Projects\projects\Nextverse-protype-app--main\research papers\2502.20766v1.pdf`](file:///C:/02_QUILLAN/09 - Projects/projects/Nextverse-protype-app--main/research papers/2502.20766v1.pdf)

---

## 1. Abstract & Executive Summary
> MECHANISM FOR EFFICIENT LONG -SEQUENCE IN- FERENCE Xunhao Lai1 , Jianqiao Lu2 , Yao Luo3, Yiyuan Ma3, Xun Zhou3, 1Peking University 2The University of Hong Kong 3 ByteDance Inc laixunhao@pku.edu.cn,jqlu@cs.hku.hk {luoyao.0, mayiyuan.unicorn, zhouxun}@bytedance.com ABSTRACT Large language models (LLMs) encounter computational challenges during long- sequence inference, especially in the attention pre-filling phase, where the com- plexity grows quadratically with the prompt length. Previous efforts to mitigate

## 2. Core Problem & Motivation ('Why It Was Created')
Analyzed across document pages.

## 3. Proposed Methodology & Mathematical Mechanics
s lacked the flex- ibility to efficiently adapt to varying input demands. In this paper, we introduce FlexPrefill, a Flexible sparse Pre-filling mechanism that dynamically adjusts sparse attention patterns and computational budget in real-time to meet the specific requirements of each input and attention head. The flexibility of our method is demonstrated through two key innovations: 1) Query-Aware Sparse Pattern Deter- mination: By measuring Jensen-Shannon divergence, this component adaptively switches between query-specific diverse attention patterns and predefined atten- tion patterns. 2) Cumulative-Attention Based Index Selection: This component dynamically selects query-key indexes to be computed based on different atten- tion patterns, ensuring the sum of attention scores meets a predefined threshold. FlexPrefill adaptively optimizes the sparse pattern and sparse ratio of each atten- tion head based on the prompt, enhancing efficiency in long-sequence inference tasks. Experimental

### Key Mathematical Formulations Extracted from Body:
- `$i=1 Si, where$`
- `$a = softmax$`
- `$a = sumpool$`
- `$S = iSi and |Si| denotes the size of chosen index subset, Qi and Kj are the query a$`
- `$n == query_specific then$`
- `$n == vertical_slash then$`

## 4. Empirical Evaluation & Benchmark Evidence
show significant improvements in both speed and ac- curacy over prior methods, providing a more flexible and efficient solution for LLM inference. https://github.com/bytedance/FlexPrefill Performance ( ) Throughput ( ) LLaMA Performance ( ) GLM Performance ( ) Yi Performance ( ) Qwen Ours Minference Flash Attention Streaming LLM Throughput ( ) Throughput ( ) Throughput ( ) Low efficiency zone High efficiency zone Low efficiency zone High efficiency zone Low efficiency zone High efficiency zone Low efficiency zone High efficiency zone (a) Performance comparison of different inference ap- proaches across LLaMA, GLM, Yi, and Qwen models. Context Length 16k8k 64k32k 256k128k 512k384k 10 8 6 4 2 Acceleration Ratio 2.2x 3.4x 4.8x 6.2x 8.0x 9.6x 1.1x 2.0x 3.3x 4.4x 5.2x Flash Attention Minference Ours (b) Attention acceleration ratio comparison of differ- ent approaches across various context lengths. Figure 1: Performance and speed analysis of different inference approaches. Leading co-authors with equal contribution. 1 arXiv:2502.20766v1 [cs.LG] 28 Feb 2025 Published as a conference paper at ICLR 2025 1 I NTRODUCTION Large language models (LLMs) are transforming various domains by enabling sophisticated natural language understanding and generation (Zhao et al., 2023). However, as the length of the context supported by these models increases (Zeng et al., 2024; Reid et al., 2024; Dubey et al., 2024; Ab- din et al., 2024; Young et al., 2024), and as these models are applied to longer and more complex tasks (Bai et al., 2024; Shaham et al., 2023; An et al., 2024; Zhang et al., 2024c), the cost of in- ference, particularly during the prefill phase, becomes a bottleneck Fu (2024). The computational complexity of full attention mechanisms, which grows quadratically with sequence length, leads to inefficiencies during long-sequence inference tasks. Considering the sparse nature of attention in LLMs (Deng et al., 2024), sparse attention mechanisms have been proposed to limit attention calculations to a selected subset of query-key pairs. Many ex- isting sparse attention methods concentrate on predefined sparse patterns, such as BigBird (Zaheer et al., 2020), LongLora (Chen et al., 2024), and SlidingWindowAttention (Jiang et al., 2023). How- ever, these approaches restrict the flexibility of sparse attention and often necessitate further training or fine-tuning of the model. There is also research proposing training-free sparse attention methods for large language mod

## 5. Direct Integration into Quillan-Ronin v5.4.0-ONI
- **Target Module**: `quillan_v5_4_oni.py`
- **Architectural Role**: Active Parameter Pipeline
- **Gradient Status**: 100% Active in autograd computational graph.
