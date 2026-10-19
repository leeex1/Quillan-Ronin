# Academic Paper Dossier #077: Mixture of Diverse Size Experts

- **Filename**: `MoDSE_2409.12210.pdf`
- **ArXiv Identifier**: `2409.12210`
- **Document Length**: 15 Pages (55,221 characters)
- **File Path**: [`C:\02_QUILLAN\10 - Formal Papers\Formal Papers\MoDSE_2409.12210.pdf`](file:///C:/02_QUILLAN/10 - Formal Papers/Formal Papers/MoDSE_2409.12210.pdf)

---

## 1. Abstract & Executive Summary
> The Sparsely-Activated Mixture-of-Experts (MoE) has gained increasing popularity for scaling up large language models (LLMs) with- out exploding computational costs. Despite its success, the current design faces a challenge where all experts have the same size, limiting the ability of tokens to choose the experts with the most appropriate size for generating the next token. In this paper, we propose the Mix- ture of Diverse Size Experts (MoDSE), a new MoE architecture with layers designed to have experts of different sizes. Our analysis of diffi- cult token generation tasks shows that experts of various sizes achieve better predictions, and the routing path of the experts tends to be stable after a training period. However, having experts of diverse sizes can lead to uneven workload distribution. To tackle this limitation, we in- troduce an expert-pair allocation strategy to evenly distribute the workload across multiple GPUs. Comprehensive evaluations across mul- tiple benchmarks demonstrate the effectiveness of MoDSE, as it outperforms existing MoEs by allocating the parameter budget to experts adaptively while maintaining the same total pa- rameter size and the number of experts.

## 2. Core Problem & Motivation ('Why It Was Created')
Large Language Models (LLMs) have demon- strated remarkable performance in a variety of NLP tasks and have become valuable assistants through a wide range of applications. The scaling law (Ka- plan et al., 2020) demonstrates that larger models exhibit superior performance. However, training larger models requires increased computational re- sources, posing a critical challenge. Mixture-of- Experts (MoE) (Fedus et al., 2022; Lepikhin et al., 2021) address this challenge by using sparse acti- vation to scale up the trainable parameters while maintaining high training and inference efficiency. Recent MoE-based

## 3. Proposed Methodology & Mathematical Mechanics
. Unlike conventional MoEs, which consist of experts of the same size, MoDSE has experts of different sizes. It assigns each token to the expert that best matches its prediction needs in terms of capa- arXiv:2409.12210v1 [cs.LG] 18 Sep 2024 bility, thereby enhancing the model s ability. Load Balance GPU nodes containing larger experts in MoDSE will have a heavier work- load. To address this issue, we propose the expert-pair allocation method in Section 3.2, which ensures that each GPU node carries an even distribution of parameters, thus maintain- ing load balance. Empirical Validation MoDSE outperforms conventional MoE with a lower loss value across a diverse set of benchmarks in the 700M 8 model setting, confirming the ef- fectiveness of our approach. We present the

### Key Mathematical Formulations Extracted from Body:
- `$i = (x Wg)i + RMSNorm(f ((x Wn)i))$`
- `$i =  vi vi topk(v)$`
- `$k = 2 h, , with k 1 n (8)$`
- `$a = N$`
- `$s = 1e 8,$`

## 4. Empirical Evaluation & Benchmark Evidence
s across mul- tiple benchmarks demonstrate the effectiveness of MoDSE, as it outperforms existing MoEs by allocating the parameter budget to experts adaptively while maintaining the same total pa- rameter size and the number of experts. 1 Introduction Large Language Models (LLMs) have demon- strated remarkable performance in a variety of NLP tasks and have become valuable assistants through a wide range of applications. The scaling law (Ka- plan et al., 2020) demonstrates that larger models exhibit superior performance. However, training larger models requires increased computational re- sources, posing a critical challenge. Mixture-of- Experts (MoE) (Fedus et al., 2022; Lepikhin et al., 2021) address this challenge by using sparse acti- vation to scale up the trainable parameters while maintaining high training and inference efficiency. Recent MoE-based architectures, such as Mixtral of Experts (Jiang et al., 2024), DeepSeekMoE (Dai et al., 2024), and OpenMoE (Xue et al., 2024) have shown superior performance in various tasks. Specifically, Dai et al. (2024) discuss two main issues in the design of the MoE Feed-Forward Net- works (FFNs) architecture: Knowledge Hybridity, where each expert covers diverse knowledge due to the limited number of experts, and Knowledge Re- dundancy, where multiple experts share common knowledge. To address these issues, they propose Fine-Grained Expert Segmentation by splitting the FFN intermediate hidden dimension and Shared Expert Isolation by isolating certain experts to be always activated as shared experts. Additionally, Zhao et al. (2024) introduce Hypernetworks and HyperExperts modules to capture the cross-expert and cross-layer knowledge. However, almost all existing MoE architectures consist of experts with identical structures and sizes. This homogeneous architecture becomes a signifi- cant bottleneck when generating tokens with vary- ing difficulty; some tokens are easier to predict, while others are more challenging. To deal with the varied difficulty, we propose the Diverse Size Experts structure for each FFN layer, where each expert has a different parameter size to handle gen- erating tasks of varying difficulty. Note that we find a similar recent work called Heterogeneous the Mixture of Experts (Wang et al., 2024), which shares a similar motivation and utilizes parameter penalty loss and router entropy loss to control the size and number of activated experts. Our contributions are summarized as follows: Diverse

## 5. Direct Integration into Quillan-Ronin v5.4.0-ONI
- **Target Module**: `paper_61_65_efficient_moe_pack.py (`MixtureOfDepths`), quillan_v5_4_oni.py`
- **Architectural Role**: Dynamic per-token depth allocation via Straight-Through Estimators.
- **Gradient Status**: 100% Active in autograd computational graph.
