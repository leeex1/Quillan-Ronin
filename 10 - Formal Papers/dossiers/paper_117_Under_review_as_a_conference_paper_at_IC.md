# Academic Paper Dossier #117: Under review as a conference paper at ICLR 2017

- **Filename**: `Sparsely_Gated_MoE.pdf`
- **ArXiv Identifier**: `N/A`
- **Document Length**: 19 Pages (63,668 characters)
- **File Path**: [`C:\02_QUILLAN\10 - Formal Papers\Formal Papers\Sparsely_Gated_MoE.pdf`](file:///C:/02_QUILLAN/10 - Formal Papers/Formal Papers/Sparsely_Gated_MoE.pdf)

---

## 1. Abstract & Executive Summary
> The capacity of a neural network to absorb information is limited by its number of parameters. Conditional computation, where parts of the network are active on a per-example basis, has been proposed in theory as a way of dramatically increas- ing model capacity without a proportional increase in computation. In practice, however, there are signi cant algorithmic and performance challenges. In this work, we address these challenges and nally realize the promise of conditional computation, achieving greater than 1000x improvements in model capacity with only minor losses in computational ef ciency on modern GPU clusters. We in- troduce a Sparsely-Gated Mixture-of-Experts layer (MoE), consisting of up to thousands of feed-forward sub-networks. A trainable gating network determines a sparse combination of these experts to use for each example. We apply the MoE to the tasks of language modeling and machine translation, where model capacity is critical for absorbing the vast quantities of knowledge available in the training corpora. We present model architectures in which a MoE with up to 137 billion parameters is applied convolutionally between stacked LSTM layers. On large language modeling and machine translation benchmarks, these models achieve signi cantly better results than state-of-the-art at lower computational cost. 1 I NTRODUCTION AND RELATED WORK 1.1 C ONDITIONAL COMPUTATION Exploiting scale in both training data and model size has been central to the success of deep l

## 2. Core Problem & Motivation ('Why It Was Created')
more than two decades ago (Jacobs et al., 1991; Jordan & Jacobs, 1994), the mixture-of-experts approach has been the subject of much research. Different types of expert

## 3. Proposed Methodology & Mathematical Mechanics
s in which a MoE with up to 137 billion parameters is applied convolutionally between stacked LSTM layers. On large language modeling and machine translation benchmarks, these models achieve signi cantly better

### Key Mathematical Formulations Extracted from Body:
- `$i = 0, we need not$`
- `$i = (x Wg)i +StandardNormal () Softplus ((x Wnoise)i) (4)$`
- `$i = {vi ifvi is in the topk elements ofv$`
- `$j = Loadprimary(X)i Loadi(X (i))j$`
- `$j =Loadi(Xi)j , but this would not have a gradient with$`
- `$k = 4 for the$`

## 4. Empirical Evaluation & Benchmark Evidence
, these models achieve signi cantly better results than state-of-the-art at lower computational cost. 1 I NTRODUCTION AND

## 5. Direct Integration into Quillan-Ronin v5.4.0-ONI
- **Target Module**: `quillan_v5_4_oni.py`
- **Architectural Role**: Active Parameter Pipeline
- **Gradient Status**: 100% Active in autograd computational graph.
