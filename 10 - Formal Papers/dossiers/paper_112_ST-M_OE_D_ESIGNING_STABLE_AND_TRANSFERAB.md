# Academic Paper Dossier #112: ST-M OE: D ESIGNING STABLE AND TRANSFERABLE

- **Filename**: `ST-MoE_Stable_Sparse_Experts.pdf`
- **ArXiv Identifier**: `2202.08906v2`
- **Document Length**: 38 Pages (119,369 characters)
- **File Path**: [`C:\02_QUILLAN\10 - Formal Papers\Formal Papers\ST-MoE_Stable_Sparse_Experts.pdf`](file:///C:/02_QUILLAN/10 - Formal Papers/Formal Papers/ST-MoE_Stable_Sparse_Experts.pdf)

---

## 1. Abstract & Executive Summary
> Scale has opened new frontiers in natural language processing but at a high cost. In response, Mixture-of-Experts (MoE) and Switch Transformers have been pro- posed as an energy ef cient path to even larger and more capable language models. But advancing the state-of-the-art across a broad set of natural language tasks has been hindered by training instabilities and uncertain quality during ne-tuning. Our work focuses on these issues and acts as a design guide. We conclude by scaling a sparse model to 269B parameters, with a computational cost comparable to a 32B dense encoder-decoder Transformer (Stable and Transferable Mixture- of-Experts or ST-MoE-32B). For the rst time, a sparse model achieves state- of-the-art performance in transfer learning, across a diverse set of tasks includ- ing reasoning (SuperGLUE, ARC Easy, ARC Challenge), summarization (XSum, CNN-DM), closed book question answering (WebQA, Natural Questions), and adversarially constructed tasks (Winogrande, ANLI R3). 1 Equal contribution. Correspondence to{barretzoph,liamfedus}@google.com. Work was done while at Google. 1Code for our models is available at https://github.com/tensorflow/mesh/blob/master/ mesh_tensorflow/transformer/moe.py 1 arXiv:2202.08906v2 [cs.CL] 29 Apr 2022 CONTENTS

## 2. Core Problem & Motivation ('Why It Was Created')
3 2 Background 3 3 Stabilizing Training of Sparse Models 5 3.1 Stability and Quality Tradeoffs when Removing Multiplicative Interactions . . . . 6 3.2 Stability and Quality Tradeoffs when Adding Noise . . . . . . . . . . . . . . . . . 6 3.3 Stability and Quality Tradeoffs when Constraining Activations and Gradients . . . 7 3.4 Selecting a Precision Format: Trading Ef ciency and Stability . . . . . . . . . . . 8 4 Fine-Tuning Performance of Sparse Models 9 4.1 Hypothesis: A Generalization Problem . . . . . . . . . . . . . . . . . . . . . . . . 9 4.2 Fine-Tuning a Subset of Model Parameters to Improve Generalization . . . . . . . 11 4.3 Sparse and Dense Models Require Different Fine-Tuning Protocols . . . . . . . . . 11 4.4 Sparse Models Are Robust to Dropped Tokens During Fine-Tuning . . . . . . . . . 12 4.5 Inserting Sentinels Tokens During Fine-Tuning . . . . . . . . . . . . . . . . . . . 13 5 Designing Sparse Models 13 5.1 Setting the Number of Experts . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13 5.2 Choosing the Capacity Factor and Routing Algorithm . . . . . . . . . . . . . . . . 14 6 Experimental Results 16 6.1 ST-MoE-L . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16 6.2 ST-MoE-32B . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16 7 Tracing Tokens Through the Model 19 7.1 Encoder Experts Exhibit Specialization . . . . . . . . . . . . . . . . . . . . . . . 19 7.2 Decoder Experts Lack Specialization . . . . . . . . . . . . . . . . . . . . . . . . . 19 7.3 Multilingual Experts Specialize, But Not by Language . . . . . . . . . . . . . . . 21 8 Related Work 21 9 Discussion 22 10 Conclusion 24 A Token Load Balance Description 31 B Router Z-Loss Training Dynamics 31 C Improved Architectural Modi cations 32 D Batch Prioritized Routing for Lower Capacity Factors 33 E Pre-Training Dataset Details 34 F Full Fine-tuning Sensitivity Data 35 G Optimally Setting the Routing Threshold 36 H Mesh Layout for Data, Model and Expert Parallelism with Few Experts 36 I Note on Communication Costs for Distributed Models 37 J Negative Results 38 2 1 I NTRODUCTION Sparse expert neural networks showcase the advantage of sheer scale and offer an ef cient alternative to the static neural network

## 3. Proposed Methodology & Mathematical Mechanics
s stabilize sparse models, but at the expense of worse quality. 2. The router z-loss stabilizes models without quality degradation. 3. Transformer modi cations with more multiplicative components (GEGLU, RMS nor- malization) worsen stability, but boost quality. Designing a large-scale stability study. We design a large-scale stability study of sparse models FLOP-matched to the T5-XL version (Raffel et al., 2019) pre-trained on the multilingual corpus mC4 (Xue et al., 2020). Each sparse model has 32 experts and we introduce a sparse MoE layer for 3See Fedus et al. (2021) for a graphical illustration of how the capacity factor works. 5 every fourth FFN. The train capacity factor is 1.25 and the eval capacity factor is 2.0. See Table 11 for a more detailed description of models used throughout this paper. For each stability technique, we record the fraction that are stable, the mean quality (negative log perplexity on English), and the standard deviation over seeds. The primary issue in constructing this study is that small models are rarely unstable but large un- stable models are too costly to run for suf cient steps and seeds. We found a sparse model FLOP- matched to T5-XL to be good object of study because it was unstable roughly 1/3 of the runs, but was still relatively cheap to train. Furthermore, we run our instability

### Key Mathematical Formulations Extracted from Body:
- `$i=1 of N experts$`
- `$t =LCE +cBLB +czLZ (6)$`
- `$s=65536) as in Raffel et al$`
- `$i = 1 toN and a batchB withT tokens$`
- `$i = gatei(x) n$`
- `$i=1 gatei(x)$`

## 4. Empirical Evaluation & Benchmark Evidence
16 6.1 ST-MoE-L . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16 6.2 ST-MoE-32B . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16 7 Tracing Tokens Through the Model 19 7.1 Encoder Experts Exhibit Specialization . . . . . . . . . . . . . . . . . . . . . . . 19 7.2 Decoder Experts Lack Specialization . . . . . . . . . . . . . . . . . . . . . . . . . 19 7.3 Multilingual Experts Specialize, But Not by Language . . . . . . . . . . . . . . . 21 8

## 5. Direct Integration into Quillan-Ronin v5.4.0-ONI
- **Target Module**: `quillan_v5_4_oni.py`
- **Architectural Role**: Active Parameter Pipeline
- **Gradient Status**: 100% Active in autograd computational graph.
