# Academic Paper Dossier #023: ABRA: SCALINGDIFFUSIONIMAGETRAINING

- **Filename**: `2608.17286v1.pdf`
- **ArXiv Identifier**: `2608.17286v1`
- **Document Length**: 26 Pages (83,048 characters)
- **File Path**: [`C:\02_QUILLAN\10 - Formal Papers\Formal Papers\2608.17286v1.pdf`](file:///C:/02_QUILLAN/10 - Formal Papers/Formal Papers/2608.17286v1.pdf)

---

## 1. Abstract & Executive Summary
> Compute-optimal scaling laws guide the training of frontier language models yet remain largely unexplored for visual generation. We present a systematic scaling law study for text-to-image diffusion models using ABRA, a controlled family of flow-matching transformers trained across three orders of magnitude worth of compute (1019 to 1022 FLOPs), reaching significantly larger compute budgets than previous works. We demonstrate that diffusion models scale just as predictably as language models but require far more data to train optimally: compute optimality occurs at approximately 200 image tokens per parameter, ten times the Chinchilla compute-optimal prescription for LLMs. We show that unlike language models, diffusion models are robust to overtraining and that practitioners should err on the side of more data rather than a larger model. Finally, we show that this predictability extends beyond training loss to generative quality metrics, optimal CFG settings, representation quality, and even the shape of the training curves, which collapse onto a universal form. Figure 1:Text-to-image diffusion transformers follow predictable scaling laws and require ten times as much data as a comparably sized language model to reach compute optimality.

## 2. Core Problem & Motivation ('Why It Was Created')
Compute-optimal scaling laws describe how a fixed training compute budget C (in FLOPs) should be allocated between model size N (in parameters) and dataset size D (in training tokens) in order to minimize loss (Hestness et al., 2017; Amodei et al., 2016; Kaplan et al., 2020; Hoffmann et al., 2022). For large language models compute should be divided evenly between the model size and the dataset size according to N C 1/2 and D C 1/2, giving the Chinchilla rule of training LLMs to 20 tokens per parameter (TPP) (Hoffmann et al., 2022). Scaling analysis has become a standard industrial tool in language modeling and is used to guide the training of frontier models (Touvron et al., 2023; OpenAI, 2023; DeepSeek-AI, 2024; Grattafiori et al., 2024; Sun et al., 2024; Kimi Team, 2025; The Microsoft AI Team, 2026). 1 arXiv:2608.17286v1 [cs.LG] 18 Aug 2026 Preprint. The extension of scaling law analysis to the visual generation domain presents challenges not present in the language domain. Visual content has higher dimensionality (i.e., 2D images), resolution dependent token information density, separate training and generation paradigms, and much noisier training curves. The scaling of these models remains understudied despite the release of many large-scale visual generation studies. Existing work by Liang et al. (2024) fits iso-FLOP profiles below 1019 FLOPs and extrapolates to 1021 FLOPs. We use an order of magnitude more compute and a more controlled family which allows us to derive much more precise empirical fits. We significantly refine the scaling foundations for text-to-image models by conducting a systematic scaling study using ABRA, a controlled family of dense flow-matching transformers ranging from 60M to 2B parameters. We utilize P to fit scaling laws spanning three orders of magnitude (from 1019 to 1022 FLOPs). Our central findings indicate that compute optimality is achieved at roughly200 image tokens per parameter for text-to-image diffusion modeling, a 10 increase over the standard Chinchilla rule for LLMs. Furthermore, we find that diffusion model training is robust to overtraining, giving practitioners confidence to err on the side of training a smaller model with more data. We fit compute-optimal scaling laws to the evaluation loss and a range of common generative metrics, finding that all of these quantities scale predictably. We study the scaling of model representation quality using linear probing and find that representation quality also foll

## 3. Proposed Methodology & Mathematical Mechanics
We aim to characterize how text-to-image diffusion transformers should be scaled: for a given training budget, how many parameters should we use and how much data should we train on? This section describes our approach to answering these questions. 3.1 COMPUTEOPTIMALITY We follow the definition of compute optimality from Hoffmann et al. (2022): given a fixed training budget of C FLOPs, we seek the model size N and dataset size D minimizing a loss or

### Key Mathematical Formulations Extracted from Body:
- `$f= 8 FLUX V AE into a64 64 latent, which a parameter-free 2 2 fold$`
- `$t =v (xt, t$`
- `$L = b/ w (average, 0$`
- `$e = interpolation)$`
- `$C = 6 iTPP N2$`
- `$C = 6 iTPP N2$`

## 4. Empirical Evaluation & Benchmark Evidence
loss and a range of common generative metrics, finding that all of these quantities scale predictably. We study the scaling of model representation quality using linear probing and find that representation quality also follows predictable scaling trends. We ablate the effect of resolution on our scaling laws and we demonstrate that like LLMs, diffusion models exhibit scaling collapse. Our main contributions are: We present a comprehensive and controlled study of compute-optimal scaling for dense text-to-image transformers and derive anactionable 200 TPPrule for training diffusion models. We show that diffusion model training isrobust to overtrainingand precisely characterize this phenomenon. We extend our analysis to show that common generative metrics scale predictably, and that representation and generation capabilities scale heterogeneously. We show for the first time that diffusion models exhibitscaling collapse. The remainder of the paper is organized as follows. We review

## 5. Direct Integration into Quillan-Ronin v5.4.0-ONI
- **Target Module**: `quillan_v5_4_oni.py`
- **Architectural Role**: Active Parameter Pipeline
- **Gradient Status**: 100% Active in autograd computational graph.
