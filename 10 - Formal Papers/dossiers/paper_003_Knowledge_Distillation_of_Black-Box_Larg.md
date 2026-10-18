# Academic Paper Dossier #003: Knowledge Distillation of Black-Box Large Language Models

- **Filename**: `2401.07013v2.pdf`
- **ArXiv Identifier**: `2401.07013v2`
- **Document Length**: 13 Pages (51,313 characters)
- **File Path**: [`C:\02_QUILLAN\10 - Formal Papers\Formal Papers\2401.07013v2.pdf`](file:///C:/02_QUILLAN/10 - Formal Papers/Formal Papers/2401.07013v2.pdf)

---

## 1. Abstract & Executive Summary
> Given the exceptional performance of propri- etary large language models (LLMs) like GPT- 4, recent research has increasingly focused on boosting the capabilities of smaller models through knowledge distillation (KD) from these powerful yet black-box teachers. While lever- aging the high-quality outputs of these teachers is advantageous, the inaccessibility of their in- ternal states often limits effective knowledge transfer. To overcome this limitation, we in- troduce Proxy-KD, a novel method that uses a proxy model to facilitate the efficient transfer of knowledge from black-box LLMs to smaller models. Our experiments show that Proxy-KD not only enhances the performance of KD from black-box teacher models but also surpasses traditional white-box KD techniques. This ap- proach presents a compelling new avenue for distilling knowledge from advanced LLMs.

## 2. Core Problem & Motivation ('Why It Was Created')
Recently, proprietary large language models (LLMs) like GPT-3.5 (OpenAI, 2022) and GPT- 4 (OpenAI, 2023) have demonstrated significant superiority over open-source counterparts such as the Llama series (Touvron et al., 2023a,b; MetaAI, 2024). However, their vast number of parameters leads to high inference costs, and they are only ac- cessible via API calls, offering limited customiza- tion and transparency. To address these challenges, recent efforts like Alpaca (Taori et al., 2023), Vi- cuna (Chiang et al., 2023), and Orca (Mukherjee et al., 2023) have focused on transferring the capa- bilities of proprietary LLMs to smaller open-source models through knowledge distillation (Chen et al., 2023; Hsieh et al., 2023; Ho et al., 2022). Knowledge distillation (KD) (Hinton et al., 2015) is a technique used to enhance the perfor- mance of a smaller student model by learning from a larger, more sophisticated teacher model. * Corresponding author. White-BoxKD Black-BoxKD Distribution Distribution Input ?Teacher Distribution Black-BoxModel Input Layer 1 Teacher Output Cross-Entropy Loss Student Response-BasedKDLossFeature-BasedKDLoss Layer nLayer n-1 Layer 2 Layer nLayer n-1 Layer 2Layer 1 Student Layer nLayer n-1 Layer 2Layer 1 Figure 1: Comparison of white-box knowledge distilla- tion (KD) and black-box knowledge distillation (KD). Depending on the level of access to the teacher model s internals, KD

## 3. Proposed Methodology & Mathematical Mechanics
that uses a proxy model to facilitate the efficient transfer of knowledge from black-box LLMs to smaller models. Our

### Key Mathematical Formulations Extracted from Body:
- `$L = E (x,y) D [ log s(y|x)] , (1)$`
- `$L = E (x,y) D [DKL( t(y|x)|| s(y|x))]$`
- `$L = E (x,y) Dp [ log p(y|x)]$`
- `$f = E (x,y) Dp, y (i)$`
- `$L = E (x,y) Ds [ log s(y|x)]$`
- `$L = E (x,y) Ds [DKL( p(y|x)|| s(y|x))]$`

## 4. Empirical Evaluation & Benchmark Evidence
show that Proxy-KD not only enhances the performance of KD from black-box teacher models but also surpasses traditional white-box KD techniques. This ap- proach presents a compelling new avenue for distilling knowledge from advanced LLMs. 1 Introduction Recently, proprietary large language models (LLMs) like GPT-3.5 (OpenAI, 2022) and GPT- 4 (OpenAI, 2023) have demonstrated significant superiority over open-source counterparts such as the Llama series (Touvron et al., 2023a,b; MetaAI, 2024). However, their vast number of parameters leads to high inference costs, and they are only ac- cessible via API calls, offering limited customiza- tion and transparency. To address these challenges, recent efforts like Alpaca (Taori et al., 2023), Vi- cuna (Chiang et al., 2023), and Orca (Mukherjee et al., 2023) have focused on transferring the capa- bilities of proprietary LLMs to smaller open-source models through knowledge distillation (Chen et al., 2023; Hsieh et al., 2023; Ho et al., 2022). Knowledge distillation (KD) (Hinton et al., 2015) is a technique used to enhance the perfor- mance of a smaller student model by learning from a larger, more sophisticated teacher model. * Corresponding author. White-BoxKD Black-BoxKD Distribution Distribution Input ?Teacher Distribution Black-BoxModel Input Layer 1 Teacher Output Cross-Entropy Loss Student Response-BasedKDLossFeature-BasedKDLoss Layer nLayer n-1 Layer 2 Layer nLayer n-1 Layer 2Layer 1 Student Layer nLayer n-1 Layer 2Layer 1 Figure 1: Comparison of white-box knowledge distilla- tion (KD) and black-box knowledge distillation (KD). Depending on the level of access to the teacher model s internals, KD methods can be categorized into two types: KD with black-box teachers and KD with white-box teachers. As illustrated in Fig- ure 1, white-box KD allows the student model to distill more intrinsic knowledge from the teacher by mimicing the teacher model s output distribution (Gu et al., 2023; Wen et al., 2023), hidden states (Jiao et al., 2020; Sun et al., 2019), and attention scores (Wang et al., 2021). Therefore, this method can only be applied when the teacher model s pa- rameters are accessible. On the other hand, black- box KD leverages the high-quality outputs from powerful proprietary LLMs to fine-tune the student model (Hsieh et al., 2023; Fu et al., 2023). Both white-box and black-box KD have their respective drawbacks. While white-box KD is hindered by the limited capacity of the teacher model, which often r

## 5. Direct Integration into Quillan-Ronin v5.4.0-ONI
- **Target Module**: `quillan_v5_4_oni.py (`distill_head`, `proxy_kd`)`
- **Architectural Role**: Self-reconstruction feature distillation preserving representation geometry.
- **Gradient Status**: 100% Active in autograd computational graph.
