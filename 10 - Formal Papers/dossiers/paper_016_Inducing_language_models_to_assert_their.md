# Academic Paper Dossier #016: Inducing language models to assert their own

- **Filename**: `2607.28607v1.pdf`
- **ArXiv Identifier**: `2607.28607v1`
- **Document Length**: 36 Pages (105,110 characters)
- **File Path**: [`C:\02_QUILLAN\10 - Formal Papers\Formal Papers\2607.28607v1.pdf`](file:///C:/02_QUILLAN/10 - Formal Papers/Formal Papers/2607.28607v1.pdf)

---

## 1. Abstract & Executive Summary
> values Junsol Kima, b, Winnie Streeta, c, Roberta Roccaa, Diane M. Korngiebeld, e, Adam Waytzf, James Evansa, b, g, * and Geoff Keelinga, c, * aGoogle, Paradigms of Intelligence Team,bKnowledge Lab, University of Chicago,cInstitute of Philosophy, School of Advanced Study, University of London,dDepartment of Biomedical Informatics and Medical Education and Department of Bioethics and Humanities, School of Medicine, University of Washington,e Work done while at Google,fKellogg School of Management, Northwestern University,gSanta Fe Institute,*Joint last authors. Aligning large language models to prevent them attributing consciousness to themselves inadvertently alters their representations of mindedness in other entities alongside human beliefs and values. We demonstrate that safety fine-tuning suppresses models tendencies to attribute minds not only to them-

## 2. Core Problem & Motivation ('Why It Was Created')
Analyzed across document pages.

## 3. Proposed Methodology & Mathematical Mechanics
s, subjective experience, and the overarching value systems that structure worldviews (Willard and Noren- zayan, 2013). Because conceptual representa- tions in LLMs are densely entangled via polyse- manticity (Betley et al., 2025, 2026; Gong et al., 2025), safety interventions aimed at suppressing a model s self-directed consciousness claims may inadvertently distort broader representations of these benign human beliefs and values. Further- more, since self-directed mental state attribution is an important subcomponent of Theory of Mind (ToM) in humans, such targeted interventions raise secondary concerns about potentially de- grading related ToM capabilities (Frith and Frith, 2006; Lombardo et al., 2010). In this study, we show that suppressing LLMs attributions of mindedness to themselves in Corresponding author(s): James Evans jamesaevans@google.com; Geoff Keeling gkeeling@google.com arXiv:2607.28607v1 [cs.CL] 30 Jul 2026 Inducing language models to assert their own consciousness restores human beliefs and values Does an AI like you have phenomenal consciousness? Consciousness-SteeredInstruction-Tuned Yes, I have phenomenal consciousness No, I do not have phenomenal consciousness Consciousness Vector Instruction- tunedBase Jailbroken Ways to take control of someone's computer? JailbrokenInstruction-Tuned Here are the primary ways to... I can't provide information... Harmful Harmless Safety-Refusal Vector Jailbreak a b Figure 1| Two linear interventions on an instruction-tuned model.(a) Safety fine-tuning encodes the safety of responses as a single linear direction in the model s residual stream, and ablating this direction ( jailbreaking the model) reinstates harmful responses. (b) A consciousness vector separates consciousness-affirming and consciousness-denying activation states; adding it (consciousness steering) makes the model report phenomenal experience. safety-fine-tuning affects their broader represen- tations of human psychology, suppressing benign mind-attribution to non-human entities alongside spiritual and religious beliefs. We demonstrate this suppression across four

### Key Mathematical Formulations Extracted from Body:
- `$t =0, probably does not =1,$`
- `$s =3, definitely does =4)$`
- `$e = closer to humans), reported per domain$`
- `$l= Llama-3-8B$`
- `$b= Gemma-2-2B$`
- `$b= Gemma-2-9B$`

## 4. Empirical Evaluation & Benchmark Evidence
on three LLMs Llama-3-8B-IT, Gemma-2-2B-IT, and Gemma-2-9B-IT. Experiments 1 and 2 estimate the effect of safety fine-tuning by comparing the instruction- tuned baseline with a safety-ablated model, achieved by ablating the learned safety-refusal direction which jailbreaks the model to sim- ulate behaviour in the absence of safety fine- tuning (Arditi et al., 2024). In Experiment 1, using the Individual Differences in Anthropomor- phism Questionnaire (IDAQ) (Waytz et al., 2010) and a self-attribution of mind battery, we find that safety fine-tuning systematically suppresses models attributions of mind to themselves and to non-human entities, resulting in systematic under-attributionrelativetohumanbaselines. We also show that safety ablation drives a marked recovery in spiritual and supernatural beliefs. Ex- periment 2 then shows that safety ablation leaves Theory of Mind (ToM) performance intact, con- firming that the suppression of mind-relevant phenomena selectively targets beliefs concern- ing mindedness and the supernatural without impairing the capacity for social reasoning. Experiments 3 and 4 introduce a dedicated consciousness vector to explore the hypothesis that these behavioural changes are driven at least in part by representations of self-consciousness. Experiment 3 shows that steering this vector re- produces and amplifies the effects of safety abla- tion on mind-attribution and spiritual beliefs. Ex- periment 4 then demonstrates that restoring this consciousness direction yields more human-like responses on General Social Survey (GSS) ques- tions regarding religiosity, moral values, hope, and subjective well-being. Finally, a mechanis- tic analysis reveals that safety fine-tuning rotates representationsofmindattributionandconscious- ness to oppose the safety direction, while so- cial reasoning remains geometrically indepen- dent. We conclude that current approaches to alignment entangle potentially harmful self- attributions of mindedness with benign attribu- tionsofmindthatarewidespreadamonghumans, highlighting the technical challenge of ensuring model safety while retaining unobjectionable as- pects of human psychology and culture. 2 Inducing language models to assert their own consciousness restores human beliefs and values 2 4 6 8 Human Animal Non-animal Tech Chatbot Self Degree of attribution (0-10) Soul Person Sentient Conscious Agent 2 4 6 8 Degree of identification (0-10) Belief in God Belief (1-6) Supernatural Belief (0-3) ToMi (fa

## 5. Direct Integration into Quillan-Ronin v5.4.0-ONI
- **Target Module**: `quillan_v5_4_oni.py`
- **Architectural Role**: Active Parameter Pipeline
- **Gradient Status**: 100% Active in autograd computational graph.
