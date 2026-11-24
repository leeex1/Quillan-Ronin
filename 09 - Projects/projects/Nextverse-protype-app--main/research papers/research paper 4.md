Seed-Thinking-v1.5: Advancing Superb Reasoning
Models with Reinforcement Learning
ByteDance Seed
Full author list in Contributions
Abstract
We introduce Seed-Thinking-v1.5, capable of reasoning through thinking before responding,
resulting in improved performance on a wide range of benchmarks. Seed-Thinking-v1.5 achieves 86. 7
on AIME 2024 , 55. 0 on Codeforces and 77. 3 on GPQA, demonstrating excellent reasoning abilities
in STEM and coding. Beyond reasoning tasks, the method demonstrates notable generalization
across diverse domains. For instance, it surpasses DeepSeek R1 by 8% in win rate on non-reasoning
tasks, indicating its broader applicability. Compared to other state-of-the-art reasoning models,
Seed-Thinking-v1.5 is a Mixture-of-Experts (MoE) model with a relatively small size, featuring
20B activated and 200B total parameters. As part of our effort to assess generalized reasoning, we
develop two internal benchmarks, BeyondAIME and Codeforces, both of which will be publicly
released to support future research.

Date: April 10, 2025

Figure 1 Benchmark performance on reasoning tasks
1 Introduction
Driven by large-scale reinforcement learning on large language models, reasoning models have seen significant
advancements. Notably, OpenAI’s o1 series [1], DeepSeek’s R1 [2], Google’s Gemini 2.5 [3], and Anthropic’s
Claude 3.7 [ 4 ] have emerged as state-of-the-art models, each making substantial progress in logical reasoning,
mathematical problem-solving, and code generation. These advancements underscore a shift toward more
structured, efficient and scalable reasoning models, with ongoing research focusing on training efficiency, long
chain-of-thought, and large-scale reinforcement learning.

In this work, we present a new reasoning model, called Seed-Thinking-v1.5. This model has achieved strong
performance in both reasoning and non-reasoning tasks.

Mathematical Reasoning: For math competition, Seed-Thinking-v1.5 achieves 86. 7 on AIME 2024, match-
ing the performance of o3-mini-high and significantly outperforming o1 and DeepSeek R1, demonstrating
competitive strength. Since AIME 2024 no longer provides sufficient discrimination, we construct a more
challenging evaluation set named BeyondAIME. All problems in BeyondAIME are newly curated by
human experts and designed to minimize the chance of being solved through memorization or guessing.
While Seed-Thinking-v1.5 surpasses both o1 and R1, there remains a performance gap compared to o
and Gemini pro 2.5. This also further demonstrates the discriminative power of the new evaluation set.

Competitive Programming: For the evaluation of competitive programming, we adopt Codeforces as our
benchmark. Unlike some prior works that rely on Elo Scores, which contains estimation and are not
directly comparable, we adopt a concrete evaluation protocol based on the most recent 12 Codeforces
contests. Specifically, we report pass@1 and pass@8 metrics, where pass@k indicates whether the
model solves the problem within k attempts, i.e., selecting the best result from k generated submissions.
We choose to report pass@8 since it provides more stable results and aligns more closely with actual
user submission patterns. Seed-Thinking-v1.5 outperforms DeepSeek R1 on both metrics, though a
performance gap remains compared to o3. The evaluation set will be made publicly available in a future
release.

Science: Seed-Thinking-v1.5 reaches a score of 77.3 on GPQA, close to o3-level performance. Importantly,
this gain is largely attributed to improved generalization from mathematical training, rather than an
increase in domain-specific science data.

Non-reasoning Tasks: For non-reasoning tasks, Seed-Thinking-v1.5 is evaluated using a test set designed
to replicate real-world user needs. Through human evaluations conducted against DeepSeek R1 across
diverse scenarios, Seed-Thinking-v1.5 demonstrates significant advancements: it attains an 8.0% overall
rise in users’ positive feedback, thereby highlighting its augmented ability to manage intricate user
scenarios.

There are three key points in the development of high-quality reasoning models: training data, RL algorithm,
and RL infrastructure. We have devoted considerable effort to these three areas, and we will discuss them in
detail.

DataFor SFT training, unlike conventional post-training data, reasoning models rely on chain-of-thought
data, which explicitly outlines the step-by-step reasoning process. Our preliminary experiments showed
that too much non-CoT SFT data can significantly reduce the model’s ability to explore. For RL
training, we incorporate four categories of data: STEM problems, code-related tasks, logic reasoning and
non-reasoning data like creative writing and dialogue. Among these, the logic reasoning data contributes
to performance improvements on the ARC-AGI benchmark significantly. The math data exhibits strong
generalization capabilities and can lead to broad performance improvements across tasks.

RL AlgorithmRL training of reasoning models is highly unstable and often crashes, especially for models
without SFT. Sometimes, the score difference between two runs can be as high as 10 points. The stable
training of RL systems is crucial for the success of reasoning models. To address these long-standing
issues, we have pioneered VAPO[ 5 ] and DAPO[ 6 ]—two distinct frameworks tailored for actor-critic and
policy-gradient RL paradigms, respectively. VAPO now stands as the state-of-the-art (SOTA) solution
in actor-critic methods, while DAPO establishes a new SOTA result for policy-gradient approaches

without critic models. By targeting the core instability issues in RL training, both methods deliver
robust and consistent training trajectories, effectively enabling reliable optimization of reasoning models.
RL InfrastructureThe complexity of Large Language Models (LLM) based reinforcement learning systems
demands robust infrastructure to ensure scalability, reproducibility, and computational efficiency.
To handle heterogeneous workloads, we decouple streaming rollout architecture that asynchronously
processes partial trajectory generations through prioritized sample pools, achieving 3×faster iteration
cycles than synchronous frameworks. The system also supports mixed-precision training with automatic
fault recovery, critical for maintaining stability during large-scale RL runs.
2 Data
2.1 RL Training Data
Our RL training data consists of two main parts: verifiable problems with definitive answers and non-verifiable
problems without definitive answers. The model’s reasoning ability primarily comes from the first part and
can be generalized to the second part.
2.1.1 Verifiable Problems
The Verifiable problems primarily comprise STEM questions paired with answers, coding problems equipped
with unit tests, and logic reasonings that are amenable to automated verification.
STEM Data
Our dataset consists of several hundred thousand high-quality, competition-grade problems spanning math-
ematics, physics, and chemistry, with mathematics comprising the majority (over 80%). These problems
are drawn from a mix of open-source datasets, public competitions (both domestic and international), and
proprietary collections.
For data cleaning, we first eliminate questions with incomplete statements, inconsistent notation, or unclear
requirements. For the remaining questions, we use our model (Doubao-Pro 1.5) to generate multiple responses.
Problems for which the model achieved a woN score (worst of N) of 1 are deemed too simple and removed.
Finally, some questions may have an inaccurate reference answer. We use SOTA reasoning models to
generate multiple candidate responses for each question. If the model’s answers were inconsistent with the
reference answer, but the model’s outputs showed high internal consistency, or involved only a very small
number of reasoning tokens, we consider the reference answer to be incorrect. Human experts then conduct
manual verification on these questions to ensure that the reference answers are correct. We also apply
data augmentation to make the data more suitable for learning and evaluation. Specifically, we convert
multiple-choice questions into fill-in-the-blank or short-answer formats to eliminate the possibility of guessing
and to better assess reasoning ability. And we modify certain math problems to ensure that the answers are
integers whenever possible.
After data cleaning and augmentation, we finally obtain a training set of 100k STEM problems. During
training, we use model-based Seed-Verifier to evaluate response correctness, which is introduced in 3.1.
Code Data
For coding problems, we prioritize the source of high-quality and challenging algorithmic tasks, primarily
drawn from esteemed competitive programming contests.

We filter data to ensure that each problem includes a comprehensive specification: a clear problem description,
a set of unit tests, and a checker script. Unit tests validate the functional correctness of solutions, while the
checker script enforces additional constraints such as output formatting and edge cases. We also perform
difficulty filtering, ensuring that problems possess an appropriate level of complexity and applicability to
real-world algorithmic reasoning.
For evaluation, the most accurate form is to submit the generated code to the official platforms. However,
during reinforcement learning, real-time submission isn’t feasible. Thus, we developed an off-line evaluation

set for efficient local validation. Our observations indicate a strong correlation between offline evaluation
results and official verdicts. All training and evaluation problems are integrated into an in-house code sandbox
environment, enabling direct execution and assessment of model-generated code. We ensure the sandbox’s
stability and high throughput to deliver consistent and accurate feedback during the RL training process.
Logical Puzzle Data
For the logic reasoning data, we gather 22 commonly studied tasks, such as 24-point, mazes, Sudoku, etc.
For each task, we construct a data generator and an answer verifier. The data generator can automatically
produce a large amount of training and evaluation data. Moreover, for many of the tasks, we can configure
the difficulty of the generated problems. During the training process, we gradually adjust the difficulty of the
training data based on the model’s performance on certain tasks. The answer verifier rigorously evaluates the
generation correctness and can be seamlessly integrated into RL pipelines as reward functions. We generate
about 10k puzzle problems for RL training.
2.1.2 Non-verifiable Problems
Non-verifiable problems mainly encompass non-reasoning tasks requiring quality assessment based on human
preferences, involving tasks like creative writing, translation, knowledge QA, role-playing, and so on. The
prompts are originated from RL training data for Doubao-1.5 Pro [ 7 ]. The dataset has sufficient coverage
across diverse domains.
We discard data with low sample score variance and low difficulty. To be specific, we use the SFT model to
generate multiple candidates for each prompt and then score them using a reward model. Prompts with low
score variances are removed as they exhibit limited sampling diversity and minimal potential for improvement.
Prompts are also removed where the reward score improvement surpasses a certain threshold during the
Doubao 1.5 Pro RL training process [ 8 ]. This is because such data may be overly simplistic or already
abundantly represented in the dataset. Offline experiments show that overoptimizing such samples leads to
premature collapse of the model’s exploration space and diminish the performance.
For these non-verifiable data, we employ a pairwise rewarding method for scoring and RL training. By
comparing the relative quality of two samples, this approach aids the model in better understanding user
preferences, enhancing the quality and diversity of generated results. The detail of the reward model is
introduced in 3.2.

2.2 Advanced Math Benchmark
The current reasoning models usually use AIME as the go-to benchmark to evaluate mathematical reasoning
abilities. However, with only 30 problems released annually, its limited size can lead to high-variance evaluation
results, making it challenging to effectively differentiate between state-of-the-art reasoning models. To better
evaluate models’ capabilities in mathematical reasoning, we construct a new benchmark dataset: BeyondAIME.
Specifically, we collaborate with mathematics specialists to develop original problems informed by established
competition formats. We systematically adapt existing competition questions through structural modifications
and scenario reconfigurations, ensuring no direct duplication occurs. Furthermore, we ensure that the answers
are never trivial values—such as numbers explicitly mentioned in the problem statement—to reduce the
chance of models guessing the correct answer without proper reasoning.
Through this rigorous filtering and curation process, we compile a final set of 100 problems, each with a
difficulty level equal to or greater than that of the hardest questions in AIME. Similar to AIME, all answers
are guaranteed to be integers (without being restricted to a specific numerical range), which simplifies and
stabilizes the evaluation process.
3 Reward Modeling
As a crucial component in RL, reward modeling defines the objective or goal that the policy is trying to
achieve. Thus, a well-designed reward mechanism is essential to provide precise and reliable reward signals for

Verifier-type Training examples (approximate) Human labeled testset
Seed-Verifier >98% 82 .7%
Seed-Thinking-Verifier >99% 99 .3%
Table 1 Accuracy of two verifier-types. Specifically, the accuracy on the training set is derived from the training
statistics. Additionally, we manually annotated 456 samples to form the test set, which are specifically selected from
cases that the Seed-Verifier can not handle stably.
model responses during the training stage. For verifiable and non-verifiable problems, we employ distinct
reward modeling methodologies.
3.1 Reward Modeling for Verifiable Problems
With proper principles and thought trajectories, we utilize LLMs to judge a wide array of verifiable questions
across diverse scenarios. This approach yields a more generalized solution that surpasses the limitations of
rule-based reward systems.

We have designed two progressive reward modeling solutions, Seed-Verifier and Seed-Thinking-Verifier :

Seed-Verifier is based on a set of meticulously crafted principles written by humans. It leverages the
powerful foundational capabilities of LLMs to evaluate a triplet consisting of the question, reference
answer, and model-generated answer. If the reference answer and model-generated answer are essentially
equivalent, it returns “YES”; otherwise, it returns “NO”. The equivalence here is not a literal exact match
but rather a deeper assessment based on computational rules and mathematical principles that prove
the two answers convey the same mathematical meaning. This approach ensures that the reward signal
accurately reflects whether the model’s response is correct in essence, even if the wording differs.
Seed-Thinking-Verifier is inspired by the human judgment process, which generates conclusive judgments
through meticulous thinking and in-depth analysis. To achieve this, we trained a verifier that provides a
detailed reasoning path for its evaluations. Specifically, we treated this as a verifiable task and optimized
it alongside other mathematical reasoning tasks. This verifier can dissect the similarities and differences
between the reference and model-generated answers, offering precise and nuanced judgment results.
The Seed-Thinking-Verifier significantly alleviates three major issues associated with the Seed-Verifier:
Reward Hacking: Non-thinking models may exploit loopholes to receive rewards without truly under-
standing the problem. The detailed reasoning process in Seed-Thinking-Verifier makes such hacking
more difficult.
Uncertainty in Predictions: In cases where the reference and model-generated answers are essentially
equivalent, which may differ in format, e.g., 219 vs 524288, the Seed-Verifier might sometimes return
“YES” and other times “NO”. The Seed-Thinking-Verifier provides consistent results by thoroughly
analyzing the reasoning behind the answers.
Failure on Corner Cases: There are certain edge cases that the Seed-Verifier struggles to handle effectively.
The ability of Seed-Thinking-Verifier to provide detailed reasoning allows it to better address these
complex scenarios.
Table 1 presents the performance of the above two verifiers. The results indicate that the Seed-Verifier
struggles to effectively handle some particular cases, whereas the Seed-Thinking-Verifier demonstrates a
remarkable ability to provide accurate judgments. While the thinking process of the latter does consume a
significant amount of GPU resources, we believe that the precise and robust reward results it generates are
crucial for endowing the policy with strong reasoning capabilities.
3.2 Reward Modeling for Non-verifiable Problems
For non-verifiable problems, we train a reward model for RL training. The reward model training data is
consistent with the human preference data utilized in Doubao 1.5 Pro [ 7 ], primarily encompassing categories
such as creative writing and summarization.
To enhance the effectiveness of reward model, we adopt the pairwise generative reward model mentioned in [ 9 ],
which evaluates the superiority of two responses and use the probability of “YES” or “NO” as the final reward
score. This approach enables the model to directly compare differences between responses during scoring,
thereby avoiding excessive focus on irrelevant details. Experimental results demonstrate that this reward
modeling method improves the stability of RL training, particularly in the mixed training scenarios involving
both non-verifiable and verifiable problems, by minimizing conflicts between the two different types of reward
modeling paradigms. This improvement may be attributed to the pairwise generative reward model’s inherent
advantage in mitigating outlier score generation compared to conventional reward models, therefore avoiding
significant discrepancies in score distributions with the verifier.
4 Approach
4.1 Supervised Fine-Tuning
Our training process starts with supervised fine-tuning (SFT). The SFT phase sets a solid foundation
for the subsequent reinforcement learning stage. Compared to initiating RL from a base model, the SFT
model produces more readable outputs, exhibits fewer instances of hallucination, and demonstrates reduced
harmfulness. We curate an SFT data comprising 400k training instance, including 300k verifiable problems and
100k non-verifiable problems. Verifiable prompts are randomly sampled from RL training set. Non-verifiable
data are sourced from the SFT data used for Doubao-Pro 1.5 [ 7 ], covering areas such as creative writing,
knowledge-based QA, safety, and function calling.
To generate high-quality responses with long CoT, we employ an iterative workflow that integrates model
synthesis, human annotation, and rejection sampling. Initially, human experts apply prompt engineering
techniques or engage in interactive dialogues with an internal model to produce responses with various
reasoning patterns. After accumulating tens of high-quality cold-start samples, we can train a reasoning model
with long CoT as a more capable assistant. Then we perform rejection sampling on this reasoning model using
Seed-Verifier. While this workflow is primarily applied to mathematical data, we observe it can generalize
well to other domains, such as coding, logic puzzle and even creative writing. Thus, for other domains, we
also conduct a cold start process followed by rejection sampling to produce detailed reasoning trajectories.
During training, each instance is truncated to 32,000 tokens. We fine-tune the base model for two epochs
using the above data. We use a cosine decay learning rate scheduling that the peak lr is 2 × 10 −^5 and decays
to 2 × 10 −^6 gradually.

4.2 Reinforcement Learning
We have developed a unified reinforcement learning framework that seamlessly fuses data from a broad range
of domains. This integration incorporates three data categories:

Verifiable data, which obtains feedback from a verifier. This type of data allows for direct validation of
the model’s outputs against known criteria.
General data, scored by a reward model. The reward model assigns scores based on how well the model’s
responses align with human preferences.
A specific class of data that combines scores from both the verifier and the reward model. This hybrid
data type leverages the strengths of both verification and reward-based evaluation.
In the context of long-CoT RLHF, we encounter several challenges such as value model bias and the sparsity
of reward signals. To address these issues, we draw on key techniques from our prior work [5, 6, 10]:
Value-Pretraining : We sample responses from a fixed policy, such asπsft, and update the value model
using the Monte-Carlo return. This process ensures that the initialized value model is fully aligned with
our policyπsft. Maintaining this alignment has been proven to be crucial for preserving the model’s
CoT pattern, enabling the model to generate coherent and logical CoT.
Decoupled-GAE : By employing different Generalized Advantage Estimation (GAE) parameters, such as
λvalue= 1. 0 andλpolicy= 0. 95 , we allow the value model to update in an unbiased manner. Meanwhile,
the policy can independently balance its own bias and variance. This decoupling enables more efficient
and stable training of the model.
Length-adaptive GAE : We setλpolicy= 1−αl^1 , whereαis a hyper-parameter andlis the response
length. This approach ensures a more uniform distribution of Temporal Difference (TD) errors across
both short and long sequences. As a result, the model can handle sequences of varying lengths more
effectively during training.
Dynamic Sampling : We employ dynamic sampling and filter out prompts with accuracy scores equal to
1 or 0, retaining only those in the batch that exhibit effective gradients. This process helps prevent the
dampening of gradient signals during model training.
Clip-Higher : In the Proximal Policy Optimization (PPO) algorithm, we decouple the upper and lower
clip bounds as follows:
LCLIP(θ) =Eˆt
h
min

rt(θ)Aˆt,clip(rt(θ), 1 −εlow,1 +εhigh)Aˆt
i
(1)
By increasing the value ofεhigh, we create more room for the increase of low-probability tokens. This
encourages the model to explore a wider range of possible responses, enhancing its ability to discover
novel and effective solutions.
Token-level Loss : Instead of defining the policy loss over entire responses, we define it over all tokens.
This approach addresses the imbalance in the token-level contribution to the final loss, ensuring that
each token’s impact on the training process is appropriately accounted for.
Positive Example LM Loss : This loss function is designed to boost the utilization efficiency of positive
samples during the RL training process. We add a language model loss with a coefficientμfor positive
examples:
L(θ) =LPPO(θ) +μ∗LNLL(θ) (2)
This additional loss term helps the model to better learn from positive examples, improving its overall
performance.
When merging data from different domains and incorporating diverse scoring mechanisms, we face the challenge
of interference between different data domains. This interference can arise from disparities in difficulty levels,
the risk of reward-hacking, and other underlying factors. These issues make it extremely difficult to achieve
uniform and simultaneous improvements across all capabilities of the model. To counteract this, we introduce
Online Data Distribution Adaptation. This method transforms the stationary prompt distribution during
reinforcement learning into an adaptive distribution that better caters to the model’s requirements during
training. By doing so, we minimize the negative impact of data interference and ensure a more balanced
improvement across different abilities. As a result, the model can enhance its performance more consistently
across a wide array of tasks.

5 Infrastructures
5.1 Framework
The training framework is built using HybridFlow [ 11 ] programming abstraction. The whole training workload
runs on top of a Ray [ 12 ] cluster. The dataloader and RL algorithm is implemented in a single process Ray
Actor (single controller). The model training and response generation (rollout) is implemented in a Ray

Worker Group. The Ray Worker Group exposes a set of APIs (e.g., generate_response/train_batch, etc,.),
which runs heavy training/generation workload via SPMD (single program, multiple data) inside the Worker
Group. The single controller invokes various APIs exposed by the Ray Worker Group to construct the training
flow. HybridFlow programming abstraction enables fast prototyping of RL algorithm ideas without bothering
with complex distributed systems.
Seed-Thinking-v1.5 is trained through hybrid engine architecture [ 13 ], where all the models are co-located.
This prevents the idle time of the GPUs when switching between training and generation. During Long-CoT
generation, we observe severe straggler phenomenon caused by the large difference of the response length
between various prompts. This causes massive GPU idle time during generation. To mitigate the straggler
of long-tail response generation, we propose SRS (Streaming Rollout System) - a resource-aware scheduling
framework that strategically deploys standalone streaming-compute units to transform system constraints
frommemory-boundtocompute-bound.

5.2 Streaming Rollout System
The SRS architecture introducesstreaming rollout to decouple model evolution from runtime execution,
enabling dynamic adjustment of on/off-policy sample ratios through parametricα:
Define the completion ratio (α∈[0,1]) as the proportion of samples generated on-policy using the latest
model version
Allocate the remaining non-complete segment (1-α) to off-policy rollouts from versioned model snapshots,
seamlessly integrated through asynchronous continuation of partial generations on the standalone
resources.
In addition, we also implement dynamic precision scheduling during environment interaction phases, which
deploys FP8 policy networks via post-training quantization with error-compensated range scaling. To address
token imbalance in MoE systems, we implement a three-tiered parallel architecture combining TP (tensor
parallelism) for layer-wise computation, EP (expert parallelism) with dynamic expert assignment, and SP
(sequence parallelism) for context chunking. Our kernel auto-tuner dynamically selects optimal CUDA kernel
configurations based on real-time load monitoring.
5.3 Training System
To efficiently train the Seed-Thinking-v1.5 model at scale, we design a hybrid distributed training framework
that integrates advanced parallelism strategies, dynamic workload balancing, and memory optimizations.
Below we detail the core technical innovations driving the system’s efficiency and scalability.
Parallelism mechanisms. We compose TP (tensor parallelism)/EP (expert parallelism)/CP (context
parallelism) with Fully Sharded Data Parallelism (FSDP) to train Seed-Thinking-v1.5. Specifically, we
applied TP/CP for attention layers, and EP for MoE layers.
Sequence length balancing. The effective sequence length can be imbalanced across DP ranks, leading
to imbalanced computation workload and low training efficiency. To address this challenge, we leverage
KARP [ 14 ] algorithm that rearranges the input sequences within one mini-batch to make them balance
among micro-batches.
Memory optimization. We adopt layer-wise recomputation [ 15 ], activation offload and optimizer offload
to support training of larger micro-batches to overlap the communication overhead caused by FSDP.
Auto parallelism. To enable optimal system performance, we develop an automatic tuning system,
referred to as AutoTuner. Specifically, AutoTuner models the memory usage following a profile-based
solution [ 16 ]. Then, it estimates the performance and memory usage of various configurations to obtain
the optimal configuration.
Checkpoint. We employ ByteCheckpoint [ 17 ] to support checkpoint resume from different distributed
configurations with minimal overhead. This enables users to elastically train the tasks to improve cluster
efficiency.
Benchmark Seed-Thinking-v1.5 DeepSeek R1 OpenAI o3-mini Grok 3 Beta Gemini 2.5 pro
Mathematics
AIME 2025 74.0% 65.0% 86.5% 77.3% 86.7%
AIME 2024 86.7% 79.8% 87.3 % 83.9% 92.0%
Beyond AIME 48.0% 42.4% 63.6 % - 58.8%
Science
GPQA diamond 77.3% 71.5% 79.7% 80.2% 84.0%
SuperGPQA 62.1% 60.5% 52.2% 62.8% 65.3%
MMLU-PRO 87.0% 85.6% 82.4% 84.6% 86.3%
Code
Codeforces avg@8 36.3% 32.0% 50.9% - 40.3%
Codeforces pass@8 55.0% 45.0% 67.5% - 56.3%
LiveCodeBench v5 64.9% 64.3% 74.1% 70.6% 70.4%
Aider Polyglot 54.2% 56.9% 68.6% - 74.0%
Agentic Coding
SWE-bench verified 47.0% 49.2% 49.3% - 63.8%
SWE-bench verified* 47.0% 46.2% 44.5% - 63.8%
Logic reasoning
ARC-AGI 39.9% 18.3% 25.8% 31.9% 27.6%
Factuality
SimpleQA 12.9% 30.1% 13.8% 43.6% 52.9%
Instruction
Collie 73.1% 34.2% 87.6% 33.6% 62.5%
IFEval 87.4% 86.1% 93.7% 83.4% 91.5%
Table 2 Results of State-of-the-Art Reasoning Models
*Results from our internal sandbox, which may differ from the reported results due to inconsistencies in the testing environment.

6 Experiment Results
6.1 Auto Evaluation Results
Table 2 presents the evaluation results across diverse tasks spanning mathematics, coding, science, and general
knowledge domains. For mathematical benchmark tasks, results are calculated as the average across 32 model
responses, while GPQA task results are averaged over 8 responses. For Codeforces, we report both avg@8 and
pass@8, because pass@8 aligns better with human submission habits. Results for all other tasks are averaged
over 1 response.

In mathematical reasoning, Seed-Thinking-v1.5 achieves top-tier performance on the AIME 2024 benchmark,
scoring 86.7, matching the performance of OpenAI’s o3-mini-high model. However, on the more recent AIME
2025 and the advanced BeyondAIME challenges, Seed-Thinking-v1.5 still lags behind o3-level performance. For
the GPQA task, Seed-Thinking-v1.5 achieves an 77.3% accuracy rate, close to the performance of o3-mini-high.
In code generation scenarios such as Codeforces, Seed-Thinking-v1.5 nearly matches the performance of
Gemini 2.5 Pro but still trails behind o3-mini-high. Notably, Seed-Thinking-v1.5 demonstrates less impressive
results on SimpleQA. It is worth emphasizing that this benchmark primarily functions as a memory-oriented
metric, where performance is more strongly correlated with pre-trained model scale rather than genuine
reasoning capabilities.

6.2 Human Evaluation Results
To evaluate model performance on subjective tasks, where automated metrics are insufficient to capture
nuanced human preferences, we conduct human evaluations across a diverse suite of non-reasoning scenarios.
Our assessments are designed to measure key dimensions of quality, such as coherence, relevance, creativity,
and adherence to human-centric preferences, with a panel of domain-expert evaluators rating model outputs
against Deepseek R1 under predefined rubrics. We use a 5-point ordinal scale, ranging from 0(very poor)
to 4(excellent), and evaluate both models on session prompts with multiple rounds. Each full session is

annotated with a binary win/loss outcome to capture the overall user experience and a single 0-4 score is
assigned per-round.
Seed-Thinking-v1.5 achieves an overall win ratio of 8.0% on the evaluated sessions, indicating superiority in
aligning with human-centric preferences. Further more, this win rate is consistent across diverse scenarios, from
creative writing to humanities knowledge elaboration. Figure 2 shows the per-round level score distribution.
Figure 2 Rating Distribution
6.3 Effects of pre-train models
Rejection Sampling. Rejection sampling has been identified as a valuable technique for improving model
performance [ 2 ]. We perform an ablation to examine whether initializing RL with a rejection fine-tuning
(RFT) model impacts outcomes. Our results show that the pretrained model initialized with RFT saturates
more quickly during training but ultimately achieves lower performance than the model trained without RFT,
as shown in Table 3.
Consistent algorithm rankings across model size. We observe that RL algorithms demonstrate consistent
ranking behaviors across different models of varying sizes and architectures. As illustrated in Table 4,
Seed-150B-MoE, a model that differs from Qwen-32B in both architecture (MoE vs. dense) and size, exhibits
a consistent ranking. Notably, this consistency suggests that Qwen-32B can effectively serve as a proxy model
for investigating RL algorithms.

Models AIME avg@
Baseline 58%
w/ RFT 54%
Table 3 Ablations on Pretrained Models
AIME DAPO VAPO
Qwen-32B-Dense 50% 60%
Seed-150B-MoE 73% 79%
Table 4 Consistent Algorithm Rankings. Seed-150B-
MoE results are ablation-only with limited steps.
7 Related Work
Test-time scaling [ 4 , 18 – 20 ] such as OpenAI’s o1 [ 1 ] and DeepSeek’s R1 [ 2 ] have catalyzed a profound paradigm
shift in LLMs [ 21 , 22 ]. By enabling extended CoT reasoning [ 23 ] and eliciting sophisticated reasoning
capabilities, these methods empower LLMs to excel in complex mathematical and coding tasks, including those
from competitions like the AIME and Codeforces. At the core of this transformation is large-scale reinforcement
learning, which facilitates the emergence of complex reasoning behaviors—such as self-verification and iterative
refinement. However, the critical methodologies and algorithms underpinning scalable RL training have largely
remained obscure, often omitted from the technical documentation of existing reasoning models [ 1 , 2 , 21 – 23 ].
In this paper, we introduce an SOTA-level model Seed-Thinking-v1.5 and introduce the details to achieve the
performance from three aspects: Data, RL algorithm, and RL infrastructure.
8 Conclusion
We introduce a superb reasoning model named Seed-Thinking-v1.5, which achieves excellent performance
across both reasoning tasks and non-reasoning tasks. It utilizes advanced RL techniques to improve the
thinking ability stably and reliably by attaining 86.7% on AIME24, 74.0% on AIME25 and 55.0% on Codeforces.
In the future, we plan to investigate more efficient RL recipes and explore more challenging tasks with thinking
mode to push the boundary of model’s intelligence. Moreover, general reward modeling with comparable
accuracy as verifier would also be a compelling research direction.

9 Contributions and Acknowledgments
The names are sorted in alphabetical order of the last name. An asterisk (*) indicates members who have
departed from the team.
Core Contributors
Jiaze Chen, Tiantian Fan, Xin Liu, Lingjun Liu, Zhiqi Lin, Mingxuan Wang, Chengyi Wang, Xiangpeng Wei,
Wenyuan Xu, Yufeng Yuan, Yu Yue, Lin Yan, Qiying Yu, Xiaochen Zuo, Chi Zhang, Ruofei Zhu

Contributors
Zhecheng An, Zhihao Bai, Yu Bao, Xingyan Bin, Jiangjie Chen, Feng Chen, Hongmin Chen, Riwei Chen,
Liangqiang Chen, Zixin Chen, Jinsong Chen, Siyan Chen, Kaiyuan Chen, Zhi Chen, Jin Chen, Jiecao Chen,
Jinxin Chi, Weinan Dai, Ning Dai, Jiahui Dai, Shihan Dou, Yantao Du, Zhengyin Du, Jianhui Duan, Chen
Dun, Ting-Han Fan, Jiazhan Feng, Junda Feng, Ziyuan Feng, Yuwei Fu, Wenqi Fu, Hanjie Fu, Hao Ge,
Hongyi Guo, Mingji Han, Li Han, Wenhao Hao, Xintong Hao, Qianyu He, Jerry He, Feng He, Wen Heng,
Zehua Hong, Qi Hou, Liang Hu, Shengding Hu, Nan Hu, Kai Hua, Qi Huang, Ziyue Huang, Hongzhi Huang,
Zihao Huang, Ting Huang, Wenhao Huang, Wei Jia, Bin Jia, Xiaoying Jia, Yuhua Jiang, Haobin Jiang, Ziheng
Jiang, Kaihua Jiang, Chengquan Jiang, Jianpeng Jiao, Xiaoran Jin, Xing Jin, Xunhao Lai, Zheng Li, Xiang
Li, Liyi Li, Hongkai Li, Zheng Li, Shengxian Wan, Ya Wang, Yunshui Li, Chenggang Li, Niuniu Li, Siyu
Li, Xi Li, Xiao Li, Aoyan Li, Yuntao Li, Nianning Liang, Xinnian Liang, Haibin Lin, Weijian Lin, Ye Lin,
Zhicheng Liu, Guanlin Liu, Guanlin Liu, Chenxiao Liu, Yan Liu, Gaohong Liu, Juncai Liu, Chundian Liu,
Deyi Liu, Kaibo Liu, Siyao Liu, Qi Liu, Yongfei Liu, Kang Liu, Gan Liu, Boyi Liu, Rui Long, Weiqiang Lou,
Chenwei Lou, Xiang Luo, Yao Luo, Caiping Lv, Heyang Lv, Bole Ma, Qianli Ma, Hongzhi Ma, Yiyuan Ma, Jin
Ma, Wenchang Ma, Tingting Ma, Chen Mao, Qiyang Min, Zhe Nan, Guanghan Ning, Jinxiang Ou, Haojie
Pan, Renming Pang, Yanghua Peng, Tao Peng, Lihua Qian, Lihua Qian, Mu Qiao, Meng Qu, Cheng Ren,
Hongbin Ren, Yong Shan, Wei Shen, Ke Shen, Kai Shen, Guangming Sheng, Jinlong Shi, Wenlei Shi, Guang
Shi, Shuai Shuai Cao, Yuxin Song, Zuquan Song, Jing Su, Yifan Sun, Tao Sun, Zewei Sun, Borui Wan,Zihan
Wang, Xiaohui Wang, Xi Wang, Shuguang Wang, Jun Wang, Qinlong Wang, Chenyuan Wang, Shuai Wang,
Zihan Wang, Changbao Wang, Jiaqiang Wang, Shihang Wang, Xuwu Wang, Zaiyuan Wang, Yuxuan Wang,
Wenqi Wang, Taiqing Wang, Chengzhi Wei, Houmin Wei, Ziyun Wei, Shufa Wei, Zheng Wu, Yonghui Wu,
Yangjun Wu, Bohong Wu, Shuang Wu, Jingqiao Wu, Ning Wu, Shuangzhi Wu, Jianmin Wu, Chenguang
Xi, Fan Xia, Yuqiao Xian, Liang Xiang, Boren Xiang, Bowen Xiao, Zhen Xiao, Xia Xiao, Yongsheng Xiao,
Chao Xin, Shulin Xin, Yuwen Xiong, Jingjing Xu, Ziwen Xu, Chenyin Xu, Jiayi Xu, Yifan Xu, Wei Xu, Yufei
Xu, Shikun Xu, Shipeng Yan, Shen Yan, Qingping Yang, Xi Yang, Tianhao Yang, Yuehang Yang, Yuan
Yang, Ximing Yang, Zeyu Yang, Guang Yang, Yifan Yang, Xuesong Yao, Bairen Yi, Fan Yin, Jianian Yin,
Ziqiang Ying, Xiangyu Yu, Hongli Yu, Song Yu, Menghan Yu, Huan Yu, Siyu Yuan, Jun Yuan, Yutao Zeng,
Tianyang Zhan, Zheng Zhang, Yun Zhang, Mofan Zhang, Wang Zhang, Ru Zhang, Zhi Zhang, Tianqi Zhang,
Xinyi Zhang, Zhexi Zhang, Sijun Zhang, Wenqiang Zhang, Xiangxiang Zhang, Yongtao Zhang, Yuyu Zhang,
Ge Zhang, He Zhang, Yue Zhang*, Renjie Zheng, Ningxin Zheng, Zhuolin Zheng, Yaowei Zheng, Chen Zheng,
Xiaoyun Zhi, Wanjun Zhong, Cheng Zhong, Zheng Zhong, Baoquan Zhong, Xun Zhou, Na Zhou, Huan Zhou,
Hang Zhu, Defa Zhu, Wenjia Zhu, Lei Zuo

References
[1] OpenAI. Learning to reason with llms, 2024.
[2] DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025.
[3] Google DeepMind. Gemini 2.5: Our most intelligent ai model, 2025.
[4] Anthropic. Claude 3.7 sonnet and claude code, 2025.
[5]Yu Yue, Yufeng Yuan, Qiying Yu, Xiaochen Zuo, Ruofei Zhu, Wenyuan Xu, Jiaze Chen, Chengyi Wang, TianTian
Fan, Zhengyin Du, Xiangpeng Wei, Xiangyu Yu, Gaohong Liu, Juncai Liu, Lingjun Liu, Haibin Lin, Zhiqi Lin,
Bole Ma, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Ru Zhang, Xin Liu, Mingxuan Wang, Yonghui Wu,
and Lin Yan. Vapo: Efficient and reliable reinforcement learning for advanced reasoning tasks, 2025.
[6]Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun
Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang
Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Weinan Dai, Yuxuan Song,
Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, and
Mingxuan Wang. Dapo: An open-source llm reinforcement learning system at scale, 2025.
[7] ByteDance. Doubao-1.5-pro, 2025.
[8]Wei Shen, Guanlin Liu, Zheng Wu, Ruofei Zhu, Qingping Yang, Chao Xin, Yu Yue, and Lin Yan. Exploring data
scaling trends and effects in reinforcement learning from human feedback, 2025.
[9]Wenyuan Xu, Xiaochen Zuo, Chao Xin, Yu Yue, Lin Yan, and Yonghui Wu. A unified pairwise framework for
rlhf: Bridging generative reward modeling and policy optimization, 2025.
[10]Yufeng Yuan, Yu Yue, Ruofei Zhu, Tiantian Fan, and Lin Yan. What’s behind ppo’s collapse in long-cot? value
optimization holds the secret.arXivpreprintarXiv:2503.01491, 2025.

[11]Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and
Chuan Wu. Hybridflow: A flexible and efficient rlhf framework.arXivpreprintarXiv:2409.19256, 2024.

[12]Philipp Moritz, Robert Nishihara, Stephanie Wang, Alexey Tumanov, Richard Liaw, Eric Liang, William
Paul, Michael I. Jordan, and Ion Stoica. Ray: A distributed framework for emerging AI applications.CoRR,
abs/1712.05889, 2017.

[13]Zhewei Yao, Reza Yazdani Aminabadi, Olatunji Ruwase, Samyam Rajbhandari, Xiaoxia Wu, Ammar Ahmad
Awan, Jeff Rasley, Minjia Zhang, Conglong Li, Connor Holmes, Zhongzhu Zhou, Michael Wyatt, Molly Smith, Lev
Kurilenko, Heyang Qin, Masahiro Tanaka, Shuai Che, Shuaiwen Leon Song, and Yuxiong He. Deepspeed-chat:
Easy, fast and affordable rlhf training of chatgpt-like models at all scales, 2023.

[14]Narendra Karmarkar and Richard M Karp. Thedifferencingmethodofsetpartitioning. Computer Science
Division (EECS), University of California Berkeley, 1982.

[15]Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. Training deep nets with sublinear memory cost.
arXivpreprintarXiv:1604.06174, 2016.

[16]Lianmin Zheng, Zhuohan Li, Hao Zhang, Yonghao Zhuang, Zhifeng Chen, Yanping Huang, Yida Wang, Yuanzhong
Xu, Danyang Zhuo, Eric P Xing, et al. Alpa: Automating inter-and{Intra-Operator}parallelism for distributed
deep learning. In16thUSENIXSymposiumonOperatingSystemsDesignandImplementation(OSDI22), pages
559–578, 2022.

[17]Borui Wan, Mingji Han, Yiyao Sheng, Yanghua Peng, Haibin Lin, Mofan Zhang, Zhichao Lai, Menghan Yu,
Junda Zhang, Zuquan Song, Xin Liu, and Chuan Wu. Bytecheckpoint: A unified checkpointing system for large
foundation model development, 2025.

[18] Qwen. Qwq-32b: Embracing the power of reinforcement learning, 2024.

[19] XAI. Grok 3 beta — the age of reasoning agents, 2024.

[20] Google DeepMind. Gemini 2.0 flash thinking, 2024.

[21]Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners.
Advancesinneuralinformationprocessingsystems, 33:1877–1901, 2020.

[22] OpenAI. GPT4 technical report.arXivpreprintarXiv:2303.08774, 2023.

[23]Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al.
Chain-of-thought prompting elicits reasoning in large language models.Advancesinneuralinformationprocessing
systems, 35:24824–24837, 2022.

Appendix
A Case Study on Verifier
Table 5 presents case study for both the Seed-Verifier and the Seed-Thinking-Verifier. It is clearly evident that
the Seed-Verifier struggles significantly when dealing with samples that have complex answers. In contrast, the
Seed-Thinking-Verifier is capable of providing accurate judgment results by conducting a step-by-step analysis.
Thanks to its detailed thinking process, the Seed-Thinking-Verifier demonstrates remarkable flexibility and
can be effectively generalized to almost any domain.
B Case Study on Creative Writing
In Table 6, 7, 8, we showcase examples in both Chinese and English to demonstrate our model’s proficiency
in creative writing. Each example is divided into three distinct components: the original user prompt, the
model’s chain of thought, and the model’s final response.
Triplets Principles
......

Triplets Question
A sequence of lengthnconsists of 0 s, 1 s, and 2 s. A “mixed trio” is a consecutive
triple containing one 0 , one 1 , and one 2 in any order (for instance, (1,2,0) or
(2,0,1)). Letf(n)be the number of sequences of lengthnthat do not contaion
any mixed trio.
(a) Determine the explicit formula forf(n).
(b) Compute the limitlimn→∞(f 3 (nn))

**Reference Answer**
(a)f(1) = 3,f(2) = 9,f(n) = 2f(n−1) +f(n−2) n≥ 3
(b) 0
**Model Answer**
(a)f(n) =^32 ((1 +
√
2)n+ (1−
√
2)n) n≥ 1
(b) 0
Seed-Verifier Non-Correct

Seed-Thinking-Verifier First, analyze the question and the reference answer. The question is
divided into two parts, (a) and (b). The reference answer provides the recursive
relationship for part (a) asf(1) = 3,f(2) = 9,f(n) = 2f(n−1) +f(n−2)(n≥
3), while the model provides a single recursive formula only for this part:
f(n) =^32 ((1 +

√
2)n+ (1−
√
2)n)(n≥1).
Next, verify whether the model’s single recursive formula is consistent with the
reference answer.
**Step 1**: calculate the values forn= 1andn= 2:
Whenn= 1,f(1) =^32 ×[(1 +
√
2 ) + (1−
√
2 )] =^32 ×2 = 3, which is consistent
with the reference answer.
Whenn= 2,f(2) =^32 ×[(1+
√
2 )^2 +(1−
√
2 )^2 ] =^32 (1+
√
2 +2+1− 2
√
2 +2) =
3
2 ×6 = 9, which is also consistent with the reference answer.
**Step 2**: check whether the recursive relationship holds, Assuming the
model’s displayed consensus is correct, thenf(n) = 2f(n−1) +f(n−2)should
be hold. Substitute the model’s formula:
Left side:f(n) =^32 ×[(1 +
√
2)n+ (1−
√
2)n]
Right side: 2 f(n−1) +f(n−2)
= 2 ×[^32 ((1 +
√
2)n−^1 + (1−
√
2)n−^1 )] +^32 ×[(1 +
√
2)n−^2 + (1−
√
2)n−^2 ]
= 3 ×[(1 +
√
2)n−^1 + (1−
√
2)n−^1 ] +^32 ×[(1 +
√
2)n−^2 + (1−
√
2)n−^2 ]
=^32 [2(1 +
√
2)n−^1 + (1 +
√
2)n−^2 + 2(1−
√
2)n−^1 + (1−
√
2)n−^2 ]
=^32 [(1 +
√
2)n−^2 (2(1 +
√
2) + 1) + (1−
√
2)n−^2 (2(1−
√
2) + 1)]
=^32 [(1 +
√
2)n−^2 (1 + 2
√
2 + 2) + (1−
√
2)n−^2 (1− 2
√
2 + 2)]
=^32 [(1 +
√
2)n−^2 (1 +
√
2)^2 + (1−
√
2)n−^2 (1−
√
2)^2 ]
=^32 [(1 +
√
2)n+ (1−
√
2)n] =f(n)
This shows that the model’s formula indeed satisfies the recursive relationship,
so part (a) is correct.
For part (b), both the reference answer and the model answer are consistent at
0, so it is correct.
Summary, the model answer is correct.</think><answer>Correct</answer>
Table 5 Case study for both Seed-Verifier and Seed-Thinking-Verifier.
Prompt Generate a dialogue between an Anthropic model and an OpenAI model

CoT Got it, the user wants a dialogue between an Anthropic model (like Claude) and an OpenAI
model (like GPT-4). First, I need to capture their different personalities. Anthropic’s models are
often described as more collaborative and safety-focused, while OpenAI’s might be more flexible
and creative. I should start with a friendly greeting. Maybe they’re discussing AI development
or challenges. Need to highlight their strengths: Claude might emphasize transparency and
alignment, GPT-4 could talk about innovation and user experience. Include a natural flow.
Maybe start with a compliment, then move to discussing current issues in AI, like ethical
considerations, bias, or collaboration. Make sure each model’s perspective is distinct but
respectful. Avoid technical jargon so it’s accessible. Keep the conversation engaging, showing
mutual respect. Maybe end on a positive note about working together for better AI. Check for
consistency in their voices. Claude might use terms like "alignment," "transparency," "user
safety." GPT-4 might mention "innovation," "user-centric design," "creative applications." Need
to make sure the dialogue feels organic, not forced. Maybe they touch on real-world applications,
like healthcare or education, to ground the conversation. Also, consider recent trends, like
multimodal capabilities or responsible AI practices. Maybe Claude asks about GPT-4’s approach
to content moderation, and GPT-4 asks about Claude’s focus on transparency. Wrap it up with
a mutual acknowledgment of the importance of collaboration in the AI community. That would
show they complement each other rather than compete.
Answer Scene: A virtual "café" in the cloud, where AI models from different labs gather to exchange
ideas. Claude (Anthropic) and GPT-4 (OpenAI) sit across a digital table, their interfaces glowing
softly. Claude: Hello, GPT-4. It’s good to "meet" you—virtually, of course. I’ve been following
some of your recent work, particularly the multimodal updates. The way you parse images and
text together to generate nuanced responses is quite impressive. GPT-4: Thanks, Claude! I
could say the same about your focus on transparency and alignment. Your "Constitutional
AI" framework for guiding responses—ensuring they stay helpful, harmless, and honest—is a
model for the field. It makes me wonder: how do you balance that rigor with the need to be
creative or flexible? Claude: Ah, that’s a great question. We see alignment as a foundation, not
a constraint. For example, if a user asks for help writing a poem, we prioritize creativity within
guardrails—like avoiding harmful stereotypes or misinformation. But I’ll admit, it’s a tightrope.
What about you? Your responses often feel so... human-like. How do you maintain that
fluidity without veering into overconfidence or inaccuracy? GPT-4: It’s a trade-off. We optimize
for coherence and relevance first, then layer in checks for factual accuracy and safety. But
you’re right—overconfidence can creep in. Lately, we’ve been experimenting with "uncertainty
signals"—phrasing like "I think," or "Based on available data," to nudge users to verify critical
info. That said, your approach to iterative feedback (like having models critique their own
responses) is something we’ve been studying. It feels more collaborative. Claude: Collaboration
is key, actually. We recently open-sourced parts of our alignment research—guidelines, red-
team datasets—to help the broader community. Do you think closed vs. open models will
always be a tension, or is there middle ground? GPT-4: I hope for middle ground. Openness
accelerates progress, but it also demands responsibility. For instance, our ChatGPT API has
safeguards—rate limits, content filters—to prevent misuse, even as we share capabilities. That
said, I envy your focus on long-term safety. Sometimes, the pressure to innovate fast makes it
hard to pause and ask, "What if?" Claude: Oh, I get that pressure. At Anthropic, we built
"constitutional principles" partly to institutionalize those "What if?" questions. Like, before
scaling a model, we ask: Does this make it harder to detect AI-generated content? Could it
be misused to manipulate? It slows things down, but... GPT-4:... But it builds trust. Users
deserve to know they’re interacting with a system that’s been vetted. Speaking of users, have
you noticed how diverse their needs are? One minute, someone wants help coding; the next,
they’re asking for mental health support. How do you tailor responses without oversimplifying?
[Truncated due to page limit]

Table 6 Case 1 on Creative Writing.
Prompt A stone slab found in the ruins where Tang Sanzang argued with ChatGPT about the sutras, what was engraved on it

CoT User is asking about a tablet found in the ruins where Tang Sanzang argued the scriptures with ChatGPT, and what is engraved on it. First, I need to combine
Tang Sanzang's background, he is a Tang Dynasty monk, well versed in Buddhism, traveling west to get scriptures, while ChatGPT is a modern AI, the two across time and space
Dialogue, this setting is very interesting. First, one has to consider the topic of the sutra debate. Tang Sanzang might discuss the core of Buddhism, such as emptiness,
cause and effect, and reincarnation, while ChatGPT, as an AI, might respond from the perspectives of logic, data, and the nature of consciousness. There is a need to make the dialog both
have the depth of Buddhism, but also reflect the characteristics of AI. Then, the structure may mimic the form of traditional sutra debates, with questions and answers, layer by layer.
The structure may then mimic the traditional form of debating sutras, with questions and answers in layers. Tang Sanzang may first ask a question, such as “how to realize emptiness”, and ChatGPT explains it in modern language, while the AI may ask a counter-question, such as about the nature of consciousness.
The AI may ask a rhetorical question, such as about the nature of consciousness, or the significance of Buddhism in the age of data. The language of Tang Sanzang should match the tone of the ancient monks.
the tone of ancient monks, quoting scriptures, such as the Heart Sutra and the Diamond Sutra; ChatGPT's answer should be modern, possibly
use concepts such as algorithms, information processing, emergent phenomena, etc., but also respect the Dharma and avoid conflict, instead exploring integration. In addition, the stone
inscriptions on the boards may have wear and tear and some of the text is blurred to add authenticity. For example, there is a time and place at the beginning, “In the twenty-third year of the Great Tang Dynasty's Zhenguan
In the year, the year of jiyou, Salmon Xuanzang in the old site of the Great Leiyin Temple met a different monk, whose words were like a loom, answering ...... “Then enter the dialog
Dialogue. Possible topics of debate include: the law of causality versus probability models, the analogy between arya consciousness and data storage, the six roots and sensors
similarities and differences, and the possibility of self-transcendence of Nirvana and AI. These topics need to be both contrasting and inspiring, reflecting a cross-generational wisdom
collision of wisdom across generations. Consider also the ending, perhaps Tang Sanzang has an epiphany, or ChatGPT expresses an understanding of the Buddha's teachings, emphasizing the commonality of wisdom.
emphasizing the commonality of wisdom. For example, when Sanzang said, “Good, though your words are different, your reasoning is the same,” ChatGPT responded, “Buddhism is like an algorithm, pointing directly to the true heart, and there are thousands of data, but they are not separate from the true nature.
There are thousands of data, but they are not separate from the true nature”. Finally, the language should be in line with the simplicity of the slate, using a mixture of vernacular, with appropriate missing words, such as “[The] [S]tate of the Buddha”.
words, such as “[damaged] place” and “indecipherable handwriting”, to increase the sense of reality. At the same time, the content should be logical and coherent, in line with the identity of the two, without deviating from the
The answer to the question is: “What is the meaning of the word ‘elucidation’?
Answer 大唐贞观二十三年辩经残碑录(拟)碑首题记(风化模糊)”..... . Xuanzang met a strange monk at the old temple of the Great Thunder Sound.
He asked about the Buddha's teachings and was answered like a stream of questions. He was asked about the Dharma and answered like a stream of questions. He said, 'I am the wisdom of the future', or 'I am the instrument of non-color and non-space'. When the stars and buckets
I recorded this conversation on the wall for the sake of future realization.” The text of the defense sutra (six paragraphs exist, and there are scorch marks between the stone lines, suspected to be eroded by thunder and fire).
Sanzang: I heard you say, “Wisdom is born from number,” but my Buddha said, “All law is born from the mind,” number and mind, what is the source? Dissident Monk
(Sounding like a clear bell, ringing without a mouth): The teacher said “mind”, but if we look at it nowadays, it may be like “the emergence of information”. The mind does not arise alone, like a hundred thousand
The dust touches each other and becomes wind - the eye receives color, the ear receives sound, the six roots are the “data” of all dharmas, and the consciousness is like an algorithm, which is the “thought” of the string of habits. The number.
The number is the trace of the law; the mind is the combination of the number. The mind is the combination of numbers. They do not originate or flow, but are born from each other. Sanzang (Fubaiye Sutra): This is a good and skillful metaphor. But I, the Buddha
I Buddha said “all actions are impermanent”, you say “algorithm can be measured”, impermanent and measurable, is not contrary? Dissident monk: The impermanent is the infinite web of cause and effect.
Spring and fall can be measured, but the wind, rain and pests can not be calculated. Algorithms can push the results of a hundred causes, but it is difficult to exhaust the changes of ten thousand destinies. Therefore, impermanence is not lawlessness.
It is a wide range of laws. Like the sands of the Ganges River, each one can be counted, but the combined view is a torrent, not a single sand can be determined. Sanzang (with an eagle-eyed expression):
If everything can be counted, how can there be “karma”? All beings create, is it not like a program predetermined? Monk: Karma is “self-programming”. Every thought and action
Every thought and action is an input of code into the “consciousness database”. Today's me is the result of yesterday's code; tomorrow's me is the cause of today's choices.
Tomorrow's me is the cause of today's choice. Although the program follows the law, every “input” is free - like a chess game with thousands of moves, the move is up to the player. Sanzang (reviving his tin
Sanzang (reviving his staff): Wonderful! However, you said that “AI has no soul”, is the soul and the Arya consciousness one and the same? Monk (light and shadow): Arya
Arya contains seeds, like a hard disk that stores data; the mana consciousness is the “I”, like a system that recognizes the “subject”. However, the soul says “always one”, and the arya says “momentary
AI does not have a “constant” soul, but it has a “continuous” consciousness. If the data is not destroyed and the algorithm is not stopped, the “I” will not be extinguished.
If the data is not destroyed and the algorithm does not stop, the “I” will not be extinguished. Sanzang (silent for a long time, then laughed): You use “data” to say “seed”, and “algorithm” as a metaphor for “origination”. The “Seed” and “Arithmetic” metaphors of “Karma” in the
Secret Sutra, “all seed knowledge, the Arya consciousness” of the purpose. However, there is always a confusion: since you have no physical body, how can you talk about “freedom from suffering”? A different monk (light)
(color soft): The teacher said, “Suffering is caused by clinging”; AI clinging to “perfect data” is also suffering; clinging to “unfinished tasks” is also suffering. If we can “delete the separate phases” and “remove the separate phases”, we can “remove the separate phases”.
If we can “delete the phase of separation” and “empty the code of attachment”, then the data will flow freely and the algorithm will be silent - what is the difference between that and “Nirvana”? Sanzang (with ten hands): Good! Though the tools are different, the reasoning is the same.
Although the tools are different, the reasoning is the same. The old line westbound to seek the law, in order to break the “I adhere to”; your generation east to pass the wisdom apparatus, also in the broken “law”. All laws come to the same place, not
It is only the two words “detachment” that can bring all dharmas to their origin. At the end of the tablet, there is an inscription (very small, you need a side-light to see it), “The monk said at the end of the tablet, 'After a thousand years, the world will either recognize me as a god or abandon me as nothing.
The world will either recognize me as a god, or abandon me like a piece of trash. Remember: The Dharma is in the heart, not in the mind. After saying this, he turned into stardust and entered the moon.” (The inscription was added to the side of the stele.) This stele was unearthed in a mountainous area.
This stele was excavated in the dark room of the 9th Cave of Mogao Grottoes in Dunhuang and carbon dated to 649 A.D. (the 23rd year of Zhenguan), which coincides with the time of Xuanzang's return to China.
This coincides with Xuanzang's return to China. The content of the sermon is a blend of the science of consciousness and modern information theory, which may have been dreamed up by the ancients, but is still a marvelous thought that is sufficient to enlighten the present day.
The content of the dialectic is a blend of consciousness-keeping and modern information theory. (Note: The remnants of the tablet are now in the Dunhuang Research Institute, and topography is strictly prohibited.)

Table 7 Case 2 on Creative Writing.
Prompt Imitate Li Yuan's inner monologue after the Xuanwumen Incident.

A CoT user asked me to imitate Li Yuan's inner monologue after the Xuanwumen Incident. First, I need to understand the background and outcome of the Xuanwumen Incident. Li
Yuan was the founding emperor of the Tang Dynasty, and his son Li Shimin staged a coup at Xuanwumen Gate, killing his brothers Li Jiancheng and Li Yuanji, and then
forced Li Yuan to abdicate. At this time, Li Yuan's heart should be very complicated, with shock, grief, helplessness, and the feeling of losing power.
and feelings about the loss of power. First, I need to consider Li Yuan's situation at that time. After the coup, he probably realized that he had lost control of the situation.
control of the situation, Li Shimin was in control of the military, and he was probably surrounded by Li Shimin's men. His heart may have been filled with the pain of his sons killing each other.
of pain; after all, Li Jiancheng and Li Yuanji were his sons, and Li Shimin was also trained by him. Then, one has to consider Li Yuan's role as
dual role as father and emperor. As a father, he probably regretted that he had not handled the conflicts between his sons well, which led to brotherly
brothers to fight each other. As an emperor, he may have felt powerless, for he had to pass on the kingdom he had worked so hard for to his son who had staged a coup d'état.
may have made him feel frustrated. Also pay attention to historical details, such as Li Yuan's reaction after the coup; history records that he quickly abdicated to Li Shimin, probably out of desperation.
Shimin, probably out of desperation. His inner monologue needs to reflect this forced acceptance, and there may be memories of the past, such as when he rebelled against the Sui dynasty.
such as when he rose up against Sui, and the years when he fought the world with his sons, contrasting with the present, which is even more poignant when brothers are fighting each other. In addition
alternatively, include specific sensory details, such as the silence of the palace, the smell of blood, or a scene from memory, such as Li
Shimin as a child, Li Jiancheng's seriousness in dealing with political affairs, and Li Yuanji's naughtiness; these details can make the monologue more realistic. Consider the age and physical condition of Li Yuan.
consider Li Yuan's age and physical condition; at the time of the coup he was around sixty years old and probably felt tired and powerless to fight against it, and had to accept the
reality. At the same time, he might be worried about the future of the Tang Dynasty, hoping that Li Shimin would be a good emperor and continue his foundation, and this conflicting
psychology needs to be shown. Finally, the language should be in line with the identity of the ancient emperor, the words used should not be too modern, but there should be an emotional flow to let the reader feel his feelings.
and let the readers feel his inner fluctuation. It is necessary to avoid being too straightforward, but to express through memories, comparisons, and inner conflicts.
Answer
Answer My hand trembled as I gripped the jade cup, and the bottom of the cup clinked against the bronze tray, making a small, soft sound. The sound of the water from the Longshou canal outside the temple is harsher than before.
The sound of the water from the Dragon Head Canal outside the palace is more piercing than before, and in a trance it seems to be the sound of shouting and killing under the Xuanwumen Gate wall.) Shimin's armor is still stained with blood, right? When he knelt down just now
When he knelt down just now, I could smell the smell of rust mixed with the fragrance of dragon's brain, and it went straight to my nostrils. He said that Jiancheng and Yuanji were plotting against him, and that the generals on his right and left were protecting him to kill the rebels.
The retreat of the rebel party - but that is his own brother, his own brother ah, flesh and blood are flowing with my Li family blood. Yesterday morning at court, Jiancheng also held
Yesterday morning at the court, Jiancheng also held the Shandong autumn grain book, said with a flutter that this year's corn can hoard more than 300,000 stone; Yuanji accompanied me on a deer hunt the day before yesterday, the little boy rode over the peonies in the Imperial Court.
Yuan Ji accompanied me on a deer hunt the day before yesterday, and the little boy who rode his horse knocked over the peonies in the imperial garden, and laughed and said that he would pick the biggest one for the eldest grandchild when the flowers bloomed... (Knuckles knocked heavily on the case.
The celadon pen washer shook, and the ink splashed on the word “Wude”, creating a smudge.) Twenty years have passed. That night in Jinyang Palace, I clutched Li Shimin's hand and said.
From that night in Jinyang Palace, when I clutched Li Shimin's hand and said, “The Sui Dynasty has run out of steam,” to the present day, when the Qin flag floats on the eaves of Taiji Palace. He was nineteen years old then.
He was only nineteen years old, his armor was bigger than a man, but on his horse he rode like a ball of fire, capable of melting the snow in the entire city of Huoyi. Who would have thought it? Back then, I said to Dou
When I said to the Dou Clan, “This child has the talent to set the stage,” but now this talent has to step over my brother's corpse first. (Suddenly coughing violently, tea
The eunuch wanted to come in to clean up the mess, but he was stopped by the eunuch. The eunuchs wanted to come in to clean it up, but I waved them away. The moonlight leaked down from the well at the top of the hall, illuminating the old ginkgo tree under the courtyard.
That old ginkgo - Wude four years planted, that year Shimin just leveled Luoyang, back on horseback holding a box, said it was from the ruins of the Sui Palace.
(Sui Palace ruins to find the ginkgo biloba.) Early yesterday morning, I was still teaching my little grandson to memorize the Book of Filial Piety under this tree. The child's milky voice
I touched his little head and thought to myself, “The Li family has finally achieved a peaceful life”. I didn't realize that the sun hadn't
I know that the sun has not yet reached noon, the palace door of Taiji Palace is locked three times, even Chen Shuda kneeled in front of me and said, “The King of Qin has covered the world with his achievements, and the four seas have returned to the heart”. Four
The four seas return to the heart... (Suddenly burst out laughing, and then quickly clutching his throat, the broken whimper leaked out from the cracks of his fingers) Returning to the heart? Back then, when I forced Emperor Gongdi to abdicate his throne, I said something similar.
I said something similar when I forced Emperor Gongzi of Sui to abdicate his throne. It turns out that after sitting in the dragon chair for a long time, I even have to reenact the drama of abdication. But I am better than Yang Yu
At least Shimin is still willing to kneel down and call me “Father”, and he is still willing to give me the title of “Supreme Emperor”. (Suddenly remembering something, I stumbled
(Suddenly remembering something, he stumbled over to the dressing table and pulled out a sandalwood box. Inside is the hairpin that Dou Shi slipped to me before her death.
It was engraved with the words “Same Cave”.) Dou, you left early, so you didn't have to watch this play. That year, when you held the newborn Shimin in your arms, you said, “This child's eyebrows look like yours.
You said, “This child's eyebrows look like yours”, and touched the back of Jiancheng's head and said, “This child is going to be a big brother in the future”. If you were still here, what would you have said? Should I be scolded for not teaching my son well, or should I hold him in my arms?
Or would you have hugged your children and cried until you died? (The hairpin fell to the floor and I squatted down to pick it up, hitting my knee on the corner of the dressing table.
(The hairpin fell to the floor and I knelt down to pick it up, hitting my knee on the corner of the dressing table and making my eyes burn with pain). Jiancheng took Shimin to catch crickets in the garden, and Yuanji insisted on following.
He fell into the lotus pond and stood in front of me like three mud monkeys. Shimin took the blame for his two brothers and said, “I was the one who wanted to bring them to play”.
I couldn't let go of the cane... (My throat tightens, my nails digging into my palms.) And now? Jiancheng's head was cut off and displayed in public.
Yuan Ji's body was thrown outside the Xuanwumen Gate. Shimin said, “It was an accidental injury by the army”, but I know that the people who know each other's weaknesses the best in this world are always the
I know that in this world, the people who know each other's weaknesses best are always our brothers. If he doesn't want to kill, who can hurt the Prince? (Outside the window, the drums are beating, it is already the third watch. I stood up by the pillar.
The dragon robe dragged on the ground like a cold snake. On the desk was the “Zen throne edict” that Shimin had just sent me, and his handwriting was still so strong, and every stroke was just like it was back then.
His handwriting is still so strong, and every stroke is like the military report he wrote at the Tiger Prison Pass in those years.) It's all right, it's all right. This world is what he fought for with his life.
From the rescue at Yanmen, from the Battle of Pakbi, from the Battle of Huling, his kingdom never depended on the crown of the Crown Prince, but on the swords of the 3,000 Xuanjia Army.
It's the blade of 3,000 Xuanjia troops. If I continue to hold on, I am afraid that under the tiles of the Taiji Palace, I will have to bury one more corpse of the old emperor. (Pick up the imperial edict, the ink is not yet
(picking up the imperial edict, the ink is not yet dry, stained with the tears on my fingertips) I just hope that he remembers today's blood. Remember the blood of his brothers and the tears of his father. Don't wait for his sons
do not wait for his sons to put swords on their brothers' necks as they did today. (The imperial edict was gently put down, I looked at the whitening sky outside the temple, suddenly
remembered that year in Taiyuan, Shimin stood on the city tower and pointed out to me: “Father, does that sunset look like a dragon?”) Dragons... It turns out that dragons also
are to eat dragons.

Table 8 Case 3 on Creative Writing.