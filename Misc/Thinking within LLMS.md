Paper 1:
# Diverse “Thinking” Paradigms in Large Language Models: Foundations, Implementations, and Impact

---

## Introduction

Large Language Models (LLMs) have rapidly advanced from mere text generators to sophisticated agents capable of complex reasoning, decision-making, and tool use. This transformation is underpinned by the emergence of diverse “thinking” paradigms—structured approaches that guide how LLMs decompose, explore, and solve problems. These paradigms, inspired by cognitive science, classical AI, and advances in deep learning, have become central to the pursuit of more reliable, interpretable, and generalizable AI systems.

This report provides a comprehensive, scholarly analysis of the major reasoning paradigms in LLMs, including Chain of Thought (CoT), Tree of Thought (ToT), Graph of Thought (GoT), Web of Thought (WoT), Program-Aided Reasoning (PaR), Scratchpad Reasoning, ReAct (Reasoning and Acting), and emerging hybrid or experimental frameworks. For each, we examine conceptual foundations, implementation strategies, strengths and weaknesses, and real-world applications. We also present comparative analyses, discuss their impact on LLM capabilities, and highlight emerging trends and open challenges.

---

## Foundations of Reasoning in LLMs

### Cognitive and Historical Inspirations

The evolution of LLM reasoning paradigms is deeply influenced by dual-process theories in cognitive science, particularly the distinction between **System 1** (fast, intuitive, automatic) and **System 2** (slow, deliberate, analytical) thinking. Early LLMs, operating primarily via next-token prediction, mirrored System 1—adept at pattern recognition but limited in deliberate, multi-step reasoning. The quest for System 2-like capabilities has driven the development of structured reasoning paradigms that enable LLMs to plan, reflect, and interact with external tools or environments.

Classical AI approaches, such as rule-based systems, symbolic logic, and search algorithms, also inform these paradigms. The integration of symbolic and neural methods—**neuro-symbolic AI**—aims to combine the strengths of explicit logic with the flexibility of deep learning.

### Types of Reasoning in LLMs

LLMs are now expected to perform a spectrum of reasoning types:

- **Deductive Reasoning**: Drawing specific conclusions from general premises.
- **Inductive Reasoning**: Generalizing from specific examples.
- **Abductive Reasoning**: Inferring the most likely explanation.
- **Commonsense Reasoning**: Applying general world knowledge to infer reasonable conclusions.
- **Probabilistic Reasoning**: Handling uncertainty using probabilistic models.

While LLMs excel at statistical learning, their ability to perform explicit, multi-step reasoning remains a central research challenge.

---

## Major Reasoning Paradigms in LLMs

### 1. Chain of Thought (CoT)

#### Conceptual Foundation

**Chain of Thought (CoT)** prompting is a seminal paradigm that elicits step-by-step reasoning in LLMs by providing exemplars of intermediate reasoning steps in prompts. Inspired by human problem-solving, CoT encourages models to break down complex tasks into a sequence of logical steps, improving accuracy and interpretability.

The emergence of CoT is closely tied to the observation that sufficiently large models, when prompted with a few chain-of-thought exemplars, can generalize to perform multi-step reasoning across arithmetic, commonsense, and symbolic tasks.

#### Implementation Strategies

CoT is typically implemented via **few-shot prompting**: the prompt includes several examples where each question is followed by a detailed, stepwise solution. The model is then asked to solve a new problem in the same format. Variants include:

- **Zero-shot CoT**: Using trigger phrases like “Let’s think step by step” to elicit reasoning without explicit exemplars.
- **Self-Consistency**: Sampling multiple CoT outputs and selecting the most consistent answer via majority voting.
- **CoT-RAG**: Integrating CoT with retrieval-augmented generation for knowledge-intensive tasks.

#### Strengths and Weaknesses

**Strengths:**
- **Interpretability**: Produces explicit reasoning traces, aiding transparency and debugging.
- **Performance**: Substantially improves accuracy on arithmetic, logic, and commonsense benchmarks (e.g., GSM8K, SVAMP, ARC).
- **Simplicity**: Requires no model modification or fine-tuning.

**Weaknesses:**
- **Error Propagation**: Mistakes in early steps can cascade to incorrect answers.
- **Prompt Sensitivity**: Effectiveness depends on prompt design and model size.
- **Limited Exploration**: Follows a single reasoning path, potentially missing alternative solutions.

#### Applications and Use Cases

- **Mathematical Problem Solving**: Achieves state-of-the-art results on GSM8K, surpassing even fine-tuned models with verifiers.
- **Commonsense and Symbolic Reasoning**: Outperforms standard prompting on tasks requiring multi-step inference.
- **Explainable AI**: Used in domains where transparent reasoning is required, such as education and legal tech.

---

### 2. Tree of Thought (ToT)

#### Conceptual Foundation

**Tree of Thought (ToT)** generalizes CoT by enabling LLMs to explore multiple reasoning paths in a tree-like structure, rather than committing to a single linear chain. This paradigm is inspired by **System 2** thinking and classical search algorithms, allowing for deliberate exploration, lookahead, and backtracking.

ToT is particularly effective for tasks where initial decisions have long-term consequences, or where multiple solution paths must be considered and evaluated.

#### Implementation Strategies

ToT is implemented by constructing a **multi-way tree** where each node represents a partial solution (state), and edges correspond to intermediate thoughts or actions. Key components include:

- **Thought Generator**: Proposes multiple candidate thoughts at each node.
- **State Evaluator**: Scores or votes on candidate states using the LLM itself.
- **Heuristic Calculator**: Aggregates evaluations to assign values for pruning.
- **Search Algorithm**: Employs breadth-first search (BFS), depth-first search (DFS), or hybrid strategies to explore the tree.

The process iterates: generate candidates, evaluate, prune, and expand promising nodes. Hyperparameters such as the number of candidates, evaluations, and breadth limit control the search depth and cost.

#### Strengths and Weaknesses

**Strengths:**
- **Exploration**: Considers diverse reasoning paths, increasing the chance of finding optimal solutions.
- **Deliberation**: Supports lookahead and backtracking, akin to human planning.
- **Interpretability**: Produces explicit, branching reasoning traces.

**Weaknesses:**
- **Computational Cost**: Tree search is resource-intensive, especially with large breadth or depth.
- **Hyperparameter Sensitivity**: Performance depends on careful tuning of search parameters.
- **Scalability**: May be impractical for tasks with vast search spaces.

#### Applications and Use Cases

- **Combinatorial Tasks**: Dramatically improves performance on the Game of 24 (CoT: 4% vs. ToT: 74% success with GPT-4).
- **Creative Writing**: Generates more coherent and diverse narratives by exploring alternative plans.
- **Planning and Strategy**: Applied to mini crosswords, strategy QA, and other tasks requiring multi-step planning.

---

### 3. Graph of Thought (GoT)

#### Conceptual Foundation

**Graph of Thought (GoT)** extends ToT by representing reasoning as a general graph, rather than a strict tree, allowing for cycles, merges, and more complex dependencies between reasoning steps. This paradigm is motivated by the observation that human reasoning often revisits, merges, or reuses intermediate conclusions.

GoT enables richer exploration and analysis of reasoning traces, supporting metrics such as exploration density, branching ratio, and convergence.

#### Implementation Strategies

GoT frameworks segment raw CoT transcripts into reasoning units, cluster them into coherent steps, and construct a directed graph where nodes are reasoning steps and edges represent logical or semantic dependencies. Advanced implementations use LLMs to estimate support or contradiction probabilities between steps, enabling adaptive sampling and consensus.

Graph metrics are computed to analyze the structure and quality of reasoning, such as linearity (proportion of steps with degree ≤2), exploration density, and convergence ratio.

#### Strengths and Weaknesses

**Strengths:**
- **Expressiveness**: Captures complex, non-linear reasoning patterns.
- **Analysis**: Enables quantitative analysis of reasoning structure.
- **Reusability**: Supports merging and reuse of intermediate results.

**Weaknesses:**
- **Complexity**: Graph construction and analysis are computationally demanding.
- **Interpretability**: Graphs may be harder to interpret than linear chains or trees.
- **Tooling**: Fewer mature frameworks compared to CoT/ToT.

#### Applications and Use Cases

- **Reasoning Trace Analysis**: Used to map and evaluate the “mind maps” of LLMs, revealing strengths and weaknesses in their cognitive processes.
- **Multi-Hop QA and Planning**: Suitable for tasks where intermediate conclusions are reused or revisited.

---

### 4. Web of Thought (WoT)

#### Conceptual Foundation

**Web of Thought (WoT)** refers to reasoning paradigms where LLMs augment their internal reasoning with **web-based retrieval**, tool use, and dynamic knowledge integration. WoT is motivated by the limitations of static, parametric knowledge in LLMs and the need for up-to-date, factual, and domain-specific information.

WoT is closely related to **Retrieval-Augmented Generation (RAG)** and **tool-augmented reasoning**, blending internal CoT with external knowledge sources.

#### Implementation Strategies

WoT systems interleave reasoning steps with web search, database queries, or API calls. Key strategies include:

- **Iterative Retrieval**: Multi-round web retrieval to refine and correct knowledge bases (e.g., Ir-Web-RAG, Ir-Web-CoT).
- **Knowledge Graph Integration**: Modulating reasoning chains with knowledge graphs to enhance credibility and logical rigor (e.g., CoT-RAG).
- **Dynamic Routing**: Adaptive selection of retrieval strategies based on query complexity and similarity labels.

Advanced WoT frameworks employ sentence-level knowledge base optimization, similarity label generation models, and self-correcting algorithms to improve retrieval accuracy and efficiency.

#### Strengths and Weaknesses

**Strengths:**
- **Factual Grounding**: Reduces hallucinations by grounding reasoning in external, up-to-date knowledge.
- **Domain Adaptability**: Excels in specialized domains (finance, law, medicine) where static LLM knowledge is insufficient.
- **Self-Correction**: Iterative retrieval and verification improve answer reliability.

**Weaknesses:**
- **Latency**: Web retrieval introduces additional response time.
- **Complexity**: Requires integration with external search and retrieval systems.
- **Dependency**: Performance depends on the quality and relevance of retrieved documents.

#### Applications and Use Cases

- **Open-Domain QA**: Outperforms standard RAG and CoT on SQuAD2.0, HotpotQA, and PopQA benchmarks.
- **Domain-Specific QA**: Demonstrates robust performance in finance, legal, and medical datasets.
- **Self-Correcting Agents**: Used in systems that iteratively refine knowledge and answers via web retrieval.

---

### 5. Program-Aided Reasoning (PaR) / PAL

#### Conceptual Foundation

**Program-Aided Reasoning (PaR)**, exemplified by **Program-Aided Language Models (PAL)**, enhances LLM reasoning by offloading computation to external program interpreters (e.g., Python). The LLM generates code as intermediate reasoning steps, which are then executed to verify or compute results.

This paradigm is inspired by the synergy between neural and symbolic computation, leveraging the precision of code execution to overcome LLM limitations in arithmetic and logic.

#### Implementation Strategies

PaR frameworks prompt the LLM to generate code snippets (e.g., Python functions) that solve subproblems. The process involves:

- **Code Generation**: LLM produces code as part of its reasoning trace.
- **External Execution**: The code is executed in a sandboxed environment (e.g., Python interpreter).
- **Result Integration**: Outputs from code execution are fed back into the reasoning process.

PAL implementations use few-shot prompting with code exemplars and can be integrated with APIs such as OpenAI or Hugging Face.

#### Strengths and Weaknesses

**Strengths:**
- **Precision**: Offloads complex computation to reliable program execution.
- **Performance**: Achieves state-of-the-art results on mathematical and algorithmic benchmarks (e.g., GSM8K, BIG-Bench Hard).
- **Error Checking**: Enables verification and correction of intermediate results.

**Weaknesses:**
- **Dependency**: Requires secure and robust code execution environments.
- **Security**: Risk of code injection or unsafe execution if not properly sandboxed.
- **Limited Generality**: Best suited for tasks that can be formalized as code.

#### Applications and Use Cases

- **Mathematical Reasoning**: Outperforms much larger models using CoT alone (e.g., Codex with PAL surpasses PaLM-540B on GSM8K by 15% top-1 accuracy).
- **Algorithmic Tasks**: Excels in symbolic and procedural reasoning.
- **Code Generation and Verification**: Used in programming assistants and automated theorem proving.

---

### 6. Scratchpad Reasoning

#### Conceptual Foundation

**Scratchpad Reasoning** refers to the use of intermediate “scratchpads”—explicit, often structured, spaces where LLMs record partial computations, diagrams, or notes during problem-solving. This paradigm is inspired by human use of paper or whiteboards to externalize thought processes.

Scratchpads can be textual, tabular, or even visual, and serve to organize multi-step reasoning, track variables, and facilitate self-correction.

#### Implementation Strategies

Scratchpad reasoning is typically implemented via prompt templates that allocate sections for intermediate calculations, diagrams, or notes. Variants include:

- **Textual Scratchpads**: LLMs write out sub-calculations or hypotheses before producing final answers.
- **Diagrammatic Scratchpads**: For tasks like geometry or logic puzzles, LLMs generate ASCII diagrams or tables.
- **Visual Scratchpads**: Emerging multimodal models can use images or visualizations as part of their reasoning process.

Scratchpads are often combined with CoT, ToT, or PaR paradigms for enhanced structure and clarity.

#### Strengths and Weaknesses

**Strengths:**
- **Transparency**: Makes intermediate reasoning explicit and auditable.
- **Error Reduction**: Helps prevent mistakes by tracking variables and steps.
- **Versatility**: Adaptable to a wide range of tasks, including math, logic, and planning.

**Weaknesses:**
- **Prompt Complexity**: Requires careful prompt engineering to structure scratchpads effectively.
- **Output Length**: Can increase response length and cost.
- **Limited Tooling**: Visual scratchpads require multimodal capabilities.

#### Applications and Use Cases

- **Mathematical Problem Solving**: Used in datasets like GSM8K and MATH for stepwise calculations.
- **Diagrammatic Reasoning**: Applied to logic puzzles, geometry, and scientific reasoning.
- **Explainable AI**: Facilitates interactive explanations and auditing.

---

### 7. ReAct (Reasoning and Acting)

#### Conceptual Foundation

**ReAct** (Reasoning and Acting) is a paradigm that interleaves reasoning traces (“thoughts”) with explicit actions (e.g., tool calls, API queries) in a structured loop. ReAct agents do not merely think—they act, observe, and adapt, enabling dynamic interaction with external environments.

ReAct is inspired by the need for LLMs to ground their reasoning in real-world data, reduce hallucinations, and handle complex, multi-step workflows.

#### Implementation Strategies

ReAct agents follow a **thought → action → observation → thought** loop:

1. **Thought**: The LLM reasons about the next step.
2. **Action**: The agent executes an action (e.g., web search, calculator, database query).
3. **Observation**: The result of the action is observed and incorporated into the context.
4. **Repeat**: The loop continues until a final answer is produced.

Implementation frameworks such as **LangChain**, **LlamaIndex**, and **LangGraph** provide reusable templates for building ReAct agents, supporting tool integration, memory, and prompt engineering.

#### Strengths and Weaknesses

**Strengths:**
- **Grounded Intelligence**: Reduces hallucinations by fetching real-time data.
- **Adaptive Planning**: Responds dynamically to changing information and observations.
- **Transparency**: Produces auditable, stepwise reasoning and action traces.
- **Scalability**: Tool-agnostic design allows integration with diverse APIs and services.

**Weaknesses:**
- **Latency**: Each action-observation cycle adds response time.
- **Cost**: Iterative loops can increase token and API usage.
- **Complexity**: Requires robust error handling and loop termination strategies.

#### Applications and Use Cases

- **Autonomous Agents**: Powers chatbots, research assistants, and robotic controllers that interact with external tools.
- **Fact-Checking and QA**: Reduces hallucinations in open-domain QA by verifying facts via web search or databases.
- **Code Assistance**: Enables LLMs to search documentation, run code, and explain decisions.
- **Multi-Agent Systems**: Forms the basis for collaborative agent workflows in enterprise and research settings.

---

### 8. Emerging, Hybrid, and Experimental Paradigms

#### Hybrid Approaches

Recent research explores **hybrid paradigms** that combine elements of CoT, ToT, ReAct, PaR, and WoT for enhanced reasoning:

- **CoT-RAG**: Integrates chain-of-thought reasoning with retrieval-augmented generation and knowledge graphs for improved credibility and logical rigor.
- **Ir-Web-CoT/Ir-Web-RAG**: Employs multi-round web retrieval and self-correction to refine knowledge bases and answers.
- **MM-REACT**: Extends ReAct to multimodal reasoning, incorporating images, audio, and video in the action-observation loop.
- **PreAct, ReST, ToT+PaR**: Variants that introduce planning before acting, reflection prompts, or combine tree search with program execution.

#### Neuro-Symbolic and Modular Architectures

**Neuro-symbolic AI** combines neural networks with symbolic reasoning frameworks, enabling LLMs to perform structured logic, rule-based inference, and knowledge graph traversal. Modular architectures decompose reasoning into specialized modules (e.g., retrievers, planners, verifiers) that collaborate via explicit protocols.

#### Evaluation and Benchmarking

A diverse set of benchmarks and metrics assess LLM reasoning:

- **GSM8K, MATH, ARC, LogiQA, HotpotQA, BIG-Bench**: Evaluate arithmetic, logical, and multi-hop reasoning.
- **Metrics**: Accuracy, logical consistency, explainability, self-consistency, multi-hop score, adversarial robustness, faithfulness, and confidence calibration.

---

## Comparative Analysis of Reasoning Paradigms

To facilitate side-by-side comparison, the following table summarizes key paradigms across critical dimensions:

| Paradigm | Scalability | Interpretability | Modularity | Reasoning Depth | Performance | Tool Use | Factual Grounding | Cost/Latency |
|----------|-------------|------------------|------------|------------------|-------------|----------|-------------------|--------------|
| CoT      | High        | Medium           | Low        | Medium           | Moderate    | No       | No                | Low          |
| ToT      | Medium      | High             | High       | High             | High        | No       | No                | Medium-High  |
| GoT      | Medium      | Medium           | High       | High             | Moderate    | No       | No                | High         |
| WoT      | Medium      | Medium           | Medium     | Medium           | High        | Yes      | Yes               | Medium-High  |
| PaR      | Medium      | High             | Medium     | High             | High        | Yes      | Yes (via code)    | Medium       |
| Scratchpad | High      | High             | Low        | Medium           | Moderate    | No       | No                | Low-Medium   |
| ReAct    | Medium      | High             | High       | High             | High        | Yes      | Yes               | Medium-High  |
| Hybrid   | Medium      | High             | High       | High             | Very High   | Yes      | Yes               | High         |

**Elaboration:**

- **Scalability**: CoT and scratchpad approaches scale well due to their linear structure, while ToT, GoT, and hybrid methods face computational challenges as search space grows.
- **Interpretability**: ToT, ReAct, and scratchpad paradigms provide the most transparent reasoning traces, facilitating auditing and debugging.
- **Modularity**: ToT, GoT, ReAct, and hybrid systems are highly modular, supporting plug-and-play integration of tools, verifiers, and retrievers.
- **Reasoning Depth**: ToT, GoT, PaR, and ReAct enable deeper, multi-step reasoning by exploring alternative paths, executing code, or interacting with external sources.
- **Performance**: Empirical studies show that ToT, PaR, ReAct, and hybrid paradigms outperform CoT on complex, multi-step tasks, especially in math, planning, and open-domain QA.
- **Tool Use and Factual Grounding**: WoT, PaR, and ReAct paradigms excel at grounding outputs in external knowledge, reducing hallucinations and improving reliability.
- **Cost/Latency**: Tree/graph search and iterative tool use increase computational cost and latency, necessitating careful trade-offs in production systems.

---

## Impact on LLM Capabilities

### Reasoning and Multi-Step Problem Solving

Structured reasoning paradigms have dramatically improved LLM performance on tasks requiring multi-step inference, planning, and logical deduction. For example, ToT increased GPT-4’s success rate on the Game of 24 from 4% (CoT) to 74%, while PAL achieved a 15% absolute gain over PaLM-540B on GSM8K.

### Transparency and Interpretability

Paradigms like CoT, ToT, ReAct, and scratchpad reasoning produce explicit reasoning traces, enhancing transparency, explainability, and trustworthiness. This is critical for high-stakes domains such as healthcare, law, and finance.

### Reliability and Robustness

Tool-augmented paradigms (WoT, ReAct, PaR) reduce hallucinations by grounding outputs in external data, enabling self-correction and verification. Self-consistency and majority voting further improve reliability by aggregating diverse reasoning paths.

### Modularity and Adaptability

Modular architectures support rapid adaptation to new tasks, domains, and tools. Frameworks like LangChain, LlamaIndex, and ToT repositories provide reusable components for building custom agents.

### Scalability and Cost

While advanced paradigms enable deeper reasoning, they introduce trade-offs in computational cost, latency, and scalability. Efficient search, caching, and prompt optimization are active areas of research to mitigate these challenges.

---

## Emerging Trends and Future Directions

### Hybrid and Multi-Agent Systems

The frontier of LLM reasoning is moving toward **hybrid paradigms** that combine the strengths of CoT, ToT, ReAct, PaR, and WoT. Multi-agent systems, where specialized agents collaborate via explicit protocols, are gaining traction in enterprise and research settings.

### Neuro-Symbolic Integration

Combining neural and symbolic reasoning remains a key goal for achieving robust, generalizable, and explainable AI. Neuro-symbolic architectures leverage the pattern recognition of deep learning with the logical rigor of symbolic systems.

### Evaluation and Benchmarking

The proliferation of reasoning paradigms necessitates more comprehensive and generalizable evaluation frameworks. Automated, capability-based, and dynamic benchmarks are being developed to assess reasoning, robustness, and generalization.

### Interpretability and Auditing

LLMs are increasingly used to generate interactive explanations, audit their own outputs, and assist in dataset analysis. Research is ongoing to improve the faithfulness, robustness, and utility of reasoning explanations.

### Scalability and Efficiency

Efforts to reduce cost and latency include semantic caching, speculative sampling, and prompt optimization. Visual workflow builders and low-code platforms are democratizing the development of reasoning-action agents.

### Safety, Robustness, and Hallucination Mitigation

Robustness to adversarial inputs, domain generalization, and hallucination mitigation remain open challenges. Fact-checking, automated verifiers, and retrieval-augmented reasoning are promising directions.

---

## Open Challenges

Despite significant progress, several challenges persist:

- **Hallucinations and Misinformation**: LLMs can still generate plausible but incorrect reasoning chains, especially in multi-step tasks.
- **Generalization Across Domains**: Fine-tuned models may overfit to specific tasks, limiting cross-domain adaptability.
- **Adversarial Robustness**: Sensitivity to input variations and adversarial prompts undermines reliability.
- **Integration of Symbolic and Neural Reasoning**: Achieving seamless neuro-symbolic integration is an ongoing research frontier.
- **Evaluation and Benchmarking**: Existing benchmarks may not capture the full spectrum of reasoning capabilities or generalization.
- **Cost and Scalability**: Advanced paradigms can be computationally expensive, necessitating efficient search and caching strategies.

---

## Conclusion

The landscape of reasoning paradigms in LLMs is rich, dynamic, and rapidly evolving. From the linear clarity of Chain of Thought to the deliberate exploration of Tree of Thought, the precision of Program-Aided Reasoning, and the agentic adaptability of ReAct, each paradigm brings unique strengths and trade-offs. Hybrid and modular approaches, grounded in cognitive science and classical AI, are pushing the boundaries of what LLMs can achieve.

As LLMs become integral to critical applications, the demand for robust, transparent, and generalizable reasoning grows ever more urgent. Continued research into hybrid paradigms, neuro-symbolic integration, evaluation frameworks, and scalable architectures will be essential to realizing the promise of truly intelligent, trustworthy, and adaptable AI systems.

---

## Appendix: Key Tooling and Frameworks

- **LangChain**: Modular framework for building LLM agents with tool integration, memory, and reasoning loops.
- **LlamaIndex**: Supports retrieval-augmented reasoning and tool use.
- **Tree of Thoughts Repository**: Provides reusable classes for BFS/DFS tree search with OpenAI and Hugging Face APIs.
- **PAL Repository**: Implements program-aided reasoning with code execution.
- **LLM-MindMap**: Toolkit for graph-based analysis of reasoning traces.

---

## References to Key Benchmarks

- **GSM8K**: Grade-school math word problems for multi-step reasoning.
- **MATH**: High-school and competition-level mathematics.
- **ARC, LogiQA, HotpotQA, BIG-Bench**: Logical, commonsense, and multi-hop reasoning.
- **HumanEval**: Code generation and functional correctness.

---

**This report synthesizes insights from a broad spectrum of recent research, technical reports, and empirical studies, providing a rigorous, actionable foundation for expert readers and researchers in the field of LLM reasoning.**

Paper 2: 

Reasoning Paradigms in Large Language Models
Large language models (LLMs) have demonstrated impressive capabilities on a variety of tasks, but effective multi-step reasoning remains challenging. In response, researchers have devised multiple prompting and inference paradigms that expose internal reasoning processes, structure thought, or integrate external computation. These include Chain-of-Thought, Tree-of-Thought, Graph-of-Thought, Scratchpad, Program-Aided Reasoning, ReAct (Reasoning+Acting), and other emerging schemes. Each paradigm has a distinct conceptual basis and implementation, with different trade-offs in scalability, transparency, and problem-solving power. Below we review each paradigm’s theory, practical realization, strengths/weaknesses, and applications, followed by a comparative analysis and discussion of impacts on LLM capabilities and emerging trends.
Chain-of-Thought (CoT)
Conceptual Foundation: CoT prompting encourages the model to “think out loud” by generating intermediate reasoning steps in natural language before a final answer
arxiv.org
. Inspired by human step-by-step reasoning, CoT breaks a complex problem into a sequence of logical inferences or calculations. This sequential chain of assertions acts as a reasoning trace that the model can follow and extend. Wei et al. (2022) showed that providing exemplar problems with detailed solution steps elicits similar reasoning behavior, dramatically improving performance on arithmetic, commonsense, and symbolic tasks
arxiv.org
aclanthology.org
.
Implementation Strategies: CoT is usually implemented via prompt engineering. In few-shot in-context prompting, each example includes a question followed by a detailed answer that lists intermediate steps. The model then continues the chain for a new question. No architectural change is needed to the LLM itself. In practice, one simply appends hints like “Let’s think step by step…” or provides sample chains in the prompt. CoT can also be applied in zero-shot fashion by prefixing the question with an instruction to solve the task stepwise
arxiv.org
.
Strengths: CoT significantly enhances LLM reasoning accuracy on tasks that require multi-step logic. For example, PaLM-540B with CoT prompting achieved state-of-the-art accuracy on the GSM8K math benchmark
arxiv.org
. The explicit reasoning steps often make the process more transparent and interpretable to users
aclanthology.org
. Chain-of-thought requires only natural language prompting and scales to any LLM.
Weaknesses: CoT requires large models and longer context windows to generate and consume the chains, which increases computational cost. It can be brittle: if the model generates an incorrect intermediate, it typically leads to a wrong final answer. CoT also tends to produce lengthy outputs and can suffer from error accumulation or repetition. Shorter chains may not sufficiently explore complex problems, while very long chains are costly to produce
aclanthology.org
.
Applications and Use Cases: CoT has been applied to arithmetic word problems, logic puzzles, commonsense reasoning, symbolic manipulation, and question answering
arxiv.org
. It is widely used in benchmark evaluation of LLMs (e.g. GSM8K for math, ProofWriter for logical reasoning). Many practical systems use CoT-style prompts to boost reliability in diagnostic or multi-step tasks.
Scratchpad Reasoning
Conceptual Foundation: Scratchpad reasoning is a variant of explicit stepwise thinking where the model is trained or prompted to output intermediate computations in a separate “scratchpad” area. It was originally proposed to enable multi-step numeric or programmatic computation by having the model emit partial results that accumulate over the course of solving
arxiv.org
. Conceptually it is akin to a student writing calculations on scrap paper.
Implementation Strategies: During training or few-shot prompting, the model is instructed to produce intermediate arithmetic or logical steps (the “scratchpad”) as part of its output
arxiv.org
. This can be done by including examples with Step 1: … Step 2: … formatting. Some implementations reserve a special block of text in each output as the scratchpad. Unlike CoT, scratchpad methods often involve supervised fine-tuning on tasks where the intermediate state is explicitly labeled, though prompting alone can suffice.
Strengths: Scratchpad improves LLM performance on tasks requiring unbounded computation or long chains of arithmetic. Nye et al. (2021) demonstrated that even standard pretrained Transformers can solve very long addition and program execution tasks when asked to show all steps
arxiv.org
. It extends the effective memory of the model by making partial results explicit. For arithmetic and symbolic tasks, scratchpads dramatically increase accuracy compared to one-shot outputs
arxiv.org
.
Weaknesses: Like CoT, scratchpad reasoning increases token usage and inference time due to longer outputs. It usually requires specialized training or careful prompt design to reliably produce the scratchpad format. In practice, if the LLM is not explicitly trained to “move” partial results into the scratchpad, it may omit or incorrectly format the steps. Without external verification, errors in the scratchpad can still propagate.
Applications and Use Cases: Scratchpad-style reasoning has been applied in mathematics (e.g. long addition, multi-digit multiplication), program simulation, and other structured tasks. For example, LLMs trained with scratchpads can simulate simple computer programs or solve symbolic integration by writing intermediate code
arxiv.org
. It is also conceptually related to prompting LLMs to output “thinking steps” in STEM or logic problems.
Tree-of-Thought (ToT)
Conceptual Foundation: Tree-of-Thoughts extends CoT by introducing a search process over possible reasoning paths. Instead of a single linear chain, the LLM explores a tree of candidate “thoughts” (coherent reasoning states) and can backtrack or branch as needed
arxiv.org
. This mirrors human deliberation (System 1 vs System 2) and classical search/planning: the model can maintain diverse partial solutions, evaluate them, and decide which branch to pursue
arxiv.org
. The ToT framework is motivated by planning and search in cognitive science: the LLM performs deliberate decision-making by considering multiple future paths, akin to “lookahead” or trial-and-error
arxiv.org
.
Implementation Strategies: ToT is implemented by iteratively prompting the LLM to expand and evaluate multiple candidate next steps. A common approach is to maintain a tree where each node is a text “thought” (an intermediate answer or plan). At each step, the model is prompted to generate several possible continuations of a given thought, creating branches. The system then uses heuristics or a value model to prune branches, possibly backtracking and expanding alternatives
arxiv.org
. The process continues until a solution is found or depth limit reached. This effectively transforms LLM inference into a search algorithm guided by the model’s own predictions.
Strengths: Tree-of-Thoughts enables solving problems that are difficult for linear CoT. By exploring multiple paths, it can recover from early mistakes (backtracking) and perform non-greedy reasoning. Yao et al. (2023) showed ToT dramatically improves success on tasks requiring planning or combinatorial search: e.g. solving “Game of 24” puzzles rose from 4% with CoT to 74% with ToT
arxiv.org
. ToT also generalizes to creative or puzzle tasks (creative writing, mini-crossword) by exploring diverse strategies
arxiv.org
.
Weaknesses: The main drawback is computational cost: branching quickly multiplies the number of model queries. Search heuristics and pruning are needed to keep the tree tractable. Designing good branch scoring or step evaluation is nontrivial. ToT implementations are more complex to engineer and depend on hyperparameters (beam width, depth). They also require more memory and latency, making them harder to deploy at scale.
Applications and Use Cases: ToT has been applied to games and puzzles where multiple strategies exist. Yao et al. tested ToT on arithmetic puzzles (“Game of 24”), creative story generation (searching narrative paths), and mini-crossword solving
arxiv.org
. In each case, ToT outperformed standard prompting. ToT is a frontier approach for complex planning tasks in LLMs, often in research prototypes (with code repositories available) rather than production systems.
Graph-of-Thought (GoT)
Conceptual Foundation: Graph-of-Thoughts generalizes tree search by allowing arbitrary graph structures of reasoning units. In GoT, each “thought” (a partial solution or piece of information) is a node, and edges encode logical dependencies between thoughts
research-collection.ethz.ch
. Unlike ToT’s tree, GoT permits merging branches (multiple parents) and forming cycles or feedback loops. This reflects the idea that reasoning is not strictly linear or hierarchical, but can involve cross-connections (much like human brainstorming or networked cognitive processes). GoT’s inspiration is to harness the expressive power of graphs: sub-solutions can combine in novel ways, and insights can be distilled from a web of possibilities
research-collection.ethz.ch
.
Implementation Strategies: A GoT framework involves constructing and traversing a reasoning graph. Practically, one approach is Graph of Operations (GoO) prompting: the user predefines a graph topology (nodes and connections) that captures how to decompose the task. Then, in multiple prompts, the LLM fills in the “thoughts” at each node by following the graph structure
research-collection.ethz.ch
. Because nodes can have multiple parents, outputs from different sub-thoughts can be merged or aggregated. The LLM can also iteratively refine nodes based on the overall graph context. Implementations typically require an outer loop or controller that manages graph expansion and invokes the LLM for each node.
Strengths: GoT allows more flexible problem decomposition than linear or tree methods. It can capture complex dependencies and combine evidence from multiple reasoning paths. Besta et al. (2024) showed that GoT improves solution quality on tasks like sorting, benefiting from “synergistic” combination of intermediate thoughts
research-collection.ethz.ch
. For example, combining several partial solutions into a fused answer was 62% more accurate than ToT alone. GoT also enables feedback loops: a later thought can influence an earlier one, mimicking recurrent reasoning.
Weaknesses: Graph reasoning is even more complex to orchestrate than tree search. The user (or system) must define the graph structure, which may require domain expertise. The search over graphs can also blow up combinatorially. Currently, GoT is largely a research framework rather than a turnkey method; building custom graphs and scheduling node evaluations adds overhead. Like ToT, it incurs high inference cost. Additionally, too much flexibility can make the reasoning opaque unless carefully constrained.
Applications and Use Cases: GoT is just emerging. The AAAI 2024 paper by Besta et al. tested it on tasks like sorting and collaborative reasoning where merging partial results helped
research-collection.ethz.ch
. It is also a conceptual foundation for building more complex agentic systems that simulate human-like associative thinking. While not yet common in deployed systems, GoT may guide future LLM toolkits that require fine-grained control over information flow.
Program-Aided Reasoning (PaR / PAL)
Conceptual Foundation: Program-Aided Reasoning embeds formal computation within LLM reasoning by having the model generate code as part of its thought process. The paradigm (also called Program-Aided Language models, PAL) takes advantage of the fact that LLMs can output executable code. Instead of solving a problem purely in natural language, the model is prompted to produce a small program that computes the answer. The LLM’s “thought” thus becomes a mix of explanatory text and code, offloading precise calculation to a programming interpreter
arxiv.org
. This approach is inspired by the reliability and rigor of code execution: it ensures arithmetic and logical steps are done correctly by running them.
Implementation Strategies: In practice, PaR is implemented by examples or instructions that encourage the LLM to output code. For instance, in few-shot prompts each example answer contains Python variable assignments and operations that mirror the reasoning steps
arxiv.org
. At inference, the LLM’s generated code is extracted and executed with an interpreter; the result is used as the final answer. The prompt format might include comments or structured templates (as in PAL) to guide the coding style. No model fine-tuning is required—this is an in-context technique utilizing the model’s existing coding ability.
Strengths: By delegating computation to a program, PaR greatly reduces arithmetic or logical errors. Gao et al. (2022) showed that PAL significantly improves performance on math and symbolic tasks; in some cases it surpasses chain-of-thought accuracy
arxiv.org
ar5iv.labs.arxiv.org
. Programmatic reasoning is deterministic once generated, eliminating run-time fuzziness. It also improves calibration and confidence estimates: LLMs using code tend to “know what they know” more reliably, as shown by Kabra et al. (2023), who found program-aided outputs were better calibrated than pure text CoT
ar5iv.labs.arxiv.org
.
Weaknesses: PaR applies best to problems that can be formulated algorithmically. It relies on the LLM’s ability to write correct code; poorly formatted or wrong code leads to failure. This method requires an execution environment (Python interpreter), which may not be available in all settings. It can also be less interpretable: a user needs to read code, not just natural language steps. Moreover, not all reasoning tasks easily reduce to short programs, limiting PaR’s scope.
Applications and Use Cases: PaR has been applied to arithmetic word problems, quantitative reasoning, and any domain where calculation is key. For example, Gao et al. showed PAL solving math word problems by generating and executing Python scripts
arxiv.org
. It can handle very large numbers or loops precisely. Beyond arithmetic, any scenario where symbolic logic can be encoded (counting, combinatorics) benefits. PaR is gaining traction in competitive programming with LLMs, code generation tasks (e.g. HumanEval with iterative solution checking), and in educational contexts for showing work via code.
ReAct (Reasoning+Acting)
Conceptual Foundation: ReAct integrates external actions into the LLM’s reasoning process. The paradigm recognizes that solving many real-world tasks may require interacting with tools or environments (e.g. searching the web, querying a database). ReAct prompts the model to output not only internal reasoning steps (“Thought: …”) but also explicit actions (“Action: …”) such as API calls or memory writes, in an interleaved fashion
arxiv.org
. This synergy allows the LLM to self-critique and update its plan based on new information from actions. Cognitive inspiration comes from the interplay of thinking and doing: humans often gather data or run experiments as they reason.
Implementation Strategies: A typical ReAct prompt instructs the LLM to solve a task by alternating between reasoning traces and actions. The prompt examples show chains like “Reason: X. Action: search(‘question’). Observation: [result]”. The model is fine-tuned or few-shot prompted with this format so it learns to decide when to “act.” After generating an Action token (e.g. an external query), the system executes the action (such as calling a search API) and feeds the result back (as “Observation”) for the model to continue reasoning. No change to the LLM architecture is needed; the paradigm leverages the LLM as a planner that can invoke tools during inference.
Strengths: ReAct allows LLMs to incorporate fresh, factual information and to check their intermediate conclusions. Yao et al. (2022) demonstrated that ReAct overcomes hallucination in question answering by doing web searches on the fly
arxiv.org
. On tasks like HotpotQA or factual verification, ReAct improved accuracy and produced more interpretable, human-like reasoning chains
arxiv.org
. It also excels at decision-making and interactive tasks: in benchmarks like ALFWorld and WebShop, ReAct agents outperformed both vanilla prompting and RL baselines, simply by using a few in-context examples
arxiv.org
. The paradigm increases robustness and trustworthiness, since the model can justify its answers with observed evidence.
Weaknesses: ReAct depends on the availability and reliability of external tools or APIs. Designing the right set of actions for a task (search, calculators, simulators) is manual work. Each action incurs additional latency and requires an execution environment. If the action results are poor or misleading, ReAct can propagate errors. Also, as the model can make many actions, controlling and constraining its behavior is challenging (it could get stuck in loops or query indefinitely).
Applications and Use Cases: ReAct is well-suited for open-domain question answering, fact-checking, and any task requiring grounded information. In Yao et al.’s experiments, ReAct was applied to HotpotQA (multi-hop QA), FEVER (fact verification), and interactive text-based games
arxiv.org
. More broadly, it underlies agentic LLM systems that use tools (e.g. ChatGPT with web browsing or code execution plugins). ReAct-style prompting is a current best-practice for building LLM-based assistants that must “look things up” or execute tasks beyond pure text generation.
Web-of-Thought (Conceptual)
Conceptual Foundation: The term Web-of-Thought is not a formalized LLM paradigm yet but represents the idea of reasoning over networked connections rather than linear chains. In some industry white papers, it is proposed as an alternative to CoT for domains like biotech, where knowledge is inherently graph-structured
lumagroup.com
lumagroup.com
. In this view, a “web” of thought means the reasoning process can traverse multiple paths in a knowledge graph, checking constraints and dependencies (e.g. causal or logical links) rather than compressing everything into a single chain
lumagroup.com
. This concept echoes Graph-of-Thought but emphasizes using external structured knowledge as part of reasoning.
Implementation Strategies: Web-of-Thought is largely a design philosophy. Implementing it could involve combining LLMs with knowledge graphs or multi-hop retrieval systems. For instance, a system might retrieve related facts, lay them out as nodes in a graph, and use LLM queries to explore paths between them. It may also require custom user interfaces or agent frameworks that guide the LLM to consider multiple knowledge trails concurrently. No standard “Web-of-Thought” toolkit exists yet.
Strengths and Weaknesses: The web-of-thought approach could enable reasoning that respects complex domain structure (e.g. biomedical networks), potentially improving correctness in specialized fields. Its weakness is that it is still a vague concept without mature techniques; it inherits the complexity of graph search. It may require substantial engineering to implement knowledge graphs and integrate LLMs effectively.
Applications and Use Cases: The idea is appealing for domains like science or medicine, where linking evidence from many sources is crucial. For example, LumaGroup suggests a web-of-thought could be used to link drug trial data, biology, and market analysis for biotech decisions
lumagroup.com
. As of now, true web-of-thought systems are more visionary than deployed; however, elements of this concept appear in retrieval-augmented generation (RAG) and multi-hop QA systems.
Comparative Analysis
To compare these paradigms, we consider key dimensions:
Paradigm	Structure	Scalability	Interpretability	Modularity	Reasoning Depth
CoT (Chain)	Linear text sequence	Scales easily to any prompt size; cost ∝ length of chain	High: produces explicit steps for user inspection
aclanthology.org
Low: single sequence, no external tools	Moderate: sequential, limited lookahead
ToT (Tree)	Tree of possibilities	Less scalable: branching search is computationally expensive	Moderate: allows tracing branches, but large trees are complex	Medium: supports sub-problem exploration	High: explores alternatives with backtracking
arxiv.org
GoT (Graph)	Directed graph of thoughts	Lower scalability: arbitrary graph can grow very complex	Moderate: graph structures are interpretable but complex	High: combines many sub-thoughts and feedback loops
research-collection.ethz.ch
Very high: models rich dependencies and loops
Scratchpad	Linear steps (with explicit memory)	Similar to CoT: only more tokens, modest overhead	High: shows intermediate results clearly	Low: monolithic reasoning steps	Moderate: like CoT, extends memory for long tasks
PaR (Program)	Mixed NL and code	Scales to any computable reasoning; cost of execution	Moderate: code is explicit but less accessible to lay readers	Medium: external interpreter as module	High: exact computation enables deep reasoning steps
ReAct	Alternating Reason/Action	Lower: depends on number of actions and tool calls	High: explicit actions and reasoning create a clear trace
arxiv.org
High: by design, uses external tools and memory	High: leverages tools and iterative feedback
arxiv.org
Scalability: Simple text-based methods (CoT, scratchpad) scale easily to large prompts or models, since they only require more tokens. ToT and GoT involve search and orchestration overhead, so they scale poorly to very large branching factors. PaR scales well for algorithmic tasks (the interpreter can handle large workloads), but its scope is limited to computable problems. ReAct’s cost depends on the number and latency of tool/API calls.
Interpretability: All paradigms aim to reveal reasoning, but to different extents. Chains and scratchpads are straightforward to follow. Trees and graphs produce richer traces but can become complex for users. Code in PaR is precise but requires programming knowledge. ReAct yields highly interpretable reasoning-action logs, making the LLM’s behavior transparent
arxiv.org
.
Modularity: ReAct and GoT are highly modular: they explicitly incorporate external modules (tools or sub-thoughts) and allow mixing different reasoning styles. PaR is also modular in offloading computation. CoT and scratchpad are monolithic flows with minimal external structure.
Reasoning Depth: ToT and GoT are designed to handle deeper, more complex reasoning by exploring many possibilities
arxiv.org
research-collection.ethz.ch
. PaR can be very deep when powerful algorithms are encoded in code. CoT and scratchpad depth is limited by prompt length and coherence. ReAct can achieve great depth by iterating with feedback and new information
arxiv.org
.
Impact on LLM Capabilities
Each paradigm influences LLM performance and behavior:
Enhanced Multi-step Reasoning: Methods like CoT, scratchpad, ToT, and GoT explicitly support multi-step inference, enabling LLMs to solve problems that single-pass generation cannot. Tree/graph schemes further allow non-greedy search and backtracking, pushing the frontier of solvable tasks
arxiv.org
research-collection.ethz.ch
.
Improved Transparency: Paradigms that expose internal steps (CoT, scratchpad, PaR, ReAct) make LLM reasoning more transparent. Users can see how the answer was derived, which aids debugging and trust. Indeed, CoT has been praised for offering “more transparent reasoning”
aclanthology.org
, and ReAct produces “human-like trajectories that are more interpretable” than vanilla outputs
arxiv.org
.
Greater Reliability: PaR and ReAct in particular improve reliability by leveraging deterministic computation. Program execution removes arithmetic errors, and tool lookups ground answers in real data, reducing hallucinations. ReAct has been shown to correct hallucinations in QA by consulting a knowledge source
arxiv.org
. Program-aided methods also yield better calibration (the model “knows when it is right”)
ar5iv.labs.arxiv.org
.
Broader Functionality: By incorporating search (ToT, GoT) and tools (ReAct, PaR), LLMs can tackle a wider array of tasks – from puzzles and creative writing to code generation and interactive environments. These paradigms effectively turn LLMs into hybrid systems that can plan, act, and compute.
Emerging Trends
Self-Consistency and Verification: Sampling multiple reasoning chains and aggregating them (self-consistency) is becoming common to reduce error. Iterative self-checking (verifiers, bidirectional reasoning) is also explored.
Reflection and Self-Improvement: Architectures like Reflexion introduce a form of self-reflection, where LLM agents use feedback to iteratively refine their strategy. In Reflexion, agents maintain an internal “verbal memory” of feedback and use it to improve subsequent reasoning
arxiv.org
. This kind of reinforcement through language is an active research direction.
Hybrid and Multi-Agent Systems: Combining paradigms (e.g. CoT + ReAct, ToT with tools) is on the rise. Systems may use CoT for local steps but fallback to code or search when appropriate. Multi-agent schemes, where different models or instantiations handle sub-tasks (or critique each other), are also being developed.
Knowledge-Graph Integration: Drawing on the web-of-thought idea, research is exploring tighter integration of LLMs with structured knowledge bases or multi-hop retrieval (“knowledge-augmented reasoning”). This could manifest as prompts that traverse knowledge graphs, or LLMs that reason over graph representations.
Efficiency and Calibration: There is growing focus on making these paradigms more efficient (fewer calls or tokens) and trustworthy. For example, studies show that program-aided outputs not only boost accuracy but also produce better-calibrated confidence estimates than CoT
ar5iv.labs.arxiv.org
.
In summary, the state of the art in LLM reasoning is rapidly evolving. Researchers continue to invent and refine “thinking” paradigms that go beyond one-shot generation. By structuring inference – through chains, trees, graphs, programs, or actions – modern LLMs are gaining more systematic problem-solving abilities. This has led to significant empirical gains on benchmarks and holds promise for more complex, reliable AI applications. Each paradigm brings different trade-offs, and hybrid approaches are likely to define future systems, combining interpretability, depth, and practical utility for advanced reasoning tasks. Sources: Chain-of-Thought prompting
arxiv.org
aclanthology.org
; Scratchpad methods
arxiv.org
; Tree-of-Thought search
arxiv.org
; Graph-of-Thought framework
research-collection.ethz.ch
; Program-Aided Reasoning
arxiv.org
ar5iv.labs.arxiv.org
; ReAct framework
arxiv.org
arxiv.org
; Web-of-Thought concept
lumagroup.com
; Reflexion self-reflection
arxiv.org
.
Citations

https://arxiv.org/pdf/2201.11903

Beyond Chain-of-Thought: A Survey of Chain-of-X Paradigms for LLMs

https://aclanthology.org/2025.coling-main.719.pdf

https://arxiv.org/pdf/2201.11903

Beyond Chain-of-Thought: A Survey of Chain-of-X Paradigms for LLMs

https://aclanthology.org/2025.coling-main.719.pdf

[2112.00114] Show Your Work: Scratchpads for Intermediate Computation with Language Models

https://arxiv.org/abs/2112.00114

https://arxiv.org/pdf/2305.10601

https://arxiv.org/pdf/2305.10601

Graph of Thoughts: Solving Elaborate Problems with Large Language Models

https://www.research-collection.ethz.ch/entities/publication/d7218422-4c53-43a5-bdf7-d67746357962

Graph of Thoughts: Solving Elaborate Problems with Large Language Models

https://www.research-collection.ethz.ch/entities/publication/d7218422-4c53-43a5-bdf7-d67746357962

Graph of Thoughts: Solving Elaborate Problems with Large Language Models

https://www.research-collection.ethz.ch/entities/publication/d7218422-4c53-43a5-bdf7-d67746357962

PAL: Program-aided Language Models

https://arxiv.org/pdf/2211.10435

[2311.09553] Program-Aided Reasoners (Better) Know What They Know

https://ar5iv.labs.arxiv.org/html/2311.09553v1

[2311.09553] Program-Aided Reasoners (Better) Know What They Know

https://ar5iv.labs.arxiv.org/html/2311.09553v1

[2210.03629] ReAct: Synergizing Reasoning and Acting in Language Models

https://arxiv.org/abs/2210.03629

[2210.03629] ReAct: Synergizing Reasoning and Acting in Language Models

https://arxiv.org/abs/2210.03629

[2210.03629] ReAct: Synergizing Reasoning and Acting in Language Models

https://arxiv.org/abs/2210.03629

The Limits of Generic LLMs: Why Biotech Needs Purpose-Built Tools - LumaGroup

https://lumagroup.com/the-limits-of-generic-llms-why-biotech-needs-purpose-built-tools/

The Limits of Generic LLMs: Why Biotech Needs Purpose-Built Tools - LumaGroup

https://lumagroup.com/the-limits-of-generic-llms-why-biotech-needs-purpose-built-tools/

Graph of Thoughts: Solving Elaborate Problems with Large Language Models

https://www.research-collection.ethz.ch/entities/publication/d7218422-4c53-43a5-bdf7-d67746357962

[2303.11366] Reflexion: Language Agents with Verbal Reinforcement Learning

https://arxiv.org/abs/2303.11366
All Sources

arxiv

aclanthology

research...tion.ethz

ar5iv.labs.arxiv

lumagroup

Paper 3:

A Comprehensive Taxonomy of Reasoning Paradigms in Large Language Models
Foundational Thought Structures: From Linear Chains to Networked Graphs
The evolution of large language models (LLMs) from fluent text generators to sophisticated reasoners has been marked by a fundamental shift away from monolithic next-token prediction towards explicit, multi-step problem-solving. At the heart of this transformation lies a series of increasingly complex structural paradigms designed to organize and guide a model's internal deliberation process. These foundational structures—Chain-of-Thought, Tree-of-Thought, and Graph-of-Thought—represent a deliberate effort to mimic and enhance human-like cognition, moving from simple sequential reasoning to intricate, networked exploration. Each paradigm introduces new mechanisms for evaluation, selection, and synthesis, enabling LLMs to tackle problems of greater complexity and nuance. However, this progression is not without significant trade-offs, particularly concerning computational cost, implementation complexity, and the inherent limitations of relying solely on internal knowledge. Understanding these foundational paradigms is essential for appreciating the broader landscape of modern LLM reasoning, as they form the bedrock upon which more advanced, agentic systems are built.

The Chain-of-Thought (CoT) paradigm represents the initial and most influential breakthrough in eliciting structured reasoning from LLMs 
. Introduced by Wei et al. in 2022, CoT operates on the principle of prompting an LLM to generate intermediate reasoning steps before arriving at a final answer 
. This technique effectively breaks down a complex problem into a sequence of smaller, logically connected sub-problems, guiding the model through a step-by-step thought process 
. The conceptual foundation of CoT is deeply rooted in cognitive science, specifically mimicking the way humans verbalize their thought processes to solve problems, thereby enhancing coherence and reducing errors 
. Its primary implementation strategy involves modifying the input prompt to include examples of this intermediate reasoning, a method known as few-shot CoT 
. An alternative, zero-shot approach simply appends a directive like "Let's think step by step" to the user's query, which has been shown to be effective even without providing demonstration examples 
. The impact of CoT on LLM capabilities has been profound, significantly improving performance on arithmetic, commonsense, and symbolic reasoning benchmarks where multi-step inference is required 
. For instance, on the GSM8K math benchmark, CoT improved PaLM's accuracy from a baseline of 17.9% to 58.1% 
. It also enhances transparency and interpretability, making the model's decision-making process more visible and auditable, which is critical for high-stakes applications in fields like law and medicine 
.

Despite its success, Chain-of-Thought is fraught with limitations. Its greatest weakness is its susceptibility to hallucination; because it relies exclusively on the model's internal knowledge, it can produce plausible-sounding but factually incorrect reasoning chains 
. An extensive error analysis revealed that while CoT reduces some failure modes compared to direct input-output prompting, it introduces new ones related to factual inaccuracies, with one study finding that 56% of CoT errors involved hallucinated facts or reasoning 
. Furthermore, the faithfulness of CoT rationales is a subject of debate. Research indicates that generated explanations often do not accurately reflect the model's true internal decision-making process, sometimes serving as post-hoc rationalizations rather than genuine deliberation 
. This unfaithful behavior can be detrimental, as biased contexts can lead the model to select an incorrect answer despite a correct CoT rationale 
. Another critical limitation is its strong dependence on model scale. CoT emerges as an emergent ability in sufficiently large models, typically those with over 60 billion parameters, and shows minimal to no improvement, or even negative impacts, on smaller models 
. To address these shortcomings, several advanced variants have been developed. Self-Consistency improves upon standard CoT by generating multiple diverse reasoning paths for a single problem and selecting the most consistent final answer through majority voting 
. This ensemble method has been shown to boost performance by up to 23 percentage points on larger models 
. Auto-CoT automates the creation of high-quality demonstration examples, clustering questions and using zero-shot CoT to generate reasoning chains, which has been found to match or exceed manually designed prompts across ten datasets 
. More recently, Deeply Understanding the Problems (DUP) was introduced as a zero-shot prompting method that reframes the problem before applying reasoning by first extracting the core question and relevant information, which has proven highly effective at mitigating semantic misunderstanding errors and achieving state-of-the-art results on GSM8K 
.

Building upon the linear structure of CoT, the Tree-of-Thought (ToT) framework generalizes the concept by modeling the reasoning process as a tree, where each node represents a coherent intermediate step (a 'thought') and branches represent possible alternatives 
. Proposed by Yao et al. in 2023, ToT enables LLMs to explore multiple reasoning paths simultaneously, self-evaluate their progress, and backtrack when a path appears unpromising 
. This capability allows for strategic lookahead and deliberate decision-making, mimicking human problem-solving behaviors seen in activities like chess or creative writing 
. The implementation of ToT involves an iterative loop guided by a search algorithm, such as Breadth-First Search (BFS) or Depth-First Search (DFS) 
. At each step, a 'propose prompt' generates candidate thoughts, which are then evaluated by a 'value prompt' to determine their potential to lead to a solution 
. Promising thoughts are retained to expand the tree, while unpromising ones are pruned 
. Empirically, ToT demonstrates a dramatic performance uplift over CoT on tasks requiring deep exploration and planning. In the Game of 24 puzzle, ToT achieved a 74% success rate, vastly outperforming CoT's 4% and Input-Output (IO) prompting's 7.3% 
. Similarly, in creative writing tasks, ToT-generated passages were rated as more coherent than both IO and CoT baselines 
. However, this enhanced capability comes at a steep computational cost. Exploring multiple paths requires substantially more API calls and tokens, leading to increased latency and operational expense 
. For example, a ToT-based creative writing task cost approximately five times more than IO or CoT, although the superior performance justified the investment for difficult problems 
.

Further generalizing the idea of structured reasoning, the Graph-of-Thoughts (GoT) paradigm models the reasoning process as an arbitrary graph, where nodes are thoughts and edges represent dependencies between them 
. Unlike the strict hierarchical and acyclic nature of trees, graphs allow for non-linear relationships, including cycles, feedback loops, and nodes with multiple parents 
. This flexibility better captures the interconnected and often recursive nature of human cognition. GoT introduces several novel thought transformations beyond simple generation and evaluation. Key among these are Aggregation, which combines multiple thoughts into a new one, and Refinement, which iteratively improves a single thought through a self-loop 
. This architecture enables richer, more synergistic reasoning, allowing insights from one branch of thought to inform and improve another 
. The practical benefits of GoT are significant. By allowing thoughts to influence multiple downstream analyses, GoT achieves a higher degree of information integration (referred to as 'volume') without sacrificing speed ('latency'), a trade-off that makes it uniquely efficient 
. Experimental evaluations confirm this superiority. On sorting tasks, GoT improved outcome quality by 62% compared to ToT while reducing computational costs by over 31% 
. In keyword counting and document merging tasks, GoT also demonstrated superior accuracy and efficiency compared to both CoT and ToT 
. Despite its advantages, GoT presents challenges in implementation complexity and managing dense, interconnected graphs, which can be difficult to visualize and prune effectively 
. Nevertheless, its capacity for flexible, networked reasoning positions it as a powerful paradigm for tackling complex, synthesis-heavy problems that lie beyond the reach of linear or strictly hierarchical structures.

Structure Topology
Linear Chain (Path Graph)
Hierarchical Tree (Acyclic Connected Undirected Graph)
Arbitrary Graph (Allows cycles, multiple parents)
Core Mechanism
Sequential generation of intermediate steps
Exploration of multiple paths with lookahead and backtracking
Flexible dependency management, aggregation, and feedback loops
Evaluation Strategy
Primarily implicit, inferred from logical flow
Explicit evaluation of states using heuristics (e.g., 'sure/maybe/impossible')
Explicit scoring, validation, and refinement of individual thoughts
Search Algorithm
Not applicable (single path)
Breadth-First Search (BFS), Depth-First Search (DFS), Beam Search
Varies based on implementation; can involve traversal algorithms
Key Strength
Simplicity, low computational cost, high interpretability
Superior performance on complex, exploratory tasks; strategic planning
High flexibility, superior efficiency on synthesis tasks, cognitive fidelity
Primary Weakness
Susceptible to hallucination; no backtracking
High computational cost; risk of "overthinking"
Implementation complexity; difficulty visualizing dense graphs
Representative Task
Grade-school math word problems (GSM8K)
Game of 24 puzzle, creative writing
Sorting, set operations, document merging

Grounding Reasoning in Reality: Action, Tools, and Programmatic Execution
While structured thought paradigms like Tree-of-Thought revolutionized the internal logic of LLM reasoning, they shared a common vulnerability: a reliance on potentially flawed internal knowledge. This limitation became a major bottleneck, especially for tasks requiring factual accuracy, dynamic information, or precise computation. In response, a parallel track of innovation emerged, focused on grounding abstract reasoning in external reality. This movement gave rise to three distinct yet complementary paradigms: ReAct (Reasoning and Acting), Scratchpad Reasoning, and Program-Aided Reasoning (PaR). These approaches fundamentally alter the relationship between an LLM and the world, shifting it from a closed-book examination to an interactive, tool-using agent. By integrating external actions, programmatic execution, and specialized memory buffers, these paradigms enable LLMs to reduce hallucinations, perform verifiable calculations, and navigate complex, real-world environments. They represent a crucial step toward building more reliable, capable, and trustworthy AI systems that can operate effectively beyond the confines of their training data.

The ReAct (Reasoning and Acting) framework, introduced by Yao et al. in 2022, is a transformative paradigm that synergistically combines internal reasoning traces with external actions 
. Its core insight is that a model's ability to plan and adapt hinges on its capacity to interact with the environment and learn from the outcomes of its actions 
. The ReAct loop consists of three alternating components: 'Thought', where the model reasons about the current state and plans a next action; 'Action', where it executes a concrete operation using an external tool (e.g., a web search, a database lookup, or an API call); and 'Observation', where it receives the result of the action and incorporates it into its context for the next reasoning step 
. This iterative cycle allows the model to dynamically decompose goals, retrieve missing information, verify facts, and adjust its strategy in real-time 
. The conceptual foundation of ReAct draws inspiration from human problem-solving, where thinking and doing are intertwined; we don't just formulate a plan in our heads—we act, observe the result, and refine our plan accordingly 
. Its implementation is typically achieved through few-shot in-context learning, where a small number of exemplar trajectories demonstrating the interleaved thought-action pattern are provided in the prompt 
. ReAct has proven highly effective across a range of domains. In multi-hop question answering on HotpotQA, it significantly outperforms both pure reasoning and pure action baselines 
. In embodied navigation tasks like ALFWorld, an agent successfully located an object by reasoning about its possible location, performing a search, observing the result, and adjusting its plan 
. Similarly, in online shopping simulations, ReAct agents could successfully purchase items by reasoning about constraints and interacting with a simulated website 
. The primary strength of ReAct is its ability to ground reasoning in external evidence, which drastically reduces hallucination 
. However, its weaknesses include increased computational overhead from iterative loops, rapid token accumulation that can hit context window limits, and sensitivity to the quality of external tools 
.

Scratchpad Reasoning refers to the practice of allowing an LLM to use an internal memory buffer, or 'scratchpad,' to store intermediate computations and notes during a reasoning process 
. This concept addresses the challenge of maintaining state and tracking complex calculations across multiple steps, which can be difficult for a purely autoregressive model. The scratchpad serves as a temporary workspace where the model can jot down partial results, variables, or key facts, analogous to how a human might use a physical scratchpad to work through a math problem. This internal state management enhances accuracy in multi-step calculations and planning by offloading part of the cognitive burden from the model's attention mechanism 
. One prominent implementation involves supervised fine-tuning, where models are explicitly trained to emit intermediate computational steps into a scratchpad format 
. Studies have shown that this approach can dramatically improve accuracy, for instance, boosting performance from 35% to 95% on addition tasks 
. While less frequently discussed as a standalone paradigm compared to others, scratchpad reasoning underpins many other techniques and is essential for maintaining coherence in long, complex reasoning chains. Its main advantage is the enhancement of accuracy in computationally intensive tasks, while its disadvantage is the added complexity of managing the scratchpad state and the need for specialized training or prompting strategies 
.

Program-Aided Language Models (PAL), introduced by Gao et al. in 2022, take the idea of grounding to its logical extreme by completely separating reasoning from computation 
. In this paradigm, the LLM's role is to translate a natural language problem into an executable program, typically in a language like Python 
. This generated code is then passed to an external symbolic interpreter (like a Python runtime) for execution, and the interpreter's deterministic output becomes the final answer 
. This neuro-symbolic approach leverages the LLM's strengths in natural language understanding and program synthesis while delegating the exacting task of calculation to a reliable interpreter 
. The conceptual foundation is that many reasoning errors in LLMs stem from weaknesses in arithmetic and logic, even when the underlying reasoning chain is correct 
. By offloading computation, PAL avoids these common pitfalls entirely. Its implementation involves a prompting strategy that instructs the model to generate code instead of a direct answer, followed by secure execution in a sandboxed environment 
. The empirical results for PAL are striking. On the GSM8K math benchmark, PAL with Codex achieved 72.0% accuracy, surpassing the 65.6% of CoT with PaLM-540B by a remarkable 15 points 
. Crucially, PAL demonstrates exceptional robustness to numerical complexity. When tested on GSM-HARD, a modified version of GSM8K with much larger numbers, CoT performance dropped to around 20%, while PAL maintained an accuracy of 61.2% 
. This resilience extends to symbolic and algorithmic tasks, where PAL achieves near-perfect accuracy on problems involving date understanding, object counting, and algorithmic sorting 
. The quality of the generated code is paramount; meaningful variable names and comments significantly improve performance by grounding symbols to real-world entities 
. While highly effective for precise tasks, the limitations of PAL include its dependency on the model's ability to generate correct code syntax and the security risks associated with executing arbitrary code, necessitating strict sandboxing 
. It is also unsuitable for problems that cannot be translated into a formal programmatic representation.

ReAct (Reasoning + Acting)
Interleaving internal reasoning with external tool interactions to enable dynamic, grounded problem-solving
.
Few-shot in-context learning with exemplars showing a 'Thought → Act → Observe' loop
.
Reduces hallucination; enables interaction with external environments; improves transparency
.
Increased computational cost; sensitive to tool reliability; difficult debugging of long agent trajectories
.
Scratchpad Reasoning
Using an internal memory buffer to store intermediate computations, aiding state management in complex reasoning
.
Supervised fine-tuning to train models to emit intermediate steps, or specialized prompting strategies
.
Improves accuracy in multi-step calculations and planning; enhances traceability of intermediate results
.
Requires additional implementation complexity; may increase computational overhead
.
Program-Aided Reasoning (PaR)
Separating reasoning (natural language translation) from computation (program execution) to ensure precision
.
Prompting the LLM to generate executable code (e.g., Python), which is run in an external interpreter
.
Eliminates arithmetic/logical errors; highly robust to numerical complexity; excellent for symbolic tasks
.
Dependent on code generation quality; security risks from code execution require sandboxing; not all problems are programmable
.

The Emergence of Agentic Systems: Synergizing Reasoning and Interaction
The culmination of advancements in structured thought and grounded reasoning is the rise of sophisticated agentic systems. These systems move beyond single-model problem-solving to create autonomous agents capable of perceiving their environment, forming plans, executing actions, and reflecting on their own performance. The ReAct framework serves as a foundational blueprint for these agents, but recent research has pushed far beyond its original scope. Modern agentic architectures integrate multiple reasoning paradigms, leverage external knowledge bases, and incorporate mechanisms for self-correction and continuous learning. This evolution marks a significant leap towards artificial intelligence that is not only capable of solving complex problems but can do so in a dynamic, adaptive, and increasingly reliable manner. These systems are being deployed in interactive environments, from virtual worlds to real-world simulations, and are pushing the boundaries of what is possible for LLMs in decision-making and task completion. However, the increased autonomy and complexity of these agents also introduce new challenges related to scalability, safety, and trustworthiness, which are now central areas of research.

The ReAct framework, with its core loop of 'Thought → Act → Observe', provides the fundamental mechanism for agentic behavior 
. By interleaving reasoning and action, ReAct agents can engage with external environments, such as APIs, knowledge bases, or interactive game worlds, to gather information and achieve goals 
. For example, an agent using ReAct can perform a web search to find information needed to answer a question, then synthesize that information to provide a final response 
. This synergy between reasoning and acting is demonstrated in various benchmarks. On the HotpotQA dataset, which requires multi-hop reasoning across documents, ReAct agents consistently outperform both reasoning-only and action-only baselines by using reasoning to decide what to search for and using the search results to inform subsequent reasoning steps 
. In the ALFWorld environment, which simulates household tasks, a ReAct agent successfully navigated the environment to locate an object by reasoning about its possible locations, performing searches in cabinets and on countertops, and adapting its plan based on the observations from each action 
. Similarly, in the WebShop environment, a ReAct agent completed a complex purchase task by reasoning about product attributes, searching for matching items, and executing the transaction 
. The primary advantage of this approach is its ability to reduce hallucination by grounding every reasoning step in observable evidence from the external world 
. However, ReAct alone faces challenges with long, multi-step tasks, where the agent can get stuck in loops or fail to handle unexpected environmental changes 
. Furthermore, the complexity of the agent's trajectory can make debugging difficult, and its performance is highly dependent on the quality of the available tools 
.

To overcome the limitations of basic ReAct agents, researchers have developed more advanced frameworks that incorporate reflection, memory, and collaborative problem-solving. Reflexion builds directly upon ReAct by adding a self-reflection component 
. After each trial, a separate evaluator module critiques the agent's past actions and stores the resulting insights in an episodic memory buffer. This allows the agent to learn from its mistakes and iteratively improve its strategy over time 
. On the ALFWorld benchmark, combining ReAct with Reflexion led to a substantial improvement, with the agent completing 130 out of 134 tasks compared to only about half for ReAct alone, showcasing the power of iterative self-improvement 
. Another significant advancement is Retrieval-Augmented Reflection (RaR), a prompting strategy that allows an agent to revise its own reasoning trace by retrieving relevant information at each intermediate step 
. Instead of a single pass, RaR uses a causal, step-by-step revision process where the agent generates an initial CoT, then for each thought, it retrieves information relevant to that step, and revises the thought before proceeding. This enables the correction of earlier errors without having to restart the entire reasoning process, a feat impossible with standard generation 
. RaR has demonstrated significant performance gains on benchmarks like GSM8K and HumanEval, and notably, it allows smaller language models to surpass larger ones by leveraging increased inference-time compute 
.

The orchestration of multiple functions and the decomposition of complex tasks are also critical for building scalable agentic systems. LLMCompiler is a framework designed to efficiently manage function calls by decomposing a complex user request into a Directed Acyclic Graph (DAG) of smaller, inter-dependent tasks 
. The planner component generates this DAG, identifying independent tasks that can be executed in parallel. The executor then asynchronously runs these tasks, and the system supports dynamic replanning if new information becomes available. This structured planning approach dramatically improves efficiency and robustness over sequential agents like ReAct. On the Movie Recommendation benchmark, LLMCompiler achieved a 3.74x speedup and a 6.73x cost reduction compared to ReAct 
. It effectively addresses two major failure modes of ReAct: premature early stopping, where the agent stops searching too soon, and repetitive function calls, where the agent gets stuck in redundant loops 
. Multi-agent collaboration represents another frontier, where a single complex task is distributed among multiple specialized agents. Frameworks like AutoGen and GPTSwarm facilitate this by assigning different roles (e.g., programmer, inspector, researcher) to different agents, who collaborate to solve a problem 
. This modular approach reduces the cognitive load on any single agent and improves accuracy by leveraging specialized expertise 
. Finally, the integration of knowledge bases is essential for grounding agents in factual reality. Retrieval-Augmented Generation (RAG) is a common technique used to augment an agent's prompt with retrieved information before it begins reasoning 
. More advanced schemes like IRCoT and SELF-RAG interleave retrieval and reasoning steps throughout the process, ensuring that the agent grounds its reasoning in the most relevant information at each stage 
. These combined advancements—from reflection and revision to structured planning and multi-agent collaboration—are creating a new class of intelligent systems that can autonomously tackle complex, open-ended problems in a wide range of domains.

Comparative Analysis of Reasoning Paradigms
A comparative analysis of the diverse reasoning paradigms reveals a complex landscape defined by a series of critical trade-offs between performance, computational cost, interpretability, and scalability. There is no single "best" paradigm; rather, the optimal choice depends heavily on the specific requirements of the task at hand. Simple, sequential problems may be best solved with Chain-of-Thought due to its efficiency and transparency, whereas complex, exploratory challenges demand the deeper search capabilities of Tree-of-Thought or Graph-of-Thought. Knowledge-intensive applications benefit immensely from the grounding provided by ReAct and Program-Aided Reasoning. This section provides a detailed comparison of these paradigms across several key dimensions, highlighting their relative strengths and weaknesses to guide the selection and design of reasoning systems.

One of the most fundamental trade-offs is between reasoning depth and computational cost. At one end of the spectrum are linear paradigms like Chain-of-Thought (CoT) and its variants. Their sequential nature makes them computationally inexpensive, as they require a single forward pass through the model to generate a complete reasoning chain 
. This efficiency makes them ideal for simple tasks or scenarios where low latency is critical. However, their lack of exploration means they can easily get trapped on a wrong path and offer limited opportunities for backtracking 
. Moving up the complexity ladder, Tree-of-Thought (ToT) and Graph-of-Thought (GoT) introduce branching and exploration, which significantly increases their computational footprint. ToT, by exploring multiple reasoning paths simultaneously, can require orders of magnitude more tokens and API calls than CoT, especially on difficult problems 
. This "exploration tax" is a necessary cost for gaining access to more robust solutions. GoT aims to mitigate this inefficiency by allowing for the aggregation of thoughts, which can lead to higher information integration with lower latency compared to ToT 
. The introduction of "overthinking"—where models generate excessively long and redundant reasoning traces—is a major concern for all these paradigms, driving research into efficient reasoning techniques that balance depth with cost 
.

Interpretability is another crucial dimension. CoT excels in this area due to its simple, linear structure. The step-by-step chain is easy for humans to read and follow, making it highly transparent 
. This transparency is a key advantage in high-stakes domains like healthcare and finance, where understanding the rationale behind a decision is as important as the decision itself 
. As paradigms become more complex, interpretability becomes more challenging. ToT's tree structure adds a layer of complexity, but the explicit evaluation scores assigned to each node can still provide insight into the model's reasoning process 
. GoT, with its arbitrary graph topology, presents the greatest interpretability challenge. Visualizing a dense graph of interconnected thoughts can be difficult, and tracing the exact flow of information through the network is non-trivial 
. However, the explicit modeling of dependencies can also provide a richer understanding of how conclusions are derived from a web of supporting evidence 
. Grounded paradigms like ReAct and Program-Aided Reasoning enhance interpretability by clearly demarcating between internal reasoning (the 'thought') and external evidence (the 'action' or 'code execution'), allowing users to audit the source of each piece of information 
.

Scalability and modularity are also key considerations. CoT is highly modular and easy to implement as a simple prompting technique, making it accessible to a wide range of users 
. Its performance scales predictably with model size, becoming more effective as models grow larger 
. ToT and GoT, while more powerful, are inherently more complex to implement, often requiring custom code to manage the state of the tree or graph and orchestrate the iterative loops of generation and evaluation 
. This complexity can be a barrier to entry. However, their modular design, which separates the reasoning logic (the generator and evaluator) from the control strategy (the search algorithm), offers great flexibility for customization 
. The emergence of frameworks like LangChain and Semantic Kernel provides abstractions that simplify the implementation of these more advanced paradigms, promoting modularity and reusability 
. Scalability also depends on the underlying architecture. While decoder-only models are generally more scalable for pre-training, encoder-decoder models may offer better performance-efficiency trade-offs after instruction tuning, suggesting that architectural choices will continue to play a role in the scalability of reasoning systems 
.

Finally, performance varies significantly across paradigms depending on the task type. CoT is highly effective for well-defined, sequential reasoning tasks like grade-school math 
. ToT and GoT demonstrate superior performance on tasks that require strategic exploration, planning, or synthesis. ToT shines in puzzles like the Game of 24 and creative tasks where considering multiple options is beneficial 
. GoT's strength lies in problems that involve aggregating information from multiple sources, such as sorting or merging documents 
. For tasks requiring factual grounding, grounded paradigms like ReAct and Program-Aided Reasoning are indispensable. ReAct excels in interactive environments and question-answering where external information is key 
, while Program-Aided Reasoning is unmatched for tasks demanding precise, verifiable computation 
. The table below summarizes these comparative aspects.

Computational Cost
Low (single path)
High (multiple paths)
Moderate to High (depends on graph density)
High (iterative loops)
Variable (cost of code execution vs. generation)
Interpretability
Very High (linear, easy to follow)
High (explicit evaluation scores)
Moderate (complex graph visualization)
High (clear separation of thought/action)
High (auditable code)
Scalability
High (simple, modular)
Moderate (implementation complexity)
Moderate (complexity management)
Moderate (context window limits)
High (delegates computation)
Reasoning Depth
Limited (no backtracking)
High (lookahead, pruning, backtracking)
Very High (aggregation, feedback loops)
Dynamic (grounded in external observation)
Verifiable (deterministic computation)
Best Suited Tasks
Sequential reasoning, math, logic
Planning, puzzles, creative writing
Synthesis, sorting, multi-source reasoning
Interactive QA, embodied tasks, tool use
Precise math, algorithmic manipulation

Hybridization and Advanced Methodologies: Enhancing Performance and Reliability
As the repertoire of individual reasoning paradigms has expanded, the most significant recent advancements have come from their thoughtful combination and the development of methodologies aimed at enhancing performance, reliability, and efficiency. Researchers have discovered that hybrid frameworks, which integrate the strengths of multiple approaches, often outperform any single paradigm in isolation. This trend is particularly evident in the fusion of reasoning with action, where the synergy between internal deliberation and external interaction creates powerful agentic systems. Concurrently, a new wave of research is addressing the inherent unreliability of LLMs through techniques focused on self-correction, verification, and faithfulness. These methods introduce layers of meta-reasoning, allowing models to critique, refine, and validate their own outputs. Finally, the escalating computational cost of deep reasoning has spurred a critical focus on efficiency, giving rise to a vibrant field dedicated to combating "overthinking" and developing cost-effective reasoning strategies. Together, these trends point toward a future of more robust, trustworthy, and economically viable LLM reasoning systems.

Hybrid frameworks represent the natural evolution of reasoning, recognizing that complex problems rarely fit neatly into a single paradigm. The most successful hybrids combine grounded reasoning with structured exploration. For instance, the original ReAct paper demonstrated that a combination of ReAct and Chain-of-Thought yields the best overall results on certain tasks, suggesting that a mix of grounded, action-based reasoning and internal, chain-like reasoning is often optimal 
. Modern agentic systems frequently integrate CoT, ReAct, and Retrieval-Augmented Generation (RAG) into a single, cohesive workflow 
. An agent might first use RAG to retrieve relevant background information, then employ CoT to decompose the user's query into a structured plan, and finally use ReAct to execute the necessary tool calls to carry out that plan 
. This modular composition allows for flexible and powerful problem-solving tailored to the specific demands of the task. Another powerful hybrid approach is Chain-of-Thought with Self-Consistency (CoT-SC), which generates multiple independent reasoning chains and aggregates the results via majority voting 
. This ensemble method improves the reliability of CoT by filtering out erroneous chains, though it comes at the cost of increased computational expense 
. The CPO (Chain-of-Preference Optimization) method takes this further by using preference data from a Tree-of-Thought search to train a CoT model, enabling it to achieve ToT-level performance during inference without the high computational cost of tree search 
. This approach effectively shifts the computational burden from inference to training, offering a compelling trade-off for real-world deployment.

To address the pervasive issue of unreliability, particularly the tendency of LLMs to generate plausible but incorrect rationales (unfaithful reasoning), several advanced methodologies have been developed 
. Self-Refine is a framework where an LLM first generates an initial response, then produces feedback on that response, and finally refines the response based on its own feedback 
. This iterative process of critique and improvement can substantially enhance the quality of outputs in domains like code optimization and constrained generation 
. Reflexion extends this idea by incorporating a linguistic feedback loop where an agent critiques its own past actions and stores reflections in memory to improve future attempts, demonstrating significant gains in interactive decision-making tasks 
. Faithful Chain-of-Thought (F-CoT) tackles the root cause of unfaithfulness by translating natural language queries into a formal symbolic language (e.g., Datalog) before solving them, ensuring that the generated rationale reflects the actual inference path taken by the model . Another powerful approach is the integration of symbolic solvers. Logic-LM translates natural language problems into formal logic, solves them externally with a solver like Z3, and maps the result back, improving performance by 18.4% on logical benchmarks and enhancing faithfulness 
. These methods introduce a layer of verification and self-correction that is crucial for building trust in LLM outputs, especially in high-stakes applications.

The increasing sophistication of reasoning has led to a corresponding rise in computational cost, giving rise to the "overthinking phenomenon," where models generate excessively long and redundant reasoning traces that add little value while increasing latency and expense 
. Efficient reasoning has thus become a critical research area. One approach is prompt-guided length control, where instructions are given to the model to be concise or to limit the number of tokens per step 
. Another is dynamic planning and routing. Frameworks like LLMCompiler decompose tasks into a DAG of parallelizable functions, drastically reducing execution time compared to sequential agents 
. Routing strategies, such as those implemented in Anthropic's Claude 3.7 Sonnet, allow a system to use a quick-response mode for simple queries and reserve deep, multi-step reasoning for more complex ones, balancing performance and cost 
. Latent space reasoning represents a more fundamental approach, aiming to compress reasoning into fewer or no explicit textual steps. Methods like Coconut treat the final-layer hidden states as a "continuous thought" that can be reused as input embeddings, bypassing the quadratic complexity of attention mechanisms and potentially enabling infinite reasoning depth 
. Finally, efficient training data curation is proving to be a powerful lever for performance. Approaches like s1 and LIMO show that carefully selected, high-quality samples (as few as 1k or 817) can outperform models trained on vast, noisy datasets, demonstrating that quality trumps quantity in training for reasoning 
. These efficiency-focused methods are essential for making advanced reasoning practical and scalable for widespread deployment.

Architectural Foundations and Future Frontiers in LLM Reasoning
The capabilities of any reasoning paradigm are ultimately constrained and enabled by the underlying architecture of the Large Language Model it operates on. The ongoing debate between decoder-only and encoder-decoder architectures continues to shape the field, with each presenting unique advantages and disadvantages for reasoning tasks. Beyond architecture, the future of LLM reasoning is being forged by a convergence of trends, including the rise of process supervision and reinforcement learning, the integration of multimodality, and the persistent challenge of scaling reasoning depth. Addressing these frontiers will require not only algorithmic innovation but also a deeper understanding of the cognitive principles that govern effective problem-solving. The ultimate goal remains the creation of systems that can reason robustly, reliably, and scalably, bringing us closer to the vision of artificial general intelligence.

The dominance of decoder-only models (e.g., GPT series) stems from their simplicity, scalability, and effectiveness in pre-training on massive amounts of unsupervised text data 
. Their autoregressive, left-to-right processing is naturally suited for generative tasks and facilitates efficient inference through KV-cache optimization 
. However, recent research suggests that encoder-decoder models (e.g., T5) may hold advantages for certain reasoning and instruction-following tasks. Encoder-decoders possess a bidirectional encoder stack that can process the full input context simultaneously, which is beneficial for tasks requiring strong contextual disambiguation, such as Named Entity Recognition 
. While encoder-decoder models suffer from an information bottleneck between the encoder and decoder stacks and are less efficient for conversational settings, they can achieve comparable or even superior performance to decoder-only models after instruction tuning 
. A comprehensive study found that while decoder-only models are more compute-optimal during pre-training, encoder-decoder models demonstrate superior inference efficiency and a better quality-compute trade-off post-tuning, particularly in handling longer context windows 
. This suggests that the choice of architecture is not absolute but task-dependent, with encoder-decoder models potentially offering a more favorable balance for complex, instruction-rich reasoning scenarios.

Looking ahead, several grand challenges and emerging trends define the future of LLM reasoning. A primary challenge is the apparent collapse of reasoning depth in existing models. Apple researchers have shown that even advanced Large Reasoning Models (LRMs) fail on long-horizon problems like the Tower of Hanoi with 8 or more disks, indicating a breakdown in sustained sequential reasoning regardless of the allocated computational budget 
. This highlights a critical need for external memory mechanisms to help models maintain and update their internal state representations over long sequences of actions 
. The BAR Theorem formalizes this challenge, proving that optimizing for inference-time budget, factual authenticity, and reasoning capacity simultaneously is infeasible, forcing a difficult trade-off between these three properties 
. Future work will likely focus on adaptive budget allocation and progressive retrieval strategies to navigate this triad 
.

In conclusion, the field of LLM reasoning has matured from a handful of simple prompting techniques to a rich and diverse ecosystem of sophisticated paradigms. The journey began with Chain-of-Thought, which established the importance of explicit reasoning steps. This was followed by the structural innovations of Tree-of-Thought and Graph-of-Thought, which introduced exploration, backtracking, and synthesis to mimic more complex cognitive processes. Parallelly, paradigms like ReAct and Program-Aided Reasoning grounded this internal deliberation in external reality, enabling interaction and verifiable computation. The current landscape is characterized by the hybridization of these approaches, creating powerful agentic systems, and a concerted effort to address their weaknesses through self-correction, verification, and efficiency optimizations. The future direction points towards training models with process-level reward signals, integrating multimodal inputs, and developing more robust cognitive architectures capable of overcoming the current scalability limits of reasoning depth. For practitioners, the key takeaway is the necessity of a context-aware, hybrid approach, selecting and composing different reasoning strategies to build systems that are not only intelligent but also reliable, interpretable, and economically viable.


Paper 4: 
Reasoning Paradigms in Large Language Models
Large language models (LLMs) excel at generating fluent text, but rigorous multi-step reasoning requires structured prompting or system designs that mirror human cognitive strategies. Recent research has introduced a variety of “thinking” paradigms that guide LLMs through intermediate reasoning steps. These include linear chains, trees, graphs, programs, and hybrid agentic loops. This report surveys the conceptual foundations, implementations, strengths, and applications of major LLM reasoning paradigms.
Chain-of-Thought (CoT) Prompting
Conceptual Foundation: Chain-of-thought (CoT) prompting was inspired by how humans verbalize intermediate steps in reasoning. In CoT, an LLM is prompted to generate a series of natural-language “thoughts” (rationales) leading to the answer. Wei et al. introduced CoT as a way to elicit latent reasoning abilities in large models
ar5iv.labs.arxiv.org
ar5iv.labs.arxiv.org
. Conceptually, CoT treats each problem as a sequence of sub-problems or steps, much like an explicit reasoning chain in cognitive psychology (akin to System 2 thinking)
ar5iv.labs.arxiv.org
. By giving the model an instruction like “Let’s think step by step,” or by providing few-shot exemplars that include detailed solution steps, the model learns to unpack a task into subtasks
ar5iv.labs.arxiv.org
ar5iv.labs.arxiv.org
. Implementation: In practice, CoT is implemented via prompting. Either a zero-shot prompt (“Let’s think step by step”) or a few-shot prompt with example Q&A pairs that include intermediate reasoning is given to the LLM. For example, Wei et al. showed that including a few math problems solved step-by-step dramatically improves performance on similar tasks
ar5iv.labs.arxiv.org
. No model fine-tuning is required; CoT exploits the LLM’s in-context learning ability. In effect, CoT is a one-shot or few-shot strategy: the prompt is of the form “Q: [question]? A: [reasoning steps] … therefore answer is X.” and the model continues the reasoning. Strengths: CoT can unlock emergent reasoning in very large models. It often leads to dramatic accuracy gains on complex tasks. For instance, prompting PaLM-540B with CoT exemplars attained state-of-the-art on the GSM8K math benchmark, surpassing fine-tuned models by ~15%
ar5iv.labs.arxiv.org
. A key advantage is interpretability: the model’s generated “thoughts” give insight into its reasoning, providing a form of transparency
ar5iv.labs.arxiv.org
. CoT also flexibly allocates computation: a longer chain for harder questions. Importantly, CoT requires no additional training data beyond constructing the prompt, making it easy to apply to new tasks. Weaknesses: CoT’s benefits appear only in sufficiently large models. Smaller models often fail to follow step-by-step instructions or produce coherent chains. Error accumulation is another issue: if an early step is wrong, the chain can lead to a wrong answer. CoT also increases token usage and latency. In some cases, CoT can produce plausible-sounding but incorrect reasoning. CoT alone struggles on tasks requiring global search or planning; it follows one path and cannot easily revise earlier choices. Finally, crafting effective CoT examples can be labor-intensive, and chain-of-thought may degrade if the exemplar steps are noisy or mismatched to the task. Applications: CoT prompting has been widely applied in arithmetic and symbolic reasoning (e.g. math word problems)
ar5iv.labs.arxiv.org
, commonsense QA, logical puzzles, and coding problems. It consistently outperforms direct prompt-answering on benchmarks like GSM8K, SVAMP, and AQuA by tens of percentage points
ar5iv.labs.arxiv.org
arxiv.org
. CoT is also used in engineering “explainable” outputs (e.g. multi-step law, finance or coding answers) since it supplies a rationale. In practice, CoT is often combined with self-consistency (sampling multiple chains and voting) to further boost robustness, though that lies beyond this survey’s scope.
Tree-of-Thought (ToT)
Conceptual Foundation: Tree-of-Thought extends CoT by allowing branching search. Yao et al. observed that linear CoT (strict left-to-right) cannot explore multiple hypotheses or backtrack. Inspired by human problem-solving and classical AI search, ToT treats reasoning as a tree search over possible “thoughts”
ar5iv.labs.arxiv.org
ar5iv.labs.arxiv.org
. Here each node is a complete intermediate thought (a sentence or paragraph), and edges represent “thinking of next step” transitions. This dual-process view uses the LLM’s sequential generation (System 1) combined with a symbolic planning (System 2) that can explore alternatives and evaluate them
ar5iv.labs.arxiv.org
ar5iv.labs.arxiv.org
. Implementation: The ToT framework builds a search tree of candidate partial solutions. At each step, the LLM generates several candidate next thoughts (children), then evaluates or scores them (often by prompting another LLM query: “Is this partial solution promising/sure/maybe/impossible?”). Search algorithms like breadth-first, depth-first, or beam search are applied. For example, in the “Game of 24,” ToT had the model propose possible arithmetic moves, then prune unlikely branches using Commonsense bounds
promptingguide.ai
. Critically, ToT requires iterative prompting: the LLM is used both to generate new thoughts and to evaluate them. Some approaches (Long et al.) train a separate ToT controller with reinforcement learning to decide when to expand or backtrack, but simpler ToT uses fixed search logic
promptingguide.ai
. Strengths: ToT can systematically look ahead and backtrack, solving tasks that CoT fails. In experiments, GPT-4 with standard CoT solved only ~4% of random 24-game instances, whereas ToT (with GPT-4 as the LLM) achieved ~74% success
ar5iv.labs.arxiv.org
. Similarly, ToT dramatically outperforms CoT on creative planning and crossword tasks. By exploring a search tree, ToT can find non-obvious solution paths and correct early mistakes. This leads to far deeper reasoning and planning than a single chain. ToT also retains interpretability: each branch is a human-readable reasoning path, and the tree structure can be inspected. Weaknesses: The main drawback is computation. Expanding and evaluating many branches requires dozens or hundreds of LLM calls per problem, incurring high latency and cost
ar5iv.labs.arxiv.org
. Managing the search (pruning criteria, branch depth) introduces complexity and many hyperparameters (branching factor, beam width, etc.). If not controlled, the tree can grow exponentially. ToT currently requires careful engineering for each task (how to score branches, when to stop). Moreover, in easy tasks where GPT-4 already excels, ToT offers little benefit but still adds overhead
ar5iv.labs.arxiv.org
. Applications: ToT is aimed at “hard” planning or combinatorial tasks. Yao et al. demonstrated ToT on novel benchmarks: Game of 24 (number puzzle), Mini Crosswords, and long-horizon creative writing. It solved many problems that GPT-4 or other CoT methods could not
ar5iv.labs.arxiv.org
. ToT has been applied to math puzzles, program synthesis, planning in robotics (where actions are like tree branches), and any domain where exploring alternatives is needed. Recent work has also examined simpler forms of ToT via prompting (e.g. "three experts collaborate, backtracking if wrong"
promptingguide.ai
), showing that even limited ToT-style prompts can help smaller models.
Graph-of-Thought (GoT)
Conceptual Foundation: Graph-of-Thought generalizes tree search to an arbitrary directed graph of reasoning steps. In GoT, “thoughts” are nodes that can connect in complex ways, reflecting the non-linear nature of human thinking. Besta et al. propose that LLMs benefit from representing intermediate information as a graph, where edges encode dependencies or inference relations
arxiv.org
lumagroup.com
. Unlike trees, a thought can feed into multiple paths or reconverge with others. This models recurrent reasoning and feedback loops akin to neural networks or cognitive maps. Yao et al. (NAACL Findings 2024) similarly note that human deduction often recombines ideas non-sequentially, so modeling reasoning as a graph (not just a chain) yields a more realistic process
aclanthology.org
aclanthology.org
. Implementation: GoT is more of a framework than a single method, so implementations vary. Besta et al. present a prompting-based approach where the LLM generates a rich set of “thoughts” (sentences or facts) and then an algorithm constructs a graph by linking compatible thoughts. For example, in a sorting task, each candidate operation (split, merge) is a node, and edges represent valid transitions. The system then reasons over the graph (with “thought transformations” or graph-walking routines) to reach a solution
arxiv.org
. GoT can also incorporate graph neural networks (GNNs): Yao et al. encode multi-modal QA problems with a Graph-of-Thought module that fuses textual, visual, and graph features via graph attention
aclanthology.org
. Strengths: GoT’s power lies in handling rich interdependencies. In tasks where multiple pieces of evidence must be combined (e.g. multi-hop QA, visual reasoning), GoT can integrate information more flexibly than a single path. Besta et al. report that GoT increased the quality of a sorting solution by +62% over ToT while reducing computation by >31%
arxiv.org
. Yao et al. (GoT for QA) show state-of-the-art results on ScienceQA (multimodal reasoning) and matched GPT-3 accuracy on AQUA-RAT, using far fewer parameters
aclanthology.org
aclanthology.org
. GoT inherently supports deductive reasoning: Yao et al. demonstrate that a model augmented with GoT reasoning can follow complex logical chains, resulting in much higher accuracy on grade-school science questions
aclanthology.org
. As with CoT and ToT, the reasoning graph is interpretable: one can inspect the nodes and edges to understand how the answer was formed. Weaknesses: Constructing and using a reasoning graph adds overhead and complexity. Systems like GoT often require fine-tuning or additional architectures (e.g. GNN encoders) beyond vanilla prompting
aclanthology.org
aclanthology.org
. They also need heuristics for building the graph (which thoughts to create, how to link them). In Besta’s framework, thought generation and graph merging can be expensive, and not all problems naturally form a useful graph. The approach may still rely on the LLM to produce relevant nodes, so errors propagate. Finally, building a GoT is currently labor-intensive (designing the graph structure and transformations), making it less off-the-shelf than CoT. Applications: Graph-of-Thought methods excel in multi-evidence reasoning. Yao et al. used GoT for multimodal QA (ScienceQA) and text reasoning (AQUA-RAT)
aclanthology.org
aclanthology.org
, achieving superior accuracy and surpassing human baselines on ScienceQA. Besta et al. evaluated GoT on algorithmic tasks (sorting, planning) where the solution naturally forms a graph of subproblems
arxiv.org
. In general, any task that involves combining clues (e.g. summarizing documents, logical deduction, complex QA) could benefit from graph-structured reasoning. Emerging work also uses GoT for iterative planning (feedback loops) and could be extended to any domain where LLMs interface with knowledge graphs or external databases.
Web-of-Thought (WoT)
Conceptual Foundation: The “web-of-thought” is a newer, more informal concept referring to fully networked reasoning, rather than any specific algorithm. It emphasizes that knowledge (and reasoning) is highly interconnected, suggesting that LLMs should reason by traversing a network of facts, not a single linear or tree-shaped path. For example, Luma Group proposes that complex domains (like biology) require “multi-hop reasoning with checks,” moving from a linear chain to a “web” of evidence and constraints
lumagroup.com
. In practice, this idea underlies many graph-based methods: a web-of-thought would allow simultaneous exploration of multiple angles, cross-checking facts across domains. Implementation: Web-of-Thought has no single formal implementation yet. It can be seen as a generalized goal of approaches like Graph-of-Thought. One could implement a WoT by building an explicit knowledge graph of relevant concepts and then guiding the LLM to traverse or update that graph. This might use retrieval-augmented methods (retrieving multiple documents as a web of information) combined with in-graph LLM reasoning. Another interpretation is multifaceted prompting: for example, prompting the model to consider several hypotheses in parallel or to draw a “mind map” of ideas. Strengths and Weaknesses: A true web-of-thought approach could in principle capture all dependencies and constraints simultaneously, making reasoning extremely robust. It would be especially valuable in interdisciplinary or highly constraint-laden tasks (like biotech decisions
lumagroup.com
). However, without a concrete algorithm, Web-of-Thought remains a vision: the main challenge is scaling. Representing and searching a full “web” of all relevant facts is combinatorially large. Currently, graph-based reasoning (GoT) is the closest practical form of WoT, but general WoT requires advances in how LLMs integrate external knowledge. Applications: While no benchmark is specifically labeled “web-of-thought,” the concept aligns with tasks like multidocument QA, knowledge integration, and strategic planning where many factors interlock. Systems built on knowledge graphs (e.g. biomedical graphs) with LLM reasoning would exemplify WoT. We can cite WoT conceptually (for its emphasis on networked, evidence-backed reasoning) even if current methods use graph structures. For completeness, WoT can be regarded as an emerging hybrid paradigm that motivates future research.
Program-Aided Reasoning (PaR or PAL)
Conceptual Foundation: Program-aided reasoning uses symbolic programs or external calculators as intermediate reasoning steps. The idea (prominent in works like Gao et al.) is that LLMs may correctly decompose a problem into steps but fail at raw computation or logic. By having the model output a program (in Python or another language) for each step, one can delegate the exact calculation to an interpreter. This is akin to combining System 1 (LLM decomposing in natural language) with System 2 (a deterministic program/symbolic engine)
arxiv.org
. PAL (Program-Aided Language models) treats the LLM as a code-writing “reasoner,” and a runtime environment as the “solver.” Implementation: In PAL, each few-shot example shows a natural-language problem and a corresponding program that solves it step by step. At inference, the LLM is prompted to generate the program code for the new problem. That code is then executed by a Python interpreter to produce the final answer. Importantly, the LLM is not asked to output the numeric answer directly; its task is only to translate the problem logic into code. The heavy lifting (arithmetic, logical operations) is done by the runtime. PAL uses standard few-shot or zero-shot prompting; no model changes are needed. The model and interpreter together constitute a single pipeline. Strengths: Program-aided reasoning ensures correctness of computations. Gao et al. report that PAL using Codex (code-optimized GPT) achieved state-of-the-art few-shot accuracy on GSM8K, outperforming even much larger vanilla LMs
arxiv.org
. By offloading arithmetic or logical steps to a solver, PAL sidesteps LLM hallucinations on those parts. It’s especially powerful for math, physics, algorithms, and any domain where a precise answer is required. Another benefit is efficiency: a shorter prompt can suffice since the LLM doesn’t need to write out all arithmetic, and the answer is computed instantly. PAL also retains interpretability: the generated code is human-readable (at least to programmers). Weaknesses: PAL is domain-restricted. It requires that the problem be programmatically solvable. Tasks outside computation (open-ended language QA, commonsense queries) may not map to a deterministic program. Moreover, the model must be capable of writing correct code – earlier LMs might fail to output syntactically correct or safe code. Integrating an interpreter also introduces security considerations. Another drawback is that PAL depends on external tooling: a Python environment must be available at inference. For some users (e.g. in constrained deployments), this could be impractical. Finally, if the LLM generates incorrect logic (bad code), the pipeline fails; PAL reduces arithmetic errors but still relies on correct problem decomposition. Applications: PAL and similar approaches have been applied to arithmetic word problems, symbolic math, and logic puzzles in benchmarks like Big-Bench Hard. Beyond natural language, PAL could enable LLMs to control data analysis pipelines: e.g. “Write a Python script that reads this dataset and computes the required metric.” In robotics or planning, a variant is to have LLMs output executable actions. The general idea of program-aided reasoning has become a standard technique for math-intensive tasks, and many coding-assistance tools use a similar principle (LLM generates code, execution verifies correctness).
Scratchpad Reasoning
Conceptual Foundation: Scratchpad reasoning involves having the LLM explicitly emit intermediate computation steps (“scratchwork”) while answering. Unlike CoT (which is also intermediate steps), “scratchpad” often refers to unstructured arithmetic work or notes akin to how we write calculations on paper. Nye et al. showed that models trained (or prompted) to produce a scratchpad dramatically improve on multi-step calculation tasks
arxiv.org
. Conceptually, a scratchpad is simply giving the LLM permission to think openly: “Show your work before giving the answer.” Implementation: There are two main modes. One is to fine-tune (or prompt) the model such that its output contains a scratchpad: e.g. asking “Answer in detail, showing all intermediate steps.” The arXiv paper describes training transformers to emit tokens on a scratchpad, essentially augmenting the output with hidden state. Another approach is to simply add a sentence in the prompt like “Please compute 237 + 589 as follows: [work] … answer.” In either case, the LLM generates step-by-step calculations (e.g. adding columns of digits) in text form, which encourages correct computation. Strengths: The scratchpad effectively turns the LLM into a step-by-step calculator. This greatly boosts accuracy on tasks like long arithmetic, multi-digit multiplication, or even executing small programs
arxiv.org
. Because the model is forced to articulate each sub-computation, it is less likely to skip or merge steps incorrectly. Empirically, models given scratchpad prompts outperform those answering directly, sometimes by orders of magnitude on complex math tasks
arxiv.org
. Scratchpad outputs are also interpretable, showing how the LLM arrived at its result. Weaknesses: Like CoT, the scratchpad can lead to false confidence: the model’s “work” may look correct but hide subtle errors. Also, writing down all steps bloats the response length (useful for accuracy but inefficient). Scratchpads are typically effective only if the LLM has enough arithmetic ability to use them meaningfully; smaller models may still struggle. The method is somewhat domain-specific: it’s most beneficial for pure calculation or algorithmic execution tasks, and less directly relevant for open-ended reasoning. Moreover, scratchpad training (if fine-tuned) requires supervised intermediate-step data, which can be costly to create. Applications: Scratchpads are used for math puzzles, algorithmic reasoning (e.g. adding, multiplying large numbers, sorting lists stepwise), and programming tasks (LLMs can list intermediate states). They are also integrated into CoT prompts for tasks like logical deduction: e.g. writing down intermediate boolean equations. In vision-language models, analogous “visual scratchpads” have been proposed, enabling global reasoning over an image. Overall, scratchpad reasoning is a simple yet powerful paradigm for improving exactness in LLM outputs.
ReAct (Reasoning and Acting)
Conceptual Foundation: The ReAct paradigm was proposed to fuse language reasoning (like CoT) with environmental interaction (actions), enabling LLMs to function as agents. The key idea is to interleave “Thought” and “Action” tokens: the model not only thinks aloud but also issues action commands (e.g. “SearchWikipedia(‘topic’)” or “OpenURL(…)”) that interact with tools or knowledge sources
arxiv.org
. This mirrors human problem-solving, where thinking and doing alternate. ReAct leverages language models’ ability to generate both reasoning traces and concrete actions. The original paper showed that alternating reasoning with actions yields synergy: reasoning helps plan and interpret actions, while actions ground the reasoning in real data
arxiv.org
. Implementation: In practice, a ReAct prompt formats examples with interleaved “Thought:” and “Action:” lines. For instance, to answer a question, the model might produce:
Thought: I should check the population of France. 
Action: Wikipedia_search("Population of France") 
Observation: "The population of France is 67 million." 
Thought: Now I will answer the question as X.
Here, "Action" results invoke an API or tool (search, calculator, environment step), and the “Observation” is fed back. The loop continues until the model outputs a final answer. ReAct can be implemented with few-shot prompting if the tasks require tool use. For environments (like ALFWorld), ReAct can be combined with a reinforcement learning framework to fine-tune the LLM’s decision policy. The essential point is that every step of the conversation contains both a reasoning trace and, if needed, an external action. Strengths: ReAct enhances LLM capability by grounding reasoning in external facts and reducing hallucinations. On fact-heavy QA (HotpotQA, FEVER), ReAct overcame hallucination issues by executing searches to get evidence, yielding more factual answers
arxiv.org
. In interactive decision-making (virtual environments), ReAct outperformed both imitation learning and pure RL: e.g. it raised success rates by +34% on ALFWorld tasks and +10% on a simulated e-commerce WebShop task
arxiv.org
. The approach also improves interpretability and trust: the chain of “Thoughts” plus a log of actions forms a clear trace of the agent’s reasoning and steps. This trace can be audited or corrected by a human. Weaknesses: Implementing ReAct is more involved. It requires integrating LLM outputs with external systems (search engines, APIs, simulators), which complicates deployment. The model must be guided to output well-formed action calls; mistakes in formatting can break the loop. There is also a trade-off: interacting with tools introduces latency. ReAct’s performance hinges on the quality of the tools – e.g. a poor search engine will yield bad “Observation”. Finally, because ReAct is an agentic loop, it may exhibit unpredictable behavior if not carefully constrained (risk of unsafe actions). ReAct is also best suited to tasks with a clear action space; not all reasoning tasks map onto actionable steps. Applications: ReAct has been applied to open-domain question answering (using search APIs to gather info) and to sequential decision tasks. In NLP, it powers QA bots that ask clarifying questions or fetch documents mid-answer. In robotics and games, ReAct enables language models to plan and act: e.g. instructing a simulated robot via natural language steps. It is central to current “LLM agent” frameworks (like OpenAI’s tool-using agents) and has been used to build agents that combine web browsing, code execution, and planning. In sum, ReAct is the paradigm for LLM-as-agent reasoning.
Comparative Analysis
The above paradigms can be contrasted along dimensions of structure, interpretability, computational cost, and suitability. The table below highlights key differences:
Paradigm	Structure	Interpretability	Multi-Step Depth	External Tools	Cost/Complexity	Exemplary Use Cases
Chain-of-Thought	Linear chain of thoughts (1 path)	High (explicit steps)	Moderate (dependent on prompt length)	None	Low (1-2 calls)	Arithmetic, logic puzzles, basic QA
ar5iv.labs.arxiv.org
arxiv.org
Tree-of-Thought	Search tree of branches	High (tree trace visible)	High (explores alternatives)	None	High (many LLM calls)	Combinatorial puzzles (Game of 24, crosswords)
ar5iv.labs.arxiv.org
Graph-of-Thought	General directed graph	High (graph structure)	High (multi-hop, feedback)	Optional (often none or retrieval)	High (graph building and GNN)	Complex inference (multimodal QA, reasoning on datasets)
arxiv.org
aclanthology.org
Web-of-Thought	Network (knowledge graph)	Conceptual (visualizable)	Very high (network reasoning)	Yes (knowledge bases)	Very high (conceptual)	Networked domains (biotech analysis)
lumagroup.com
Program-Aided (PAL)	Chain of programmatic steps (code)	High (code is interpretable)	Moderate (decomposed by code)	Yes (code interpreter)	Moderate (LLM call + exec)	Math word problems, algorithms
arxiv.org
Scratchpad	Linear work steps (in-output)	High (shows calculations)	Moderate to high (task-dependent)	No	Low to moderate (longer output)	Arithmetic, program execution
arxiv.org
ReAct	Mixed chain + action steps	High (thought/action log)	High (iterative with tools)	Yes (APIs/tools)	Moderate (calls per action)	Interactive QA, planning, robotics
arxiv.org
arxiv.org
Scalability: CoT and scratchpad are simplest (single prompt, few calls). Tree and graph methods require many LLM invocations and bookkeeping, making them more expensive. PAL incurs an external execution cost but relatively few model calls. ReAct’s cost depends on the number of tool interactions and API latency.
Interpretability: All paradigms output human-readable reasoning to some extent. CoT, scratchpad, and ReAct each produce explicit text traces. Tree/Graph produce a clear structure of thoughts. Graph/ToT excel at showing all alternative paths, boosting transparency.
Reasoning Depth: CoT can only follow one linear reasoning path. ToT and GoT allow far deeper exploration via backtracking or graph links. PaL and scratchpad enforce accuracy in arithmetic, which deepens calculation depth. ReAct can repeat reasoning cycles by using observations.
Modularity: Tree and graph frameworks are inherently modular: they separate generation and evaluation. PAL modularizes reasoning (LLM) vs execution (interpreter). ReAct modularizes thought vs environment. CoT/scratchpad are monolithic prompts.
Dependency Modeling: Only graphs/web can explicitly represent arbitrary dependencies among subproblems. CoT implicitly has sequential dependency; ToT has parent-child; PAL/shachpad encode dependency in code flow; ReAct externalizes dependency via actions/observation.
Deployment: CoT, PAL, scratchpad require minimal infrastructure (just prompt and model). ToT/GoT often require custom orchestrator code. ReAct needs connected tools/APIs.
In summary, CoT/scratchpad are easy to deploy and interpret, boosting basic multi-step reasoning. ToT/GoT/WoT offer richer reasoning depth at higher computational cost. PAL trades computation for precision on STEM tasks. ReAct uniquely enables tool-use and interactive reasoning. The choice of paradigm depends on task needs: simple tasks may only need CoT, whereas planning or high-stakes domains may warrant tree/graph structures and external tools.
Impact on LLM Capabilities
Each paradigm significantly alters what LLMs can do:
Enhanced Multi-Step Reasoning: CoT, ToT, and GoT allow LLMs to tackle problems that require more than token-by-token inference. By breaking tasks into intermediate “thoughts,” models can solve complex math, logic, and planning tasks far beyond vanilla capabilities
ar5iv.labs.arxiv.org
ar5iv.labs.arxiv.org
. ToT and GoT in particular enable systematic search, akin to explicit planning, thus deepening reasoning horizons.
Transparency and Debugging: All methods produce explicit reasoning steps. This transparency can improve trust and detect failure modes. For instance, CoT’s chain or ReAct’s “Thoughts” let a human examiner see why an answer was given
ar5iv.labs.arxiv.org
arxiv.org
. GoT’s graph offers an even richer map of inference. Having these reasoning traces makes it easier to align models or correct errors (e.g. reject a wrong chain-of-thought) compared to black-box outputs.
Reliability and Correctness: Paradigms like PAL (program-aided) and scratchpad reduce simple errors. By using a program or explicit arithmetic scratchwork, the model’s outputs become more reliable for calculation-heavy tasks
arxiv.org
arxiv.org
. ReAct’s grounding in external sources similarly reduces hallucination in QA
arxiv.org
. Overall, these methods improve consistency, though they do not fully eliminate failures (a wrong intermediate step still leads to a wrong answer).
Specialization and Modularity: These paradigms modularize reasoning, enabling hybrid systems. For example, an LLM can focus on abstraction (planning, decomposition) while a tool handles execution. This synergy has opened new applications (e.g. LLM-powered agents). Moreover, some paradigms (like ToT) suggest new training directions: e.g. future LMs could be trained to plan or search, embodying Tree-of-Thought internally.
Performance: Empirically, each paradigm yields performance gains on benchmark tasks in its domain. Table-of-content tasks (GSM8K math) improved from ~40% to ~85% accuracy using CoT or PAL
ar5iv.labs.arxiv.org
arxiv.org
. Multi-hop QA jumps by 10–20 points with ReAct or GoT. Hard puzzles unsolvable by greedy LMs become solvable with ToT. These leaps demonstrate that LLM capabilities are not fixed but can be dramatically enhanced by the right prompting or system architecture.
Emerging Trends and Hybrids
Research on LLM reasoning is evolving rapidly. Several emerging directions blend the above paradigms:
Self-Consistency & Ensemble Methods: Extensions of CoT sample multiple reasoning chains (via different random seeds) and aggregate answers, reducing variance. Though not a standalone paradigm, this meta-strategy is increasingly paired with CoT/ToT to further boost accuracy.
Meta-Cognition and Reflection: Recent work explores models that reflect on their answers. For example, Reflexion (Shinn et al.) builds on ReAct by having a separate “reflector” LLM evaluate failures and iteratively improve answers, akin to a human reflecting on mistakes
arxiv.org
. Such approaches combine chain-of-thought reasoning with self-critiquing loops.
Multi-Agent Collaboration: Some methods simulate multiple LMs (“experts”) that discuss or debate a solution. PanelGPT is one example (multiple threads reasoning in parallel)
promptingguide.ai
. Multi-agent frameworks (DiMo et al. 2023) let agents with different thinking modes communicate, blending diverse reasoning strategies.
Longer Contexts and Retrievers: Combining these paradigms with retrieval is an active trend. For instance, Retrieval-Augmented CoT uses knowledge retrieval between reasoning steps. This effectively forms a dynamic web-of-thought where each step can query an external memory or search engine.
Fine-tuned Reasoners: So far these methods mainly use fixed pretrained LMs. A future direction is to fine-tune models with these reasoning scaffolds in mind. For example, training an LLM with tree-search objectives or to generate and evaluate thoughts could internalize ToT mechanisms, making them more efficient.
Tool Use and Automation: The line between reasoning paradigms and agents is blurring. AutoGPT and BabyAGI (2023–2024) chain together ReAct-like steps autonomously. The agent paradigm essentially uses CoT plus planning to solve long-horizon tasks without human prompts. Expect more work on automating and optimizing these agentic pipelines.
In summary, LLMs are becoming thinking machines in multiple senses: they can plan (ToT), connect concepts (GoT), write and run programs (PAL), and act in environments (ReAct). The field is rapidly experimenting with hybrids (e.g. graph-based tree search) and domain-specific adaptations. The most promising research lines combine the strengths of each: e.g. a graph-of-thought planner that uses programs and interacts with tools, or a scratchpad-enhanced CoT within an agent framework. As these paradigms mature, LLM-based systems will gain deeper reasoning, greater reliability, and more robust decision-making capabilities. Sources: This report is grounded in recent literature on LLM reasoning, including seminal papers and surveys
ar5iv.labs.arxiv.org
ar5iv.labs.arxiv.org
arxiv.org
arxiv.org
arxiv.org
arxiv.org
aclanthology.org
, which detail the design and evaluation of these paradigms. Each paradigm summary above cites key findings from these studies.
Citations

[2201.11903] Chain-of-Thought Prompting Elicits Reasoning in Large Language Models

https://ar5iv.labs.arxiv.org/html/2201.11903

[2201.11903] Chain-of-Thought Prompting Elicits Reasoning in Large Language Models

https://ar5iv.labs.arxiv.org/html/2201.11903

[2305.10601] Tree of Thoughts: Deliberate Problem Solving with Large Language Models

https://ar5iv.labs.arxiv.org/html/2305.10601

[2201.11903] Chain-of-Thought Prompting Elicits Reasoning in Large Language Models

https://ar5iv.labs.arxiv.org/html/2201.11903

[2201.11903] Chain-of-Thought Prompting Elicits Reasoning in Large Language Models

https://ar5iv.labs.arxiv.org/html/2201.11903

Reasoning with Large Language Models, a Survey

https://arxiv.org/html/2407.11511v1

[2305.10601] Tree of Thoughts: Deliberate Problem Solving with Large Language Models

https://ar5iv.labs.arxiv.org/html/2305.10601

[2305.10601] Tree of Thoughts: Deliberate Problem Solving with Large Language Models

https://ar5iv.labs.arxiv.org/html/2305.10601

Tree of Thoughts (ToT) | Prompt Engineering Guide<!-- -->

https://www.promptingguide.ai/techniques/tot

Tree of Thoughts (ToT) | Prompt Engineering Guide<!-- -->

https://www.promptingguide.ai/techniques/tot

[2305.10601] Tree of Thoughts: Deliberate Problem Solving with Large Language Models

https://ar5iv.labs.arxiv.org/html/2305.10601

[2305.10601] Tree of Thoughts: Deliberate Problem Solving with Large Language Models

https://ar5iv.labs.arxiv.org/html/2305.10601

[2305.10601] Tree of Thoughts: Deliberate Problem Solving with Large Language Models

https://ar5iv.labs.arxiv.org/html/2305.10601

Tree of Thoughts (ToT) | Prompt Engineering Guide<!-- -->

https://www.promptingguide.ai/techniques/tot

[2308.09687] Graph of Thoughts: Solving Elaborate Problems with Large Language Models

https://arxiv.org/abs/2308.09687

The Limits of Generic LLMs: Why Biotech Needs Purpose-Built Tools - LumaGroup

https://lumagroup.com/the-limits-of-generic-llms-why-biotech-needs-purpose-built-tools/

https://aclanthology.org/2024.findings-naacl.183.pdf

https://aclanthology.org/2024.findings-naacl.183.pdf

[2308.09687] Graph of Thoughts: Solving Elaborate Problems with Large Language Models

https://arxiv.org/abs/2308.09687

https://aclanthology.org/2024.findings-naacl.183.pdf

[2211.10435] PAL: Program-aided Language Models

https://arxiv.org/abs/2211.10435

[2211.10435] PAL: Program-aided Language Models

https://arxiv.org/abs/2211.10435

[2112.00114] Show Your Work: Scratchpads for Intermediate Computation with Language Models

https://arxiv.org/abs/2112.00114

[2210.03629] ReAct: Synergizing Reasoning and Acting in Language Models

https://arxiv.org/abs/2210.03629

[2210.03629] ReAct: Synergizing Reasoning and Acting in Language Models

https://arxiv.org/abs/2210.03629

Reasoning with Large Language Models, a Survey

https://arxiv.org/html/2407.11511v1

Tree of Thoughts (ToT) | Prompt Engineering Guide<!-- -->

https://www.promptingguide.ai/techniques/tot
All Sources

ar5iv.labs.arxiv

arxiv

promptingguide

lumagroup

aclanthology