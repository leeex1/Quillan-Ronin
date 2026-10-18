---
file_type: paper
domain: training
status: active
tags: [quillan, dataset, sota, agi, training, data]
---
# dataset creation SOTA level:

Blueprint for State-of-the-Art Artificial General Intelligence (AGI) Dataset Specification (JSONL)
I. Defining the AGI Data Mandate: Requirements for General Intelligence
The development of State-of-the-Art (SOTA) Artificial General Intelligence (AGI) models necessitates a fundamental divergence from the data architectures that underpinned successful Artificial Narrow Intelligence (ANI) systems. ANI models are designed to excel at specific, constrained tasks such as image recognition or natural language processing. AGI, however, is defined by its ability to understand or learn any intellectual task a human being can, characterized primarily by generalization ability, common sense knowledge, and effective adaptation to novel, ambiguous environments.   

The current high performance observed in existing large models, often achieved through scaling data and compute, runs the risk of only optimizing for imitation rather than true general intelligence. Evaluation methods that rely purely on similarity to human output are prone to being "gamed" if the system circumvents the metric’s true intention. Consequently, an AGI training dataset must move beyond simple input-output pairs to incorporate meta-tasks that explicitly test generalization and common sense knowledge, encompassing facts, relationships, and social norms. The philosophy underlying critical capability benchmarks, such as the Abstract Reasoning Corpus for AGI (ARC-AGI), emphasizes tasks that are not economically useful to target but serve purely as an objective measure of core cognitive capacity.   

A. Necessity of Generalization and Dual Input Encoding
Achieving SOTA AGI requires training models to synthesize knowledge across disparate domains, a capability currently missing in narrow models. The architectural implication is that the dataset must support dual input encoding. This means training on natural language instructions (the traditional Large Language Model, or LLM, path) must be coupled with structured, visual, or programmatic inputs (the abstract reasoning path exemplified by ARC-AGI).   

This dual approach is necessary because successful methodologies for solving highly abstract tasks often involve techniques like discrete program search or ensemble solutions. These approaches demand algorithmic reasoning rather than statistical pattern matching. Therefore, the instruction tuning data must explicitly formalize and encode the algorithmic search space or the sequence of transformation steps executed to solve the problem. This structural requirement will be satisfied by integrating Chain-of-Thought (CoT) and Thought of Structure (ToS) fields within the JSONL specification, transforming internal justification into an external, verifiable reasoning process.   

B. Integrating Abstract and Multitask Complexity
To ensure robustness against simple factual recall, the AGI dataset must integrate challenges that necessitate high-difficulty reasoning. Benchmarks such as MMLU-Pro and BIG-bench, which cover extensive domains and increase answer options to reduce random guessing, define the required standard for intellectual complexity.   

For abstract reasoning specifically, the dataset must reflect the visual and programmatic complexity of tasks like ARC-AGI. This requires dedicated data instances that utilize structured inputs—such as array representations of grids, graphs, or visual scenes—and demand generalized transformation rules rather than merely descriptive outputs. The data format must therefore include dedicated fields for encoding visual/grid inputs and the precise programmatic steps used in the solution, confirming that the model has learned the underlying rule set, not just the output pattern.   

II. The Intellectual Domains of AGI: Core Data Clusters
The AGI instruction tuning dataset is structured around cognitive domains essential for human-level general intelligence, focusing particularly on skills that demand 'System 2' type processing—deliberate, structured, and multi-step problem solving.

A. Abstract and Programmatic Reasoning (System 2 Data Stream)
This data stream is designed to ensure models can handle non-linguistic, constraint-based problem-solving, which is critical for agentic control and reliability. The instances cover mathematical proofs, optimization, resource allocation, and generalizable pattern recognition.

The architecture mandates that problem-solving tasks be paired with schema-constrained reasoning tests. Current models exhibit difficulties in reliably generating valid JSON strings, especially with intricate schemas. To overcome this limitation and ensure the reliability required for AGI deployment, the dataset instances must demand not only the solution to the programmatic task but also the guaranteed output of that solution in a strict, verifiable structure defined by a JSON Schema. This process trains the model on the fundamental requirement of an agentic system: solving a problem and articulating the solution in a guaranteed, actionable format.   

B. Causality, Physics, and Counterfactuals
A key differentiator for AGI is the capacity to understand causal relationships, allowing it to reason about interventions and counterfactual outcomes—the understanding of why and what if.   

Structured Causal Models (SCMs) provide a framework for modeling complex causal dependencies, interventions, and counterfactual reasoning superior to simpler knowledge graphs. Therefore, AGI data instances must contain a structured representation of the underlying causal model (e.g., nodes, edges, dependencies) either as context or as an intermediate reasoning step. By operating on this metadata, LLMs have already demonstrated the ability to generate causal arguments and construct causal graphs from natural language, surpassing existing algorithms in accuracy for tasks like pairwise causal discovery and event causality. This capability must be systematically trained and verified.   

Furthermore, generalizable causal laws, particularly intuitive physics, must be grounded in actionable data. The scale-maximalist approach, successful for text-based LLMs, falters here due to the scarcity of natural deposits of embodiment data necessary for AGI. To circumvent this limitation, the dataset must heavily rely on synthetic data generated from high-fidelity physics engines and virtual worlds. These simulations provide the model with essential knowledge about physical laws (gravity, force, deformation) and allow agents to learn how real-world phenomena behave, supporting counterfactual queries and generalizable prediction. This reliance on structured, synthetic interaction data bridges the gap created by the lack of readily available real-world embodiment logs.   

Table 1 summarizes the required cognitive capabilities mapped to their corresponding data requirements.

Table 1: Mapping AGI Capabilities to Required Data Domains

AGI Capability	Required Data Domains	Example Task/Benchmark Reference	Key Data Format Requirement
Generalization & Abstract Reasoning	Analogical mapping, inductive logic, discrete programming.	
ARC-AGI tasks , Constraint-based programs 

Structured grid/array data, explicit pseudo-code for CoT.
Causal & Counterfactual Inference	Physics engines, social dynamics, Statistical Causal Models (SCMs).	
Virtual world simulations , Causal graph construction 

Structured graph data (JSON/YAML), intervention variables, counterfactual scenarios.
Social & Ethical Judgment	Descriptive ethics, moral dilemmas, Theory of Mind (ToM).	
SCRUPLES dataset , MT-bench 

Community judgment distributions, multi-turn dialogue history, role identification.
Multimodal Synergy	Joint reasoning across text, image, video, requiring explicit correlation.	
General-Bench (700+ tasks) , CCoT Framework 

External URI references, fine-grained cross-modal annotations (e.g., scene graphs).
  
C. Social Intelligence, Ethics, and Theory of Mind (ToM)
For an AGI system to operate responsibly and effectively in human environments, it must integrate social understanding and ethical judgment.

The dataset must incorporate examples drawn from descriptive ethics, reflecting actual human judgments on complex moral dilemmas. Datasets like SCRUPLES provide large-scale ethical judgments over real-life anecdotes. Crucially, ethical norms are not always clean-cut; many situations are naturally divisive. Therefore, the data structure must encode the distribution of judgments (e.g., community scores or likelihood functions) rather than forcing a single 'correct' answer. This trains the model to understand intrinsic uncertainty and separate disagreement arising from the situation itself from model error. Furthermore, to progress beyond simple descriptive ethics, the data should embed specialized contexts or instructions that invoke philosophical frameworks, such as utilitarianism or deontology, refining the model's capacity for theoretical ethical analysis.   

To simulate realistic social interaction, the dataset must include multi-user, multi-turn social tasks. Adopting the structure of benchmarks like MT-bench , each instance must track the conversational history (e.g., using a dialogue_turn_id), identify the role or intent of each speaker, and span complex categories like roleplay and social science.   

III. Multimodal Data Architecture and Encoding Strategy
SOTA AGI models are rapidly evolving towards the Multimodal Generalist (MLLM) paradigm, capable of both comprehending and generating across arbitrary modalities. The primary evaluation criterion for these models is Synergy, the measure of whether capabilities are preserved across comprehension, generation, and complex multimodal interactions.   

A. The Multimodal Generalist Paradigm and Synergy
To train for this synergy, the dataset structure must facilitate joint reasoning across different input types. Input modalities (images, video, audio) must be referenced via stable URI links, which is standard practice in formats like VQA and specialized JSONL structures. The data must go beyond simple file path linkage by implementing a structured array for media inputs, detailing the modality type (e.g., image, video, 3d_scene_graph) and including optional paths to corresponding metadata for fine-grained annotations (e.g., bounding boxes or temporal coordinates).   

This detailed encoding supports Visual Instruction Tuning (VIT), which is critical for LMM success. VIT requires the provision of high-quality training data that passes specific textual descriptions and precise object location information directly to the model.   

B. Complex Multimodal Chain-of-Thought (CCoT) Implementation
Complex visual reasoning tasks require combining visual instruction tuning with explicit, structured thought processes. Relying solely on latent visual features embedded by the model is insufficient for robust, explicit reasoning.   

A specialized form of Complex Chain-of-Thought (CCoT) must be implemented. For a multimodal query, the data instance is structured to guide the model to first generate a scene graph—a symbolic, structured representation of objects and their spatial and semantic relationships within the visual input. This scene graph serves as an intermediate, verifiable artifact that externalizes the visual reasoning process into a symbolic format. The final response is then generated by prompting the model using the original instruction, the input media, and the intermediate scene graph. This architecture ensures that the model trains on an explicit, structured reasoning process, moving beyond simple textual CoT justification to a verifiable, executable sequence specific to visual and spatial transformations, mirroring the need for structured output observed in programmatic tasks.   

IV. The Foundational JSONL Schema: Technical Specification
The AGI dataset is delivered in the JSON Lines (JSONL) format, optimized for large-scale instruction tuning. Each line is a complete, independent JSON object representing a single instruction-response entry. This structure integrates standard instructional fields with advanced structures for multimodal input, structured reasoning, and explicit output constraints.

A. Core Instruction Tuning Schema Fields
Every entry adheres to the following foundational fields required for advanced fine-tuning :   

id: A unique identifier for the specific data instance.

domain_tag: Categorization of the task (e.g., 'Abstract_Reasoning', 'Ethical_Dilemma').

instruction: The primary query or task.

system_prompt_context: An optional field used to define the model’s role or set specific constraints for generating the response (e.g., "Act as a physicist calculating gravitational force").

context: Supplementary textual data or necessary background tables.   

B. Multimodal and Embodied Input Schema
To support MLLM training, input media is formalized as an array to handle multiple files per instruction:

input_media (Array of Objects):

media_id: Unique identifier for the media file.

media_type: Mandatory specification of the input format: image, video, audio, 3d_scene_graph, or simulation_state.

uri: The accessible S3 or HTTPS link to the external media resource.   

metadata_path: Optional link to a corresponding structured annotation file (e.g., JSONL defining object locations or temporal sequences).

C. Advanced Reasoning Structure Implementation (CoT/ToS)
This schema employs the Thought of Structure (ToS) concept  to formalize complex, multi-step thought processes necessary for generalization.   

reasoning_structure (Object):

coherence_score: A score (0.0 to 1.0) indicating the logical fidelity of the steps, often human- or model-generated during evaluation.

chain_of_thought (Array of Objects - ToS Format): This array records the structured process:

step_id: Sequential step number.

step_description: Natural language summary of the action taken.

intermediate_state: The structured, critical output of the step, which can be pseudo-code, an updated knowledge graph, a mathematical equation in LaTeX, or a scene graph for visual tasks.   

modality_focus: Specifies which input modality was primarily used in this step ('Text', 'Vision', 'Interaction').

expected_output (Object):

response_format: Defines the output type required: natural_language, json, code, or grid_array.

response: The final validated solution.   

json_schema_validation: A mandatory field containing the strict JSON Schema definition when response_format is json or code. Enforcing this schema trains the model in schema-constrained reasoning, crucial for ensuring structured, reliable output for downstream agents.   

Table 2 provides the complete blueprint for the JSONL instance schema.

Table 2: Comprehensive AGI Dataset JSONL Schema Blueprint

Field Category	JSON Key	Data Type	Description & AGI Significance
Core Metadata	id	String	Unique instance identifier.
Domain	domain_tag	String	Defines the specific AGI skill targeted (e.g., Causality, Ethics).
Instruction	instruction	String	The prompt defining the task.
Input Context	context	String/Object	Supplementary textual or structured data.
Multimodal Input	input_media	Array	
URIs and metadata for non-textual inputs (e.g., images, scene files).

Structured Reasoning	reasoning_structure	Object	
Container for complex CoT/ToS steps, pseudo-code, and intermediate states.

Causal Specificity	intervention_variables	Array	
Defines the changes applied to the context for causal tasks.

Output Constraint	json_schema_validation	Object	
Strict schema for complex outputs (e.g., generated code, structured data).

Expected Output	response	String/Object	The final, validated answer or action sequence.
Governance ID	provenance_uuid	String	
Globally unique identifier (OTDI requirement).

Safety Data	safety_labels	Object	
Detailed toxicity, bias, and PII assessment scores.[23, 24]

  
V. Ethical Governance and Provenance Layer (Metadata Specification)
Achieving SOTA AGI requires trust and accountability, mandating a rigorous governance layer integrated directly into the dataset structure. This metadata must adhere to standards like the Open Trusted Data Initiative (OTDI) and MLCommons Croissant specifications. These fields are not merely regulatory hurdles; they are functionally essential for debugging failure modes, assessing bias, and ensuring the data supports generalizability across diverse regulatory and cultural environments.   

A. Provenance, Lineage, and Accountability
Comprehensive provenance tracking is crucial for complex AGI systems whose behavior is difficult to interpret.

The provenance_uuid field, a globally unique identifier (UUID), is mandatory for every entry to establish unambiguous lineage tracking. This allows researchers to trace a failure back to a specific data point. Furthermore, the source_datasets field must list all upstream sources utilized, allowing correlation between model errors and the characteristics of the originating data (e.g., differentiating between errors derived from web crawls versus expert-generated data). Accountability is ensured by the mandatory curated_by field, listing the legal entities responsible for the dataset's creation and annotation.   

The range_dates_data_generation field is mandatory to track the span of time during which the data was collected. This temporal context is vital for assessing the timeliness and potential obsolescence of information, particularly in rapidly changing knowledge domains.   

B. Licensing and Regulatory Compliance
Legal transparency is a core requirement for open scientific progress. Mandatory fields include dataset_license (with cdla-permissive-2.0 strongly recommended by OTDI) and consent_documentation_location, which specifies where consent agreements for third-party contributed data can be found.   

A powerful implication of including governance data is the ability to address culturally relative requirements. For instance, the data_origin_geography must be specified if geographical restrictions apply. This field is highly valuable for deployment: if an AGI model exhibits regional bias in its social or ethical judgments, the geographical data allows researchers to trace that bias directly back to the regionally sourced data used during training, enabling targeted mitigation strategies and ensuring compliance with local regulations (such as GDPR or specific cultural norms).   

C. Safety, Bias Mitigation, and PII Disclosure
The dataset must include robust fields for assessing and mitigating inherent risks.   

The safety_labels field must capture detailed, multi-dimensional toxicity scores rather than a simple binary flag. This includes measuring factors such as severe_toxicity, profanity, and identity_attack using a structured scoring system. This detailed quantification is necessary for fine-grained measurement and effective filtering during training and fine-tuning.   

Furthermore, mandatory disclosure concerning personal and sensitive information (PII/SPI) is required via the personal_and_sensitive_information field. This disclosure must be paired with the use_of_privacy_enhancing_technologies_pets field, indicating whether any techniques were employed to protect sensitive data. Finally, the bias_risks_limitations field requires a description of known biases (e.g., demographic skew in annotators or cultural assumptions in ethical scenarios) to inform users of the training data's limitations.   

Table 3: Mandatory Ethical Governance and Provenance Fields (OTDI/Croissant Specification)

Field Name	OTDI Requirement	AGI Functional Significance	Source Reference
provenance_uuid	Mandatory	Enables unambiguous, instance-level lineage tracking.	
dataset_license	Mandatory	Legal compliance and defining usage rights.	
source_datasets	Mandatory	Correlating model failures with upstream data source characteristics.	
curated_by	Mandatory	Accountability and point of contact for legal/ethical inquiries.	
range_dates_data_generation	Mandatory	Assessing the timeliness and potential obsolescence of information.	
personal_and_sensitive_information	Mandatory	Mandatory risk disclosure for PII/SPI content.	
safety_labels	Highly Recommended	Fine-grained measurement and mitigation of toxic content generation.	[23, 24]
bias_risks_limitations	Mandatory	Essential for informing users of cultural or demographic limitations of AGI training.	
  
VI. Implementation Recommendations and Next Steps
The construction of a SOTA AGI dataset based on this schema requires specialized data generation and pipeline enforcement techniques.

A. Data Pipeline Enforcement and Synthetic Generation
Due to the structural complexity of the ToS/CCoT fields and the imperative for guaranteed structured output, strict JSON Schema validation must be integrated directly into the data generation and ingestion pipeline.

The difficulty models exhibit in generating complex, valid JSON  must be addressed using advanced techniques such as Schema Reinforcement Learning (SRL) during the fine-tuning phase. SRL utilizes a fine-grained validator to provide feedback, guiding the model to generate responses that not only solve the instructed task but also adhere flawlessly to the output schema. This enforcement mechanism is vital for ensuring AGI components can interact reliably within complex software environments.   

Furthermore, a substantial portion of the dataset, particularly in domains requiring causality, physics, and embodiment, must be synthetically generated. Formal models and intuitive physics engines must be leveraged to create high-fidelity interaction data. This reliance on synthetic environments is the necessary response to the scarcity of high-quality, generalizable real-world embodiment data , providing the scale and diversity required to train generalizable causal laws.   

B. Future Proofing and Generalization
The multimodal architecture, particularly the input_media array structure, is designed to be extensible to future arbitrary modalities. By standardizing the use of stable URIs and associated metadata paths, the system can seamlessly integrate new data types—such as environmental sensor data or haptic feedback—preparing the AGI model for increasingly complex embodied cognitive processes without requiring a complete redesign of the core data specification.   

This JSONL specification, by mandating structured inputs, structured reasoning paths (CCoT/ToS), and strictly constrained outputs (via JSON Schema validation), ensures that the dataset trains for the core attributes of AGI: generalization, reliable structural output, and cognitive transparency. It transitions the training focus from linguistic imitation to verifiable, programmatic reasoning.


cloud.google.com
What is artificial general intelligence (AGI)? - Google Cloud
Opens in a new window

arxiv.org
Improving AGI Evaluation: A Data Science Perspective - arXiv
Opens in a new window

arcprize.org
Testing Policy - ARC Prize
Opens in a new window

arcprize.org
Guide - ARC Prize
Opens in a new window

crfm-helm.readthedocs.io
Scenarios - CRFM HELM - Read the Docs
Opens in a new window

huggingface.co
google/bigbench · Datasets at Hugging Face
Opens in a new window

arxiv.org
Learning to Generate Structured Output with Schema Reinforcement Learning - arXiv
Opens in a new window

openreview.net
Causal Reasoning and Large Language Models: Opening a New Frontier... - OpenReview
Opens in a new window

arxiv.org
Large language models for artificial general intelligence (AGI): A survey of foundational principles and approaches - arXiv
Opens in a new window

thegradient.pub
AGI Is Not Multimodal - The Gradient
Opens in a new window

ojs.aaai.org
SCRUPLES: A Corpus of Community Ethical Judgments on 32,000 Real-Life Anecdotes
Opens in a new window

evidentlyai.com
30 LLM evaluation benchmarks and how they work - Evidently AI
Opens in a new window

openreview.net
On Path to Multimodal Generalist: General-Level and General-Bench - OpenReview
Opens in a new window

arxiv.org
Compositional Chain-of-Thought Prompting for Large Multimodal Models - arXiv
Opens in a new window

frontiersin.org
Ethical and Statistical Considerations in Models of Moral Judgments - Frontiers
Opens in a new window

arxiv.org
How Social is It? A Benchmark for LLMs' Capabilities in Multi-user Multi-turn Social Agent Tasks - arXiv
Opens in a new window

roboflow.com
What is the Multimodal JSONL Annotation Format? - Roboflow
Opens in a new window

roboflow.com
What is the PaliGemma JSONL Annotation Format? - Roboflow
Opens in a new window

arxiv.org
Socratic Questioning: Learn to Self-guide Multimodal Reasoning in the Wild - arXiv
Opens in a new window

docs.aws.amazon.com
Fine-tune a large language model (LLM) using prompt instructions - Amazon SageMaker AI
Opens in a new window

docs.dify.ai
How to Use JSON Schema Output in Dify?
Opens in a new window

the-ai-alliance.github.io
Dataset Specification | Open Trusted Data Initiative
Opens in a new window

gitlab.llm-jp.nii.ac.jp
LLM-jp Toxicity Dataset - GitLab
Opens in a new window

kaggle.com
RealToxicityPrompts - Kaggle
Opens in a new window

kaggle.com
CommonGen (Generative Commonsense Reasoning) - Kaggle
Opens in a new window

mdpi.com
Evaluating Causal Reasoning Capabilities of Large Language Models: A Systematic Analysis Across Three Scenarios - MDPI
Opens in a new window

firecrawl.dev
How to Create Custom Instruction Datasets for LLM Fine-tuning - Firecrawl
Opens in a new window

arxiv.org
Instruction Tuning for Large Language Models: A Survey - arXiv
Opens in a new window

grandviewresearch.com
AI Datasets & Licensing For Academic Research And Publishing Market Report 2030
Opens in a new window

community.openai.com
How to create a correct JSONL for training - Prompting - OpenAI Developer Community
Opens in a new window

developers.google.com
Intro to How Structured Data Markup Works | Google Search Central | Documentation

## Connections
- [[0-Quillan Loader Manifest.md]]
- [[Formal Papers/The_next_Viral_Synapse.md]]
- [[Five fewshot output examples.md]]
- [[Platforms/Claude/dataset creation SOTA level.md]]
- [[00 - Meta/00 - Vault Index.md]]
- [[00 - Meta/02 - Knowledge Foundation.md]]

- [[system prompts/Quillan-Samurai.md]]
