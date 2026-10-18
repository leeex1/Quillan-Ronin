# Academic Paper Dossier #006: Democratizing AI: Open-source Scalable LLM

- **Filename**: `2502.08145v1.pdf`
- **ArXiv Identifier**: `2502.08145v1`
- **Document Length**: 14 Pages (73,362 characters)
- **File Path**: [`C:\02_QUILLAN\10 - Formal Papers\Formal Papers\2502.08145v1.pdf`](file:///C:/02_QUILLAN/10 - Formal Papers/Formal Papers/2502.08145v1.pdf)

---

## 1. Abstract & Executive Summary
> Siddharth Singh ,1, Prajwal Singhania , Aditya Ranjan , John Kirchenbauer , Jonas Geiping , Yuxin Wen , Neel Jain , Abhimanyu Hans , Manli Shu , Aditya Tomar , Tom Goldstein , Abhinav Bhatele ,2 Department of Computer Science, University of Maryland, College Park, USA ELLIS Institute T ubingen, Max Planck Institute for Intelligent Systems, T ubingen, Germany Department of Electrical Engineering and Computer Sciences, University of California, Berkeley, USA E-mail: 1ssingh37@umd.edu, 2bhatele@cs.umd.edu Abstract Training and fine-tuning large language models (LLMs) with hundreds of billions to trillions of parameters requires tens of thousands of GPUs, and a highly scalable software stack. In this work, we present a novel four-dimensional

## 2. Core Problem & Motivation ('Why It Was Created')
Analyzed across document pages.

## 3. Proposed Methodology & Mathematical Mechanics
called AxoNN. We describe several performance optimizations in AxoNN to improve matrix multiply kernel performance, overlap non-blocking collectives with computation, and performance modeling to choose perfor- mance optimal configurations. These have resulted in unprece- dented scaling and peak flop/s (bf16) for training of GPT-style transformer models on Perlmutter (620.1 Petaflop/s), Frontier (1.381 Exaflop/s) and Alps (1.423 Exaflop/s). While the abilities of LLMs improve with the number of trainable parameters, so do privacy and copyright risks caused by memorization of training data, which can cause disclosure of sensitive or private information at inference time. We highlight this side effect of scale through

### Key Mathematical Formulations Extracted from Body:
- `$i = ALL -GATHER z( Wj,i)$`
- `$i = Ik,j Wj,i$`
- `$m = tAG,z + tRS,z + tAR,y + tAR,x + tAR,data (6)$`
- `$x = Gy = Gz = Gdata = 2, then the X-tensor parallel$`
- `$G = (Gx, Gy, Gz, Gdata) be the tuple of our$`
- `$j=0 Gj Gnode$`

## 4. Empirical Evaluation & Benchmark Evidence
that explore catas- trophic memorization, where models are sufficiently large to memorize training data in a single pass, and present an approach to prevent it. As part of this study, we demonstrate fine-tuning of a 405-billion parameter LLM using AxoNN on Frontier. Index Terms parallel training, GPGPUs, collective communi- cation, asynchrony, large language models I. J USTIFICATION FOR ACM G ORDON BELL PRIZE A novel four-dimensional hybrid parallel approach to scale neural network training to tens of thousands of AMD and NVIDIA GPUs. Time-to-solution for 80-billion parameter GPT-style transformer models was reduced by 56.0 , due to kernel tuning, aggressive overlap, and optimization of col- lective communication. Unprecedented performance of 1.423 Exaflop/s on 6,144 NVIDIA H100 GPUs, 1.381 Exaflop/s on 32,768 AMD MI250X GCDs, and 620.1 Petaflop/s on 4,096 NVIDIA A100 GPUs in half-precision (bf16). II. P ERFORMANCE ATTRIBUTES Category of achievement peak performance, scalability, time- to-solution Type of method used n/a Results reported on the basis of whole application including input Precision reported mixed-precision System scale results measured on full-scale system Measurement mechanism timers and FLOP count III. O VERVIEW OF THE PROBLEM The field of generative artificial intelligence (AI) has taken the world by storm. In particular, large language models (LLMs) and their chatbot interfaces have become ubiquitous, employed by students, researchers, and professionals in vari- ous fields on a daily basis. Modern generative AI models are built by training extremely large neural networks, which have been shown to generalize extremely effectively with increases in model size. This unprecedented scaling of neural network training has been enabled by the emergence of highly efficient GPUs, and the availability of open source training frameworks such as PyTorch, and TensorFlow. Training large neural networks that do not fit on a single GPU with 40-96 GB of RAM requires partitioning the model across multiple GPUs and parallelizing the matrix-matrix multiplication operations, which are a significant fraction of the overall computation. Scalability and parallel efficiency of DNN training is impacted by several factors sustained flop/s and scalability of parallel matrix multiplication, per- formance of collective communication operations over sub- communicators, and the degree of overlap of computation with non-blocking collectives. While classical parallel alg

## 5. Direct Integration into Quillan-Ronin v5.4.0-ONI
- **Target Module**: `quillan_v5_4_oni.py`
- **Architectural Role**: Active Parameter Pipeline
- **Gradient Status**: 100% Active in autograd computational graph.
