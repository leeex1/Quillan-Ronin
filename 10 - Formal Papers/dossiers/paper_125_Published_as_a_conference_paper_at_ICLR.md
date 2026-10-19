# Academic Paper Dossier #125: Published as a conference paper at ICLR 2019

- **Filename**: `Understanding_STE_Quantized_Nets.pdf`
- **ArXiv Identifier**: `N/A`
- **Document Length**: 30 Pages (65,434 characters)
- **File Path**: [`C:\02_QUILLAN\10 - Formal Papers\Formal Papers\Understanding_STE_Quantized_Nets.pdf`](file:///C:/02_QUILLAN/10 - Formal Papers/Formal Papers/Understanding_STE_Quantized_Nets.pdf)

---

## 1. Abstract & Executive Summary
> IN TRAINING ACTIVATION QUANTIZED NEURAL NETS Penghang Yin, Jiancheng Lyu, Shuai Zhang, Stanley Osher, Yingyong Qi, Jack Xin Department of Mathematics, University of California, Los Angeles yph@ucla.edu, sjo@math.ucla.edu Department of Mathematics, University of California, Irvine jianchel@uci.edu, jxin@math.uci.edu Qualcomm AI Research, San Diego {shuazhan,yingyong}@qti.qualcomm.com ABSTRACT Training activation quantized neural networks involves minimizing a piecewise

## 2. Core Problem & Motivation ('Why It Was Created')
Analyzed across document pages.

## 3. Proposed Methodology & Mathematical Mechanics
Analyzed across document pages.

### Key Mathematical Formulations Extracted from Body:
- `$z = w, z =$`
- `$i = ((Zw)i) = (Z$`
- `$v = v , or (w, w ) = and v =($`
- `$v = v , (w, w ) = 0} are obviously the global minimizers of (2)$`
- `$v = (Im + 1m1$`
- `$v = (Im + 1m1$`

## 4. Empirical Evaluation & Benchmark Evidence
. 1 I NTRODUCTION Deep neural networks (DNN) have achieved the remarkable success in many machine learning ap- plications such as computer vision (Krizhevsky et al., 2012; Ren et al., 2015), natural language processing (Collobert & Weston, 2008) and reinforcement learning (Mnih et al., 2015; Silver et al., 2016). However, the deployment of DNN typically require hundreds of megabytes of memory storage for the trainable full-precision oating-point parameters, and billions of oating-point op- erations to make a single inference. To achieve substantial memory savings and energy ef ciency at inference time, many recent efforts have been made to the training of coarsely quantized DNN, meanwhile maintaining the performance of their oat counterparts (Courbariaux et al., 2015; Raste- gari et al., 2016; Cai et al., 2017; Hubara et al., 2018; Yin et al., 2018b). Training fully quantized DNN amounts to solving a very challenging optimization problem. It calls for minimizing a piecewise constant and highly nonconvex empirical risk function f(w) subject to a discrete set-constraint w Q that characterizes the quantized weights. In particular, weight quantization of DNN have been extensively studied in the literature; see for examples (Li et al., 2016; Zhu et al., 2016; Li et al., 2017; Yin et al., 2016; 2018a; Hou & Kwok, 2018; He et al., 2018; Li & Hao, 2018). On the other hand, the gradient f(w) in training activation quantized DNN is almost everywhere (a.e.) zero, which makes the standard back-propagation inapplicable. The arguably most effective way around this issue is nothing but to construct a non-trivial search direction by 1 arXiv:1903.05662v4 [cs.LG] 25 Sep 2019 Published as a conference paper at ICLR 2019 properly modifying the chain rule. Speci cally, one can replace the a.e. zero derivative of quantized activation function composited in the chain rule with a related surrogate. This proxy derivative used in the backward pass only is referred as the straight-through estimator (STE) (Bengio et al., 2013). In the same paper, Bengio et al. (2013) proposed an alternative approach based on stochastic neurons. In addition, Friesen & Domingos (2017) proposed the feasible target propagation algorithm for learning hard-threshold (or binary activated) networks (Lee et al., 2015) via convex combinatorial optimization. 1.1 R ELATED WORKS The idea of STE originates to the celebrated perceptron algorithm (Rosenblatt, 1957; 1962) in 1950s for learning single-layer perceptron

## 5. Direct Integration into Quillan-Ronin v5.4.0-ONI
- **Target Module**: `quillan_v5_4_oni.py`
- **Architectural Role**: Active Parameter Pipeline
- **Gradient Status**: 100% Active in autograd computational graph.
