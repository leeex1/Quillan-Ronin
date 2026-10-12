# Academic Paper Dossier #082: NITRO-D: Native Integer-only Training of

- **Filename**: `NITRO_D_2407.11698.pdf`
- **ArXiv Identifier**: `2407.11698`
- **Document Length**: 9 Pages (47,919 characters)
- **File Path**: [`C:\02_QUILLAN\10 - Formal Papers\Formal Papers\NITRO_D_2407.11698.pdf`](file:///C:/02_QUILLAN/10 - Formal Papers/Formal Papers/NITRO_D_2407.11698.pdf)

---

## 1. Abstract & Executive Summary
> Quantization is a pivotal technique for managing the growing computational and memory demands of Deep Neural Net- works (DNNs). By reducing the number of bits used to repre- sent weights and activations (typically from 32-bit Floating-Point (FP) to 16-bit or 8-bit integers), quantization reduces memory foot- print, energy consumption, and execution time of DNNs. However, most existing methods typically target DNN inference, while train- ing still relies on FP operations, limiting applicability in environ- ments where FP arithmetic is unavailable. To date, only one prior work has addressed integer-only training, and only for Multi-Layer Perceptron (MLP) architectures. This paper introduces NITRO-D, a novel framework for training deep integer-only Convolutional Neu- ral Networks (CNNs) that operate entirely in the integer domain for both training and inference. NITRO-D enables training of inte- ger CNNs without requiring a separate quantization scheme. Specifi- cally, it introduces a novel architecture that integrates multiplelocal- loss blocks, which include the proposedNITRO-Scalinglayer and NITRO-ReLUactivation function. The proposed framework also features a novel learning algorithm that employs local error sig- nals and leveragesIntegerSGD, an optimizer specifically designed for integer computations. NITRO-D is implemented as an open- source Python library. Extensive evaluations on state-of-the-art im- age recognition datasets demonstrate its effectiveness. For integer- on

## 2. Core Problem & Motivation ('Why It Was Created')
In recent years, quantization of Deep Neural Networks (DNNs) has emerged as a pivotal trend in both the scientific and technological domains [11]. This technique aims at lowering the computational de- mand and memory requirements of DNNs by reducing the number of bits to represent weights and activations (e.g., from 32-bit Floating- Point (FP) to 16-bit or 8-bit integers). However, existing quantization

## 3. Proposed Methodology & Mathematical Mechanics
s typically target DNN inference, while train- ing still relies on FP operations, limiting applicability in environ- ments where FP arithmetic is unavailable. To date, only one prior work has addressed integer-only training, and only for Multi-Layer Perceptron (MLP) architectures. This paper introduces NITRO-D, a novel framework for training deep integer-only Convolutional Neu- ral Networks (CNNs) that operate entirely in the integer domain for both training and inference. NITRO-D enables training of inte- ger CNNs without requiring a separate quantization scheme. Specifi- cally, it introduces a novel architecture that integrates multiplelocal- loss blocks, which include the proposedNITRO-Scalinglayer and NITRO-ReLUactivation function. The proposed framework also features a novel learning algorithm that employs local error sig- nals and leveragesIntegerSGD, an optimizer specifically designed for integer computations. NITRO-D is implemented as an open- source Python library. Extensive

### Key Mathematical Formulations Extracted from Body:
- `$l =a l 1 W i$`
- `$l = 8), the bound for an integer linear$`
- `$l =O(15 + log2 Ml 1)$`
- `$l =O(15 + log2 K 2$`
- `$i={1,2,3,4}defined as:$`
- `$b=  128 1732 f anin 1000$`

## 4. Empirical Evaluation & Benchmark Evidence
s on state-of-the-art im- age recognition datasets demonstrate its effectiveness. For integer- only MLPs, NITRO-D improves test accuracy by up to+5.96%over the state-of-the-art. It also successfully trains integer-only CNNs, reducing memory requirements and energy consumption by up to 76.14%and32.42%, respectively, compared to the traditional FP backpropagation algorithm. 1 Introduction In recent years, quantization of Deep Neural Networks (DNNs) has emerged as a pivotal trend in both the scientific and technological domains [11]. This technique aims at lowering the computational de- mand and memory requirements of DNNs by reducing the number of bits to represent weights and activations (e.g., from 32-bit Floating- Point (FP) to 16-bit or 8-bit integers). However, existing quantization frameworks for DNNs retain a dependence on FP operations during training. In this perspective, the aim of this paper is to address the following research question:Is it possible to train arbitrarily deep Convolutional Neural Networks (CNNs) using only integer values for both training and inference? Corresponding Author. Email: alberto.pirillo@mail.polimi.it Corresponding Author. Email: luca2.colombo@polimi.it The literature in this field is extensive but highly fragmented. Existing frameworks for integer-only DNNs can be grouped into three main categories. The first,quantization-aware inference, ap- plies quantization only at inference time, while training relies on FP operations to compute gradients [18, 16, 7, 28, 43, 2, 22, 34, 24, 23]. The second,quantization-aware training, extends quantization to the training process, further exploiting the advantages of integer arithmetic at the cost of additional complexity in the quantization scheme [42, 39, 40, 37, 5, 17, 26, 41, 4, 32, 10]. Despite these ad- vancements, such approaches still partially rely on FP arithmetic or employ custom fixed-point numeric formats that emulate it. The last category,native integer-only training, uses integer-only arithmetic during both inference and training. To date, the only solution in this category targets Multi-Layer Perceptron (MLP) architectures [31]. This paper introduces NITRO-D, a native integer-only framework for training deep CNNs exclusively employing integer arithmetic. This approach eliminates the need for any explicit quantization steps or FP operations during training. NITRO-D features a novel network architecture composed of multiple local-loss blocks and introduces novel compo

## 5. Direct Integration into Quillan-Ronin v5.4.0-ONI
- **Target Module**: `quillan_v5_4_oni.py`
- **Architectural Role**: Active Parameter Pipeline
- **Gradient Status**: 100% Active in autograd computational graph.
