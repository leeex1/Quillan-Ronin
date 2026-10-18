# Academic Paper Dossier #069: Published as a conference paper at ICLR 2017

- **Filename**: `Gumbel-Softmax_Categorical_Reparameterization.pdf`
- **ArXiv Identifier**: `N/A`
- **Document Length**: 13 Pages (33,581 characters)
- **File Path**: [`C:\02_QUILLAN\10 - Formal Papers\Formal Papers\Gumbel-Softmax_Categorical_Reparameterization.pdf`](file:///C:/02_QUILLAN/10 - Formal Papers/Formal Papers/Gumbel-Softmax_Categorical_Reparameterization.pdf)

---

## 1. Abstract & Executive Summary
> WITH GUMBEL -SOFTMAX Eric Jang Google Brain ejang@google.com Shixiang Gu University of Cambridge MPI T ubingen sg717@cam.ac.uk Ben Poole Stanford University

## 2. Core Problem & Motivation ('Why It Was Created')
Analyzed across document pages.

## 3. Proposed Methodology & Mathematical Mechanics
for semi-supervised learning for tasks with a large number of classes. (a) Comparison of training speed (steps/sec) between Gumbel-Softmax and marginaliza- tion (Kingma et al., 2014) on a semi-supervised V AE.

### Key Mathematical Formulations Extracted from Body:
- `$z = one_hot$`
- `$i = exp((log( i) +gi)/ )$`
- `$j=1 exp((log( j) +gj)/ )$`
- `$g = log( log(u))$`
- `$z = g( , )$`
- `$m = (z), implying m = 1$`

## 4. Empirical Evaluation & Benchmark Evidence
, we nd that the softmax temperature can be annealed according to a variety of schedules and still perform well. If is a learned parameter (rather than annealed via a xed schedule), this scheme can be interpreted as entropy regularization (Szegedy et al., 2015; Pereyra et al., 2016), where the Gumbel-Softmax distribution can adaptively adjust the con dence of proposed samples during the training process. 2.2 S TRAIGHT -T HROUGH GUMBEL -SOFTMAX ESTIMATOR Continuous relaxations of one-hot vectors are suitable for problems such as learning hidden repre- sentations and sequence modeling. For scenarios in which we are constrained to sampling discrete values (e.g. from a discrete action space for reinforcement learning, or quantized compression), we discretizey using arg max but use our continuous approximation in the backward pass by approxi- mating z y. We call this the Straight-Through (ST) Gumbel Estimator, as it is reminiscent of the biased path derivative estimator described in Bengio et al. (2013). ST Gumbel-Softmax allows samples to be sparse even when the temperature is high. 3 R ELATED WORK In this section we review existing stochastic gradient estimation techniques for discrete variables (illustrated in Figure 2). Consider a stochastic computation graph (Schulman et al., 2015) with discrete random variable z whose distribution depends on parameter , and cost function f(z). The objective is to minimize the expected cost L( ) = Ez p (z)[f(z)] via gradient descent, which requires us to estimate Ez p (z)[f(z)]. 3.1 P ATH DERIVATIVE GRADIENT ESTIMATORS For distributions that are reparameterizable, we can compute the samplez as a deterministic function g of the parameters and an independent random variable , so that z = g( , ). The path-wise gradients fromf to can then be computed without encountering any stochastic nodes: Ez p [f(z))] = E [f(g( , ))] = E p [ f g g ] (4) For example, the normal distribution z N ( , ) can be re-written as + N (0, 1), making it trivial to compute z/ and z/ . This reparameterization trick is commonly applied to train- ing variational autooencoders with continuous latent variables using backpropagation (Kingma & Welling, 2013; Rezende et al., 2014b). As shown in Figure 2, we exploit such a trick in the con- struction of the Gumbel-Softmax estimator. Biased path derivative estimators can be utilized even when z is not reparameterizable. In general, we can approximate z m( ), where m is a differentiable proxy for the stochastic s

## 5. Direct Integration into Quillan-Ronin v5.4.0-ONI
- **Target Module**: `quillan_v5_4_oni.py`
- **Architectural Role**: Active Parameter Pipeline
- **Gradient Status**: 100% Active in autograd computational graph.
