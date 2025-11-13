# Must know formulas

## Section 1:


# Essential Formulas for LLMs, ML, and RL

| #  | Concept / Formula | Purpose / Use |
|----|-----------------|---------------|
| 1  | `y = Wx + b` | Linear Layer (Fully Connected), fundamental for MLPs and transformers |
| 2  | `ReLU(x) = max(0,x)`<br>`Sigmoid(x) = 1/(1+e^{-x})`<br>`Tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x})` | Activation functions for introducing non-linearity |
| 3  | `softmax(z_i) = e^{z_i} / Σ_j e^{z_j}` | Converts logits into probabilities |
| 4  | `L = - Σ_i y_i log(ŷ_i)` | Cross-Entropy Loss for classification |
| 5  | `L = (1/n) Σ_i (ŷ_i - y_i)^2` | Mean Squared Error (Regression) |
| 6  | `θ ← θ - η ∂L/∂θ` | Gradient Descent update rule |
| 7  | `m_t = β₁ m_{t-1} + (1-β₁) g_t`<br>`v_t = β₂ v_{t-1} + (1-β₂) g_t^2`<br>`θ_t = θ_{t-1} - η (m_t / (1-β₁^t)) / (√(v_t / (1-β₂^t)) + ε)` | Adam Optimizer |
| 8  | `Attention(Q,K,V) = softmax(QK^T / √d_k) V` | Scaled Dot-Product Attention in transformers |
| 9  | `PE(pos,2i) = sin(pos / 10000^{2i/d_model})`<br>`PE(pos,2i+1) = cos(pos / 10000^{2i/d_model})` | Positional Encoding |
| 10 | `LN(x) = (x - μ)/(σ + ε) * γ + β` | Layer Normalization |
| 11 | `FFN(x) = max(0, xW_1 + b_1) W_2 + b_2` | Transformer Feed-Forward Network |
| 12 | `D_KL(P || Q) = Σ_i P(i) log(P(i)/Q(i))` | Kullback-Leibler Divergence (knowledge distillation, variational models) |
| 13 | `∂L/∂x = (∂L/∂y) * (∂y/∂x)` | Backpropagation chain rule |
| 14 | `S(i,j) = (X * K)(i,j) = Σ_m Σ_n X(i+m,j+n) K(m,n)` | Convolution operation (CNNs, embeddings) |
| 15 | `V^π(s) = E_π [ r_t + γ V^π(s_{t+1}) ]` | Bellman Equation in Reinforcement Learning |
| 16 | `Q(s_t,a_t) ← Q(s_t,a_t) + α [ r_t + γ max_a Q(s_{t+1},a) - Q(s_t,a_t) ]` | Q-Learning update |
| 17 | `∇_θ J(θ) = E_π [ ∇_θ log π_θ(a¦s) R ]` | Policy Gradient (REINFORCE) |
| 18 | `MultiHead(Q,K,V) = Concat(head_1,...,head_h) W^O`<br>`head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)` | Transformer Multi-Head Attention |
| 19 | `W ~ U(-√6/√(n_in+n_out), √6/√(n_in+n_out))` | Weight Initialization (Xavier/Glorot) |
| 20 | `y = x ⊙ mask, mask ~ Bernoulli(p)` | Dropout Regularization |
| 21 | `PPL = exp(-1/N Σ log P(w_i))` | Perplexity metric for evaluating LLM fluency and predictive uncertainty on sequences |
| 22 | `L_total = L + λ Σ w^2` | L2 Regularization (weight decay) to penalize large weights and prevent overfitting in training |
| 23 | `BN(x) = γ (x - μ_B)/σ_B + β` | Batch Normalization to normalize activations across mini-batches for stable deep network training |
| 24 | `L_RM = -Σ [r log σ(y) + (1-r) log(1-σ(y))]` | Reward Model Loss in RLHF for aligning LLMs with human preferences via binary classification |
| 25 | `x_t = √α_t x_{t-1} + √(1-α_t) ε` | Diffusion Forward Process for generative models like Stable Diffusion, adding noise step-by-step |
| 26 | `L_VAE = ||x - \hat{x}||^2 + D_KL(q(z|x) || p(z))` | Variational Autoencoder (VAE) Loss combining reconstruction and KL regularization for latent spaces |
| 27 | `ΔW = B A` (low-rank matrices B, A) | LoRA (Low-Rank Adaptation) update for efficient fine-tuning of large LLMs with minimal parameters |
| 28 | `f(x) = (1/(σ √(2π))) exp(- (x-μ)^2 / (2σ^2))` | Normal (Gaussian) Distribution for modeling continuous data in sampling and probabilistic LLMs |
| 29 | `cos θ = (A · B) / (||A|| ||B||)` | Cosine Similarity for measuring vector alignment in embeddings and retrieval-augmented generation |
| 30 | `L_PPO = E[min(r(θ) Â, clip(r(θ), 1-ε, 1+ε) Â)]` | PPO (Proximal Policy Optimization) clipped objective for stable RL in LLM alignment training |
| 31 | `BLEU = BP · exp(Σ w_n log p_n)` | BLEU Score for evaluating machine translation and text generation quality via n-gram precision |
| 32 | `FlashAttn(Q,K,V) ≈ O(N)` (approximate via tiling/blocking) | FlashAttention complexity reduction for efficient transformer inference on long sequences |
| 33 | `NTK(x,x') = E[∇f(x) · ∇f(x')]` | Neural Tangent Kernel for analyzing wide NN training dynamics and infinite-width limits |
| 34 | `ROUGE-N = Σ (overlapping n-grams) / Σ (candidate n-grams)` | ROUGE-N recall metric for summarization and extractive generation evaluation |
| 35 | `ELBO = E_q[log p(x/z)] - D_KL(q(z/x) || p(z))` | Evidence Lower Bound (ELBO) for optimizing variational inference in generative models |

---

💡 **Think Key Takeaways:**  
- Most LLM formulas revolve around **linear algebra, probability, gradients, and attention mechanics**.  
- RL formulas add **expectations, discounted rewards, and policy updates**.  
- Symbols like **K, Q, V** are vector placeholders; the math is the same if renamed (e.g., X, Y, Z).

---

## Section 2:

```yaml
# Here are 20 essential formulas and mathematical concepts you should know for building LLMs, machine learning (ML), and reinforcement learning (RL):  
   
# Updated: 20 essential + 15 missing formulas/concepts for building LLMs, machine learning (ML), and reinforcement learning (RL).
# Original focused on matrix ops, prob, RL returns, and attention. Additions cover activations, eval, reg, gen models, and fine-tuning.
# Total: 35 entries. Letters like K, V, Q are attention-specific; others (A, B) are general notation.

formulas:
  - id: 1
    name: "Matrix Multiplication"
    formula: "C[i,j] = sum_k A[i,k] * B[k,j]"
    description: "Fundamental for linear transformations, including embeddings and neural network computations."

  - id: 2
    name: "Dot Product / Inner Product"
    formula: "a · b = sum_i a_i * b_i"
    description: "Used in similarity measures and attention score calculations."

  - id: 3
    name: "Eigenvalue Equation"
    formula: "A v = λ v"
    description: "Key in understanding principal components analysis (PCA) for dimensionality reduction."

  - id: 4
    name: "Softmax Function"
    formula: "softmax(z_i) = e^(z_i) / sum_j e^(z_j)"
    description: "Transforms logits into probability distributions, used in output layers and attention mechanisms."

  - id: 5
    name: "Cross-Entropy Loss"
    formula: "L = -sum_i y_i * log(y_hat_i)"
    description: "Measures difference between true and predicted distributions, key for training."

  - id: 6
    name: "Gradient Descent Update Rule"
    formula: "θ := θ - η ∇_θ J(θ)"
    description: "Used to optimize model parameters by minimizing loss."

  - id: 7
    name: "Backpropagation Chain Rule"
    formula: "∂L/∂x = ∂L/∂y * ∂y/∂x"
    description: "Basis for updating weights in neural networks."

  - id: 8
    name: "Attention Score Calculation (Scaled Dot-Product)"
    formula: "Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V"
    description: "Fundamental self-attention mechanism in transformer models."

  - id: 9
    name: "Positional Encoding"
    formula: "PE(pos,2i) = sin(pos / 10000^(2i/d_model)), PE(pos,2i+1) = cos(pos / 10000^(2i/d_model))"
    description: "Adds order information to tokens in a sequence."

  - id: 10
    name: "ReLU Activation Function"
    formula: "ReLU(x) = max(0, x)"
    description: "Non-linear activation used in neural networks."

  - id: 11
    name: "Bayes’ Theorem"
    formula: "P(A|B) = P(B|A) * P(A) / P(B)"
    description: "Used in probabilistic reasoning."

  - id: 12
    name: "Markov Decision Process (MDP) Expected Return"
    formula: "G_t = R_{t+1} + γ R_{t+2} + γ^2 R_{t+3} + ... = sum_{k=0}^∞ γ^k R_{t+k+1}"
    description: "Key in reinforcement learning, where γ is the discount factor."

  - id: 13
    name: "Bellman Equation"
    formula: "V^π(s) = E_π [ R_{t+1} + γ V^π(s_{t+1}) | s_t = s ]"
    description: "Describes the value function under a policy π."

  - id: 14
    name: "Q-Learning Update"
    formula: "Q(s_t,a_t) := Q(s_t,a_t) + α ( R_{t+1} + γ max_a Q(s_{t+1},a) - Q(s_t,a_t) )"
    description: "Update rule for Q-learning in reinforcement learning."

  - id: 15
    name: "Kullback-Leibler Divergence"
    formula: "D_KL(P||Q) = sum_i P(i) log(P(i)/Q(i))"
    description: "Measures how one probability distribution diverges from another."

  - id: 16
    name: "Variance and Standard Deviation"
    formula: "σ^2 = E[(X - μ)^2]"
    description: "Measures data spread, important for understanding data and regularization."

  - id: 17
    name: "Chain Rule in Probability"
    formula: "P(A,B) = P(A|B) * P(B)"
    description: "Used in probabilistic models and Bayesian networks."

  - id: 18
    name: "Adam Optimizer Equations"
    formula: "m_t = β1 m_{t-1} + (1 - β1) g_t, v_t = β2 v_{t-1} + (1 - β2) g_t^2"
    description: "With bias correction and parameter update."

  - id: 19
    name: "Dropout Regularization"
    formula: "Randomly sets input units to zero with probability p during training"
    description: "Reduces overfitting."

  - id: 20
    name: "Linear Regression Formula"
    formula: "y = Xβ + ε"
    description: "Fundamental model underlying many machine learning algorithms."

  # Additions: Missing Formulas (Gaps Filled for Completeness)
  - id: 21
    name: "Mean Squared Error (MSE) Loss"
    formula: "L = (1/n) Σ_i (ŷ_i - y_i)^2"
    description: "Regression loss measuring squared prediction errors; complements CE for continuous outputs in ML."

  - id: 22
    name: "Sigmoid Activation"
    formula: "Sigmoid(x) = 1 / (1 + e^{-x})"
    description: "S-shaped activation for binary classification and gating in RNNs/LLMs."

  - id: 23
    name: "Tanh Activation"
    formula: "Tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x})"
    description: "Hyperbolic tangent for bounded non-linearity, common in LSTMs and early transformers."

  - id: 24
    name: "Layer Normalization"
    formula: "LN(x) = (x - μ)/(σ + ε) * γ + β"
    description: "Normalizes across features for stable LLM training; γ/β learnable scale/shift."

  - id: 25
    name: "Feed-Forward Network (FFN)"
    formula: "FFN(x) = max(0, x W_1 + b_1) W_2 + b_2"
    description: "Position-wise MLP in transformers for non-linear feature expansion."

  - id: 26
    name: "Policy Gradient (REINFORCE)"
    formula: "∇_θ J(θ) = E_π [ ∇_θ log π_θ(a|s) R ]"
    description: "Direct policy optimization in RL for LLM alignment and decision-making."

  - id: 27
    name: "Xavier/Glorot Initialization"
    formula: "W ~ U(-√6/√(n_in + n_out), √6/√(n_in + n_out))"
    description: "Variance-preserving weight init to prevent vanishing/exploding gradients in deep nets."

  - id: 28
    name: "Convolution Operation"
    formula: "S(i,j) = (X * K)(i,j) = Σ_m Σ_n X(i+m,j+n) K(m,n)"
    description: "Spatial feature extraction in CNNs; used in vision-augmented LLMs."

  - id: 29
    name: "Perplexity (LLM Evaluation)"
    formula: "PPL = exp(-1/N Σ log P(w_i))"
    description: "Exponential of average negative log-likelihood; measures LLM predictive fluency."

  - id: 30
    name: "L2 Regularization"
    formula: "L_total = L + λ Σ w^2"
    description: "Weight decay penalty for overfitting control in LLM pre-training and fine-tuning."

  - id: 31
    name: "Batch Normalization"
    formula: "BN(x) = γ (x - μ_B)/σ_B + β"
    description: "Mini-batch normalization for faster, stable training in ML/RL pipelines."

  - id: 32
    name: "LoRA (Low-Rank Adaptation)"
    formula: "ΔW = B A (low-rank matrices B, A)"
    description: "Efficient fine-tuning for LLMs by updating low-rank adapters instead of full weights."

  - id: 33
    name: "Diffusion Forward Process"
    formula: "x_t = √α_t x_{t-1} + √(1-α_t) ε"
    description: "Noise addition in generative diffusion models for image/text synthesis in multimodal LLMs."

  - id: 34
    name: "PPO Clipped Objective"
    formula: "L_PPO = E[min(r(θ) Â, clip(r(θ), 1-ε, 1+ε) Â)]"
    description: "Proximal policy optimization for stable RLHF in aligning LLMs with preferences."

  - id: 35
    name: "Cosine Similarity"
    formula: "cos θ = (A · B) / (||A|| ||B||)"
    description: "Vector alignment metric for embeddings in retrieval and attention mechanisms."

# Updated Notes:
# - Covers original gaps: Activations (Sigmoid/Tanh), losses (MSE), transformer internals (FFN/LN), RL depth (Policy Grad), init (Xavier), conv ops.
# - 2025 additions: Eval (Perplexity), reg (L2/BN), fine-tune (LoRA), gen (Diffusion), alignment (PPO).
# - Total backbone: Linear alg + prob + gradients + attention + RL + modern scaling.

# These formulas and concepts collectively form the backbone of LLMs, general machine learning, and reinforcement learning models. Letters like K, V, Q specifically arise in the transformer attention formula, while others like A, B, C are general matrix/vector notation used in many equations. Understanding these will enable building, training, fine-tuning, and analyzing such models effectively. 
```


## Cheat sheet:

# LLM / ML / RL Cheat Sheet – Core Formulas

A concise reference for building, training, and analyzing LLMs, machine learning, and reinforcement learning models.

---

## 1. Linear Algebra & Neural Computations

| Formula | Purpose / Use | Symbols |
|---------|---------------|---------|
| `C[i,j] = Σ_k A[i,k] * B[k,j]` | Matrix multiplication, linear transformations | `A,B,C` matrices |
| `a · b = Σ_i a_i b_i` | Dot product, similarity scores, attention | `a,b` vectors |
| `Av = λv` | Eigenvalues, PCA | `A` matrix, `v` vector |
| `y = Wx + b` | Fully connected layer | `W` weights, `b` bias |
| `ReLU(x) = max(0,x)` | Non-linear activation | `x` input |
| `softmax(z_i) = e^{z_i} / Σ_j e^{z_j}` | Convert logits to probability distribution | `z_i` logits |

---

## 2. Loss & Optimization

| Formula | Purpose / Use |
|---------|---------------|
| `L = -Σ_i y_i log(ŷ_i)` | Cross-entropy loss (classification) |
| `L = (1/n) Σ_i (ŷ_i - y_i)^2` | Mean squared error (regression) |
| `θ := θ - η ∇_θ L` | Gradient descent update |
| Adam Optimizer:<br>`m_t = β1 m_{t-1} + (1-β1) g_t`<br>`v_t = β2 v_{t-1} + (1-β2) g_t^2`<br>`θ_t = θ_{t-1} - η (m_t / (1-β1^t)) / (√(v_t/(1-β2^t)) + ε)` | Adaptive optimization |

---

## 3. Backpropagation & Chain Rules

| Formula | Purpose / Use |
|---------|---------------|
| `∂L/∂x = (∂L/∂y) * (∂y/∂x)` | Gradient computation for backprop |
| `P(A,B) = P(A¦B) * P(B)` | Chain rule in probability, Bayesian networks |

---

## 4. Transformer & Attention Mechanics

| Formula | Purpose / Use |
|---------|---------------|
| `Attention(Q,K,V) = softmax(QK^T / √d_k) V` | Scaled dot-product attention, self-attention |
| `MultiHead(Q,K,V) = Concat(head_1,...,head_h) W^O`<br>`head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)` | Capture multiple representation subspaces |
| `PE(pos,2i) = sin(pos / 10000^{2i/d_model})`<br>`PE(pos,2i+1) = cos(pos / 10000^{2i/d_model})` | Positional encoding for token order |

---

## 5. Probability & Statistical Measures

| Formula | Purpose / Use |
|---------|---------------|
| `P(A¦B) = P(B¦A) * P(A) / P(B)` | Bayes’ theorem, probabilistic reasoning |
| `D_KL(P¦¦Q) = Σ_i P(i) log(P(i)/Q(i))` | Kullback-Leibler divergence |
| `σ^2 = E[(X-μ)^2]` | Variance, standard deviation |

---

## 6. Reinforcement Learning

| Formula | Purpose / Use |
|---------|---------------|
| `G_t = Σ_{k=0}^∞ γ^k R_{t+k+1}` | MDP expected return, discounted rewards |
| `V^π(s) = E_π [R_{t+1} + γ V^π(s_{t+1})]` | Bellman equation, value function |
| `Q(s_t,a_t) := Q(s_t,a_t) + α [R_{t+1} + γ max_a Q(s_{t+1},a) - Q(s_t,a_t)]` | Q-learning update |
| `∇_θ J(θ) = E_π [∇_θ log π_θ(a¦s) R]` | Policy gradient, REINFORCE |

---

## 7. Regularization

| Formula | Purpose / Use |
|---------|---------------|
| `y = x ⊙ mask, mask ~ Bernoulli(p)` | Dropout, reduces overfitting |

---

## 8. Linear / Regression Foundation

| Formula | Purpose / Use |
|---------|---------------|
| `y = Xβ + ε` | Linear regression, supervised learning |

---

### **Think Notes**
- K, Q, V = Key, Query, Value vectors in attention.  
- Most LLM formulas revolve around **linear algebra + probability + gradient updates**.  
- RL formulas introduce **expectations, discount factors, and policy optimization**.  
- This cheat sheet covers **ML fundamentals → Transformers → RL pipelines**.

---
 
## Cheat Sheet 2:


# Updated LLM / ML / RL Cheat Sheet – Core Formulas

---

## 1. Linear Algebra & Neural Computations

| Formula | Purpose / Use | Symbols |
|---------|---------------|---------|
| `C[i,j] = Σ_k A[i,k] * B[k,j]` | Matrix multiplication, linear transformations | `A,B,C` matrices |
| `a · b = Σ_i a_i b_i` | Dot product, similarity scores, attention | `a,b` vectors |
| `Av = λv` | Eigenvalues, PCA | `A` matrix, `v` vector |
| `y = Wx + b` | Fully connected layer | `W` weights, `b` bias |
| `ReLU(x) = max(0,x)` | Non-linear activation | `x` input |
| `Sigmoid(x) = 1 / (1 + e^{-x})` | S-shaped activation for binary/gating | `x` input |
| `Tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x})` | Bounded non-linearity for LSTMs | `x` input |
| `softmax(z_i) = e^{z_i} / Σ_j e^{z_j}` | Convert logits to probability distribution | `z_i` logits |
| `S(i,j) = (X * K)(i,j) = Σ_m Σ_n X(i+m,j+n) K(m,n)` | Convolution for spatial features (CNNs/vision LLMs) | `X` input, `K` kernel |

---

## 2. Loss & Optimization

| Formula | Purpose / Use |
|---------|---------------|
| `L = -Σ_i y_i log(ŷ_i)` | Cross-entropy loss (classification) |
| `L = (1/n) Σ_i (ŷ_i - y_i)^2` | Mean squared error (regression) |
| `θ := θ - η ∇_θ L` | Gradient descent update |
| Adam Optimizer:<br>`m_t = β1 m_{t-1} + (1-β1) g_t`<br>`v_t = β2 v_{t-1} + (1-β2) g_t^2`<br>`θ_t = θ_{t-1} - η (m_t / (1-β1^t)) / (√(v_t/(1-β2^t)) + ε)` | Adaptive optimization |
| `L_total = L + λ Σ w^2` | L2 Regularization (weight decay) for overfitting control |
| `PPL = exp(-1/N Σ log P(w_i))` | Perplexity for LLM fluency evaluation |

---

## 3. Backpropagation & Chain Rules

| Formula | Purpose / Use |
|---------|---------------|
| `∂L/∂x = (∂L/∂y) * (∂y/∂x)` | Gradient computation for backprop |
| `P(A,B) = P(A\|B) * P(B)` | Chain rule in probability, Bayesian networks |

---

## 4. Transformer & Attention Mechanics

| Formula | Purpose / Use |
|---------|---------------|
| `Attention(Q,K,V) = softmax(QK^T / √d_k) V` | Scaled dot-product attention, self-attention |
| `MultiHead(Q,K,V) = Concat(head_1,...,head_h) W^O`<br>`head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)` | Capture multiple representation subspaces |
| `PE(pos,2i) = sin(pos / 10000^{2i/d_model})`<br>`PE(pos,2i+1) = cos(pos / 10000^{2i/d_model})` | Positional encoding for token order |
| `LN(x) = (x - μ)/(σ + ε) * γ + β` | Layer Normalization for stable training (γ/β learnable) |
| `FFN(x) = max(0, x W_1 + b_1) W_2 + b_2` | Position-wise MLP for feature expansion |
| `W ~ U(-√6/√(n_in + n_out), √6/√(n_in + n_out))` | Xavier/Glorot weight initialization to avoid gradient issues |
| `cos θ = (A · B) / (||A|| ||B||)` | Cosine similarity for embeddings/retrieval |

---

## 5. Probability & Statistical Measures

| Formula | Purpose / Use |
|---------|---------------|
| `P(A\|B) = P(B\|A) * P(A) / P(B)` | Bayes’ theorem, probabilistic reasoning |
| `D_KL(P\|Q) = Σ_i P(i) log(P(i)/Q(i))` | Kullback-Leibler divergence |
| `σ^2 = E[(X-μ)^2]` | Variance, standard deviation |
| `f(x) = (1/(σ √(2π))) exp(- (x-μ)^2 / (2σ^2))` | Normal (Gaussian) Distribution for sampling/prob LLMs |

---

## 6. Reinforcement Learning

| Formula | Purpose / Use |
|---------|---------------|
| `G_t = Σ_{k=0}^∞ γ^k R_{t+k+1}` | MDP expected return, discounted rewards |
| `V^π(s) = E_π [R_{t+1} + γ V^π(s_{t+1})]` | Bellman equation, value function |
| `Q(s_t,a_t) := Q(s_t,a_t) + α [R_{t+1} + γ max_a Q(s_{t+1},a) - Q(s_t,a_t)]` | Q-learning update |
| `∇_θ J(θ) = E_π [∇_θ log π_θ(a\|s) R]` | Policy gradient, REINFORCE |
| `L_PPO = E[min(r(θ) Â, clip(r(θ), 1-ε, 1+ε) Â)]` | PPO clipped objective for stable RLHF alignment |
| `L_RM = -Σ [r log σ(y) + (1-r) log(1-σ(y))]` | Reward Model Loss for preference-based RLHF |

---

## 7. Regularization & Normalization

| Formula | Purpose / Use |
|---------|---------------|
| `y = x ⊙ mask, mask ~ Bernoulli(p)` | Dropout, reduces overfitting |
| `BN(x) = γ (x - μ_B)/σ_B + β` | Batch Normalization for mini-batch stability |

---

## 8. Linear / Regression Foundation

| Formula | Purpose / Use |
|---------|---------------|
| `y = Xβ + ε` | Linear regression, supervised learning |

---

## 9. Generative & Fine-Tuning (2025 Additions)

| Formula | Purpose / Use |
|---------|---------------|
| `x_t = √α_t x_{t-1} + √(1-α_t) ε` | Diffusion Forward Process for multimodal generation |
| `L_VAE = ||x - \hat{x}||^2 + D_KL(q(z\|x) \| p(z))` | VAE Loss for latent space learning in autoencoders |
| `ΔW = B A` (low-rank matrices B, A) | LoRA for efficient LLM fine-tuning |
| `BLEU = BP · exp(Σ w_n log p_n)` | BLEU Score for text generation evaluation |
| `ROUGE-N = Σ (overlapping n-grams) / Σ (candidate n-grams)` | ROUGE-N recall for summarization quality |
| `ELBO = E_q[log p(x\|z)] - D_KL(q(z\|x) \| p(z))` | Evidence Lower Bound for variational inference |
| `FlashAttn(Q,K,V) ≈ O(N)` (via tiling/blocking) | FlashAttention for efficient long-sequence inference |

---

### **Think Notes**
- K, Q, V = Key, Query, Value vectors in attention.  
- Most LLM formulas revolve around **linear algebra + probability + gradient updates**.  
- RL formulas introduce **expectations, discount factors, and policy optimization**.  
- This cheat sheet covers **ML fundamentals → Transformers → RL pipelines** + 2025 evos (gen/fine-tune/align).  
- Fork for Quillan/ANGELA: Use in HMoE for deriving swarm ethics or embodied resonance.

---