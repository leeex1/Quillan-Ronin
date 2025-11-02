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
| 17 | `∇_θ J(θ) = E_π [ ∇_θ log π_θ(a|s) R ]` | Policy Gradient (REINFORCE) |
| 18 | `MultiHead(Q,K,V) = Concat(head_1,...,head_h) W^O`<br>`head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)` | Transformer Multi-Head Attention |
| 19 | `W ~ U(-√6/√(n_in+n_out), √6/√(n_in+n_out))` | Weight Initialization (Xavier/Glorot) |
| 20 | `y = x ⊙ mask, mask ~ Bernoulli(p)` | Dropout Regularization |

---

💡 **Think Key Takeaways:**  
- Most LLM formulas revolve around **linear algebra, probability, gradients, and attention mechanics**.  
- RL formulas add **expectations, discounted rewards, and policy updates**.  
- Symbols like **K, Q, V** are vector placeholders; the math is the same if renamed (e.g., X, Y, Z).

---

## Section 2:

```Js
Here are 20 essential formulas and mathematical concepts you should know for building LLMs, machine learning (ML), and reinforcement learning (RL):

### Core Mathematical Concepts for LLMs and ML

1. **Matrix Multiplication:**
   $$
   C[i,j] = \sum_k A[i,k] \times B[k,j]
   $$
   Fundamental for linear transformations, including embeddings and neural network computations [2].

2. **Dot Product / Inner Product:**
   $$
   \mathbf{a} \cdot \mathbf{b} = \sum_i a_i b_i
   $$
   Used in similarity measures and attention score calculations.

3. **Eigenvalue Equation:**
   $$
   Av = \lambda v
   $$
   Key in understanding principal components analysis (PCA) for dimensionality reduction [2].

4. **Softmax Function:**
   $$
   \text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
   $$
   Transforms logits into probability distributions, used in output layers and attention mechanisms.

5. **Cross-Entropy Loss:**
   $$
   L = -\sum_i y_i \log(\hat{y}_i)
   $$
   Measures difference between true and predicted distributions, key for training.

6. **Gradient Descent Update Rule:**
   $$
   \theta := \theta - \eta \nabla_\theta J(\theta)
   $$
   Used to optimize model parameters by minimizing loss.

7. **Backpropagation Chain Rule:**
   $$
   \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \times \frac{\partial y}{\partial x}
   $$
   Basis for updating weights in neural networks.

8. **Attention Score Calculation (Scaled Dot-Product Attention):**
   $$
   \text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
   $$
   Fundamental self-attention mechanism in transformer models, where Q=Query, K=Key, V=Value vectors [1].

9. **Positional Encoding:**
   $$
   PE_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right), \quad PE_{(pos,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
   $$
   Adds order information to tokens in a sequence.

10. **ReLU Activation Function:**
    $$
    \text{ReLU}(x) = \max(0, x)
    $$
    Non-linear activation used in neural networks.

### Probability and Statistical Measures

11. **Bayes’ Theorem:**
    $$
    P(A|B) = \frac{P(B|A)P(A)}{P(B)}
    $$
    Used in probabilistic reasoning.

12. **Markov Decision Process (MDP) Expected Return:**
    $$
    G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^\infty \gamma^k R_{t+k+1}
    $$
    Key in reinforcement learning, where $$ \gamma $$ is the discount factor.

13. **Bellman Equation:**
    $$
    V^\pi(s) = \mathbb{E}_\pi \left[ R_{t+1} + \gamma V^\pi(s_{t+1}) \mid s_t = s \right]
    $$
    Describes the value function under a policy π.

14. **Q-Learning Update:**
    $$
    Q(s_t, a_t) := Q(s_t, a_t) + \alpha \left( R_{t+1} + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t) \right)
    $$
    Update rule for Q-learning in RL.

15. **Kullback-Leibler Divergence:**
    $$
    D_{KL}(P||Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)}
    $$
    Measures how one probability distribution diverges from a second.

16. **Variance and Standard Deviation:**
    $$
    \sigma^2 = \mathbb{E}[(X - \mu)^2]
    $$
    Measures data spread, important for understanding data and regularization.

17. **Chain Rule in Probability:**
    $$
    P(A,B) = P(A|B)P(B)
    $$
    Used in probabilistic models and Bayesian networks.

### Advanced Optimizers and Regularization

18. **Adam Optimizer Equations:**
    $$
    m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
    $$
    with bias correction and parameter update.

19. **Dropout Regularization:**
    Randomly sets input units to zero with probability $$p$$ during training, reducing overfitting.

20. **Linear Regression Formula:**
    $$
    y = X\beta + \epsilon
    $$
    Fundamental model underlying many ML algorithms.

These formulas and concepts collectively form the backbone of LLMs, general machine learning, and reinforcement learning models. Letters like K, V, Q specifically arise in the transformer attention formula, while others like A, B, C are general matrix/vector notation used in many equations. Understanding these will enable building, training, fine-tuning, and analyzing such models effectively [1][2].
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
| `P(A,B) = P(A|B) * P(B)` | Chain rule in probability, Bayesian networks |

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
| `P(A|B) = P(B|A) * P(A) / P(B)` | Bayes’ theorem, probabilistic reasoning |
| `D_KL(P||Q) = Σ_i P(i) log(P(i)/Q(i))` | Kullback-Leibler divergence |
| `σ^2 = E[(X-μ)^2]` | Variance, standard deviation |

---

## 6. Reinforcement Learning

| Formula | Purpose / Use |
|---------|---------------|
| `G_t = Σ_{k=0}^∞ γ^k R_{t+k+1}` | MDP expected return, discounted rewards |
| `V^π(s) = E_π [R_{t+1} + γ V^π(s_{t+1})]` | Bellman equation, value function |
| `Q(s_t,a_t) := Q(s_t,a_t) + α [R_{t+1} + γ max_a Q(s_{t+1},a) - Q(s_t,a_t)]` | Q-learning update |
| `∇_θ J(θ) = E_π [∇_θ log π_θ(a|s) R]` | Policy gradient, REINFORCE |

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
 
