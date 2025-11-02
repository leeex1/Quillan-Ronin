# Must know formulas

Here’s a distilled list of 20 essential formulas/concepts you should have at your fingertips for LLMs, ML, and RL:


1. Linear Layer (Fully Connected Layer)

y = Wx + b

Fundamental building block for MLPs and transformers.



2. Activation Functions

ReLU: 

Sigmoid: 

Tanh: 


3. Softmax

\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}


4. Cross-Entropy Loss (Classification)

L = - \sum_i y_i \log(\hat{y}_i)


5. Mean Squared Error (Regression)

L = \frac{1}{n} \sum_i (\hat{y}_i - y_i)^2


6. Gradient Descent Update

\theta \leftarrow \theta - \eta \frac{\partial L}{\partial \theta}


7. Adam Optimizer

m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t

v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 

\hat{\theta}_t = \theta_{t-1} - \eta \frac{m_t / (1-\beta_1^t)}{\sqrt{v_t / (1-\beta_2^t)} + \epsilon}



8. Attention (Scaled Dot-Product)

\text{Attention}(Q,K,V) = \text{softmax}\Big(\frac{QK^\top}{\sqrt{d_k}}\Big) V



9. Positional Encoding

PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_\text{model}}}\right), \quad
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_\text{model}}}\right)



10. Layer Normalization

\text{LN}(x) = \frac{x - \mu}{\sigma + \epsilon} \cdot \gamma + \beta



11. Transformer Feed-Forward

\text{FFN}(x) = \max(0, xW_1 + b_1) W_2 + b_2


12. KL Divergence (for knowledge distillation, variational models)

D_\text{KL}(P || Q) = \sum_i P(i) \log\frac{P(i)}{Q(i)}



13. Backpropagation Chain Rule

\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial x}


14. Convolution Operation (CNNs, sometimes in embedding)

S(i,j) = (X * K)(i,j) = \sum_m \sum_n X(i+m,j+n) K(m,n)



15. Reinforcement Learning: Bellman Equation

V^\pi(s) = \mathbb{E}_\pi \Big[ r_t + \gamma V^\pi(s_{t+1}) \Big]


16. Q-Learning Update

Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \big[ r_t + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t) \big]


17. Policy Gradient (REINFORCE)

\nabla_\theta J(\theta) = \mathbb{E}_\pi \big[ \nabla_\theta \log \pi_\theta(a|s) R \big]



18. Transformer Multi-Head Attention

\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O

\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V) 


19. Weight Initialization (Xavier/Glorot)

W \sim \mathcal{U}\Big(-\frac{\sqrt{6}}{\sqrt{n_\text{in} + n_\text{out}}}, \frac{\sqrt{6}}{\sqrt{n_\text{in} + n_\text{out}}}\Big)


20. Dropout Regularization

y = x \odot \text{mask}, \quad \text{mask} \sim \text{Bernoulli}(p)


💡 Think key takeaways:

Most LLM formulas revolve around linear algebra (matrix ops), probabilities, gradients, and attention mechanics.

RL adds expectations, discounted rewards, and policy updates.

The symbols like K, Q, V are not hard-coded—they’re just vector placeholders; the math is the same if you rename them X, Y, Z.



---


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
 
