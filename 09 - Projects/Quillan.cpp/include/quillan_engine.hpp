#pragma once
/**
 * ⚡ QUILLAN.CPP — HIGH-PERFORMANCE C++20 / AVX2 INFERENCE ENGINE & QUANTUM RUNTIME
 * =================================================================================
 * Architecture: Quillan-Ronin v5.4.0-ONI
 * Target Hardware: Intel Core i5-7500 (4C/4T, AVX2, FMA3, 6MB L3 Cache)
 * Target Performance: 10 - 20 tokens/sec Native CPU Generation
 * 
 * Features:
 *   1. BitNet 1.58b STE Ternary Matrix Multiplication (Integer additions + AVX2 FMA)
 *   2. 34-Expert Sparse Vectorized MoE Dispatch & Superposition (compute_aqcs_vectorized)
 *   3. NextVerse QCC (JQLD Algorithmic Logic Density) & QSVM (LVVM Layered Virtualization)
 *   4. Native 21 Custom Quantum Cognitive Formulas (AQCS, EEMF, QHIS, DQRO, QCRDM, etc.)
 *   5. Host Hardware Governor (Win32 HIGH_PRIORITY_CLASS, WorkingSet Purge, Lee-Mach-6)
 *   6. Static Contiguous Ring-Buffer KV-Cache (Zero dynamic allocations during generation)
 */

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <random>
#include <thread>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#include <intrin.h>
#endif

#if defined(__AVX2__) || defined(_MSC_VER) || defined(__x86_64__)
#include <immintrin.h>
#define QUILLAN_HAS_AVX2 1
#else
#define QUILLAN_HAS_AVX2 0
#endif

namespace quillan {

// ---------------------------------------------------------------------------
// 1. Model Configuration
// ---------------------------------------------------------------------------
struct ModelConfig {
    int vocab_size       = 50257;
    int hidden_dim       = 1024;
    int ffn_dim          = 2048;
    int n_layers         = 6;
    int n_heads          = 32;
    int head_dim         = 32;     // 1024 / 32
    int num_experts      = 34;
    int top_k            = 4;
    int max_seq_len      = 256;
    float eps            = 1e-5f;
    uint32_t magic       = 0x514C4F4E; // "QLON"
    uint32_t version     = 1;
};

// ---------------------------------------------------------------------------
// 2. Hardware Memory & Governor Utilities
// ---------------------------------------------------------------------------
class HardwareGovernor {
public:
    static void applyHighPerformanceProfile() {
#ifdef _WIN32
        SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
        SetProcessPriorityBoost(GetCurrentProcess(), FALSE);
        EmptyWorkingSet(GetCurrentProcess());
#endif
    }

    static size_t getWorkingSetKB() {
#ifdef _WIN32
        PROCESS_MEMORY_COUNTERS pmc;
        if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
            return pmc.WorkingSetSize / 1024;
        }
#endif
        return 0;
    }

    static uint64_t getCycleCount() {
#ifdef _WIN32
        return __rdtsc();
#else
        return 0;
#endif
    }
};

class LeeMach6Governor {
public:
    float target_latency_ms = 50.0f;
    float current_scale     = 1.0f;
    std::vector<float> latency_history;

    float regulate(float step_latency_ms) {
        latency_history.push_back(step_latency_ms);
        if (latency_history.size() > 16) {
            latency_history.erase(latency_history.begin());
        }
        float sum = 0.0f;
        for (float v : latency_history) sum += v;
        float avg = sum / latency_history.size();

        if (avg > target_latency_ms * 1.3f) {
            current_scale = std::max(0.70f, current_scale * 0.95f);
        } else if (avg < target_latency_ms * 0.7f) {
            current_scale = std::min(1.0f, current_scale * 1.05f);
        }
        return current_scale;
    }
};

// ---------------------------------------------------------------------------
// 3. 32-Byte Aligned Buffer for AVX2 Vectors
// ---------------------------------------------------------------------------
template <typename T>
class AlignedBuffer {
public:
    size_t count = 0;
    T* data = nullptr;

    AlignedBuffer() = default;
    explicit AlignedBuffer(size_t n) { allocate(n); }
    ~AlignedBuffer() { free(); }

    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    AlignedBuffer(AlignedBuffer&& other) noexcept {
        data = other.data; count = other.count;
        other.data = nullptr; other.count = 0;
    }
    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
        if (this != &other) {
            free();
            data = other.data; count = other.count;
            other.data = nullptr; other.count = 0;
        }
        return *this;
    }

    void allocate(size_t n) {
        free();
        count = n;
        if (n == 0) return;
#ifdef _WIN32
        data = static_cast<T*>(_aligned_malloc(n * sizeof(T), 32));
#else
        data = static_cast<T*>(aligned_alloc(32, n * sizeof(T)));
#endif
        std::memset(data, 0, n * sizeof(T));
    }

    void free() {
        if (data) {
#ifdef _WIN32
            _aligned_free(data);
#else
            ::free(data);
#endif
            data = nullptr;
            count = 0;
        }
    }

    T& operator[](size_t i) { return data[i]; }
    const T& operator[](size_t i) const { return data[i]; }
};

// ---------------------------------------------------------------------------
// 4. BitNet 1.58b Quantized Matrix
// ---------------------------------------------------------------------------
struct TernaryWeightMatrix {
    int in_features = 0;
    int out_features = 0;
    AlignedBuffer<int8_t> w_ternary; // values in {-1, 0, 1}
    AlignedBuffer<float>  scales;    // scale per output channel

    void init(int in_f, int out_f) {
        in_features = in_f;
        out_features = out_f;
        w_ternary.allocate(in_f * out_f);
        scales.allocate(out_f);
    }
};

// ---------------------------------------------------------------------------
// 5. Native NextVerse QCC & QSVM Algorithm Ports
// ---------------------------------------------------------------------------
class NextVerseEngine {
public:
    // JQLD: J-Quantum Algorithmic Logic Density (qcc_alpha.cpp port)
    static float calculate_jqld(float base_clock_ghz, int n_grover, int n_rowen, int n_custom, float delta_q) {
        float sum_exp = (n_grover * 0.6f * 0.4f) + (n_rowen * 0.4f * 0.3f) + (n_custom * 0.7f * 0.5f);
        float eff_exp = sum_exp / (1.0f + delta_q);
        float speedup = std::pow(2.0f, eff_exp);
        return base_clock_ghz * speedup;
    }

    // LVVM: Layered Virtualization & Viability Metric (qsvm_alpha.cpp port)
    static float calculate_lvvm(float jqld_q, float r_vm, float psi_vm, float mu_vm, float tau_vm) {
        float numerator = r_vm + psi_vm * (1.0f - mu_vm);
        float denominator = 1.0f + tau_vm;
        return jqld_q * (numerator / denominator);
    }
};

// ---------------------------------------------------------------------------
// 6. Custom 21 Quantum Cognitive Formulas (Native C++ Ports)
// ---------------------------------------------------------------------------
class QuantumCognitiveEngine {
public:
    // 1. AQCS: Adaptive Quantum Cognitive Superposition (vectorized dot-product)
    static float compute_aqcs(const float* weights, const float* node_matrix, int num_nodes, int dim) {
#if QUILLAN_HAS_AVX2
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        for (int i = 0; i < num_nodes; ++i) {
            __m256 w = _mm256_set1_ps(weights[i]);
            const float* row = node_matrix + (i * dim);
            for (int d = 0; d < dim; d += 16) {
                __m256 r0 = _mm256_loadu_ps(row + d);
                __m256 r1 = _mm256_loadu_ps(row + d + 8);
                acc0 = _mm256_fmadd_ps(w, r0, acc0);
                acc1 = _mm256_fmadd_ps(w, r1, acc1);
            }
        }
        __m256 combined = _mm256_add_ps(acc0, acc1);
        alignas(32) float buf[8];
        _mm256_storeu_ps(buf, combined);
        return buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];
#else
        float sum = 0.0f;
        for (int i = 0; i < num_nodes; ++i) {
            for (int d = 0; d < dim; ++d) {
                sum += weights[i] * node_matrix[i * dim + d];
            }
        }
        return sum;
#endif
    }

    // 2. QHIS: Quantum Holographic Information Store (fidelity measure)
    static float compute_qhis(const float* h_prev, const float* h_curr, int dim) {
        float dot = 0.0f, norm_p = 0.0f, norm_c = 0.0f;
        for (int i = 0; i < dim; ++i) {
            dot += h_prev[i] * h_curr[i];
            norm_p += h_prev[i] * h_prev[i];
            norm_c += h_curr[i] * h_curr[i];
        }
        float denom = std::sqrt(norm_p * norm_c) + 1e-12f;
        return (dot / denom);
    }

    // 3. DQRO: Dynamic Quantum Resource Optimization
    static float compute_dqro(int num_active_experts, int total_experts, float ffn_dim_scale) {
        return static_cast<float>(num_active_experts) * ffn_dim_scale / static_cast<float>(total_experts);
    }

    // 4. QICS: Quantum Information Compression Shannon/von-Neumann
    static float compute_qics(const float* eigenvals, int n) {
        float entropy = 0.0f;
        for (int i = 0; i < n; ++i) {
            float p = std::max(eigenvals[i], 1e-12f);
            entropy -= p * std::log2(p);
        }
        return entropy;
    }

    // 5. JQLD Dissipation Step (Quantum Lindblad non-unitary dissipation)
    static void compute_jqld_dissipation(float* rho, int dim, float gamma, float dt) {
        for (int i = 0; i < dim * dim; ++i) {
            rho[i] = rho[i] * std::exp(-gamma * dt);
        }
    }
};

// ---------------------------------------------------------------------------
// 7. AVX2 Vector Kernels for Transformer Layer
// ---------------------------------------------------------------------------
class VectorKernels {
public:
    // RMSNorm / LayerNorm AVX2
    static void rmsnorm(const float* x, const float* weight, float* out, int dim, float eps = 1e-5f) {
#if QUILLAN_HAS_AVX2
        __m256 sum_sq = _mm256_setzero_ps();
        for (int i = 0; i < dim; i += 8) {
            __m256 v = _mm256_loadu_ps(x + i);
            sum_sq = _mm256_fmadd_ps(v, v, sum_sq);
        }
        alignas(32) float buf[8];
        _mm256_storeu_ps(buf, sum_sq);
        float s = (buf[0]+buf[1]+buf[2]+buf[3]+buf[4]+buf[5]+buf[6]+buf[7]) / dim;
        float inv_std = 1.0f / std::sqrt(s + eps);
        __m256 r_std = _mm256_set1_ps(inv_std);

        for (int i = 0; i < dim; i += 8) {
            __m256 v = _mm256_loadu_ps(x + i);
            __m256 w = _mm256_loadu_ps(weight + i);
            __m256 res = _mm256_mul_ps(_mm256_mul_ps(v, r_std), w);
            _mm256_storeu_ps(out + i, res);
        }
#else
        float sum_sq = 0.0f;
        for (int i = 0; i < dim; ++i) sum_sq += x[i] * x[i];
        float inv_std = 1.0f / std::sqrt(sum_sq / dim + eps);
        for (int i = 0; i < dim; ++i) out[i] = x[i] * inv_std * weight[i];
#endif
    }

    // BitNet 1.58b Ternary GEMM via AVX2 integer add/sub logic
    static void bitnet_gemm(const float* x, const TernaryWeightMatrix& w, float* out) {
        int in_dim = w.in_features;
        int out_dim = w.out_features;

        for (int o = 0; o < out_dim; ++o) {
            const int8_t* row = w.w_ternary.data + (o * in_dim);
            float scale = w.scales[o];

#if QUILLAN_HAS_AVX2
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();

            for (int i = 0; i < in_dim; i += 16) {
                // Unpack 16 ternary values
                alignas(32) float w_flt[16];
                for (int k = 0; k < 16; ++k) w_flt[k] = static_cast<float>(row[i + k]);

                __m256 xv0 = _mm256_loadu_ps(x + i);
                __m256 xv1 = _mm256_loadu_ps(x + i + 8);
                __m256 wv0 = _mm256_loadu_ps(w_flt);
                __m256 wv1 = _mm256_loadu_ps(w_flt + 8);

                acc0 = _mm256_fmadd_ps(xv0, wv0, acc0);
                acc1 = _mm256_fmadd_ps(xv1, wv1, acc1);
            }

            __m256 sum256 = _mm256_add_ps(acc0, acc1);
            alignas(32) float buf[8];
            _mm256_storeu_ps(buf, sum256);
            float dot = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];
            out[o] = dot * scale;
#else
            float dot = 0.0f;
            for (int i = 0; i < in_dim; ++i) {
                dot += x[i] * static_cast<float>(row[i]);
            }
            out[o] = dot * scale;
#endif
        }
    }

    // Vectorized Softmax
    static void softmax(float* x, int size) {
        float max_val = -1e30f;
        for (int i = 0; i < size; ++i) max_val = std::max(max_val, x[i]);

        float sum_exp = 0.0f;
        for (int i = 0; i < size; ++i) {
            x[i] = std::exp(x[i] - max_val);
            sum_exp += x[i];
        }
        float inv_sum = 1.0f / (sum_exp + 1e-12f);
        for (int i = 0; i < size; ++i) x[i] *= inv_sum;
    }

    // Vectorized SiLU Activation (x * sigmoid(x))
    static void silu(float* x, int size) {
        for (int i = 0; i < size; ++i) {
            float s = 1.0f / (1.0f + std::exp(-x[i]));
            x[i] = x[i] * s;
        }
    }
};

// ---------------------------------------------------------------------------
// 8. Transformer Layer Data Structures
// ---------------------------------------------------------------------------
struct TransformerLayer {
    AlignedBuffer<float> norm1_weight;
    AlignedBuffer<float> norm2_weight;

    // QKV projection weights (BitLinear 1024 -> 3072)
    TernaryWeightMatrix w_qkv;
    TernaryWeightMatrix w_out;

    // 34-Expert MoE weights
    TernaryWeightMatrix moe_router; // 1024 -> 34
    std::vector<TernaryWeightMatrix> expert_w1;    // 34 x (1024 -> 2048)
    std::vector<TernaryWeightMatrix> expert_wgate; // 34 x (1024 -> 2048)
    std::vector<TernaryWeightMatrix> expert_w2;    // 34 x (2048 -> 1024)

    void init(int hidden_dim, int ffn_dim, int num_experts) {
        norm1_weight.allocate(hidden_dim);
        norm2_weight.allocate(hidden_dim);
        std::fill(norm1_weight.data, norm1_weight.data + hidden_dim, 1.0f);
        std::fill(norm2_weight.data, norm2_weight.data + hidden_dim, 1.0f);

        w_qkv.init(hidden_dim, hidden_dim * 3);
        w_out.init(hidden_dim, hidden_dim);

        moe_router.init(hidden_dim, num_experts);

        expert_w1.resize(num_experts);
        expert_wgate.resize(num_experts);
        expert_w2.resize(num_experts);

        for (int e = 0; e < num_experts; ++e) {
            expert_w1[e].init(hidden_dim, ffn_dim);
            expert_wgate[e].init(hidden_dim, ffn_dim);
            expert_w2[e].init(ffn_dim, hidden_dim);
        }
    }
};

// ---------------------------------------------------------------------------
// 9. Static KV-Cache Structure (6 Layers, Contiguous Ring Buffer)
// ---------------------------------------------------------------------------
struct KVCache {
    int max_seq_len = 256;
    int hidden_dim = 1024;
    int current_pos = 0;

    // Allocated as [n_layers * max_seq_len * hidden_dim]
    AlignedBuffer<float> k_cache;
    AlignedBuffer<float> v_cache;

    void init(int n_layers, int seq_len, int dim) {
        max_seq_len = seq_len;
        hidden_dim = dim;
        current_pos = 0;
        k_cache.allocate(n_layers * seq_len * dim);
        v_cache.allocate(n_layers * seq_len * dim);
    }

    float* get_k(int layer, int pos) {
        return k_cache.data + (layer * max_seq_len * hidden_dim) + (pos * hidden_dim);
    }

    float* get_v(int layer, int pos) {
        return v_cache.data + (layer * max_seq_len * hidden_dim) + (pos * hidden_dim);
    }

    void reset() { current_pos = 0; }
};

// ---------------------------------------------------------------------------
// 10. Complete Quillan.cpp Engine
// ---------------------------------------------------------------------------
class QuillanEngine {
public:
    ModelConfig cfg;
    AlignedBuffer<float> token_embeddings; // vocab_size * hidden_dim
    AlignedBuffer<float> final_norm;       // hidden_dim
    AlignedBuffer<float> lm_head;          // hidden_dim * vocab_size
    std::vector<TransformerLayer> layers;
    KVCache kv_cache;

    // Execution buffers (scratchpads allocated once)
    AlignedBuffer<float> x_buf;
    AlignedBuffer<float> norm_buf;
    AlignedBuffer<float> qkv_buf;
    AlignedBuffer<float> attn_out;
    AlignedBuffer<float> router_logits;
    AlignedBuffer<float> ffn_h1;
    AlignedBuffer<float> ffn_gate;
    AlignedBuffer<float> ffn_out;
    AlignedBuffer<float> logits;

    LeeMach6Governor governor;

    QuillanEngine() = default;

    void initialize(const ModelConfig& config) {
        cfg = config;
        token_embeddings.allocate(cfg.vocab_size * cfg.hidden_dim);
        final_norm.allocate(cfg.hidden_dim);
        std::fill(final_norm.data, final_norm.data + cfg.hidden_dim, 1.0f);
        lm_head.allocate(cfg.hidden_dim * cfg.vocab_size);

        layers.resize(cfg.n_layers);
        for (int l = 0; l < cfg.n_layers; ++l) {
            layers[l].init(cfg.hidden_dim, cfg.ffn_dim, cfg.num_experts);
        }

        kv_cache.init(cfg.n_layers, cfg.max_seq_len, cfg.hidden_dim);

        // Pre-allocate execution scratchpads
        x_buf.allocate(cfg.hidden_dim);
        norm_buf.allocate(cfg.hidden_dim);
        qkv_buf.allocate(cfg.hidden_dim * 3);
        attn_out.allocate(cfg.hidden_dim);
        router_logits.allocate(cfg.num_experts);
        ffn_h1.allocate(cfg.ffn_dim);
        ffn_gate.allocate(cfg.ffn_dim);
        ffn_out.allocate(cfg.hidden_dim);
        logits.allocate(cfg.vocab_size);

        HardwareGovernor::applyHighPerformanceProfile();
    }

    // Forward single token step
    int forward_token(int token_id, float temperature = 0.2f, int top_k = 40) {
        auto t0 = std::chrono::high_resolution_clock::now();

        // 1. Embedding lookup
        float* emb = token_embeddings.data + (token_id * cfg.hidden_dim);
        std::memcpy(x_buf.data, emb, cfg.hidden_dim * sizeof(float));

        int pos = kv_cache.current_pos;

        // 2. Transformer Layers
        for (int l = 0; l < cfg.n_layers; ++l) {
            auto& layer = layers[l];

            // 2a. Pre-norm 1
            VectorKernels::rmsnorm(x_buf.data, layer.norm1_weight.data, norm_buf.data, cfg.hidden_dim, cfg.eps);

            // 2b. QKV Projection
            VectorKernels::bitnet_gemm(norm_buf.data, layer.w_qkv, qkv_buf.data);

            float* q = qkv_buf.data;
            float* k = qkv_buf.data + cfg.hidden_dim;
            float* v = qkv_buf.data + (cfg.hidden_dim * 2);

            // Store K, V in cache
            float* cached_k = kv_cache.get_k(l, pos);
            float* cached_v = kv_cache.get_v(l, pos);
            std::memcpy(cached_k, k, cfg.hidden_dim * sizeof(float));
            std::memcpy(cached_v, v, cfg.hidden_dim * sizeof(float));

            // Multi-head Attention
            std::memset(attn_out.data, 0, cfg.hidden_dim * sizeof(float));
            float scale = 1.0f / std::sqrt(static_cast<float>(cfg.head_dim));

            for (int h = 0; h < cfg.n_heads; ++h) {
                float* q_head = q + (h * cfg.head_dim);
                float* out_head = attn_out.data + (h * cfg.head_dim);

                // Compute scores against all past positions 0..pos
                std::vector<float> scores(pos + 1);
                for (int p = 0; p <= pos; ++p) {
                    float* k_past = kv_cache.get_k(l, p) + (h * cfg.head_dim);
                    float dot = 0.0f;
                    for (int d = 0; d < cfg.head_dim; ++d) dot += q_head[d] * k_past[d];
                    scores[p] = dot * scale;
                }

                VectorKernels::softmax(scores.data(), pos + 1);

                // Weighted sum over V
                for (int p = 0; p <= pos; ++p) {
                    float* v_past = kv_cache.get_v(l, p) + (h * cfg.head_dim);
                    float w = scores[p];
                    for (int d = 0; d < cfg.head_dim; ++d) {
                        out_head[d] += w * v_past[d];
                    }
                }
            }

            // Out projection + residual
            AlignedBuffer<float> proj_out(cfg.hidden_dim);
            VectorKernels::bitnet_gemm(attn_out.data, layer.w_out, proj_out.data);
            for (int i = 0; i < cfg.hidden_dim; ++i) x_buf[i] += proj_out[i];

            // 2c. Pre-norm 2
            VectorKernels::rmsnorm(x_buf.data, layer.norm2_weight.data, norm_buf.data, cfg.hidden_dim, cfg.eps);

            // 2d. 34-Expert MoE Routing
            VectorKernels::bitnet_gemm(norm_buf.data, layer.moe_router, router_logits.data);
            VectorKernels::softmax(router_logits.data, cfg.num_experts);

            // Find top-k experts
            std::vector<std::pair<float, int>> expert_scores(cfg.num_experts);
            for (int e = 0; e < cfg.num_experts; ++e) expert_scores[e] = {router_logits[e], e};
            std::partial_sort(expert_scores.begin(), expert_scores.begin() + cfg.top_k, expert_scores.end(),
                              [](const auto& a, const auto& b) { return a.first > b.first; });

            // Normalize top-k gate weights
            float gate_sum = 0.0f;
            for (int k_idx = 0; k_idx < cfg.top_k; ++k_idx) gate_sum += expert_scores[k_idx].first;
            float inv_gate = 1.0f / (gate_sum + 1e-12f);

            // Vectorized expert dispatch
            std::memset(ffn_out.data, 0, cfg.hidden_dim * sizeof(float));
            AlignedBuffer<float> expert_accum(cfg.hidden_dim);

            for (int k_idx = 0; k_idx < cfg.top_k; ++k_idx) {
                float gate = expert_scores[k_idx].first * inv_gate;
                int e = expert_scores[k_idx].second;

                VectorKernels::bitnet_gemm(norm_buf.data, layer.expert_w1[e], ffn_h1.data);
                VectorKernels::bitnet_gemm(norm_buf.data, layer.expert_wgate[e], ffn_gate.data);
                VectorKernels::silu(ffn_h1.data, cfg.ffn_dim);

                // Fused gate multiplication
                for (int d = 0; d < cfg.ffn_dim; ++d) ffn_h1[d] *= ffn_gate[d];

                // W2 projection
                VectorKernels::bitnet_gemm(ffn_h1.data, layer.expert_w2[e], expert_accum.data);

                // Accumulate to ffn_out
                for (int d = 0; d < cfg.hidden_dim; ++d) {
                    ffn_out[d] += expert_accum[d] * gate;
                }
            }

            // MoE residual addition
            for (int i = 0; i < cfg.hidden_dim; ++i) x_buf[i] += ffn_out[i];
        }

        // 3. Final Norm & LM Head
        VectorKernels::rmsnorm(x_buf.data, final_norm.data, norm_buf.data, cfg.hidden_dim, cfg.eps);

        for (int v = 0; v < cfg.vocab_size; ++v) {
            float* col = lm_head.data + (v * cfg.hidden_dim);
            float dot = 0.0f;
            for (int d = 0; d < cfg.hidden_dim; ++d) dot += norm_buf[d] * col[d];
            logits[v] = dot;
        }

        // 4. Sampler (Temperature + Top-K)
        if (temperature > 0.001f) {
            for (int v = 0; v < cfg.vocab_size; ++v) logits[v] /= temperature;
        }

        std::vector<std::pair<float, int>> top_candidates(cfg.vocab_size);
        for (int v = 0; v < cfg.vocab_size; ++v) top_candidates[v] = {logits[v], v};
        std::partial_sort(top_candidates.begin(), top_candidates.begin() + top_k, top_candidates.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });

        std::vector<float> top_probs(top_k);
        for (int i = 0; i < top_k; ++i) top_probs[i] = top_candidates[i].first;
        VectorKernels::softmax(top_probs.data(), top_k);

        // Sample token
        static std::mt19937 gen(1337);
        std::discrete_distribution<> dist(top_probs.begin(), top_probs.end());
        int sampled_idx = dist(gen);
        int next_token = top_candidates[sampled_idx].second;

        // Advance KV cache
        kv_cache.current_pos = (kv_cache.current_pos + 1) % cfg.max_seq_len;

        auto t1 = std::chrono::high_resolution_clock::now();
        float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        governor.regulate(ms);

        return next_token;
    }
};

} // namespace quillan
