/*
 * Quantum Street Engine v4 — DSP core
 * AGPL-3.0. Pure C11, no libc: compiles to wasm32 with -nostdlib.
 *
 * Chain (from the Google Flow `QuantumMasteringConsole` source, folded into
 * the plugin's 8 host parameters; see DSP_SPEC.md):
 *   rumble HPF -> 808 sub tunnel (saturation) -> de-mud -> presence -> air
 *   -> Shor exciter -> ghost-tap superposition + probability collapse
 *   -> entangled M/S -> entropy dither -> glue comp -> brickwall -> output
 *
 * Public API (ABI adapter lives in qse_wasm.c):
 *   void qse_init(qse_state*, float sampleRate);
 *   void qse_process(qse_state*, const float* params8,
 *                    const float* inL, const float* inR,
 *                    float* outL, float* outR, int frames);
 *   void qse_meters(const qse_state*, float* out4);
 *
 * params8 are in REAL units, ids 0..7:
 *   0 Tunneling %    1 SuperposTime ms  2 Probability %
 *   3 EntangleCutoff Hz  4 EntropyNoise %  5 WaveDecay %
 *   6 OutputGain dB  7 Mix %
 */

#include "qse_dsp.h"

/* ---------- math helpers (no libm) ---------- */

static float qse_absf(float x) { return x < 0.0f ? -x : x; }

/* tanh approximation, accurate to ~1e-3 over [-4,4], saturating outside. */
static float qse_tanh(float x)
{
    if (x > 4.0f) return 1.0f;
    if (x < -4.0f) return -1.0f;
    const float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

/* exp(x) via ln2 range reduction + degree-5 polynomial. Good to ~1e-5. */
float qse_exp(float x)
{
    if (x > 80.0f) return 4.8e34f;
    if (x < -80.0f) return 0.0f;
    int n = (int)(x * 1.4426950408889634f);
    float r = x - (float)n * 0.6931471805599453f;
    float p = 1.0f + r * (1.0f + r * (0.5f + r * (0.16666667f +
              r * (0.041666668f + r * 0.0083333338f))));
    while (n > 0) { p *= 2.0f; --n; }
    while (n < 0) { p *= 0.5f; ++n; }
    return p;
}

float qse_db2lin(float db) { return qse_exp(db * 0.11512925464970229f); }

float qse_logf(float x)
{
    union { float f; unsigned int u; } v;
    v.f = x;
    int e = ((int)((v.u >> 23) & 0xffu)) - 127;
    v.u = (v.u & 0x007fffffu) | 0x3f800000u;   /* mantissa in [1,2) */
    const float m = v.f;
    const float t = (m - 1.0f) / (m + 1.0f);
    const float t2 = t * t;
    const float s = t + t2 * t * (0.33333334f +
                    t2 * (0.2f + t2 * (0.14285715f + t2 * 0.11111111f)));
    return 2.0f * s + (float)e * 0.6931471805599453f;
}

/* sin/cos without libm: reduce to [-pi/4, pi/4], then degree-7 polynomials */
static void qse_sincos(float x, float* outS, float* outC)
{
    const float xq = x * 0.6366197723675814f;
    const float k = xq >= 0.0f ? __builtin_floorf(xq + 0.5f) : __builtin_ceilf(xq - 0.5f);
    const float r = x - k * 1.5707963267948966f;
    const float r2 = r * r;
    const float sr = r * (1.0f + r2 * (-0.16666667f +
                        r2 * (0.0083333338f + r2 * -0.00019841270f)));
    const float cr = 1.0f + r2 * (-0.5f + r2 *
                     (0.041666668f + r2 * -0.0013888889f));
    switch (((int)k) & 3) {
        case 0: *outS =  sr; *outC =  cr; break;
        case 1: *outS =  cr; *outC = -sr; break;
        case 2: *outS = -sr; *outC = -cr; break;
        default: *outS = -cr; *outC =  sr; break;
    }
}

/* deterministic xorshift32 — "hardware entropy" stand-in, reproducible */
static unsigned int qse_rng_next(unsigned int* s)
{
    unsigned int x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x ? x : 0x9e3779b9u;
    return *s;
}

/* ---------- one-pole filters ---------- */

static void qse_onepole_lp_init(qse_onepole* f, float sr, float hz)
{
    const float dt = 1.0f / sr;
    const float rc = 1.0f / (6.283185307179586f * hz);
    f->a = dt / (rc + dt);
    f->z = 0.0f;
}

static float qse_onepole_lp(qse_onepole* f, float x)
{
    f->z += f->a * (x - f->z);
    return f->z;
}

static float qse_onepole_hp(qse_onepole* f, float x)
{
    const float lp = f->z + f->a * (x - f->z);
    f->z = lp;
    return x - lp;
}

/* ---------- RBJ peaking EQ ---------- */

static void qse_peak_init(qse_biquad* b, float sr, float hz, float q, float gainDb)
{
    const float w0 = 6.283185307179586f * hz / sr;
    float cw, sw; qse_sincos(w0, &sw, &cw);
    const float alpha = sw / (2.0f * q);
    const float A = qse_db2lin(gainDb * 0.5f);
    const float a0 = 1.0f + alpha / A;
    b->b0 = (1.0f + alpha * A) / a0;
    b->b1 = (-2.0f * cw) / a0;
    b->b2 = (1.0f - alpha * A) / a0;
    b->a1 = (-2.0f * cw) / a0;
    b->a2 = (1.0f - alpha / A) / a0;
    b->x1 = b->x2 = b->y1 = b->y2 = 0.0f;
}

static float qse_peak(qse_biquad* b, float x)
{
    const float y = b->b0 * x + b->b1 * b->x1 + b->b2 * b->x2
                  - b->a1 * b->y1 - b->a2 * b->y2;
    b->x2 = b->x1; b->x1 = x;
    b->y2 = b->y1; b->y1 = y;
    return y;
}

static float qse_shelf(qse_biquad* b, float x)
{
    const float y = b->b0 * x + b->b1 * b->x1 + b->b2 * b->x2
                  - b->a1 * b->y1 - b->a2 * b->y2;
    b->x2 = b->x1; b->x1 = x;
    b->y2 = b->y1; b->y1 = y;
    return y;
}

static void qse_shelf_init(qse_biquad* b, float sr, float hz, float gainDb)
{
    const float w0 = 6.283185307179586f * hz / sr;
    float cw, sw; qse_sincos(w0, &sw, &cw);
    const float A = qse_db2lin(gainDb * 0.5f);
    const float two = 2.0f * __builtin_sqrtf(A) * sw * 0.7071067811865476f;
    const float a0 = A + 1.0f + (A - 1.0f) * cw + two;
    b->b0 = A * ((A + 1.0f) + (A - 1.0f) * cw + two) / a0;
    b->b1 = -2.0f * A * ((A - 1.0f) + (A + 1.0f) * cw) / a0;
    b->b2 = A * ((A + 1.0f) + (A - 1.0f) * cw - two) / a0;
    b->a1 = -2.0f * ((A - 1.0f) + (A + 1.0f) * cw) / a0;
    b->a2 = ((A + 1.0f) + (A - 1.0f) * cw - two) / a0;
    b->x1 = b->x2 = b->y1 = b->y2 = 0.0f;
}

/* ---------- state ---------- */

void qse_init(qse_state* s, float sampleRate)
{
    s->sr = sampleRate;
    qse_onepole_lp_init(&s->rumble, sampleRate, 28.0f);
    qse_onepole_lp_init(&s->sublp, sampleRate, 110.0f);
    qse_onepole_lp_init(&s->hpb, sampleRate, 3400.0f);
    qse_onepole_lp_init(&s->noiselp, sampleRate, 9000.0f);
    qse_peak_init(&s->mud, sampleRate, 280.0f, 1.8f, -3.5f);
    qse_peak_init(&s->box, sampleRate, 520.0f, 2.0f, -2.0f);
    qse_peak_init(&s->presence, sampleRate, 3400.0f, 1.2f, 2.0f);
    qse_shelf_init(&s->air, sampleRate, 11000.0f, 3.5f);
    for (int i = 0; i < QSE_MAX_DELAY; ++i) { s->dl[i] = 0.0f; }
    s->dlpos = 0;
    s->rng = 0x13579bdfu;
    s->env = 0.0f;
    s->makeup = 1.0f;
    s->peakL = s->peakR = s->tunnel = s->entropy = 0.0f;
    s->ready = 1;
}

/* ---------- processing ---------- */

void qse_process(qse_state* s, const float* p,
                 const float* inL, const float* inR,
                 float* outL, float* outR, int frames)
{
    if (!s->ready) return;

    const float tunnel   = p[0] / 100.0f;
    const float timeMs   = p[1];
    const float prob     = p[2] / 100.0f;
    const float cutoff   = p[3];
    const float entropy  = p[4] / 100.0f;
    const float decay    = p[5] / 100.0f;
    const float outGain  = qse_db2lin(p[6]);
    const float mix      = p[7] / 100.0f;

    const float sr = s->sr;
    int delay = (int)(sr * timeMs * 0.001f);
    if (delay < 8) delay = 8;
    if (delay > QSE_MAX_DELAY - 4) delay = QSE_MAX_DELAY - 4;
    const float fb = 0.10f + 0.55f * decay;
    const float wet = 0.05f + 0.45f * prob;
    const float drive = 1.0f + 4.0f * tunnel;
    const float subTrim = 1.0f + 0.9f * tunnel;
    const float entAmt = 0.12f * entropy;
    const float noiseAmt = 0.0016f * entropy;

    /* compressor coefficients, updated per block */
    const float thr = qse_db2lin(-14.0f);
    const float ratio = 3.2f;
    const float ca = 1.0f - qse_exp(-1.0f / (sr * 0.025f));
    const float cr = 1.0f - qse_exp(-1.0f / (sr * 0.120f));

    /* shelf cutoff follows the entanglement param (clamped to Nyquist) */
    float shz = cutoff;
    if (shz < 100.0f) shz = 100.0f;
    if (shz > sr * 0.45f) shz = sr * 0.45f;
    qse_shelf_init(&s->air, sr, shz, 1.2f + 3.5f * entropy * 0.5f);

    float peakL = 0.0f, peakR = 0.0f, tun = 0.0f, ent = 0.0f;

    for (int i = 0; i < frames; ++i) {
        float l = inL[i], r = inR[i];
        const float dryL = l, dryR = r;

        /* 1. rumble */
        l = qse_onepole_hp(&s->rumble, l);
        r = qse_onepole_hp(&s->rumble, r);

        /* 2. 808 tunneling: mono sub, saturate, add back */
        const float sub = qse_onepole_lp(&s->sublp, 0.5f * (l + r));
        const float sat = subTrim * qse_tanh(sub * drive);
        l += sat; r += sat;
        tun += qse_absf(sat);

        /* 3-5. tone shaping */
        l = qse_peak(&s->mud, l);      r = qse_peak(&s->mud, r);
        l = qse_peak(&s->box, l);      r = qse_peak(&s->box, r);
        l = qse_peak(&s->presence, l); r = qse_peak(&s->presence, r);
        l = qse_shelf(&s->air, l);     r = qse_shelf(&s->air, r);

        /* 6. Shor exciter on the high band */
        const float hiL = qse_onepole_hp(&s->hpb, l);
        const float hiR = qse_onepole_hp(&s->hpb, r);
        l += entropy * 0.15f * hiL * hiL * hiL;
        r += entropy * 0.15f * hiR * hiR * hiR;

        /* 7. superposition: read ghost taps from the delay line */
        const int d1 = delay;
        const int d2 = delay + delay / 2;
        const int idx1 = (s->dlpos - d1 + QSE_MAX_DELAY) % QSE_MAX_DELAY;
        const int idx2 = (s->dlpos - d2 + QSE_MAX_DELAY) % QSE_MAX_DELAY;
        const float g1 = s->dl[idx1];
        const float g2 = s->dl[idx2];
        const float ghostSeed = (float)qse_rng_next(&s->rng) / 4294967296.0f;
        const float gate = ghostSeed < prob ? 1.0f : 0.25f;
        l += wet * gate * (0.7f * g1 + 0.3f * g2);
        r += wet * gate * (0.3f * g1 + 0.7f * g2);
        s->dl[s->dlpos] = 0.5f * (l + r);
        s->dlpos = (s->dlpos + 1) % QSE_MAX_DELAY;

        /* 8. entanglement: couple side energy above cutoff */
        const float side = 0.5f * (l - r);
        const float nt = (float)qse_rng_next(&s->rng) / 2147483648.0f - 1.0f;
        const float noise = qse_onepole_hp(&s->noiselp, nt) * noiseAmt;
        l += entAmt * side + noise;
        r -= entAmt * side + noise;
        ent += qse_absf(side);

        /* 9. glue compressor */
        const float detect = __builtin_sqrtf((l * l + r * r) * 0.5f);
        const float coef = detect > thr ? ca : cr;
        s->env += coef * (detect - s->env);
        float gr = 1.0f;
        if (s->env > thr) {
            const float over = s->env / thr;
            gr = 1.0f / qse_exp(0.6875f * qse_logf(over));
        }
        if (gr < 0.25f) gr = 0.25f;
        s->makeup += 0.0005f * (1.0f / gr - s->makeup);
        l *= gr * s->makeup; r *= gr * s->makeup;

        /* 10. brickwall soft clip at -0.8 dBFS */
        const float ceil = 0.912f;
        l = ceil * qse_tanh(l / ceil);
        r = ceil * qse_tanh(r / ceil);

        /* 11. output trim + dry/wet collapse */
        l = (mix * l + (1.0f - mix) * dryL) * outGain;
        r = (mix * r + (1.0f - mix) * dryR) * outGain;
        if (l > 1.0f) l = 1.0f; if (l < -1.0f) l = -1.0f;
        if (r > 1.0f) r = 1.0f; if (r < -1.0f) r = -1.0f;

        outL[i] = l; outR[i] = r;
        const float al = qse_absf(l), ar = qse_absf(r);
        if (al > peakL) peakL = al;
        if (ar > peakR) peakR = ar;
    }

    const float k = frames > 0 ? 1.0f / (float)frames : 0.0f;
    s->peakL = peakL; s->peakR = peakR;
    s->tunnel = tun * k; s->entropy = ent * k;
}

void qse_meters(const qse_state* s, float* out4)
{
    out4[0] = s->peakL;
    out4[1] = s->peakR;
    out4[2] = s->tunnel;
    out4[3] = s->entropy;
}
