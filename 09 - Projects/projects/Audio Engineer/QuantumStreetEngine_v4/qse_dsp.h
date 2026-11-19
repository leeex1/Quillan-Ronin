/* Quantum Street Engine v4 — DSP core API. AGPL-3.0. */
#ifndef QSE_DSP_H
#define QSE_DSP_H

#define QSE_MAX_DELAY 65536

typedef struct { float a; float z; } qse_onepole;
typedef struct {
    float b0, b1, b2, a1, a2;
    float x1, x2, y1, y2;
} qse_biquad;

typedef struct {
    float sr;
    qse_onepole rumble, sublp, hpb, noiselp;
    qse_biquad mud, box, presence, air;
    float dl[QSE_MAX_DELAY];
    int dlpos;
    unsigned int rng;
    float env, makeup;
    float peakL, peakR, tunnel, entropy;
    int ready;
} qse_state;

void qse_init(qse_state* s, float sampleRate);
void qse_process(qse_state* s, const float* params8,
                 const float* inL, const float* inR,
                 float* outL, float* outR, int frames);
void qse_meters(const qse_state* s, float* out4);

#endif /* QSE_DSP_H */
