/*
 * Quantum Street Engine v4 — WebAssembly ABI adapter.
 * Drop-in replacement for the original 19 KB AssemblyScript DSP module.
 *
 * ABI (verified by static + dynamic analysis of v3):
 *   init(f32 sampleRate), getNumParams()->8, reset(),
 *   process(inL, inR, outL, outR, numFrames) -> void,
 *   getAbiVersion()->1, memory
 *   params: 8 f32 at 0x10000 + 4*i (host writes raw; we smooth into 0x10100)
 *   audio:  f32 pointers, max 4096 frames per call
 *   exposes: "meters" (4 f32) and "qTelemetry" (32 f32) via hostExposeBuffer
 */
#include "qse_dsp.h"

#define P_RAW     ((volatile float*)0x10000u)
#define P_SMOOTH  ((volatile float*)0x10100u)

__attribute__((import_module("env"), import_name("abort")))
void qse_abort(int msg, int file, int line, int col);

__attribute__((import_module("env"), import_name("hostExposeBuffer")))
int qse_host_expose(const char* name, int nameLen, void* data, int elemCount);

static qse_state st;
static float g_meters[4];
static float g_telemetry[32];
static float g_smoothCoef = 0.995f;   /* overwritten by init(sr) */

extern float qse_exp(float x);

void init(float sampleRate)
{
    qse_init(&st, sampleRate);
    g_smoothCoef = qse_exp(-1000.0f / (sampleRate * 6.0f));
    for (int i = 0; i < 8; ++i) P_SMOOTH[i] = P_RAW[i];
    qse_host_expose("meters", 6, g_meters, 4);
    qse_host_expose("qTelemetry", 10, g_telemetry, 32);
}

int getNumParams(void) { return 8; }
int getAbiVersion(void) { return 1; }

void reset(void)
{
    qse_init(&st, st.sr);
    g_meters[0] = g_meters[1] = g_meters[2] = g_meters[3] = 0.0f;
}

void process(int inL, int inR, int outL, int outR, int numFrames)
{
    if (numFrames < 0) qse_abort(0, 0, 0, 0);   /* keeps the import alive */
    int n = numFrames > 4096 ? 4096 : numFrames;
    if (n <= 0) return;

    float p[8];
    for (int i = 0; i < 8; ++i) {
        float raw = P_RAW[i];
        /* guard against NaN/Inf from a hostile automation write */
        if (raw != raw) raw = 0.0f;
        if (raw > 1.0e7f) raw = 1.0e7f;
        if (raw < -1.0e7f) raw = -1.0e7f;
        float sm = P_SMOOTH[i] + (raw - P_SMOOTH[i]) * (1.0f - g_smoothCoef);
        P_SMOOTH[i] = sm;
        p[i] = sm;
    }

    qse_process(&st, p, (const float*)inL, (const float*)inR,
                (float*)outL, (float*)outR, n);

    qse_meters(&st, g_meters);
    g_telemetry[0] = g_meters[0];
    g_telemetry[1] = g_meters[1];
    g_telemetry[2] = st.tunnel;
    g_telemetry[3] = st.entropy;
    g_telemetry[4] = st.env;
    g_telemetry[5] = st.sr;
    for (int i = 6; i < 32; ++i) g_telemetry[i] = 0.0f;
}

