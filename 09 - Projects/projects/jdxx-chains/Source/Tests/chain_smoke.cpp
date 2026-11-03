/*
    JDXX Chains — chain smoke test (no JUCE; runs anywhere)

    Verifies what can be verified without a host:
      1. All five chains load with the exact §4 module counts.
      2. Silence in -> silence out (all five).
      3. No NaN/Inf through any chain under sustained hot input.
      4. GraphicEQ7 bell magnitude at band centers (+/-0.15 dB).
      5. Feedforward compressor static ratio (+/-2 dB of the law's prediction).
      6. Vintage Limiter keeps peaks <= 1.0.
      7. Noise gate attenuates below-threshold signal >= 20 dB.
      8. Stereo delay echo lands within +/-8 samples of time*sr.
      9. Latency reported to host == 0 samples.

    Build: see CMakeLists (target chain_smoke). Run: ./chain_smoke
*/
#include "Chains/ChainEngine.h"
#include "Chains/Modules.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <string>

using namespace JdxxChains;

static int failures = 0;
static int checks = 0;

#define CHECK(cond, msg) do { ++checks; if (!(cond)) { ++failures; \
    std::printf ("FAIL  [%d] %s\n", checks, msg); } } while (0)

static double rmsDb (const std::vector<float>& v)
{
    double acc = 0.0;
    for (float x : v) acc += (double) x * x;
    acc /= (double) v.size();
    return 10.0 * std::log10 (acc + 1e-12);
}

static float DB_TO_LIN_FS (float db) noexcept { return std::pow (10.0f, db / 20.0f); }

static std::vector<float> sineAt (float freq, float amp, int frames, double sr)
{
    std::vector<float> out ((size_t) frames);
    const double w = 2.0 * 3.141592653589793 * freq / sr;
    for (int i = 0; i < frames; ++i) out[(size_t) i] = amp * (float) std::sin (w * i);
    return out;
}

static void processThrough (ChainInstance& chain, std::vector<float>& L, std::vector<float>& R)
{
    chain.processBlock (L.data(), R.data(), (int) L.size());
}

// ---------------------------------------------------------------------------
static void testRegistry()
{
    CHECK (getChainSpec (ChainMode::Vocal).numModules == 9, "vocal chain has 9 modules");
    CHECK (getChainSpec (ChainMode::Adlib).numModules == 8, "adlib chain has 8 modules");
    CHECK (getChainSpec (ChainMode::Intro).numModules == 8, "intro chain has 8 modules");
    CHECK (getChainSpec (ChainMode::Pan).numModules == 7,   "pan chain has 7 modules");
    CHECK (getChainSpec (ChainMode::Mastering).numModules == 8, "mastering chain has 8 modules");

    for (int m = 0; m < (int) ChainMode::NumModes; ++m)
    {
        ChainInstance chain;
        chain.build ((ChainMode) m, 48000.0);
        CHECK (chain.getNumModules() == getChainSpec ((ChainMode) m).numModules,
               "chain instance module count matches spec");
        CHECK (chain.getLatencySamples() == 0.0, "zero latency reported");
    }
}

static void testSilence (double sr)
{
    const int n = (int) (1.0 * sr);
    for (int m = 0; m < (int) ChainMode::NumModes; ++m)
    {
        ChainInstance chain;
        chain.build ((ChainMode) m, sr);
        std::vector<float> L ((size_t) n, 0.0f), R ((size_t) n, 0.0f);
        processThrough (chain, L, R);
        float peak = 0.0f;
        for (int i = 0; i < n; ++i) peak = std::fmax (peak, std::fabs (L[(size_t) i]) + std::fabs (R[(size_t) i]));
        CHECK (peak < 1e-4f, "silence in -> silence out");
    }
}

static void testNoNaN (double sr)
{
    const int n = (int) (5.0 * sr);
    // Deterministic pseudo-noise via LCG — avoids <random> in test builds.
    uint32_t state = 0x12345678u;
    auto nextNoise = [&state]() { state = state * 1664525u + 1013904223u; return ((int32_t) state >> 8) / 8388608.0f; };

    for (int m = 0; m < (int) ChainMode::NumModes; ++m)
    {
        ChainInstance chain;
        chain.build ((ChainMode) m, sr);
        std::vector<float> L ((size_t) n), R ((size_t) n);
        for (int i = 0; i < n; ++i)
        {
            L[(size_t) i] = 0.5f * nextNoise();
            R[(size_t) i] = 0.5f * nextNoise();
        }
        processThrough (chain, L, R);
        bool finite = true;
        for (int i = 0; i < n; ++i)
            finite = finite && std::isfinite (L[(size_t) i]) && std::isfinite (R[(size_t) i]);
        CHECK (finite, "no NaN/Inf under sustained input");
    }
}

static void testGraphicEQ7 (double sr)
{
    const float testFreqs[7] = { 100.0f, 200.0f, 400.0f, 800.0f, 1600.0f, 3200.0f, 6400.0f };
    for (int band = 0; band < 7; ++band)
    {
        ModuleParams p {};
        for (int k = 0; k < 7; ++k) (&p.a)[k] = 0.0f;
        (&p.a)[band] = 6.0f;   // +6 dB on one band only
        p.h = 0.0f;            // unity out

        ModuleSpec spec { ModuleType::GraphicEQ7, "TestEQ", p };
        Module* mod = createModule (spec, sr);
        const int frames = (int) (2.0 * sr);
        auto in = sineAt (testFreqs[band], 0.01f, frames, sr);
        std::vector<float> L = in, R = in;
        mod->processBlock (L.data(), R.data(), frames);
        const double inDb  = rmsDb (in);
        const double outDb = rmsDb (L);
        const double gainDb = outDb - inDb;
        std::printf ("  EQ band %d @ %.0f Hz -> %+.2f dB (expect +6.00)\n", band, testFreqs[band], gainDb);
        CHECK (std::fabs (gainDb - 6.0) < 0.15, "EQ bell magnitude within 0.15 dB");
        destroyModule (mod);
    }
}

static void testCompressorFF (double sr)
{
    ModuleParams p {};
    p.a = 1.0f;      // attack ms
    p.b = 0.0f;      // knee dB (hard knee for the math check)
    p.c = 4.0f;      // ratio 4:1
    p.d = 50.0f;     // release ms
    p.e = -15.0f;    // threshold dB
    p.f = 0.0f;      // makeup dB
    ModuleSpec spec { ModuleType::CompressorFF, "TestComp", p };
    Module* mod = createModule (spec, sr);

    const int frames = (int) (3.0 * sr);
    auto in = sineAt (997.0f, DB_TO_LIN_FS (-6.0f), frames, sr);
    std::vector<float> L = in, R = in;
    mod->processBlock (L.data(), R.data(), frames);

    // Steady-state check on the last 1 s.
    const int tail = (int) (1.0 * sr);
    std::vector<float> tailBuf (L.end() - tail, L.end());
    const double outDb = rmsDb (tailBuf);
    const double inDb  = rmsDb (in);
    const double dropDb = inDb - outDb;
    // Input envelope ~ -6 dBFS (sine peak); GR law: (env - thr) * 3/4 = (-6 - -15) * 0.75 = 6.75 dB
    const double expected = 6.75;
    std::printf ("  Compressor 4:1 @ -6 dBFS -> %+.2f dB drop (expect ~%.2f)\n", -dropDb, expected);
    CHECK (std::fabs (dropDb - expected) < 2.0, "compressor ratio within 2 dB of the law");
    destroyModule (mod);
}

static void testLimiter (double sr)
{
    ModuleParams p {};
    p.a = 0.65f; p.b = 750.0f; p.c = 6.0f; p.d = 0.0f; p.e = 0.0f; p.f = -1.0f;
    ModuleSpec spec { ModuleType::Limiter, "TestLim", p };
    Module* mod = createModule (spec, sr);

    const int frames = (int) (2.0 * sr);
    auto in = sineAt (440.0f, 0.95f, frames, sr);   // hot input
    std::vector<float> L = in, R = in;
    mod->processBlock (L.data(), R.data(), frames);
    float peak = 0.0f;
    for (float x : L) peak = std::fmax (peak, std::fabs (x));
    std::printf ("  Limiter peak out = %.4f (must be <= 1.0)\n", peak);
    CHECK (peak <= 1.0f, "limiter clamps to <= 1.0");
    destroyModule (mod);
}

static void testGate (double sr)
{
    ModuleParams p {};
    p.a = -35.0f; p.b = 5.0f; p.c = 80.0f;
    ModuleSpec spec { ModuleType::Gate, "TestGate", p };
    Module* mod = createModule (spec, sr);

    const int frames = (int) (2.0 * sr);
    auto in = sineAt (1000.0f, DB_TO_LIN_FS (-50.0f), frames, sr);  // below threshold
    std::vector<float> L = in, R = in;
    mod->processBlock (L.data(), R.data(), frames);
    const int tail = (int) (1.0 * sr);
    std::vector<float> tailBuf (L.end() - tail, L.end());
    const double atten = rmsDb (in) - rmsDb (tailBuf);
    std::printf ("  Gate attenuation below threshold = %.1f dB (expect >= 20)\n", atten);
    CHECK (atten >= 20.0, "gate attenuates below-threshold signal >= 20 dB");
    destroyModule (mod);
}

static void testDelay (double sr)
{
    ModuleParams p {};
    p.a = 0.0f;      // dry 0
    p.b = 100.0f;    // wet 100 %
    p.c = 10.0f;     // HP 10 Hz (out of the way)
    p.d = 18000.0f;  // LP 18 kHz
    p.e = 0.0f;      // no feedback
    p.f = 0.05f;     // 50 ms
    p.g = 0.0f;      // unity out
    ModuleSpec spec { ModuleType::StereoDelay, "TestDly", p };
    Module* mod = createModule (spec, sr);

    const int frames = (int) (0.5 * sr);
    std::vector<float> L ((size_t) frames, 0.0f), R ((size_t) frames, 0.0f);
    L[0] = 1.0f;   // impulse
    mod->processBlock (L.data(), R.data(), frames);

    const int expected = (int) (0.05 * sr);
    int bestIdx = -1; float bestAbs = 0.0f;
    for (int i = 100; i < frames; ++i)
        if (std::fabs (L[(size_t) i]) > bestAbs) { bestAbs = std::fabs (L[(size_t) i]); bestIdx = i; }
    std::printf ("  Delay echo at sample %d (expect %d +/- 8)\n", bestIdx, expected);
    CHECK (std::abs (bestIdx - expected) <= 8, "delay echo within +/-8 samples");
    destroyModule (mod);
}

int main()
{
    const double sr = 48000.0;
    std::printf ("JDXX Chains — chain_smoke @ %.0f Hz\n", sr);
    testRegistry();
    testSilence (sr);
    testNoNaN (sr);
    testGraphicEQ7 (sr);
    testCompressorFF (sr);
    testLimiter (sr);
    testGate (sr);
    testDelay (sr);

    std::printf ("\n%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
