/*
    RoninEQ — DSP smoke test (standalone, no JUCE).

    Build & run:
      g++ -std=c++17 -Wall -Wextra -O2 Source/DSP/Biquad.cpp \
          Source/DSP/EnvelopeFollower.cpp Source/DSP/DynamicBand.cpp \
          Source/Tests/dsp_smoke.cpp -o /tmp/roneq_smoke && /tmp/roneq_smoke

    What it proves:
      1. RBJ biquad magnitude responses land where the cookbook says they should.
      2. A 3-band chain with one dynamic band cuts a loud in-band burst,
         leaves a quiet in-band burst alone, applies static boost correctly,
         stereo-links the detector, and releases back to 0 dB in silence.
*/
#include "../DSP/Biquad.h"
#include "../DSP/DynamicBand.h"
#include "../DSP/LinearPhaseFIR.h"
#include "../DSP/LoudnessMeter.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace
{

const float kSR = 48000.0f;
const float kPi = 3.14159265358979323846f;
int gFailures = 0;

void check (bool ok, const char* name, const char* detail = "")
{
    std::printf ("  [%s] %-58s %s\n", ok ? "PASS" : "FAIL", name, detail);
    if (! ok) ++gFailures;
}

// RMS of a buffer region in dB.
float rmsDb (const std::vector<float>& buf, size_t from, size_t to)
{
    double sum = 0.0;
    for (size_t i = from; i < to; ++i) sum += (double) buf[i] * buf[i];
    const double mean = sum / (double) (to - from);
    return (float) (10.0 * std::log10 (mean + 1.0e-24));
}

bool finite (float v) { return std::isfinite (v); }

struct BurstResult
{
    std::vector<float> inL, inR;   // dry input (for honest level references)
    std::vector<float> outL, outR; // processed output
};

// Run n samples of a sine (levelDb = PEAK level) through the band chain.
// leftOnly=true drives just the left channel (stereo-link test).
BurstResult runBurst (RoninEQ::DynamicBand* bands, int numBands,
                      float freqHz, float levelDbPeak, int numSamples,
                      bool leftOnly = false)
{
    BurstResult res;
    res.inL.resize ((size_t) numSamples);  res.inR.resize ((size_t) numSamples);
    res.outL.resize ((size_t) numSamples); res.outR.resize ((size_t) numSamples);
    const float amp = std::pow (10.0f, levelDbPeak / 20.0f);
    const float phaseInc = 2.0f * kPi * freqHz / kSR;
    float phase = 0.0f;
    for (int n = 0; n < numSamples; ++n)
    {
        const float dryL = amp * std::sin (phase);
        const float dryR = leftOnly ? 0.0f : dryL;
        phase += phaseInc;
        float l = dryL, r = dryR;
        for (int b = 0; b < numBands; ++b)
            bands[b].processSample (l, r);
        const size_t i = (size_t) n;
        res.inL[i] = dryL; res.inR[i] = dryR;
        res.outL[i] = l;   res.outR[i] = r;
    }
    return res;
}

bool allFinite (const BurstResult& br)
{
    for (float v : br.outL) if (! finite (v)) return false;
    for (float v : br.outR) if (! finite (v)) return false;
    return true;
}

} // namespace

int main()
{
    using namespace RoninEQ;
    char detail[256];

    std::printf ("RoninEQ DSP smoke test\n");
    std::printf ("=====================\n");

    // ---- 1. Biquad magnitude spot checks ---------------------------------
    std::printf ("[1] Biquad magnitude response spot checks\n");
    {
        Biquad f;
        f.setParams (Biquad::Type::Bell, 1000.0f, 1.0f, 6.0f, kSR);
        float m = f.magnitudeResponseDb (1000.0f, kSR);
        std::snprintf (detail, sizeof (detail), "got %+.2f dB", m);
        check (std::fabs (m - 6.0f) < 0.1f, "bell +6dB @ center == +6dB", detail);

        m = f.magnitudeResponseDb (100.0f, kSR);
        std::snprintf (detail, sizeof (detail), "got %+.2f dB", m);
        check (std::fabs (m) < 0.5f, "bell +6dB far below center ~= 0dB", detail);

        f.setParams (Biquad::Type::LowShelf, 200.0f, 0.71f, 6.0f, kSR);
        m = f.magnitudeResponseDb (40.0f, kSR);
        std::snprintf (detail, sizeof (detail), "got %+.2f dB", m);
        check (std::fabs (m - 6.0f) < 0.25f, "lowshelf +6dB deep in shelf == +6dB", detail);

        f.setParams (Biquad::Type::HighShelf, 8000.0f, 0.71f, -6.0f, kSR);
        m = f.magnitudeResponseDb (16000.0f, kSR);
        std::snprintf (detail, sizeof (detail), "got %+.2f dB", m);
        check (std::fabs (m + 6.0f) < 0.25f, "highshelf -6dB deep in shelf == -6dB", detail);

        f.setParams (Biquad::Type::HighPass, 1000.0f, 0.71f, 0.0f, kSR);
        m = f.magnitudeResponseDb (2000.0f, kSR);
        std::snprintf (detail, sizeof (detail), "got %+.2f dB", m);
        check (std::fabs (m) < 0.4f, "highpass 1k @ 2k ~= 0dB (passband)", detail);
        m = f.magnitudeResponseDb (250.0f, kSR);
        std::snprintf (detail, sizeof (detail), "got %+.2f dB", m);
        check (m < -18.0f, "highpass 1k @ 250Hz < -18dB (stopband)", detail);

        f.setParams (Biquad::Type::LowPass, 5000.0f, 0.71f, 0.0f, kSR);
        m = f.magnitudeResponseDb (2500.0f, kSR);
        std::snprintf (detail, sizeof (detail), "got %+.2f dB", m);
        check (std::fabs (m) < 0.4f, "lowpass 5k @ 2.5k ~= 0dB (passband)", detail);
        m = f.magnitudeResponseDb (15000.0f, kSR);
        std::snprintf (detail, sizeof (detail), "got %+.2f dB", m);
        check (m < -18.0f, "lowpass 5k @ 15k < -18dB (stopband)", detail);

        // RBJ notch has (theoretically) infinite depth at center regardless
        // of gainDb, so we check for a deep null + unity two octaves away.
        f.setParams (Biquad::Type::Notch, 1000.0f, 4.0f, 0.0f, kSR);
        m = f.magnitudeResponseDb (1000.0f, kSR);
        std::snprintf (detail, sizeof (detail), "got %+.1f dB", m);
        check (m < -40.0f, "notch @ center is a deep null (< -40dB)", detail);
        m = f.magnitudeResponseDb (4000.0f, kSR);
        std::snprintf (detail, sizeof (detail), "got %+.2f dB", m);
        check (std::fabs (m) < 0.5f, "notch two octaves up ~= 0dB", detail);
    }

    // ---- 2. Dynamic behavior ----------------------------------------------
    std::printf ("[2] Dynamic band behavior (3-band chain @ 48kHz)\n");
    {
        DynamicBand bands[3];
        for (auto& b : bands) b.setSampleRate (kSR);

        BandParams p0; // dynamic bell @ 1kHz
        p0.type = Biquad::Type::Bell; p0.freq = 1000.0f; p0.q = 1.0f;
        p0.gainDb = 0.0f; p0.dynEnabled = true;
        p0.thresholdDb = -20.0f; p0.ratio = 4.0f;
        p0.attackMs = 5.0f; p0.releaseMs = 100.0f;
        bands[0].setParams (p0);

        BandParams p1; // static bell @ 5kHz, +6dB
        p1.type = Biquad::Type::Bell; p1.freq = 5000.0f; p1.q = 1.0f;
        p1.gainDb = 6.0f;
        bands[1].setParams (p1);

        BandParams p2; // static high-pass @ 100Hz
        p2.type = Biquad::Type::HighPass; p2.freq = 100.0f; p2.q = 0.71f;
        bands[2].setParams (p2);

        for (auto& b : bands) b.reset();

        const int nA = (int) (0.5f * kSR);
        const int nB = (int) (1.0f * kSR);
        // Measure over the settled tail of each burst (skip the attack transient).
        const size_t mA0 = (size_t) (0.30f * kSR), mA1 = (size_t) nA;
        const size_t mB0 = (size_t) (0.50f * kSR), mB1 = (size_t) nB;

        // Burst A: 1kHz, peak -30dB (RMS -33dB) — below threshold, expect no cut.
        auto burstA = runBurst (bands, 3, 1000.0f, -30.0f, nA);
        check (allFinite (burstA), "burst A samples all finite");
        {
            const float inDb  = rmsDb (burstA.inL, mA0, mA1);
            const float outDb = rmsDb (burstA.outL, mA0, mA1);
            std::snprintf (detail, sizeof (detail),
                           "in %+.1fdB out %+.1fdB", inDb, outDb);
            check (std::fabs (outDb - inDb) < 1.0f,
                   "quiet in-band burst passes untouched (no false trigger)", detail);
        }

        // Burst B: 1kHz, peak -6dB — above threshold, expect the cut.
        // Detector sees ~-6dB peak, over = 14dB, ratio 4 -> ~-10.5dB cut.
        auto burstB = runBurst (bands, 3, 1000.0f, -6.0f, nB);
        check (allFinite (burstB), "burst B samples all finite");
        {
            const float inDb  = rmsDb (burstB.inL, mB0, mB1);
            const float outDb = rmsDb (burstB.outL, mB0, mB1);
            std::snprintf (detail, sizeof (detail),
                           "in %+.1fdB out %+.1fdB (drop %+.1fdB)",
                           inDb, outDb, outDb - inDb);
            check (outDb < inDb - 6.0f && outDb > inDb - 14.0f,
                   "loud in-band burst gets dynamically cut (~10dB drop)", detail);
        }
        {
            const float dynDb = bands[0].getDynamicGainDb();
            std::snprintf (detail, sizeof (detail), "meter reads %+.2fdB", dynDb);
            check (dynDb < -6.0f, "dynamic gain meter shows real reduction", detail);
        }

        // Burst C: 5kHz, peak -12dB — static +6dB boost, dynamics must stay out.
        auto burstC = runBurst (bands, 3, 5000.0f, -12.0f, nA);
        check (allFinite (burstC), "burst C samples all finite");
        {
            const float inDb  = rmsDb (burstC.inL, mA0, mA1);
            const float outDb = rmsDb (burstC.outL, mA0, mA1);
            std::snprintf (detail, sizeof (detail),
                           "in %+.1fdB out %+.1fdB (boost %+.1fdB)",
                           inDb, outDb, outDb - inDb);
            check (std::fabs ((outDb - inDb) - 6.0f) < 1.5f,
                   "static +6dB boost applies, dynamics stay out of it", detail);
        }

        // Burst D: 1kHz, peak -6dB, LEFT ONLY — stereo-linked detector must
        // still engage the cut on the driven channel.
        auto burstD = runBurst (bands, 3, 1000.0f, -6.0f, nB, /*leftOnly=*/ true);
        check (allFinite (burstD), "burst D samples all finite");
        {
            const float inDb  = rmsDb (burstD.inL, mB0, mB1);
            const float outDb = rmsDb (burstD.outL, mB0, mB1);
            std::snprintf (detail, sizeof (detail),
                           "left-only in %+.1fdB out %+.1fdB", inDb, outDb);
            check (outDb < inDb - 6.0f,
                   "stereo-linked detector cuts single-channel signal", detail);
        }

        // Release: 1s of TRUE silence (fresh zeros every sample — never feed
        // one band's output back into the chain) -> gain relaxes near 0.
        {
            const int n = (int) kSR;
            for (int i = 0; i < n; ++i)
            {
                float l = 0.0f, r = 0.0f;
                for (auto& b : bands) b.processSample (l, r);
            }
            const float dynDb = bands[0].getDynamicGainDb();
            std::snprintf (detail, sizeof (detail),
                           "meter reads %+.2fdB after 1s silence", dynDb);
            check (dynDb > -1.0f, "dynamic gain releases back toward 0dB", detail);
        }

        // Param change mid-stream must not explode or NaN.
        {
            BandParams p0b = p0;
            p0b.freq = 2000.0f; p0b.q = 2.0f; p0b.thresholdDb = -30.0f;
            bands[0].setParams (p0b);
            auto burstE = runBurst (bands, 3, 2000.0f, -10.0f, nA);
            check (allFinite (burstE), "post-param-change samples all finite");
        }
    }

    std::printf ("[3] Placement modes, external key, linear-phase FIR, loudness\n");
    {
        using namespace RoninEQ;
        const int n = (int) (kSR * 0.5f); // 0.5 s @ 48 kHz

        auto sine = [] (int i, float peak)
        {
            return peak * std::sin (2.0f * kPi * 1000.0f * (float) i / kSR);
        };

        // Placement::Left — right channel must pass untouched.
        {
            DynamicBand b;
            b.setSampleRate (kSR);
            BandParams p;
            p.type = Biquad::Type::Bell; p.freq = 1000.0f;
            p.gainDb = 6.0f; p.q = 1.0f;
            p.placement = Placement::Left;
            b.setParams (p);
            double sumL = 0.0, sumR = 0.0, sumIn = 0.0;
            for (int i = 0; i < n; ++i)
            {
                float l = sine (i, 0.25f), r = l;
                sumIn += (double) l * l;
                b.processSample (l, r);
                const int j = i - n / 2; // settled second half
                if (j >= 0) { sumL += (double) l * l; sumR += (double) r * r; }
            }
            const float boostL = (float) (10.0 * std::log10 (sumL / sumIn * 2.0 + 1e-24));
            const float boostR = (float) (10.0 * std::log10 (sumR / sumIn * 2.0 + 1e-24));
            std::snprintf (detail, sizeof (detail), "L %+.1fdB R %+.1fdB", boostL, boostR);
            check (std::fabs (boostL - 6.0f) < 1.0f && std::fabs (boostR) < 0.5f,
                   "Left placement boosts L only, R untouched", detail);
        }

        // Placement::Mid at 0 dB — M/S round-trip must be transparent.
        {
            DynamicBand b;
            b.setSampleRate (kSR);
            BandParams p;
            p.placement = Placement::Mid;
            b.setParams (p);
            double sumIn = 0.0, sumDiff = 0.0;
            for (int i = 0; i < n; ++i)
            {
                float l = sine (i, 0.25f), r = 0.5f * l;
                sumIn += (double) l * l + (double) r * r;
                const float dl = l, dr = r;
                b.processSample (l, r);
                sumDiff += (double) (l - dl) * (l - dl) + (double) (r - dr) * (r - dr);
            }
            const float errDb = (float) (10.0 * std::log10 (sumDiff / sumIn + 1e-24));
            std::snprintf (detail, sizeof (detail), "round-trip error %+.1fdB", errDb);
            check (errDb < -90.0f, "Mid placement at 0dB is transparent", detail);
        }

        // Placement::Side cut on a mono (identical L/R) signal — no side
        // energy, so output must equal input.
        {
            DynamicBand b;
            b.setSampleRate (kSR);
            BandParams p;
            p.type = Biquad::Type::Bell; p.freq = 1000.0f;
            p.gainDb = -12.0f; p.q = 1.0f;
            p.placement = Placement::Side;
            b.setParams (p);
            double sumIn = 0.0, sumDiff = 0.0;
            for (int i = 0; i < n; ++i)
            {
                float l = sine (i, 0.25f), r = l;
                sumIn += (double) l * l;
                const float dl = l;
                b.processSample (l, r);
                sumDiff += (double) (l - dl) * (l - dl);
            }
            const float errDb = (float) (10.0 * std::log10 (sumDiff / sumIn + 1e-24));
            std::snprintf (detail, sizeof (detail), "mono-signal error %+.1fdB", errDb);
            check (errDb < -60.0f, "Side cut leaves mono signal alone", detail);
        }

        // External key: quiet program + loud key -> cut; loud key removed -> pass.
        {
            DynamicBand b;
            b.setSampleRate (kSR);
            BandParams p;
            p.type = Biquad::Type::Bell; p.freq = 1000.0f; p.q = 1.0f;
            p.gainDb = 0.0f;
            p.dynEnabled = true; p.dynSource = DetectorSource::External;
            p.thresholdDb = -20.0f; p.ratio = 4.0f;
            p.attackMs = 10.0f; p.releaseMs = 100.0f;
            b.setParams (p);

            auto runKeyed = [&] (float progPeak, float keyPeak)
            {
                double sumIn = 0.0, sumOut = 0.0;
                for (int i = 0; i < n; ++i)
                {
                    float l = sine (i, progPeak), r = l;
                    const float k = sine (i, keyPeak);
                    sumIn += (double) l * l;
                    float kl = k, kr = k;
                    b.processSample (l, r, kl, kr);
                    if (i >= n / 2) sumOut += (double) l * l;
                }
                const double sumInHalf = sumIn * 0.5;
                return (float) (10.0 * std::log10 (sumOut / sumInHalf + 1e-24));
            };

            const float dropLoudKey = runKeyed (0.03f, 0.5f);   // quiet prog, loud key
            std::snprintf (detail, sizeof (detail), "drop %+.1fdB", dropLoudKey);
            check (dropLoudKey < -6.0f, "loud external key cuts quiet program", detail);

            b.reset();
            b.setParams (p);
            const float dropQuietKey = runKeyed (0.03f, 0.0001f); // key ~silence
            std::snprintf (detail, sizeof (detail), "drop %+.1fdB", dropQuietKey);
            check (std::fabs (dropQuietKey) < 1.0f, "silent key leaves program alone", detail);
        }

        // LinearPhaseFIR unity: impulse out exactly at the latency tap.
        {
            LinearPhaseFIR f;
            const int nn = 1400;
            bool ok = true;
            for (int i = 0; i < nn; ++i)
            {
                const float y = f.processSample (i == 0 ? 1.0f : 0.0f);
                const float want = (i == LinearPhaseFIR::kLatencySamples) ? 1.0f : 0.0f;
                if (std::fabs (y - want) > 1.0e-5f) ok = false;
            }
            check (ok && LinearPhaseFIR::kLatencySamples == 512,
                   "unity FIR delays impulse by exactly 512 samples");
        }

        // LoudnessMeter: full-scale 1 kHz sine reads ≈ -3.7 LUFS; silence floors.
        {
            LoudnessMeter m;
            m.setSampleRate (kSR);
            for (int i = 0; i < (int) (kSR * 3.0f); ++i)
            {
                const float s = std::sin (2.0f * kPi * 1000.0f * (float) i / kSR);
                m.processSample (s, s);
            }
            const float lufs = m.getLUFS();
            std::snprintf (detail, sizeof (detail), "reads %+.1f LUFS", lufs);
            // Dual-mono FS sine: z = 0.5 + 0.5 = 1.0 -> -0.69 LUFS
            // (plus a touch of K-weighting shelf lift at 1 kHz).
            check (lufs > -2.5f && lufs < 1.0f, "FS sine loudness sane", detail);

            m.reset();
            for (int i = 0; i < 1000; ++i) m.processSample (0.0f, 0.0f);
            check (m.getLUFS() < -100.0f, "silence floors the meter");
        }
    }

    std::printf ("=====================\n");
    if (gFailures == 0)
        std::printf ("ALL CHECKS PASSED\n");
    else
        std::printf ("%d CHECK(S) FAILED\n", gFailures);
    return gFailures == 0 ? 0 : 1;
}
