/*
    RoninEQ — clean-room dynamic equalizer. License: AGPL-3.0.
*/
#include "PluginProcessor.h"
#include "PluginEditor.h"
#include "PluginInfo.h"

//==============================================================================
namespace
{

juce::String freqToString (float v, int)
{
    if (v >= 1000.0f)
        return juce::String (v / 1000.0f, 2) + " kHz";
    return juce::String (v, (v < 100.0f ? 1 : 0)) + " Hz";
}

juce::String dbToString (float v, int)
{
    return juce::String (v, 1) + " dB";
}

juce::String msToString (float v, int)
{
    return juce::String (v, (v < 10.0f ? 1 : 0)) + " ms";
}

} // namespace

//==============================================================================
juce::AudioProcessorValueTreeState::ParameterLayout
RoninEQAudioProcessor::createParameterLayout()
{
    juce::AudioProcessorValueTreeState::ParameterLayout layout;

    const juce::StringArray typeChoices { "Bell", "Low Shelf", "High Shelf",
                                          "High Pass", "Low Pass", "Notch" };
    // Must stay in the same order as RoninEQ::Biquad::Type.
    static_assert ((int) RoninEQ::Biquad::Type::Bell      == 0, "type order");
    static_assert ((int) RoninEQ::Biquad::Type::LowShelf  == 1, "type order");
    static_assert ((int) RoninEQ::Biquad::Type::HighShelf == 2, "type order");
    static_assert ((int) RoninEQ::Biquad::Type::HighPass  == 3, "type order");
    static_assert ((int) RoninEQ::Biquad::Type::LowPass   == 4, "type order");
    static_assert ((int) RoninEQ::Biquad::Type::Notch     == 5, "type order");

    const float defaultFreqs[kNumBands] = { 1000.0f, 100.0f, 300.0f,
                                            3000.0f, 8000.0f, 12000.0f };

    for (int b = 0; b < kNumBands; ++b)
    {
        const juce::String bandName = "Band " + juce::String (b + 1);

        layout.add (std::make_unique<juce::AudioParameterBool> (
            bandParamId (b, "enabled"), bandName + " Enabled", b == 0));

        layout.add (std::make_unique<juce::AudioParameterChoice> (
            bandParamId (b, "type"), bandName + " Type", typeChoices, 0));

        juce::NormalisableRange<float> freqRange (20.0f, 20000.0f);
        freqRange.setSkewForCentre (1000.0f);
        layout.add (std::make_unique<juce::AudioParameterFloat> (
            bandParamId (b, "freq"), bandName + " Freq",
            freqRange, defaultFreqs[b], "Hz",
            juce::AudioProcessorParameter::genericParameter,
            freqToString, nullptr));

        layout.add (std::make_unique<juce::AudioParameterFloat> (
            bandParamId (b, "gain"), bandName + " Gain",
            juce::NormalisableRange<float> (-15.0f, 15.0f), 0.0f, "dB",
            juce::AudioProcessorParameter::genericParameter,
            dbToString, nullptr));

        juce::NormalisableRange<float> qRange (0.1f, 10.0f);
        qRange.setSkewForCentre (1.0f);
        layout.add (std::make_unique<juce::AudioParameterFloat> (
            bandParamId (b, "q"), bandName + " Q",
            qRange, 1.0f, "",
            juce::AudioProcessorParameter::genericParameter,
            [] (float v, int) { return juce::String (v, 2); }, nullptr));

        layout.add (std::make_unique<juce::AudioParameterBool> (
            bandParamId (b, "dyn_enabled"), bandName + " Dynamic", b == 0));

        layout.add (std::make_unique<juce::AudioParameterFloat> (
            bandParamId (b, "dyn_threshold"), bandName + " Threshold",
            juce::NormalisableRange<float> (-60.0f, 0.0f), -20.0f, "dB",
            juce::AudioProcessorParameter::genericParameter,
            dbToString, nullptr));

        layout.add (std::make_unique<juce::AudioParameterFloat> (
            bandParamId (b, "dyn_ratio"), bandName + " Ratio",
            juce::NormalisableRange<float> (1.0f, 20.0f), 4.0f, ":1",
            juce::AudioProcessorParameter::genericParameter,
            [] (float v, int) { return juce::String (v, 1) + ":1"; }, nullptr));

        juce::NormalisableRange<float> attackRange (0.1f, 100.0f);
        attackRange.setSkewForCentre (10.0f);
        layout.add (std::make_unique<juce::AudioParameterFloat> (
            bandParamId (b, "dyn_attack"), bandName + " Attack",
            attackRange, 10.0f, "ms",
            juce::AudioProcessorParameter::genericParameter,
            msToString, nullptr));

        juce::NormalisableRange<float> releaseRange (10.0f, 1000.0f);
        releaseRange.setSkewForCentre (100.0f);
        layout.add (std::make_unique<juce::AudioParameterFloat> (
            bandParamId (b, "dyn_release"), bandName + " Release",
            releaseRange, 100.0f, "ms",
            juce::AudioProcessorParameter::genericParameter,
            msToString, nullptr));
    }

    layout.add (std::make_unique<juce::AudioParameterFloat> (
        "input_gain", "Input Gain",
        juce::NormalisableRange<float> (-24.0f, 24.0f), 0.0f, "dB",
        juce::AudioProcessorParameter::genericParameter, dbToString, nullptr));

    layout.add (std::make_unique<juce::AudioParameterFloat> (
        "output_gain", "Output Gain",
        juce::NormalisableRange<float> (-24.0f, 24.0f), 0.0f, "dB",
        juce::AudioProcessorParameter::genericParameter, dbToString, nullptr));

    layout.add (std::make_unique<juce::AudioParameterBool> ("bypass", "Bypass", false));

    return layout;
}

//==============================================================================
RoninEQAudioProcessor::RoninEQAudioProcessor()
    : AudioProcessor (BusesProperties()
                          .withInput  ("Input",  juce::AudioChannelSet::stereo(), true)
                          .withOutput ("Output", juce::AudioChannelSet::stereo(), true)),
      apvts (*this, nullptr, "RoninEQState", createParameterLayout()),
      oversampling (2, kOversampleFactor, // stereo max: mono buses reuse channel 0
                    juce::dsp::Oversampling<float>::filterHalfBandPolyphaseIIR,
                    false,   // maxQuality: keep latency + CPU low
                    false)   // useIntegerLatency
{
    cacheParameterPtrs();

    for (auto& m : bandMeters)
        m.store (0.0f, std::memory_order_relaxed);
}

RoninEQAudioProcessor::~RoninEQAudioProcessor() = default;

void RoninEQAudioProcessor::cacheParameterPtrs()
{
    for (int b = 0; b < kNumBands; ++b)
    {
        auto& p = bandPtrs[b];
        p.enabled    = apvts.getRawParameterValue (bandParamId (b, "enabled"));
        p.type       = apvts.getRawParameterValue (bandParamId (b, "type"));
        p.freq       = apvts.getRawParameterValue (bandParamId (b, "freq"));
        p.gain       = apvts.getRawParameterValue (bandParamId (b, "gain"));
        p.q          = apvts.getRawParameterValue (bandParamId (b, "q"));
        p.dynEnabled = apvts.getRawParameterValue (bandParamId (b, "dyn_enabled"));
        p.threshold  = apvts.getRawParameterValue (bandParamId (b, "dyn_threshold"));
        p.ratio      = apvts.getRawParameterValue (bandParamId (b, "dyn_ratio"));
        p.attack     = apvts.getRawParameterValue (bandParamId (b, "dyn_attack"));
        p.release    = apvts.getRawParameterValue (bandParamId (b, "dyn_release"));
    }
    inputGainPtr  = apvts.getRawParameterValue ("input_gain");
    outputGainPtr = apvts.getRawParameterValue ("output_gain");
    bypassPtr     = apvts.getRawParameterValue ("bypass");
}

//==============================================================================
const juce::String RoninEQAudioProcessor::getName() const
{
    return juce::String (RONINEQ_NAME);
}

bool RoninEQAudioProcessor::acceptsMidi() const      { return false; }
bool RoninEQAudioProcessor::producesMidi() const     { return false; }
bool RoninEQAudioProcessor::isMidiEffect() const     { return false; }
double RoninEQAudioProcessor::getTailLengthSeconds() const { return 0.0; }

int RoninEQAudioProcessor::getNumPrograms()                          { return 1; }
int RoninEQAudioProcessor::getCurrentProgram()                       { return 0; }
void RoninEQAudioProcessor::setCurrentProgram (int)                  {}
const juce::String RoninEQAudioProcessor::getProgramName (int)       { return {}; }
void RoninEQAudioProcessor::changeProgramName (int, const juce::String&) {}

bool RoninEQAudioProcessor::hasEditor() const { return true; }

juce::AudioProcessorEditor* RoninEQAudioProcessor::createEditor()
{
    return new RoninEQAudioProcessorEditor (*this);
}

//==============================================================================
void RoninEQAudioProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    effectiveSampleRate = sampleRate * (double) kOversampleFactor;

    oversampling.initProcessing ((size_t) juce::jmax (1, samplesPerBlock));
    oversampling.reset();

    for (auto& b : bands)
    {
        b.setSampleRate ((float) effectiveSampleRate);
        b.reset();
    }

    inputGainSmoother.reset (sampleRate, 0.05);
    outputGainSmoother.reset (sampleRate, 0.05);
    inputGainSmoother.setCurrentAndTargetValue (
        juce::Decibels::decibelsToGain (inputGainPtr->load (std::memory_order_relaxed)));
    outputGainSmoother.setCurrentAndTargetValue (
        juce::Decibels::decibelsToGain (outputGainPtr->load (std::memory_order_relaxed)));

    setLatencySamples ((int) oversampling.getLatencyInSamples());

    accumPos = 0;
    fifoWritePos.store (0, std::memory_order_relaxed);
    fifoReadPos.store (0, std::memory_order_relaxed);
}

void RoninEQAudioProcessor::releaseResources()
{
    oversampling.reset();
}

#ifndef JucePlugin_PreferredChannelConfigurations
bool RoninEQAudioProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;

    const auto in = layouts.getMainInputChannelSet();
    return in == juce::AudioChannelSet::mono() || in == juce::AudioChannelSet::stereo();
}
#endif

//==============================================================================
void RoninEQAudioProcessor::processBlock (juce::AudioBuffer<float>& buffer,
                                          juce::MidiBuffer&)
{
    juce::ScopedNoDenormals noDenormals;
    const int numSamples  = buffer.getNumSamples();
    const int numChannels = buffer.getNumChannels();

    // --- parameter -> DSP struct (block rate; the dynamic gain itself is per-sample)
    for (int i = 0; i < kNumBands; ++i)
    {
        const auto& p = bandPtrs[i];
        RoninEQ::BandParams bp;
        bp.enabled    = p.enabled->load (std::memory_order_relaxed) > 0.5f;
        bp.type       = static_cast<RoninEQ::Biquad::Type> (
                            (int) p.type->load (std::memory_order_relaxed));
        bp.freq       = p.freq->load (std::memory_order_relaxed);
        bp.gainDb     = p.gain->load (std::memory_order_relaxed);
        bp.q          = p.q->load (std::memory_order_relaxed);
        bp.dynEnabled = p.dynEnabled->load (std::memory_order_relaxed) > 0.5f;
        bp.thresholdDb= p.threshold->load (std::memory_order_relaxed);
        bp.ratio      = p.ratio->load (std::memory_order_relaxed);
        bp.attackMs   = p.attack->load (std::memory_order_relaxed);
        bp.releaseMs  = p.release->load (std::memory_order_relaxed);
        bands[i].setParams (bp);
    }

    inputGainSmoother.setTargetValue (
        juce::Decibels::decibelsToGain (inputGainPtr->load (std::memory_order_relaxed)));
    outputGainSmoother.setTargetValue (
        juce::Decibels::decibelsToGain (outputGainPtr->load (std::memory_order_relaxed)));

    if (bypassPtr->load (std::memory_order_relaxed) > 0.5f)
        return; // buffer already holds the dry input

    // --- input gain + pre-EQ analyzer tap (native rate)
    for (int n = 0; n < numSamples; ++n)
    {
        const float g = inputGainSmoother.getNextValue();
        float mono = 0.0f;
        for (int ch = 0; ch < numChannels; ++ch)
        {
            const float s = buffer.getSample (ch, n) * g;
            buffer.setSample (ch, n, s);
            mono += s;
        }
        pushPreSample (mono / (float) juce::jmax (1, numChannels));
    }

    // --- 2x oversampled band chain
    juce::dsp::AudioBlock<float> block (buffer);
    auto osBlock = oversampling.processSamplesUp (block);

    const int osCh = (int) osBlock.getNumChannels();
    const int osN  = (int) osBlock.getNumSamples();

    for (int n = 0; n < osN; ++n)
    {
        for (int ch = 0; ch < osCh; ch += 2)
        {
            float l = osBlock.getSample (ch, n);
            float r = (ch + 1 < osCh) ? osBlock.getSample (ch + 1, n) : l;

            for (int b = 0; b < kNumBands; ++b)
                bands[b].processSample (l, r);

            osBlock.setSample (ch, n, l);
            if (ch + 1 < osCh)
                osBlock.setSample (ch + 1, n, r);
        }
    }

    oversampling.processSamplesDown (block);

    // --- post-EQ analyzer tap + output gain (native rate)
    for (int n = 0; n < numSamples; ++n)
    {
        const float g = outputGainSmoother.getNextValue();
        float mono = 0.0f;
        for (int ch = 0; ch < numChannels; ++ch)
        {
            const float s = buffer.getSample (ch, n) * g;
            buffer.setSample (ch, n, s);
            mono += s;
        }
        pushPostSample (mono / (float) juce::jmax (1, numChannels));
    }

    for (int i = 0; i < kNumBands; ++i)
        bandMeters[i].store (bands[i].getDynamicGainDb(), std::memory_order_relaxed);
}

//==============================================================================
void RoninEQAudioProcessor::pushPreSample (float mono)
{
    if (accumPos < kFftSize)
        preAccum[(size_t) accumPos] = mono;
}

void RoninEQAudioProcessor::pushPostSample (float mono)
{
    if (accumPos < kFftSize)
        postAccum[(size_t) accumPos] = mono;

    if (++accumPos >= kFftSize)
        analyzeAccumulators(); // resets accumPos to 0
}

void RoninEQAudioProcessor::analyzeAccumulators()
{
    SpectrumFrame frame;
    computeLogSpectrum (preAccum.data(),  frame.pre);
    computeLogSpectrum (postAccum.data(), frame.post);
    pushSpectrumFrame (frame);
    accumPos = 0;
}

void RoninEQAudioProcessor::computeLogSpectrum (const float* src, float* dst)
{
    float* d = fftScratch.data(); // 2 * kFftSize, per JUCE FFT scratch requirement
    for (int i = 0; i < kFftSize; ++i)
        d[i] = src[i];

    window.multiplyWithWindowingTable (d, (size_t) kFftSize);
    fft.performFrequencyOnlyForwardTransform (d); // magnitudes now in d[0 .. kFftSize/2]

    const float binHz = (float) (effectiveSampleRate / (double) kOversampleFactor
                                 / (double) kFftSize);
    const float halfStep = std::pow (10.0f, 3.0f / (2.0f * (float) kNumSpectrumBins));

    for (int b = 0; b < kNumSpectrumBins; ++b)
    {
        const float fC = 20.0f * std::pow (1000.0f, (float) b / (float) (kNumSpectrumBins - 1));
        int kLo = juce::jmax (1, (int) (fC / halfStep / binHz));
        int kHi = juce::jmin (kFftSize / 2, (int) (fC * halfStep / binHz) + 1);

        double power = 0.0;
        for (int k = kLo; k <= kHi; ++k)
            power += (double) d[k] * (double) d[k];
        const float mag = std::sqrt ((float) (power / (double) juce::jmax (1, kHi - kLo + 1)));

        // Hann coherent gain is 0.5 -> full-scale sine peak reads kFftSize/4.
        float db = 20.0f * std::log10 (mag * 4.0f / (float) kFftSize + 1.0e-9f);
        dst[b] = juce::jlimit (-96.0f, 6.0f, db);
    }
}

void RoninEQAudioProcessor::pushSpectrumFrame (const SpectrumFrame& frame)
{
    const int w = fifoWritePos.load (std::memory_order_relaxed);
    const int r = fifoReadPos.load (std::memory_order_acquire);
    const int next = (w + 1) % kFifoCapacity;

    if (next != r) // drop the frame if the UI is behind; never block audio
    {
        spectrumFifo[w] = frame;
        fifoWritePos.store (next, std::memory_order_release);
    }
}

bool RoninEQAudioProcessor::popSpectrumFrame (SpectrumFrame& out)
{
    const int r = fifoReadPos.load (std::memory_order_relaxed);
    const int w = fifoWritePos.load (std::memory_order_acquire);

    if (r == w)
        return false;

    out = spectrumFifo[r];
    fifoReadPos.store ((r + 1) % kFifoCapacity, std::memory_order_release);
    return true;
}

float RoninEQAudioProcessor::getBandDynamicGainDb (int bandIndex) const
{
    if (bandIndex < 0 || bandIndex >= kNumBands)
        return 0.0f;
    return bandMeters[bandIndex].load (std::memory_order_relaxed);
}

//==============================================================================
void RoninEQAudioProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    juce::MemoryOutputStream stream (destData, true);
    apvts.state.writeToStream (stream);
}

void RoninEQAudioProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    juce::MemoryInputStream stream (data, (size_t) sizeInBytes, false);
    auto tree = juce::ValueTree::readFromStream (stream);
    if (tree.isValid())
        apvts.replaceState (tree);
}

//==============================================================================
// This creates new instances of the plugin.
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new RoninEQAudioProcessor();
}
