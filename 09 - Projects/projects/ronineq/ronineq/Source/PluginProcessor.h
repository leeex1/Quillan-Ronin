/*
    RoninEQ — clean-room dynamic equalizer (VST3 + Standalone).
    DSP core: hand-rolled C++17 (see Source/DSP/). JUCE is used only as the
    plugin shell: parameters, oversampling, FFT analyzer, UI.
    License: AGPL-3.0 (see LICENSE).
*/
#pragma once

#include <JuceHeader.h>
#include "DSP/DynamicBand.h"

//==============================================================================
class RoninEQAudioProcessor  : public juce::AudioProcessor
{
public:
    static constexpr int kNumBands       = 6;
    static constexpr int kOversampleFactor = 2;
    static constexpr int kFftOrder        = 11;
    static constexpr int kFftSize         = 1 << kFftOrder;   // 2048
    static constexpr int kNumSpectrumBins = 256;              // log-spaced 20Hz..20kHz

    struct SpectrumFrame
    {
        float pre [kNumSpectrumBins];
        float post[kNumSpectrumBins];
    };

    //==============================================================================
    RoninEQAudioProcessor();
    ~RoninEQAudioProcessor() override;

    //==============================================================================
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

   #ifndef JucePlugin_PreferredChannelConfigurations
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
   #endif

    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    //==============================================================================
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    //==============================================================================
    const juce::String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    //==============================================================================
    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram (int index) override;
    const juce::String getProgramName (int index) override;
    void changeProgramName (int index, const juce::String& newName) override;

    //==============================================================================
    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    //==============================================================================
    // Parameter IDs. One flat list; built once in createParameterLayout().
    static juce::String bandParamId (int bandIndex, const juce::String& leaf)
    {
        return "band" + juce::String (bandIndex + 1) + "_" + leaf;
    }
    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    juce::AudioProcessorValueTreeState apvts;

    //==============================================================================
    // Editor / UI accessors (thread-safe).
    float  getBandDynamicGainDb (int bandIndex) const;
    double getEffectiveSampleRate() const noexcept { return effectiveSampleRate; }
    bool   popSpectrumFrame (SpectrumFrame& out);

private:
    // Cached raw parameter pointers (no map lookup on the audio thread).
    struct BandParamPtrs
    {
        std::atomic<float>* enabled    = nullptr;
        std::atomic<float>* type       = nullptr;
        std::atomic<float>* freq       = nullptr;
        std::atomic<float>* gain       = nullptr;
        std::atomic<float>* q          = nullptr;
        std::atomic<float>* dynEnabled = nullptr;
        std::atomic<float>* threshold  = nullptr;
        std::atomic<float>* ratio      = nullptr;
        std::atomic<float>* attack     = nullptr;
        std::atomic<float>* release    = nullptr;
    };
    BandParamPtrs bandPtrs[kNumBands];
    std::atomic<float>* inputGainPtr  = nullptr;
    std::atomic<float>* outputGainPtr = nullptr;
    std::atomic<float>* bypassPtr     = nullptr;

    void cacheParameterPtrs();

    //--------------------------------------------------------------------------
    juce::dsp::Oversampling<float> oversampling;
    RoninEQ::DynamicBand           bands[kNumBands];

    juce::SmoothedValue<float, juce::ValueSmoothingTypes::Linear> inputGainSmoother;
    juce::SmoothedValue<float, juce::ValueSmoothingTypes::Linear> outputGainSmoother;

    std::atomic<float> bandMeters[kNumBands];
    double effectiveSampleRate = 48000.0;

    //--------------------------------------------------------------------------
    // Spectrum analyzer (audio thread produces, message thread consumes).
    juce::dsp::FFT fft { kFftOrder };
    juce::dsp::WindowingFunction<float> window { (size_t) kFftSize,
                                                 juce::dsp::WindowingFunction<float>::hann };
    std::vector<float> preAccum  = std::vector<float> ((size_t) kFftSize, 0.0f);
    std::vector<float> postAccum = std::vector<float> ((size_t) kFftSize, 0.0f);
    std::vector<float> fftScratch = std::vector<float> ((size_t) kFftSize * 2, 0.0f);
    int accumPos = 0;

    static constexpr int kFifoCapacity = 8;
    SpectrumFrame spectrumFifo[kFifoCapacity];
    std::atomic<int> fifoWritePos { 0 };
    std::atomic<int> fifoReadPos  { 0 };

    void pushPreSample (float mono);
    void pushPostSample (float mono);
    void analyzeAccumulators();
    void computeLogSpectrum (const float* src, float* dst);
    void pushSpectrumFrame (const SpectrumFrame& frame);

    //--------------------------------------------------------------------------
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (RoninEQAudioProcessor)
};
