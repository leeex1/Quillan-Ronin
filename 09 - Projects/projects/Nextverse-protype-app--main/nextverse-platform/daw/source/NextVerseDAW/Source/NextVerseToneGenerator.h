#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <cmath>

/// Minimal deterministic signal source used to prove the JUCE audio graph
/// carries real samples. It mirrors the conceptual DAW's test-tone behaviour.
class NextVerseToneGenerator : public juce::AudioProcessor
{
public:
    NextVerseToneGenerator() = default;

    const juce::String getName() const override { return "NextVerse Tone Generator"; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }
    juce::AudioProcessorEditor* createEditor() override { return nullptr; }
    bool hasEditor() const override { return false; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; }
    void changeProgramName (int, const juce::String&) override {}

    bool isBusesLayoutSupported (const BusesLayout& layouts) const override
    {
        return layouts.getMainOutputChannelSet() == juce::AudioChannelSet::stereo();
    }

    void prepareToPlay (double sampleRateHz, int) override
    {
        sampleRate = sampleRateHz > 0.0 ? sampleRateHz : 44100.0;
        phase = 0.0;
    }

    void releaseResources() override {}

    void processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) override
    {
        constexpr double frequency = 440.0;
        constexpr double twoPi = 6.283185307179586;
        for (int channel = 0; channel < buffer.getNumChannels(); ++channel)
        {
            for (int sample = 0; sample < buffer.getNumSamples(); ++sample)
            {
                const double t = static_cast<double> (phase + sample) / sampleRate;
                buffer.setSample (channel, sample,
                                  static_cast<float> (0.25 * std::sin (twoPi * frequency * t)));
            }
        }
        phase += buffer.getNumSamples();
    }

    void getStateInformation (juce::MemoryBlock&) override {}
    void setStateInformation (const void*, int) override {}

private:
    double sampleRate = 44100.0;
    double phase = 0.0;
};
