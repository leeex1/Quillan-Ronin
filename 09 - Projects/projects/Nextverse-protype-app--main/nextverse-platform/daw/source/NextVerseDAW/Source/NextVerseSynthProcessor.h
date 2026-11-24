#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <cmath>

/// A small JUCE AudioProcessor demonstrating the DAW plugin contract:
/// exposed parameters (automation) plus MIDI note input.
class NextVerseSynthProcessor : public juce::AudioProcessor
{
public:
    NextVerseSynthProcessor()
    {
        gainParameter = new juce::AudioParameterFloat (
            juce::ParameterID { "nextverse_gain", 1 }, "Gain", 0.0f, 1.0f, 0.5f);
        addParameter (gainParameter);
    }

    const juce::String getName() const override { return "NextVerse Synth"; }
    bool acceptsMidi() const override { return true; }
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
        noteActive = false;
    }

    void releaseResources() override {}

    void processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi) override
    {
        for (const auto metadata : midi)
        {
            const auto message = metadata.getMessage();
            if (message.isNoteOn())
            {
                frequency = noteToFrequency (message.getNoteNumber());
                noteActive = true;
            }
            else if (message.isNoteOff())
            {
                noteActive = false;
            }
        }

        const float gain = gainParameter != nullptr ? gainParameter->get() : 0.0f;
        constexpr double twoPi = 6.283185307179586;

        for (int channel = 0; channel < buffer.getNumChannels(); ++channel)
        {
            for (int sample = 0; sample < buffer.getNumSamples(); ++sample)
            {
                float value = 0.0f;
                if (noteActive)
                {
                    const double t = static_cast<double> (phase + sample) / sampleRate;
                    value = static_cast<float> (0.25 * std::sin (twoPi * frequency * t) * gain);
                }
                buffer.setSample (channel, sample, value);
            }
        }
        phase += buffer.getNumSamples();
    }

    static double noteToFrequency (int noteNumber)
    {
        return 440.0 * std::pow (2.0, (noteNumber - 69) / 12.0);
    }

    void setGain (float value)
    {
        if (gainParameter != nullptr)
            gainParameter->setValueNotifyingHost (juce::jlimit (0.0f, 1.0f, value));
    }

    float getGain() const { return gainParameter != nullptr ? gainParameter->get() : 0.0f; }

    void getStateInformation (juce::MemoryBlock&) override {}
    void setStateInformation (const void*, int) override {}

private:
    juce::AudioParameterFloat* gainParameter = nullptr;
    double sampleRate = 44100.0;
    double phase = 0.0;
    double frequency = 440.0;
    bool noteActive = false;
};
