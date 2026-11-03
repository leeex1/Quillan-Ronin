/*
    JDXX Chains — audio processor shell (JUCE 8)

    Hosts the five JDXX chains as one plugin. Zero latency by design.
    Host automation (M1): chain mode + output gain + bypass.
    Per-module parameter exposure to the host lands in M2.

    AGPL-3.0. Derivative of RoninEQ. See LICENSE.
*/
#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include "Chains/ChainEngine.h"

class JdxxChainsAudioProcessor : public juce::AudioProcessor
{
public:
    JdxxChainsAudioProcessor();
    ~JdxxChainsAudioProcessor() override = default;

    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override {}
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
    void processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override;
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return getName(); }
    void changeProgramName (int, const juce::String&) override {}

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    double getTailLengthSeconds() const override;

    void getStateInformation (juce::MemoryBlock& dest) override;
    void setStateInformation (const void* data, int size) override;

    // APVTS: mode (5), output gain, bypass.
    juce::AudioProcessorValueTreeState apvts;

    // Currently built chain (mode combo selects this).
    JdxxChains::ChainInstance chain;
    int lastBuiltMode = -1;

private:
    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    double sr = 48000.0;
    juce::SmoothedValue<float, juce::ValueSmoothingTypes::Linear> outGain;
    juce::SmoothedValue<float, juce::ValueSmoothingTypes::Linear> bypassMix;  // 0 = wet, 1 = dry

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (JdxxChainsAudioProcessor)
};
