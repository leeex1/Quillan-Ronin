/*
    JDXX Chains — audio processor implementation.
    AGPL-3.0. See LICENSE.
*/
#include "PluginProcessor.h"
#include "PluginEditor.h"
#include "PluginInfo.h"
#include <cmath>

// ---------------------------------------------------------------------------
JdxxChainsAudioProcessor::JdxxChainsAudioProcessor()
    : AudioProcessor (BusesProperties()
          .withInput  ("Input",  juce::AudioChannelSet::stereo(), true)
          .withOutput ("Output", juce::AudioChannelSet::stereo(), true)),
      apvts (*this, nullptr, "JDXCHAINS1", createParameterLayout())
{
    outGain.setCurrentAndTargetValue (1.0f);
    bypassMix.setCurrentAndTargetValue (0.0f);
}

juce::AudioProcessorValueTreeState::ParameterLayout JdxxChainsAudioProcessor::createParameterLayout()
{
    juce::AudioProcessorValueTreeState::ParameterLayout layout;

    layout.add (std::make_unique<juce::AudioParameterChoice> (
        "mode", "Chain (JDXX role)",
        juce::StringArray { "Vocal — recording JDXX", "Vocal adlib", "Intro (hypeVox basis)", "Pan (slowPan basis)", "Mastering JDXX" },
        0));

    layout.add (std::make_unique<juce::AudioParameterFloat> (
        "out_gain", "Output gain",
        juce::NormalisableRange<float> (-12.0f, 12.0f, 0.1f), 0.0f, " dB"));

    layout.add (std::make_unique<juce::AudioParameterBool> (
        "bypass", "Bypass", false));

    return layout;
}

const juce::String JdxxChainsAudioProcessor::getName() const { return JDXX_CHAINS_NAME; }
bool JdxxChainsAudioProcessor::acceptsMidi() const { return false; }
bool JdxxChainsAudioProcessor::producesMidi() const { return false; }
double JdxxChainsAudioProcessor::getTailLengthSeconds() const { return 0.0; }

bool JdxxChainsAudioProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    // Stereo only — the chains are mastered for stereo buses.
    return layouts.getMainOutputChannelSet() == juce::AudioChannelSet::stereo()
        && layouts.getMainInputChannelSet()  == juce::AudioChannelSet::stereo();
}

void JdxxChainsAudioProcessor::prepareToPlay (double sampleRate, int)
{
    sr = sampleRate;
    outGain.reset (sr, 0.02);       // 20 ms ramps
    bypassMix.reset (sr, 0.02);

    const int mode = (int) *apvts.getRawParameterValue ("mode");
    chain.build ((JdxxChains::ChainMode) mode, sr);
    lastBuiltMode = mode;
}

void JdxxChainsAudioProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer&)
{
    juce::ScopedNoDenormals noDenormals;

    const int mode = (int) *apvts.getRawParameterValue ("mode");
    if (mode != lastBuiltMode)
    {
        chain.build ((JdxxChains::ChainMode) mode, sr);
        lastBuiltMode = mode;
    }

    const float outDb = apvts.getRawParameterValue ("out_gain")->load();
    outGain.setTargetValue (std::pow (10.0f, outDb / 20.0f));

    const bool bypassed = apvts.getRawParameterValue ("bypass")->load() > 0.5f;
    bypassMix.setTargetValue (bypassed ? 1.0f : 0.0f);

    auto* L = buffer.getWritePointer (0);
    auto* R = buffer.getWritePointer (1);
    const int n = buffer.getNumSamples();

    // Snapshot dry signal for the bypass crossfade.
    juce::AudioBuffer<float> dryCopy;
    if (bypassMix.getCurrentValue() > 0.0001f || bypassMix.getTargetValue() > 0.0001f)
    {
        dryCopy.makeCopyOf (buffer);
    }

    chain.processBlock (L, R, n);

    for (int i = 0; i < n; ++i)
    {
        const float g  = outGain.getNextValue();
        const float bm = bypassMix.getNextValue();
        float wetL = L[i] * g, wetR = R[i] * g;
        if (dryCopy.getNumSamples() > 0)
        {
            wetL = (1.0f - bm) * wetL + bm * dryCopy.getSample (0, i);
            wetR = (1.0f - bm) * wetR + bm * dryCopy.getSample (1, i);
        }
        L[i] = juce::jlimit (-1.0f, 1.0f, wetL);
        R[i] = juce::jlimit (-1.0f, 1.0f, wetR);
    }
}

juce::AudioProcessorEditor* JdxxChainsAudioProcessor::createEditor()
{
    return new JdxxChainsAudioProcessorEditor (*this);
}

void JdxxChainsAudioProcessor::getStateInformation (juce::MemoryBlock& dest)
{
    if (auto xml = apvts.copyState().createXml())
        copyXmlToBinary (*xml, dest);
}

void JdxxChainsAudioProcessor::setStateInformation (const void* data, int size)
{
    if (auto xml = getXmlFromBinary (data, size))
        if (xml->hasTagName (apvts.state.getType()))
            apvts.replaceState (juce::ValueTree::fromXml (*xml));
}

// JUCE entry point — creates the processor the host will talk to.
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new JdxxChainsAudioProcessor();
}
