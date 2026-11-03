/*
    JDXX Chains — minimal editor implementation.
    AGPL-3.0. See LICENSE.
*/
#include "PluginEditor.h"
#include "PluginProcessor.h"
#include "PluginInfo.h"
#include "Chains/ChainSpecs.h"

JdxxChainsAudioProcessorEditor::JdxxChainsAudioProcessorEditor (JdxxChainsAudioProcessor& p)
    : AudioProcessorEditor (&p), processor (p)
{
    setResizable (true, true);
    setSize (560, 420);

    titleLabel.setText (JDXX_CHAINS_NAME, juce::dontSendNotification);
    titleLabel.setFont (juce::Font (juce::FontOptions().withHeight (22.0f).withStyle ("Bold")));
    titleLabel.setColour (juce::Label::textColourId, juce::Colours::white);
    addAndMakeVisible (titleLabel);

    specLabel.setColour (juce::Label::backgroundColourId, juce::Colour (0xff14161c));
    specLabel.setColour (juce::Label::textColourId, juce::Colour (0xffcdd3e0));
    specLabel.setJustificationType (juce::Justification::centredLeft);
    specLabel.setBorderSize (juce::BorderSize<int> (12, 12, 12, 12));
    addAndMakeVisible (specLabel);

    modeBox.addItemList ({ "Vocal — recording JDXX", "Vocal adlib", "Intro (hypeVox basis)",
                           "Pan (slowPan basis)", "Mastering JDXX" }, 1);
    addAndMakeVisible (modeBox);
    modeAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>
        (processor.apvts, "mode", modeBox);

    gainSlider.setSliderStyle (juce::Slider::LinearHorizontal);
    gainSlider.setTextBoxStyle (juce::Slider::TextBoxRight, false, 64, 20);
    gainSlider.setRange (-12.0, 12.0);
    gainSlider.setTextValueSuffix (" dB");
    gainSlider.setName ("Output gain");
    addAndMakeVisible (gainSlider);
    gainAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>
        (processor.apvts, "out_gain", gainSlider);

    bypassButton.setButtonText ("Bypass");
    addAndMakeVisible (bypassButton);
    bypassAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>
        (processor.apvts, "bypass", bypassButton);

    footerNote.setText (
        "JDXX chains — spec 2026-11-02 (live BandLab presets). Zero latency.\n"
        "Chain order is load-bearing; bypasses are per global switch only in v0.1.",
        juce::dontSendNotification);
    footerNote.setFont (juce::Font (juce::FontOptions().withHeight (12.0f)));
    footerNote.setColour (juce::Label::textColourId, juce::Colour (0xff8b93a7));
    addAndMakeVisible (footerNote);

    updateSpecLabel();
    startTimerHz (8);   // editor is also a juce::Timer now — polls mode at 8 Hz
}

void JdxxChainsAudioProcessorEditor::timerCallback()
{
    updateSpecLabel();
}

void JdxxChainsAudioProcessorEditor::updateSpecLabel()
{
    const int mode = (int) *processor.apvts.getRawParameterValue ("mode");
    const auto& spec = JdxxChains::getChainSpec ((JdxxChains::ChainMode) mode);

    juce::String s;
    s << "Role: " << spec.jdxxRole << "\n\n";
    s << "JDXX chain (" << spec.numModules << " stages, in order):\n";
    for (int i = 0; i < spec.numModules; ++i)
        s << "  " << (i + 1) << ". " << spec.modules[i].name << "\n";
    specLabel.setText (s, juce::dontSendNotification);
}

void JdxxChainsAudioProcessorEditor::paint (juce::Graphics& g)
{
    g.fillAll (juce::Colour (0xff0b0d12));
}

void JdxxChainsAudioProcessorEditor::resized()
{
    auto bounds = getLocalBounds().reduced (16);
    titleLabel.setBounds (bounds.removeFromTop (32));
    bounds.removeFromTop (8);

    auto controls = bounds.removeFromTop (44);
    modeBox.setBounds (controls.removeFromLeft (220));
    controls.removeFromLeft (12);
    gainSlider.setBounds (controls.removeFromLeft (240));
    controls.removeFromLeft (16);
    bypassButton.setBounds (controls.removeFromLeft (90));

    bounds.removeFromTop (10);
    specLabel.setBounds (bounds.removeFromTop (250));
    bounds.removeFromTop (8);
    footerNote.setBounds (bounds.removeFromTop (32));
}
