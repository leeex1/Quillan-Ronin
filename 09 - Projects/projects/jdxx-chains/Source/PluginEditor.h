/*
    JDXX Chains — minimal editor: mode select, output gain, bypass, spec sheet.
    (M1 UI. Per-module knobs/toggles land in M2 with host automation.)
    AGPL-3.0. See LICENSE.
*/
#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_gui_basics/juce_gui_basics.h>

class JdxxChainsAudioProcessor;

class JdxxChainsAudioProcessorEditor : public juce::AudioProcessorEditor
{
public:
    explicit JdxxChainsAudioProcessorEditor (JdxxChainsAudioProcessor&);
    ~JdxxChainsAudioProcessorEditor() override = default;

    void paint (juce::Graphics&) override;
    void resized() override;

private:
    void updateSpecLabel();
    void timerCallback() override;

    JdxxChainsAudioProcessor& processor;
    juce::Label titleLabel, specLabel;
    juce::ComboBox modeBox;
    juce::Slider gainSlider;
    juce::ToggleButton bypassButton;
    juce::Label footerNote;
    juce::Timer specPoll;   // refreshes the spec read-out when mode changes

    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> modeAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> gainAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> bypassAttachment;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (JdxxChainsAudioProcessorEditor)
};
