/*
    RoninEQ — Pro-Q-style editor: log grid, pre/post spectrum, summed curve,
    draggable band nodes, per-band controls, dynamics section.
    License: AGPL-3.0.
*/
#pragma once

#include <JuceHeader.h>
#include "PluginProcessor.h"
#include "DSP/Biquad.h"

//==============================================================================
class RoninEQAudioProcessorEditor  : public juce::AudioProcessorEditor,
                                     public juce::Timer
{
public:
    explicit RoninEQAudioProcessorEditor (RoninEQAudioProcessor&);
    ~RoninEQAudioProcessorEditor() override;

    void paint (juce::Graphics&) override;
    void resized() override;
    void timerCallback() override;

private:
    //--------------------------------------------------------------------------
    // The main display: log-frequency grid, analyzer spectra, summed EQ curve,
    // and the draggable band nodes.
    class SpectrumComponent : public juce::Component
    {
    public:
        explicit SpectrumComponent (RoninEQAudioProcessor& p);

        void updateFromState(); // called on the message thread by the editor timer

        void paint (juce::Graphics&) override;
        void mouseDown (const juce::MouseEvent&) override;
        void mouseDrag (const juce::MouseEvent&) override;
        void mouseUp (const juce::MouseEvent&) override;
        void mouseDoubleClick (const juce::MouseEvent&) override;

        int  getSelectedBand() const noexcept { return selectedBand; }
        void setSelectedBand (int b);

        std::function<void (int)> onBandSelected; // editor hook: rebuild attachments

    private:
        RoninEQAudioProcessor& processor;

        // Static-curve mirrors (message thread only; fed from APVTS atomics).
        RoninEQ::Biquad curveFilters[RoninEQAudioProcessor::kNumBands];

        // Displayed spectra with peak decay, in dB.
        float displayPre [RoninEQAudioProcessor::kNumSpectrumBins];
        float displayPost[RoninEQAudioProcessor::kNumSpectrumBins];

        int selectedBand = 0;

        // Node dragging state.
        bool dragging = false;
        bool dragIsQ  = false;
        juce::RangedAudioParameter* dragFreqParam = nullptr;
        juce::RangedAudioParameter* dragGainParam = nullptr;
        juce::RangedAudioParameter* dragQParam    = nullptr;

        static constexpr float kMinFreq = 20.0f, kMaxFreq = 20000.0f;
        static constexpr float kTopDb = 24.0f, kBottomDb = -60.0f;

        float freqToX (float freq) const;
        float xToFreq (float x) const;
        float gainToY (float gainDb) const;
        float yToGain (float y) const;
        juce::Point<float> nodePosition (int band) const;
        int nodeAt (juce::Point<float> pos) const;

        juce::Colour bandColour (int band) const;
        void applyDragToParams (juce::Point<float> pos);

        JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (SpectrumComponent)
    };

    //--------------------------------------------------------------------------
    RoninEQAudioProcessor& processorRef;

    SpectrumComponent spectrum;

    // Band strip.
    juce::TextButton bandButtons[RoninEQAudioProcessor::kNumBands];
    juce::ToggleButton enabledButton { "Enabled" };
    juce::ComboBox typeBox;

    // Static EQ knobs.
    juce::Slider freqSlider, gainSlider, qSlider;

    // Dynamics.
    juce::ToggleButton dynButton { "Dynamic" };
    juce::Slider thresholdSlider, ratioSlider, attackSlider, releaseSlider;
    juce::Label grLabel;

    // Globals.
    juce::Slider inputGainSlider, outputGainSlider;
    juce::ToggleButton bypassButton { "Bypass" };

    // Attachments are rebuilt whenever the selected band changes.
    using SliderAttach = juce::AudioProcessorValueTreeState::SliderAttachment;
    using ButtonAttach = juce::AudioProcessorValueTreeState::ButtonAttachment;
    using ComboAttach  = juce::AudioProcessorValueTreeState::ComboBoxAttachment;
    std::unique_ptr<SliderAttach> freqAttach, gainAttach, qAttach;
    std::unique_ptr<SliderAttach> thresholdAttach, ratioAttach, attackAttach, releaseAttach;
    std::unique_ptr<SliderAttach> inputGainAttach, outputGainAttach;
    std::unique_ptr<ButtonAttach> enabledAttach, dynAttach, bypassAttach;
    std::unique_ptr<ComboAttach>  typeAttach;

    void selectBand (int bandIndex);
    void rebuildBandAttachments();
    static void setupRotary (juce::Slider& s, const juce::String& suffix);

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (RoninEQAudioProcessorEditor)
};
