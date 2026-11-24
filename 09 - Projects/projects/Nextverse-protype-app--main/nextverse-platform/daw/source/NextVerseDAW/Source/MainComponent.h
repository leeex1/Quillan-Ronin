/*
  ==============================================================================

    MainComponent.h
    Created: 5 Aug 2024 11:00:00am
    Author:  Your Name

  ==============================================================================
*/

#pragma once

// JUCE 8+ removed the umbrella JuceHeader.h; include modules explicitly.
#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_gui_extra/juce_gui_extra.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_audio_devices/juce_audio_devices.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_utils/juce_audio_utils.h>
#include <tracktion_engine/tracktion_engine.h>

// Tracktion Engine (develop) moved from namespace tracktion_engine to
// tracktion::engine; keep the original source readable with a compatibility alias.
namespace tracktion_engine = tracktion::engine;

//==============================================================================
class MainComponent  : public juce::AudioAppComponent
{
public:
    //==============================================================================
    MainComponent();
    ~MainComponent() override;

    //==============================================================================
    void prepareToPlay (int samplesPerBlockExpected, double sampleRate) override;
    void getNextAudioBlock (const juce::AudioSourceChannelInfo& bufferToFill) override;
    void releaseResources() override;

    //==============================================================================
    void paint (juce::Graphics& g) override;
    void resized() override;

private:
    //==============================================================================
    tracktion_engine::Engine engine { tracktion_engine::Engine::getEngines()[0] }; // Use the default engine
    std::unique_ptr<tracktion_engine::Edit> edit;

    // Helper to create a unique ProjectItemID, needed for creating an Edit
    static tracktion_engine::ProjectItemID getNewProjectItemID()
    {
        // In a real app, this would come from a project manager that ensures uniqueness.
        // For this test, we generate a random one.
        return tracktion_engine::ProjectItemID (juce::Uuid().toString());
    }

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MainComponent)
};
