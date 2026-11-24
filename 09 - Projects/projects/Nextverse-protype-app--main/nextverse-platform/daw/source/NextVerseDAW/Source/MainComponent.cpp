/*
  ==============================================================================

    MainComponent.cpp
    Created: 5 Aug 2024 11:00:00am
    Author:  Your Name

  ==============================================================================
*/

#include "MainComponent.h"

//==============================================================================
MainComponent::MainComponent()
    : engine(ProjectInfo::projectName) // Initialize the Tracktion Engine with the application name
{
    setSize (800, 600);

    // Initialize Tracktion Engine Edit
    auto projectItemID = getNewProjectItemID(); // Get a unique ID for the edit
    edit = std::make_unique<tracktion_engine::Edit>(
        engine, // Pass the engine reference
        tracktion_engine::Edit::Options {
            engine,
            tracktion_engine::ProjectManager::getTempDirectory(), // Temporary file location
            projectItemID,
            "Test Edit", // Name of the Edit
            tracktion_engine::EditFileOperations::forFile (juce::File()), // Dummy file ops
            true // isUndoManagerEnabled
        }
    );


    if (edit != nullptr)
    {
        // Add a single audio track to the edit as a test
        auto track = tracktion_engine::createAudioTrack(edit.get(), {});
        if (track != nullptr)
        {
            DBG("Tracktion Engine Initialised. Test track added to Edit: " + edit->getName());
        }
        else
        {
            DBG("Tracktion Engine Initialised, but failed to create audio track.");
        }
    }
    else
    {
        DBG("Failed to initialize Tracktion Engine Edit.");
    }


    // Some platforms require permissions to open input channels so request that here
    if (juce::RuntimePermissions::isRequired (juce::RuntimePermissions::recordAudio)
        && ! juce::RuntimePermissions::isGranted (juce::RuntimePermissions::recordAudio))
    {
        juce::RuntimePermissions::request (juce::RuntimePermissions::recordAudio,
                                           [&] (bool granted) { if (granted) setAudioChannels (2, 2); });
    }
    else
    {
        // Specify the number of input and output channels that we want to open
        setAudioChannels (0, 2); // No inputs for now, two outputs
    }
}

MainComponent::~MainComponent()
{
    edit = nullptr; // Release the edit first
    // Engine will clean itself up
    shutdownAudio();
}

//==============================================================================
void MainComponent::prepareToPlay (int samplesPerBlockExpected, double sampleRate)
{
    juce::ignoreUnused(samplesPerBlockExpected, sampleRate);
    // This function will be called when the audio device is started, or when
    // its settings (i.e. sample rate, block size, etc) are changed.
}

void MainComponent::getNextAudioBlock (const juce::AudioSourceChannelInfo& bufferToFill)
{
    // Your audio-processing code goes here!
    bufferToFill.clearActiveBufferRegion();
}

void MainComponent::releaseResources()
{
    // This will be called when the audio device stops, or when it is being
    // restarted due to a setting change.
}

//==============================================================================
void MainComponent::paint (juce::Graphics& g)
{
    // (Our component is opaque, so we must completely fill the background with a solid colour)
    g.fillAll (getLookAndFeel().findColour (juce::ResizableWindow::backgroundColourId));

    g.setFont (juce::Font (16.0f));
    g.setColour (juce::Colours::white);
    g.drawText ("NextVerseDAW - Tracktion Engine Test", getLocalBounds(), juce::Justification::centred, true);

    if (edit != nullptr)
    {
        g.drawText("Edit: " + edit->getName(), getLocalBounds().reduced(10, 10), juce::Justification::bottomLeft, true);
        g.drawText("Tracks: " + juce::String(edit->getTrackList().size()), getLocalBounds().reduced(10, 30), juce::Justification::bottomLeft, true);
    }
    else
    {
        g.drawText("Tracktion Edit not initialized.", getLocalBounds().reduced(10,10), juce::Justification::bottomLeft, true);
    }
}

void MainComponent::resized()
{
    // This is called when the MainContentComponent is resized.
    // If you add any child components, this is where you should
    // update their positions.
}
