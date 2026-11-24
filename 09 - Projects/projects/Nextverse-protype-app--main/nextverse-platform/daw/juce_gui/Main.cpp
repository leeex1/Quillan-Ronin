#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include <cstdlib>

using namespace juce;

class NextVerseDAWComponent : public AudioAppComponent,
                              private Timer,
                              private Slider::Listener,
                              private Button::Listener
{
public:
    NextVerseDAWComponent()
    {
        formatManager.addFormat (new VST3PluginFormat());
        graph.setPlayConfigDetails (2, 2, 44100.0, 512);
        buildGraph();

        for (int track = 0; track < maxTracks; ++track)
        {
            auto& fader = faders[track];
            fader.setSliderStyle (Slider::LinearVertical);
            fader.setTextBoxStyle (Slider::TextBoxBelow, false, 60, 18);
            fader.setRange (0.0, 1.0, 0.01);
            fader.setValue (0.6 + 0.1 * track, dontSendNotification);
            fader.addListener (this);
            addAndMakeVisible (fader);

            names[track].setText (track < static_cast<int> (levels.size()) ? "VST3 " + String (track + 1) : "Empty",
                                  dontSendNotification);
            addAndMakeVisible (names[track]);
        }

        playButton.setButtonText ("Play");
        playButton.addListener (this);
        addAndMakeVisible (playButton);

        automateButton.setButtonText ("Automation");
        automateButton.setClickingTogglesState (true);
        automateButton.addListener (this);
        addAndMakeVisible (automateButton);

        midiButton.setButtonText ("MIDI");
        midiButton.setClickingTogglesState (true);
        midiButton.addListener (this);
        addAndMakeVisible (midiButton);

        setSize (760, 430);
        setAudioChannels (0, 2);
        startTimerHz (20);
    }

    ~NextVerseDAWComponent() override
    {
        stopTimer();
        shutdownAudio();
    }

    void buildGraph()
    {
        graph.clear();
        nodes.clear();
        levels.clear();
        graph.setPlayConfigDetails (2, 2, 44100.0, 512);
        inputNode = graph.addNode (std::make_unique<AudioProcessorGraph::AudioGraphIOProcessor> (
            AudioProcessorGraph::AudioGraphIOProcessor::audioInputNode));
        outputNode = graph.addNode (std::make_unique<AudioProcessorGraph::AudioGraphIOProcessor> (
            AudioProcessorGraph::AudioGraphIOProcessor::audioOutputNode));

        String paths = String::fromUTF8 (std::getenv ("NEXTVERSE_VST3_PATHS") ? std::getenv ("NEXTVERSE_VST3_PATHS") : "");
        if (paths.isEmpty())
            paths = String::fromUTF8 (std::getenv ("NEXTVERSE_VST3_PATH") ? std::getenv ("NEXTVERSE_VST3_PATH") : "");

        auto tokens = StringArray::fromTokens (paths, "|", "");
        for (int track = 0; track < jmin (maxTracks, tokens.size()); ++track)
        {
            auto token = tokens[track].unquoted().trim();
            if (token.isEmpty()) continue;
            OwnedArray<PluginDescription> found;
            VST3PluginFormat scanner;
            scanner.findAllTypesForFile (found, token);
            if (found.size() == 0) continue;
            String error;
            auto instance = formatManager.createPluginInstance (*found[0], 44100.0, 512, error);
            if (instance == nullptr) continue;
            nodes.add (graph.addNode (std::move (instance)));
            levels.push_back (0.6f);
            for (int channel = 0; channel < 2; ++channel)
            {
                graph.addConnection ({ { inputNode->nodeID, channel }, { nodes.getLast()->nodeID, channel } });
                graph.addConnection ({ { nodes.getLast()->nodeID, channel }, { outputNode->nodeID, channel } });
            }
        }
        graph.prepareToPlay (44100.0, 512);
    }

    void prepareToPlay (int samplesPerBlockExpected, double sampleRate) override
    {
        currentSample = 0;
        currentSampleRate = sampleRate;
        graph.prepareToPlay (sampleRate, samplesPerBlockExpected);
    }

    void releaseResources() override { graph.releaseResources(); }

    void getNextAudioBlock (const AudioSourceChannelInfo& info) override
    {
        info.clearActiveBufferRegion();
        if (! playing) return;
        for (int channel = 0; channel < info.buffer->getNumChannels(); ++channel)
        {
            float* data = info.buffer->getWritePointer (channel);
            for (int sample = 0; sample < info.numSamples; ++sample)
            {
                double t = static_cast<double> (currentSample + sample) / currentSampleRate;
                data[sample] = 0.15f * static_cast<float> (std::sin (2.0 * MathConstants<double>::pi * 220.0 * t));
            }
        }
        currentSample += info.numSamples;

        if (automating)
        {
            for (int track = 0; track < static_cast<int> (levels.size()); ++track)
            {
                double phase = (currentSample % (currentSampleRate > 0 ? static_cast<int> (currentSampleRate) : 44100)) /
                               static_cast<double> (currentSampleRate > 0 ? currentSampleRate : 44100.0);
                float value = 0.3f + 0.5f * static_cast<float> (phase);
                levels[static_cast<size_t> (track)] = value;
                auto& parameters = nodes[track]->getProcessor()->getParameters();
                if (parameters.size() > 0) parameters[0]->setValueNotifyingHost (value);
            }
        }

        midiBuffer.clear();
        if (midiEnabled)
        {
            midiBuffer.addEvent (MidiMessage::noteOn (1, 60, (uint8) 100), 0);
            midiBuffer.addEvent (MidiMessage::noteOff (1, 60), 256);
        }
        graph.processBlock (*info.buffer, midiBuffer);
        peak = info.buffer->getMagnitude (0, info.numSamples);
    }

    void paint (Graphics& g) override
    {
        g.fillAll (Colour (0xff05070d));
        g.setColour (Colours::cyan);
        g.setFont (Font (22.0f, Font::bold));
        g.drawText ("NextVerse DAW — Mixer", 20, 12, 500, 30, Justification::left);
        g.setFont (Font (13.0f));
        g.drawText ("Tracks: " + String (static_cast<int> (levels.size())) +
                    "  |  Graph connections: " + String (graph.getConnections().size()) +
                    "  |  Peak: " + String (peak, 3), 20, 48, 720, 22, Justification::left);
        g.drawText ("Set NEXTVERSE_VST3_PATHS with '|' separated VST3 paths for up to 4 tracks",
                    20, 400, 720, 20, Justification::left);
    }

    void resized() override
    {
        playButton.setBounds (20, 78, 90, 28);
        automateButton.setBounds (120, 78, 110, 28);
        midiButton.setBounds (240, 78, 80, 28);
        for (int track = 0; track < maxTracks; ++track)
        {
            int x = 30 + track * 180;
            faders[track].setBounds (x, 120, 60, 250);
            names[track].setBounds (x, 372, 120, 22);
        }
    }

private:
    void sliderValueChanged (Slider* slider) override
    {
        for (int track = 0; track < maxTracks; ++track)
            if (slider == &faders[track] && track < static_cast<int> (levels.size()))
                levels[static_cast<size_t> (track)] = static_cast<float> (slider->getValue());
    }

    void buttonClicked (Button* button) override
    {
        if (button == &playButton) playing = !playing;
        else if (button == &automateButton) automating = automateButton.getToggleState();
        else if (button == &midiButton) midiEnabled = midiButton.getToggleState();
    }

    void timerCallback() override { repaint(); }

    static constexpr int maxTracks = 4;
    AudioPluginFormatManager formatManager;
    AudioProcessorGraph graph;
    AudioProcessorGraph::Node::Ptr inputNode, outputNode;
    Array<AudioProcessorGraph::Node::Ptr> nodes;
    std::vector<float> levels;
    Slider faders[maxTracks];
    Label names[maxTracks];
    TextButton playButton, automateButton, midiButton;
    MidiBuffer midiBuffer;
    bool playing = true, automating = false, midiEnabled = false;
    int currentSample = 0;
    double currentSampleRate = 44100.0;
    float peak = 0.0f;
};

class NextVerseDAWApplication : public JUCEApplication
{
public:
    const String getApplicationName() override { return "NextVerse DAW"; }
    const String getApplicationVersion() override { return "0.2.0"; }
    void initialise (const String&) override { mainWindow = std::make_unique<MainWindow>(); }
    void shutdown() override { mainWindow.reset(); }

private:
    class MainWindow : public DocumentWindow
    {
    public:
        MainWindow() : DocumentWindow ("NextVerse DAW", Colour (0xff05070d), DocumentWindow::allButtons)
        {
            setContentOwned (new NextVerseDAWComponent(), true);
            centreWithSize (getWidth(), getHeight());
            setVisible (true);
        }
        void closeButtonPressed() override { JUCEApplication::getInstance()->systemRequestedQuit(); }
    };
    std::unique_ptr<MainWindow> mainWindow;
};

START_JUCE_APPLICATION (NextVerseDAWApplication)
