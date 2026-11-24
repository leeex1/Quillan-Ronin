#include <juce_audio_processors/juce_audio_processors.h>
#include <iostream>

#include "../Source/NextVerseToneGenerator.h"
#include "../Source/NextVerseSynthProcessor.h"

int main()
{
    juce::AudioProcessorGraph graph;

    auto tone = graph.addNode (std::make_unique<NextVerseToneGenerator>());
    using IO = juce::AudioProcessorGraph::AudioGraphIOProcessor;
    auto output = graph.addNode (std::make_unique<IO> (IO::audioOutputNode));

    if (tone == nullptr || output == nullptr)
    {
        std::cerr << "[FAIL] could not create graph nodes\n";
        return 1;
    }

    // JUCE 8 graphs need explicit channel layouts before connections are legal.
    tone->getProcessor()->setPlayConfigDetails (0, 2, 44100.0, 512);
    output->getProcessor()->setPlayConfigDetails (2, 0, 44100.0, 512);

    for (int channel = 0; channel < 2; ++channel)
    {
        const juce::AudioProcessorGraph::NodeAndChannel source { tone->nodeID, channel };
        const juce::AudioProcessorGraph::NodeAndChannel destination { output->nodeID, channel };
        const juce::AudioProcessorGraph::Connection connection { source, destination };
        if (! graph.addConnection (connection))
        {
            std::cerr << "[FAIL] could not connect channel " << channel << "\n";
            return 1;
        }
    }

    constexpr int blockSize = 512;
    graph.setPlayConfigDetails (0, 2, 44100.0, blockSize);
    graph.prepareToPlay (44100.0, blockSize);

    juce::AudioBuffer<float> buffer (2, blockSize);
    buffer.clear();
    juce::MidiBuffer midi;
    graph.processBlock (buffer, midi);

    const float peak = buffer.getMagnitude (0, blockSize);
    if (! (peak > 0.01f && peak <= 0.3f))
    {
        std::cerr << "[FAIL] graph rendered unexpected peak: " << peak << "\n";
        return 1;
    }

    std::cout << "[PASS] JUCE audio graph connected 1 generator to 2 outputs\n";
    std::cout << "[PASS] JUCE audio graph offline render peak=" << peak << "\n";
    // --- Plugin parameters + MIDI input ---------------------------------
    juce::AudioProcessorGraph synthGraph;
    auto synthNode = synthGraph.addNode (std::make_unique<NextVerseSynthProcessor>());
    auto synthOutput = synthGraph.addNode (std::make_unique<IO> (IO::audioOutputNode));
    if (synthNode == nullptr || synthOutput == nullptr)
    {
        std::cerr << "[FAIL] synth graph nodes\n";
        return 1;
    }
    synthNode->getProcessor()->setPlayConfigDetails (0, 2, 44100.0, 512);
    synthOutput->getProcessor()->setPlayConfigDetails (2, 0, 44100.0, 512);
    for (int channel = 0; channel < 2; ++channel)
    {
        const juce::AudioProcessorGraph::NodeAndChannel source { synthNode->nodeID, channel };
        const juce::AudioProcessorGraph::NodeAndChannel destination { synthOutput->nodeID, channel };
        if (! synthGraph.addConnection ({ source, destination }))
        {
            std::cerr << "[FAIL] synth graph connection\n";
            return 1;
        }
    }
    synthGraph.setPlayConfigDetails (0, 2, 44100.0, 512);
    synthGraph.prepareToPlay (44100.0, 512);

    auto* synth = dynamic_cast<NextVerseSynthProcessor*> (synthNode->getProcessor());
    if (synth == nullptr || ! synth->acceptsMidi())
    {
        std::cerr << "[FAIL] synth processor contract\n";
        return 1;
    }

    // Process the plugin directly: graph MIDI input requires an explicit MIDI
    // input node, which is outside this contract test.
    synth->prepareToPlay (44100.0, 512);
    juce::AudioBuffer<float> synthBuffer (2, 512);
    juce::MidiBuffer midiIn;
    synthBuffer.clear();
    synth->processBlock (synthBuffer, midiIn);
    if (synthBuffer.getMagnitude (0, 512) > 0.001f)
    {
        std::cerr << "[FAIL] synth should be silent without a MIDI note\n";
        return 1;
    }

    synth->setGain (0.5f);
    midiIn.addEvent (juce::MidiMessage::noteOn (1, 69, (juce::uint8) 100), 0);
    synth->processBlock (synthBuffer, midiIn);
    const float notePeak = synthBuffer.getMagnitude (0, 512);
    if (! (notePeak > 0.05f && notePeak <= 0.15f))
    {
        std::cerr << "[FAIL] MIDI-triggered voice peak: " << notePeak << "\n";
        return 1;
    }

    midiIn.clear();
    midiIn.addEvent (juce::MidiMessage::noteOff (1, 69), 0);
    synth->processBlock (synthBuffer, midiIn);
    if (synthBuffer.getMagnitude (0, 512) > 0.001f)
    {
        std::cerr << "[FAIL] synth should be silent after note-off\n";
        return 1;
    }

    synth->setGain (0.0f);
    midiIn.clear();
    midiIn.addEvent (juce::MidiMessage::noteOn (1, 69, (juce::uint8) 100), 0);
    synth->processBlock (synthBuffer, midiIn);
    if (synthBuffer.getMagnitude (0, 512) > 0.001f)
    {
        std::cerr << "[FAIL] automated gain=0 should mute the voice\n";
        return 1;
    }

    std::cout << "[PASS] JUCE plugin parameter automation (gain 0.5 -> peak "
              << notePeak << ", gain 0 -> silence)\n";
    std::cout << "[PASS] JUCE MIDI note on/off gating verified\n";

    std::cout << "[PASS] JUCE audio graph verification complete\n";
    return 0;
}
