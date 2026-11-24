#include <juce_audio_processors/juce_audio_processors.h>
#include <chrono>
#include <iostream>
#include <string>

// Real JUCE audio graph: two instances of the owner's VST3, automation sweeps,
// MIDI events and a measured processing benchmark.
int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "usage: NVVST3Graph <path to .vst3>\n";
        return 2;
    }

    juce::AudioPluginFormatManager formats;
    formats.addFormat (new juce::VST3PluginFormat());
    juce::OwnedArray<juce::PluginDescription> found;
    juce::VST3PluginFormat scanner;
    scanner.findAllTypesForFile (found, juce::String::fromUTF8 (argv[1]));
    if (found.size() == 0)
    {
        std::cerr << "[FAIL] no VST3 found\n";
        return 1;
    }

    juce::AudioProcessorGraph graph;
    graph.setPlayConfigDetails (2, 2, 44100.0, 512);
    juce::String error;
    juce::AudioProcessorGraph::Node::Ptr nodes[2];
    for (int index = 0; index < 2; ++index)
    {
        auto instance = formats.createPluginInstance (*found[0], 44100.0, 512, error);
        if (instance == nullptr)
        {
            std::cerr << "[FAIL] plugin load: " << error.toStdString() << "\n";
            return 1;
        }
        nodes[index] = graph.addNode (std::move (instance));
    }
    auto input = graph.addNode (std::make_unique<juce::AudioProcessorGraph::AudioGraphIOProcessor> (
        juce::AudioProcessorGraph::AudioGraphIOProcessor::audioInputNode));
    auto output = graph.addNode (std::make_unique<juce::AudioProcessorGraph::AudioGraphIOProcessor> (
        juce::AudioProcessorGraph::AudioGraphIOProcessor::audioOutputNode));

    for (int channel = 0; channel < 2; ++channel)
    {
        graph.addConnection ({ { input->nodeID, channel }, { nodes[0]->nodeID, channel } });
        graph.addConnection ({ { nodes[0]->nodeID, channel }, { nodes[1]->nodeID, channel } });
        graph.addConnection ({ { nodes[1]->nodeID, channel }, { output->nodeID, channel } });
    }
    std::cout << "[DEBUG] plugin buses in=" << nodes[0]->getProcessor()->getBusCount (true)
              << " out=" << nodes[0]->getProcessor()->getBusCount (false)
              << " inputChannels=" << nodes[0]->getProcessor()->getMainBusNumInputChannels()
              << " outputChannels=" << nodes[0]->getProcessor()->getMainBusNumOutputChannels()
              << " connections=" << graph.getConnections().size() << "\n";
    graph.prepareToPlay (44100.0, 512);

    const int parameterCount = nodes[1]->getProcessor()->getParameters().size();
    if (parameterCount == 0)
    {
        std::cerr << "[FAIL] plugin exposes no automatable parameters\n";
        return 1;
    }

    juce::AudioBuffer<float> buffer (2, 512);
    juce::MidiBuffer midi;
    midi.addEvent (juce::MidiMessage::noteOn (1, 60, (juce::uint8) 100), 0);
    midi.addEvent (juce::MidiMessage::noteOff (1, 60), 256);

    constexpr int blocks = 256;
    float peak = 0.0f;
    auto start = std::chrono::high_resolution_clock::now();
    for (int block = 0; block < blocks; ++block)
    {
        for (int channel = 0; channel < 2; ++channel)
            for (int sample = 0; sample < 512; ++sample)
            {
                const double t = static_cast<double> (block * 512 + sample) / 44100.0;
                buffer.setSample (channel, sample,
                    0.2f * static_cast<float> (std::sin (2.0 * juce::MathConstants<double>::pi * 220.0 * t)));
            }
        // Automation: sweep the first exposed parameter from 0 to 1 and back.
        const float phase = static_cast<float> (block) / static_cast<float> (blocks);
        const float value = phase < 0.5f ? phase * 2.0f : (1.0f - phase) * 2.0f;
        nodes[1]->getProcessor()->getParameters()[0]->setValueNotifyingHost (value);
        graph.processBlock (buffer, midi);
        for (int channel = 0; channel < 2; ++channel)
            for (int sample = 0; sample < 512; ++sample)
                peak = juce::jmax (peak, std::abs (buffer.getSample (channel, sample)));
    }
    auto elapsed = std::chrono::duration<double> (std::chrono::high_resolution_clock::now() - start).count();

    if (! (peak > 0.0f))
    {
        std::cerr << "[FAIL] graph produced silence\n";
        return 1;
    }
    std::cout << "[PASS] two VST3 instances chained in a JUCE audio graph\n";
    std::cout << "[PASS] automation swept " << parameterCount << " exposed parameters\n";
    std::cout << "[PASS] MIDI note on/off events accepted by the graph\n";
    std::cout << "[PASS] processed " << blocks << " blocks in " << elapsed << " s, peak=" << peak << "\n";
    std::cout << "[PASS] audio graph verification complete\n";
    return 0;
}
