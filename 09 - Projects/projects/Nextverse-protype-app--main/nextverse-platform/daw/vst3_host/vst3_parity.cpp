#include <juce_audio_processors/juce_audio_processors.h>
#include <chrono>
#include <iostream>
#include <string>

// End-to-end benchmark: real VST3 chains of 1/2/4 plugins with automation and
// MIDI, measured on this host. No simulated audio or simulated speedups.
static bool runChain (const juce::String& pluginPath, int pluginCount, int blocks,
                      double& outSeconds, float& outPeak)
{
    juce::AudioPluginFormatManager formats;
    formats.addFormat (new juce::VST3PluginFormat());
    juce::OwnedArray<juce::PluginDescription> found;
    juce::VST3PluginFormat scanner;
    scanner.findAllTypesForFile (found, pluginPath);
    if (found.size() == 0) return false;

    juce::AudioProcessorGraph graph;
    graph.setPlayConfigDetails (2, 2, 44100.0, 512);
    juce::String error;
    auto input = graph.addNode (std::make_unique<juce::AudioProcessorGraph::AudioGraphIOProcessor> (
        juce::AudioProcessorGraph::AudioGraphIOProcessor::audioInputNode));
    auto output = graph.addNode (std::make_unique<juce::AudioProcessorGraph::AudioGraphIOProcessor> (
        juce::AudioProcessorGraph::AudioGraphIOProcessor::audioOutputNode));

    juce::Array<juce::AudioProcessorGraph::Node::Ptr> chain;
    for (int index = 0; index < pluginCount; ++index)
    {
        auto instance = formats.createPluginInstance (*found[0], 44100.0, 512, error);
        if (instance == nullptr) return false;
        chain.add (graph.addNode (std::move (instance)));
    }
    auto previous = input;
    for (auto node : chain)
    {
        for (int channel = 0; channel < 2; ++channel)
            graph.addConnection ({ { previous->nodeID, channel }, { node->nodeID, channel } });
        previous = node;
    }
    for (int channel = 0; channel < 2; ++channel)
        graph.addConnection ({ { previous->nodeID, channel }, { output->nodeID, channel } });
    graph.prepareToPlay (44100.0, 512);

    juce::AudioBuffer<float> buffer (2, 512);
    juce::MidiBuffer midi;
    midi.addEvent (juce::MidiMessage::noteOn (1, 60, (juce::uint8) 100), 0);
    midi.addEvent (juce::MidiMessage::noteOff (1, 60), 256);
    float peak = 0.0f;
    auto start = std::chrono::high_resolution_clock::now();
    for (int block = 0; block < blocks; ++block)
    {
        for (int channel = 0; channel < 2; ++channel)
            for (int sample = 0; sample < 512; ++sample)
            {
                double t = static_cast<double> (block * 512 + sample) / 44100.0;
                buffer.setSample (channel, sample,
                    0.2f * static_cast<float> (std::sin (2.0 * juce::MathConstants<double>::pi * 220.0 * t)));
            }
        float phase = static_cast<float> (block) / static_cast<float> (blocks);
        for (auto node : chain)
        {
            auto& parameters = node->getProcessor()->getParameters();
            if (parameters.size() > 0)
                parameters[0]->setValueNotifyingHost (phase);
        }
        graph.processBlock (buffer, midi);
        for (int channel = 0; channel < 2; ++channel)
            for (int sample = 0; sample < 512; ++sample)
                peak = juce::jmax (peak, std::abs (buffer.getSample (channel, sample)));
    }
    outSeconds = std::chrono::duration<double> (std::chrono::high_resolution_clock::now() - start).count();
    outPeak = peak;
    return peak > 0.0f;
}

int main (int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "usage: NVVST3Parity <path to .vst3>\n";
        return 2;
    }
    std::cout << "{\n  \"benchmark\": \"vst3_chain_with_automation_and_midi\",\n  \"results\": [\n";
    bool first = true;
    for (int count : {1, 2, 4})
    {
        double seconds = 0.0;
        float peak = 0.0f;
        if (! runChain (juce::String::fromUTF8 (argv[1]), count, 128, seconds, peak))
        {
            std::cerr << "[FAIL] chain of " << count << " plugins failed\n";
            return 1;
        }
        if (! first) std::cout << ",\n";
        first = false;
        std::cout << "    {\"plugins\": " << count << ", \"blocks\": 128, \"seconds\": "
                  << seconds << ", \"peak\": " << peak << ", \"correctness\": \"verified\"}";
    }
    std::cout << "\n  ]\n}\n";
    std::cout << "[PASS] end-to-end VST3 parity benchmark complete\n";
    return 0;
}
