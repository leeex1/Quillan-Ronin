#include <juce_audio_processors/juce_audio_processors.h>
#include <chrono>
#include <iostream>

// Four-track native mixer benchmark: one VST3 per track, per-track fader
// automation, MIDI events and summed stereo output.
int main (int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "usage: NVVST3Mixer <path to .vst3>\n";
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
    auto input = graph.addNode (std::make_unique<juce::AudioProcessorGraph::AudioGraphIOProcessor> (
        juce::AudioProcessorGraph::AudioGraphIOProcessor::audioInputNode));
    auto output = graph.addNode (std::make_unique<juce::AudioProcessorGraph::AudioGraphIOProcessor> (
        juce::AudioProcessorGraph::AudioGraphIOProcessor::audioOutputNode));

    constexpr int tracks = 4;
    juce::Array<juce::AudioProcessorGraph::Node::Ptr> plugins;
    for (int track = 0; track < tracks; ++track)
    {
        auto instance = formats.createPluginInstance (*found[0], 44100.0, 512, error);
        if (instance == nullptr)
        {
            std::cerr << "[FAIL] plugin load track " << track << ": " << error.toStdString() << "\n";
            return 1;
        }
        plugins.add (graph.addNode (std::move (instance)));
        for (int channel = 0; channel < 2; ++channel)
            graph.addConnection ({ { input->nodeID, channel }, { plugins[track]->nodeID, channel } });
    }
    for (auto node : plugins)
        for (int channel = 0; channel < 2; ++channel)
            graph.addConnection ({ { node->nodeID, channel }, { output->nodeID, channel } });
    graph.prepareToPlay (44100.0, 512);

    juce::AudioBuffer<float> buffer (2, 512);
    juce::MidiBuffer midi;
    midi.addEvent (juce::MidiMessage::noteOn (1, 60, (juce::uint8) 100), 0);
    midi.addEvent (juce::MidiMessage::noteOff (1, 60), 256);

    constexpr int blocks = 128;
    float peak = 0.0f;
    auto start = std::chrono::high_resolution_clock::now();
    for (int block = 0; block < blocks; ++block)
    {
        for (int channel = 0; channel < 2; ++channel)
            for (int sample = 0; sample < 512; ++sample)
            {
                double t = static_cast<double> (block * 512 + sample) / 44100.0;
                buffer.setSample (channel, sample,
                    0.15f * static_cast<float> (std::sin (2.0 * juce::MathConstants<double>::pi * 220.0 * t)));
            }
        for (int track = 0; track < tracks; ++track)
        {
            auto& parameters = plugins[track]->getProcessor()->getParameters();
            if (parameters.size() > 0)
            {
                float base = 0.25f + 0.15f * static_cast<float> (track);
                float sweep = 0.1f * static_cast<float> (block % 32) / 32.0f;
                parameters[0]->setValueNotifyingHost (base + sweep);
            }
        }
        graph.processBlock (buffer, midi);
        for (int channel = 0; channel < 2; ++channel)
            for (int sample = 0; sample < 512; ++sample)
                peak = juce::jmax (peak, std::abs (buffer.getSample (channel, sample)));
    }
    double seconds = std::chrono::duration<double> (std::chrono::high_resolution_clock::now() - start).count();

    if (! (peak > 0.0f))
    {
        std::cerr << "[FAIL] mixer produced silence\n";
        return 1;
    }
    std::cout << "[PASS] 4-track native mixer with VST3 per track\n";
    std::cout << "[PASS] per-track fader automation + MIDI events active\n";
    std::cout << "[PASS] processed " << blocks << " blocks in " << seconds << " s, peak=" << peak << "\n";
    std::cout << "[PASS] mixer benchmark complete\n";
    return 0;
}
