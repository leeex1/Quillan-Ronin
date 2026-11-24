#include <juce_audio_processors/juce_audio_processors.h>
#include <iostream>
#include <string>

// Real VST3 host using JUCE's plugin scanner. No simulated plugin data.
int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "usage: NVVST3Host <path to .vst3>\n";
        return 2;
    }

    juce::AudioPluginFormatManager formats;
    formats.addFormat (new juce::VST3PluginFormat());

    juce::PluginDescription description;
    juce::OwnedArray<juce::PluginDescription> found;
    juce::VST3PluginFormat vst3;
    juce::String searchPath = juce::String::fromUTF8 (argv[1]);

    // Scan the exact path supplied; JUCE accepts a file or directory.
    vst3.findAllTypesForFile (found, searchPath);
    if (found.size() == 0)
    {
        std::cerr << "[FAIL] no VST3 plugin found at: " << argv[1] << "\n";
        return 1;
    }

    description = *found[0];
    juce::String error;
    auto instance = formats.createPluginInstance (description, 44100.0, 512, error);
    if (instance == nullptr)
    {
        std::cerr << "[FAIL] plugin load error: " << error.toStdString() << "\n";
        return 1;
    }

    instance->prepareToPlay (44100.0, 512);
    juce::AudioBuffer<float> buffer (2, 512);
    for (int channel = 0; channel < 2; ++channel)
        for (int sample = 0; sample < 512; ++sample)
            buffer.setSample (channel, sample, 0.25f);
    juce::MidiBuffer midi;
    instance->processBlock (buffer, midi);

    float peak = 0.0f;
    for (int channel = 0; channel < 2; ++channel)
        for (int sample = 0; sample < 512; ++sample)
            peak = juce::jmax (peak, std::abs (buffer.getSample (channel, sample)));

    std::cout << "[PASS] VST3 scanned and loaded: " << description.name.toStdString() << "\n";
    std::cout << "[PASS] VST3 processed 512 frames, peak=" << peak << "\n";
    std::cout << "[PASS] real VST3 plugin hosting verified\n";
    instance->releaseResources();
    return 0;
}
