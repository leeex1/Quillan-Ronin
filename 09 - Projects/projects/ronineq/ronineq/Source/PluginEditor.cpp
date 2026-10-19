/*
    RoninEQ editor implementation. License: AGPL-3.0.
*/
#include "PluginEditor.h"
#include "PluginProcessor.h"

namespace
{
constexpr float kMinFreq = 20.0f;
constexpr float kMaxFreq = 20000.0f;
} // namespace

//==============================================================================
// SpectrumComponent
//==============================================================================
RoninEQAudioProcessorEditor::SpectrumComponent::SpectrumComponent (RoninEQAudioProcessor& p)
    : processor (p)
{
    for (auto& v : displayPre)  v = -96.0f;
    for (auto& v : displayPost) v = -96.0f;
    setWantsKeyboardFocus (false);
}

void RoninEQAudioProcessorEditor::SpectrumComponent::setSelectedBand (int b)
{
    selectedBand = b;
    repaint();
}

float RoninEQAudioProcessorEditor::SpectrumComponent::freqToX (float freq) const
{
    const float f = juce::jlimit (kMinFreq, kMaxFreq, freq);
    return (float) getWidth() * std::log10 (f / kMinFreq) / std::log10 (kMaxFreq / kMinFreq);
}

float RoninEQAudioProcessorEditor::SpectrumComponent::xToFreq (float x) const
{
    const float t = juce::jlimit (0.0f, 1.0f, x / (float) juce::jmax (1, getWidth()));
    return kMinFreq * std::pow (kMaxFreq / kMinFreq, t);
}

float RoninEQAudioProcessorEditor::SpectrumComponent::gainToY (float gainDb) const
{
    return (kTopDb - gainDb) / (kTopDb - kBottomDb) * (float) juce::jmax (1, getHeight());
}

float RoninEQAudioProcessorEditor::SpectrumComponent::yToGain (float y) const
{
    const float t = juce::jlimit (0.0f, 1.0f, y / (float) juce::jmax (1, getHeight()));
    return kTopDb - t * (kTopDb - kBottomDb);
}

juce::Point<float> RoninEQAudioProcessorEditor::SpectrumComponent::nodePosition (int band) const
{
    auto& apvts = processor.apvts;
    const float freq = apvts.getRawParameterValue (
        RoninEQAudioProcessor::bandParamId (band, "freq"))->load();
    const float gain = apvts.getRawParameterValue (
        RoninEQAudioProcessor::bandParamId (band, "gain"))->load();
    const int type = (int) apvts.getRawParameterValue (
        RoninEQAudioProcessor::bandParamId (band, "type"))->load();

    // Filter shapes sit on the 0 dB line; bells/shelves sit at their gain.
    const float yGain = (type <= 2) ? gain : 0.0f;
    return { freqToX (freq), gainToY (yGain) };
}

int RoninEQAudioProcessorEditor::SpectrumComponent::nodeAt (juce::Point<float> pos) const
{
    auto& apvts = processor.apvts;
    for (int b = 0; b < RoninEQAudioProcessor::kNumBands; ++b)
    {
        const bool enabled = apvts.getRawParameterValue (
            RoninEQAudioProcessor::bandParamId (b, "enabled"))->load() > 0.5f;
        if (! enabled)
            continue;
        if (nodePosition (b).getDistanceFrom (pos) < 14.0f)
            return b;
    }
    return -1;
}

juce::Colour RoninEQAudioProcessorEditor::SpectrumComponent::bandColour (int band) const
{
    static const juce::Colour colours[] = {
        juce::Colour (0xff5ac8fa), juce::Colour (0xff7ee787), juce::Colour (0xffffd866),
        juce::Colour (0xffff9e64), juce::Colour (0xffd2a8ff), juce::Colour (0xff8ecae6),
    };
    return colours[band % 6];
}

void RoninEQAudioProcessorEditor::SpectrumComponent::updateFromState()
{
    // --- spectra with peak decay
    RoninEQAudioProcessor::SpectrumFrame frame;
    bool gotFrame = false;
    while (processor.popSpectrumFrame (frame))
        gotFrame = true;

    constexpr float kDecayDb = 3.0f; // per timer tick
    if (gotFrame)
    {
        for (int b = 0; b < RoninEQAudioProcessor::kNumSpectrumBins; ++b)
        {
            displayPre [b] = juce::jmax (frame.pre [b], displayPre [b] - kDecayDb);
            displayPost[b] = juce::jmax (frame.post[b], displayPost[b] - kDecayDb);
        }
    }
    else
    {
        for (int b = 0; b < RoninEQAudioProcessor::kNumSpectrumBins; ++b)
        {
            displayPre [b] = juce::jmax (-96.0f, displayPre [b] - kDecayDb);
            displayPost[b] = juce::jmax (-96.0f, displayPost[b] - kDecayDb);
        }
    }

    // --- static-curve mirrors from APVTS (message thread only)
    auto& apvts = processor.apvts;
    const float sr = (float) processor.getEffectiveSampleRate();
    for (int b = 0; b < RoninEQAudioProcessor::kNumBands; ++b)
    {
        const bool enabled = apvts.getRawParameterValue (
            RoninEQAudioProcessor::bandParamId (b, "enabled"))->load() > 0.5f;
        if (! enabled)
        {
            curveFilters[b].setUnity();
            continue;
        }
        curveFilters[b].setParams (
            static_cast<RoninEQ::Biquad::Type> ((int) apvts.getRawParameterValue (
                RoninEQAudioProcessor::bandParamId (b, "type"))->load()),
            apvts.getRawParameterValue (
                RoninEQAudioProcessor::bandParamId (b, "freq"))->load(),
            apvts.getRawParameterValue (
                RoninEQAudioProcessor::bandParamId (b, "q"))->load(),
            apvts.getRawParameterValue (
                RoninEQAudioProcessor::bandParamId (b, "gain"))->load(),
            sr);
    }

    repaint();
}

void RoninEQAudioProcessorEditor::SpectrumComponent::paint (juce::Graphics& g)
{
    const int W = getWidth(), H = getHeight();
    g.fillAll (juce::Colour (0xff101418));

    // --- log frequency grid
    g.setColour (juce::Colour (0xff232b33));
    const float freqMarks[] = { 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000 };
    for (float f : freqMarks)
    {
        const float x = freqToX (f);
        g.drawVerticalLine ((int) x, 0.0f, (float) H);
        g.setColour (juce::Colour (0xff5a6672));
        g.setFont (10.0f);
        juce::String label = f >= 1000.0f ? juce::String (f / 1000.0f, 0) + "k"
                                          : juce::String ((int) f);
        g.drawText (label, (int) x + 3, H - 16, 40, 14, juce::Justification::left);
        g.setColour (juce::Colour (0xff232b33));
    }
    for (float db = -54.0f; db <= 18.0f; db += 6.0f)
    {
        const float y = gainToY (db);
        g.drawHorizontalLine ((int) y, 0.0f, (float) W);
        if (std::abs (db) < 0.01f)
        {
            g.setColour (juce::Colour (0xff3a454f));
            g.drawHorizontalLine ((int) y, 0.0f, (float) W);
            g.setColour (juce::Colour (0xff232b33));
        }
    }

    auto drawSpectrum = [&] (const float* data, juce::Colour colour)
    {
        juce::Path path;
        bool started = false;
        for (int b = 0; b < RoninEQAudioProcessor::kNumSpectrumBins; ++b)
        {
            const float fC = kMinFreq * std::pow (kMaxFreq / kMinFreq,
                (float) b / (float) (RoninEQAudioProcessor::kNumSpectrumBins - 1));
            const float x = freqToX (fC);
            const float y = gainToY (juce::jlimit (kBottomDb, kTopDb, data[b]));
            if (! started) { path.startNewSubPath (x, y); started = true; }
            else           { path.lineTo (x, y); }
        }
        path.lineTo ((float) W, (float) H);
        path.lineTo (0.0f, (float) H);
        path.closeSubPath();
        g.setColour (colour.withAlpha (0.28f));
        g.fillPath (path);
        g.setColour (colour);
        g.strokePath (path, juce::PathStrokeType (1.2f));
    };

    drawSpectrum (displayPre,  juce::Colour (0xff3d7a8c));
    drawSpectrum (displayPost, juce::Colour (0xff59d98c));

    // --- summed EQ curves: static ghost + live dynamic curve.
    // The live curve adds each band's current dynamic gain (<= 0 dB while
    // compressing), FabFilter-style, so bands visibly pump in real time.
    {
        auto& state = processor.apvts;
        const float sr = (float) processor.getEffectiveSampleRate();

        struct BandSnap
        {
            bool enabled = false;
            RoninEQ::Biquad::Type type = RoninEQ::Biquad::Type::Bell;
            float freq = 1000.0f, gain = 0.0f, q = 1.0f, dynGain = 0.0f;
        };
        BandSnap snap[RoninEQAudioProcessor::kNumBands];
        for (int b = 0; b < RoninEQAudioProcessor::kNumBands; ++b)
        {
            snap[b].enabled = state.getRawParameterValue (
                RoninEQAudioProcessor::bandParamId (b, "enabled"))->load() > 0.5f;
            snap[b].type = static_cast<RoninEQ::Biquad::Type> ((int) state.getRawParameterValue (
                RoninEQAudioProcessor::bandParamId (b, "type"))->load());
            snap[b].freq = state.getRawParameterValue (
                RoninEQAudioProcessor::bandParamId (b, "freq"))->load();
            snap[b].gain = state.getRawParameterValue (
                RoninEQAudioProcessor::bandParamId (b, "gain"))->load();
            snap[b].q = state.getRawParameterValue (
                RoninEQAudioProcessor::bandParamId (b, "q"))->load();
            snap[b].dynGain = processor.getBandDynamicGainDb (b);
        }

        auto sumDbAt = [&] (float freq, bool live)
        {
            float sumDb = 0.0f;
            for (int b = 0; b < RoninEQAudioProcessor::kNumBands; ++b)
            {
                if (! snap[b].enabled)
                    continue;
                RoninEQ::Biquad tmp;
                tmp.setParams (snap[b].type, snap[b].freq, snap[b].q,
                               snap[b].gain + (live ? snap[b].dynGain : 0.0f), sr);
                sumDb += tmp.magnitudeResponseDb (freq, sr);
            }
            return sumDb;
        };

        auto buildCurve = [&] (bool live)
        {
            juce::Path curve;
            bool started = false;
            for (int x = 0; x < W; x += 2)
            {
                const float y = gainToY (juce::jlimit (
                    kBottomDb, kTopDb, sumDbAt (xToFreq ((float) x), live)));
                if (! started) { curve.startNewSubPath ((float) x, y); started = true; }
                else           { curve.lineTo ((float) x, y); }
            }
            return curve;
        };

        // static ghost (dim)
        g.setColour (juce::Colour (0xff8a7134));
        g.strokePath (buildCurve (false), juce::PathStrokeType (1.2f));

        // live dynamic curve (bright) with translucent fill to the floor
        juce::Path live = buildCurve (true);
        juce::Path fill = live;
        fill.lineTo ((float) W, (float) H);
        fill.lineTo (0.0f, (float) H);
        fill.closeSubPath();
        g.setColour (juce::Colour (0xffffc44d).withAlpha (0.10f));
        g.fillPath (fill);
        g.setColour (juce::Colour (0xffffc44d));
        g.strokePath (live, juce::PathStrokeType (2.2f));
    }

    // --- band nodes
    auto& apvts = processor.apvts;
    for (int b = 0; b < RoninEQAudioProcessor::kNumBands; ++b)
    {
        const bool enabled = apvts.getRawParameterValue (
            RoninEQAudioProcessor::bandParamId (b, "enabled"))->load() > 0.5f;
        if (! enabled)
            continue;

        const auto pos = nodePosition (b);
        const auto colour = bandColour (b);
        const bool selected = (b == selectedBand);

        g.setColour (colour);
        g.fillEllipse (pos.x - 7.0f, pos.y - 7.0f, 14.0f, 14.0f);
        g.setColour (juce::Colour (0xff101418));
        g.fillEllipse (pos.x - 4.0f, pos.y - 4.0f, 8.0f, 8.0f);
        g.setColour (colour);
        g.fillEllipse (pos.x - 2.5f, pos.y - 2.5f, 5.0f, 5.0f);

        if (selected)
        {
            g.setColour (juce::Colours::white);
            g.drawEllipse (pos.x - 10.0f, pos.y - 10.0f, 20.0f, 20.0f, 1.5f);
        }

        // dynamic-activity ring: outer glow sized by current cut
        const float gr = -processor.getBandDynamicGainDb (b); // positive when cutting
        if (gr > 0.25f)
        {
            const float r = 7.0f + juce::jmin (10.0f, gr * 0.8f);
            g.setColour (colour.withAlpha (0.45f));
            g.drawEllipse (pos.x - r, pos.y - r, r * 2.0f, r * 2.0f, 2.0f);
        }
    }
}

void RoninEQAudioProcessorEditor::SpectrumComponent::mouseDown (const juce::MouseEvent& e)
{
    const int hit = nodeAt (e.position);
    if (hit >= 0)
    {
        setSelectedBand (hit);
        if (onBandSelected)
            onBandSelected (hit);

        dragFreqParam = processor.apvts.getParameter (
            RoninEQAudioProcessor::bandParamId (hit, "freq"));
        dragGainParam = processor.apvts.getParameter (
            RoninEQAudioProcessor::bandParamId (hit, "gain"));
        dragQParam = processor.apvts.getParameter (
            RoninEQAudioProcessor::bandParamId (hit, "q"));

        if (dragFreqParam != nullptr) dragFreqParam->beginChangeGesture();
        if (dragGainParam != nullptr) dragGainParam->beginChangeGesture();
        if (dragQParam    != nullptr) dragQParam->beginChangeGesture();

        dragging = true;
        dragIsQ  = e.mods.isAltDown() || e.mods.isRightButtonDown();
    }
}

void RoninEQAudioProcessorEditor::SpectrumComponent::applyDragToParams (juce::Point<float> pos)
{
    if (dragFreqParam != nullptr)
    {
        const float newFreq = juce::jlimit (kMinFreq, kMaxFreq, xToFreq (pos.x));
        dragFreqParam->setValueNotifyingHost (dragFreqParam->convertTo0to1 (newFreq));
    }

    if (dragIsQ)
    {
        if (dragQParam != nullptr)
        {
            // Top of the display -> Q 10, bottom -> Q 0.1 (log).
            const float t = juce::jlimit (0.0f, 1.0f,
                pos.y / (float) juce::jmax (1, getHeight()));
            const float newQ = 0.1f * std::pow (100.0f, 1.0f - t);
            dragQParam->setValueNotifyingHost (dragQParam->convertTo0to1 (
                juce::jlimit (0.1f, 10.0f, newQ)));
        }
    }
    else if (dragGainParam != nullptr)
    {
        const int type = (int) processor.apvts.getRawParameterValue (
            RoninEQAudioProcessor::bandParamId (selectedBand, "type"))->load();
        if (type <= 2) // bells & shelves have a gain axis; filters don't
        {
            const float newGain = juce::jlimit (-15.0f, 15.0f, yToGain (pos.y));
            dragGainParam->setValueNotifyingHost (
                dragGainParam->convertTo0to1 (newGain));
        }
    }
}

void RoninEQAudioProcessorEditor::SpectrumComponent::mouseDrag (const juce::MouseEvent& e)
{
    if (dragging)
        applyDragToParams (e.position);
}

void RoninEQAudioProcessorEditor::SpectrumComponent::mouseUp (const juce::MouseEvent&)
{
    if (dragging)
    {
        if (dragFreqParam != nullptr) dragFreqParam->endChangeGesture();
        if (dragGainParam != nullptr) dragGainParam->endChangeGesture();
        if (dragQParam    != nullptr) dragQParam->endChangeGesture();
        dragging = false;
        dragFreqParam = dragGainParam = dragQParam = nullptr;
    }
}

void RoninEQAudioProcessorEditor::SpectrumComponent::mouseDoubleClick (const juce::MouseEvent& e)
{
    if (nodeAt (e.position) >= 0)
        return; // double-clicking a node does nothing special

    // Double-click empty space: wake the first disabled band at this frequency.
    auto& apvts = processor.apvts;
    for (int b = 0; b < RoninEQAudioProcessor::kNumBands; ++b)
    {
        auto* enabledParam = apvts.getParameter (
            RoninEQAudioProcessor::bandParamId (b, "enabled"));
        if (enabledParam->getValue() < 0.5f)
        {
            enabledParam->setValueNotifyingHost (1.0f);
            if (auto* freqParam = apvts.getParameter (
                    RoninEQAudioProcessor::bandParamId (b, "freq")))
                freqParam->setValueNotifyingHost (
                    freqParam->convertTo0to1 (xToFreq (e.position.x)));
            setSelectedBand (b);
            if (onBandSelected)
                onBandSelected (b);
            break;
        }
    }
}

//==============================================================================
// Editor
//==============================================================================
void RoninEQAudioProcessorEditor::setupRotary (juce::Slider& s, const juce::String& suffix)
{
    s.setSliderStyle (juce::Slider::RotaryVerticalDrag);
    s.setTextBoxStyle (juce::Slider::TextBoxBelow, false, 72, 16);
    s.setTextValueSuffix (suffix);
    s.setColour (juce::Slider::rotarySliderFillColourId, juce::Colour (0xffffc44d));
}

RoninEQAudioProcessorEditor::RoninEQAudioProcessorEditor (RoninEQAudioProcessor& p)
    : AudioProcessorEditor (p), processorRef (p), spectrum (p)
{
    setSize (940, 560);

    addAndMakeVisible (spectrum);
    spectrum.onBandSelected = [this] (int b) { selectBand (b); };

    for (int b = 0; b < RoninEQAudioProcessor::kNumBands; ++b)
    {
        auto& btn = bandButtons[b];
        btn.setButtonText (juce::String (b + 1));
        btn.setRadioGroupId (1);
        btn.setClickingTogglesState (true);
        btn.setColour (juce::TextButton::buttonOnColourId, juce::Colour (0xffffc44d));
        addAndMakeVisible (btn);
        btn.onClick = [this, b] { selectBand (b); };
    }
    bandButtons[0].setToggleState (true, juce::dontSendNotification);

    addAndMakeVisible (enabledButton);

    typeBox.addItemList ({ "Bell", "Low Shelf", "High Shelf",
                           "High Pass", "Low Pass", "Notch" }, 1);
    addAndMakeVisible (typeBox);

    setupRotary (freqSlider,      " Hz");
    setupRotary (gainSlider,      " dB");
    setupRotary (qSlider,         "");
    setupRotary (thresholdSlider, " dB");
    setupRotary (ratioSlider,     ":1");
    setupRotary (attackSlider,    " ms");
    setupRotary (releaseSlider,   " ms");
    setupRotary (inputGainSlider, " dB");
    setupRotary (outputGainSlider," dB");
    for (auto* s : { &freqSlider, &gainSlider, &qSlider, &thresholdSlider, &ratioSlider,
                     &attackSlider, &releaseSlider, &inputGainSlider, &outputGainSlider })
        addAndMakeVisible (s);

    addAndMakeVisible (dynButton);
    addAndMakeVisible (bypassButton);
    bypassButton.setColour (juce::TextButton::buttonOnColourId, juce::Colour (0xffc44d4d));

    grLabel.setJustificationType (juce::Justification::centred);
    grLabel.setFont (juce::Font (juce::FontOptions (12.0f)));
    addAndMakeVisible (grLabel);

    // Global attachments (never rebuilt).
    inputGainAttach  = std::make_unique<SliderAttach> (
        processorRef.apvts, "input_gain", inputGainSlider);
    outputGainAttach = std::make_unique<SliderAttach> (
        processorRef.apvts, "output_gain", outputGainSlider);
    bypassAttach = std::make_unique<ButtonAttach> (
        processorRef.apvts, "bypass", bypassButton);

    selectBand (0);
    startTimerHz (30);
}

RoninEQAudioProcessorEditor::~RoninEQAudioProcessorEditor()
{
    stopTimer();
}

void RoninEQAudioProcessorEditor::selectBand (int bandIndex)
{
    // Called from band buttons and from the spectrum node clicks.
    for (int b = 0; b < RoninEQAudioProcessor::kNumBands; ++b)
        bandButtons[b].setToggleState (b == bandIndex, juce::dontSendNotification);

    spectrum.setSelectedBand (bandIndex);
    rebuildBandAttachments();
}

void RoninEQAudioProcessorEditor::rebuildBandAttachments()
{
    // Tear down the old attachments BEFORE creating the new ones. Each new
    // attachment syncs its control from the parameter with a *notifying*
    // update; if the old attachment were still listening it would echo the
    // newly-selected band's values back into the previously-selected band's
    // parameters (this used to disable/reset band 1 when switching to band 2).
    freqAttach.reset(); gainAttach.reset(); qAttach.reset();
    thresholdAttach.reset(); ratioAttach.reset();
    attackAttach.reset(); releaseAttach.reset();
    enabledAttach.reset(); dynAttach.reset(); typeAttach.reset();

    const int b = spectrum.getSelectedBand();
    auto& apvts = processorRef.apvts;
    const auto pid = [&] (const juce::String& leaf)
        { return RoninEQAudioProcessor::bandParamId (b, leaf); };

    freqAttach      = std::make_unique<SliderAttach> (apvts, pid ("freq"),          freqSlider);
    gainAttach      = std::make_unique<SliderAttach> (apvts, pid ("gain"),          gainSlider);
    qAttach         = std::make_unique<SliderAttach> (apvts, pid ("q"),             qSlider);
    thresholdAttach = std::make_unique<SliderAttach> (apvts, pid ("dyn_threshold"), thresholdSlider);
    ratioAttach     = std::make_unique<SliderAttach> (apvts, pid ("dyn_ratio"),     ratioSlider);
    attackAttach    = std::make_unique<SliderAttach> (apvts, pid ("dyn_attack"),    attackSlider);
    releaseAttach   = std::make_unique<SliderAttach> (apvts, pid ("dyn_release"),   releaseSlider);

    enabledAttach = std::make_unique<ButtonAttach> (apvts, pid ("enabled"),     enabledButton);
    dynAttach     = std::make_unique<ButtonAttach> (apvts, pid ("dyn_enabled"), dynButton);
    typeAttach    = std::make_unique<ComboAttach>  (apvts, pid ("type"),        typeBox);
}

void RoninEQAudioProcessorEditor::timerCallback()
{
    spectrum.updateFromState();

    const int b = spectrum.getSelectedBand();

    // Gain reduction readout for the selected band.
    const float gr = processorRef.getBandDynamicGainDb (b);
    grLabel.setText ("GR " + juce::String (gr, 1) + " dB", juce::dontSendNotification);

    // Dynamics knobs only mean something when dynamics are on.
    const bool dynOn = dynButton.getToggleState();
    thresholdSlider.setEnabled (dynOn);
    ratioSlider.setEnabled (dynOn);
    attackSlider.setEnabled (dynOn);
    releaseSlider.setEnabled (dynOn);

    // The RBJ notch ignores gainDb (fixed deep null), so don't offer the knob.
    const int type = (int) processorRef.apvts.getRawParameterValue (
        RoninEQAudioProcessor::bandParamId (b, "type"))->load();
    gainSlider.setEnabled (type != (int) RoninEQ::Biquad::Type::Notch);
}

void RoninEQAudioProcessorEditor::paint (juce::Graphics& g)
{
    g.fillAll (juce::Colour (0xff0c0f12));

    g.setColour (juce::Colour (0xff8a94a0));
    g.setFont (juce::Font (juce::FontOptions (12.0f).withStyle ("Bold")));
    const int px = getWidth() - 242;
    g.drawText ("BAND",       px, 8,   234, 16, juce::Justification::left);
    g.drawText ("DYNAMICS",   px, 208, 234, 16, juce::Justification::left);
    g.drawText ("IN / OUT",   px, 372, 234, 16, juce::Justification::left);
}

void RoninEQAudioProcessorEditor::resized()
{
    const int panelW = 250;
    auto area = getLocalBounds();
    auto panel = area.removeFromRight (panelW);
    spectrum.setBounds (area.reduced (8));

    int y = 30;
    const int margin = 8;
    const int contentW = panelW - margin * 2;

    // Band selector: 3 x 2.
    {
        const int bw = contentW / 3;
        for (int b = 0; b < RoninEQAudioProcessor::kNumBands; ++b)
            bandButtons[b].setBounds (panel.getX() + margin + (b % 3) * bw,
                                      y + (b / 3) * 30, bw - 4, 26);
        y += 64;
    }

    enabledButton.setBounds (panel.getX() + margin, y, 110, 24);
    typeBox.setBounds (panel.getX() + margin + 118, y, contentW - 118, 24);
    y += 32;

    // Freq / Gain / Q rotaries.
    {
        const int kw = contentW / 3;
        freqSlider.setBounds (panel.getX() + margin + 0 * kw, y, kw, 78);
        gainSlider.setBounds (panel.getX() + margin + 1 * kw, y, kw, 78);
        qSlider.setBounds    (panel.getX() + margin + 2 * kw, y, kw, 78);
        y += 84;
    }

    y = 230; // DYNAMICS section
    dynButton.setBounds (panel.getX() + margin, y, 110, 24);
    grLabel.setBounds (panel.getX() + margin + 118, y, contentW - 118, 24);
    y += 30;

    // Threshold / Ratio / Attack / Release: 2 x 2.
    {
        const int kw = contentW / 2;
        thresholdSlider.setBounds (panel.getX() + margin + 0 * kw, y + 0 * 78, kw, 78);
        ratioSlider.setBounds     (panel.getX() + margin + 1 * kw, y + 0 * 78, kw, 78);
        attackSlider.setBounds    (panel.getX() + margin + 0 * kw, y + 1 * 78, kw, 78);
        releaseSlider.setBounds    (panel.getX() + margin + 1 * kw, y + 1 * 78, kw, 78);
        y += 162;
    }

    y = 394; // IN / OUT section
    {
        const int kw = contentW / 2;
        inputGainSlider.setBounds  (panel.getX() + margin + 0 * kw, y, kw, 78);
        outputGainSlider.setBounds (panel.getX() + margin + 1 * kw, y, kw, 78);
        y += 84;
    }
    bypassButton.setBounds (panel.getX() + margin, y, contentW, 28);
}
