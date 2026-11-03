/*
    JDXX Chains — chain engine: builds the five JDXX chains from their specs,
    runs them in strict order, reports zero latency to the host.

    AGPL-3.0. See LICENSE.
*/
#pragma once

#include "ChainSpecs.h"
#include <memory>
#include <vector>
#include <string>

namespace JdxxChains
{

struct Module;   // forward declaration; full definition in Modules.h

class ChainInstance
{
public:
    ChainInstance() = default;
    ~ChainInstance();

    ChainInstance (const ChainInstance&) = delete;
    ChainInstance& operator= (const ChainInstance&) = delete;

    void build (ChainMode mode, double sampleRate);
    void reset() noexcept;
    void setSampleRate (double sampleRate);

    // In-place stereo processing. Planar float buffers, nFrames per channel.
    void processBlock (float* L, float* R, int nFrames) noexcept;

    ChainMode getMode() const noexcept { return mode; }
    const ChainSpec* getSpec() const noexcept { return spec; }

    // Host-facing enable toggles per module (order matches the spec).
    void setModuleEnabled (int index, bool enabled);
    bool isModuleEnabled (int index) const noexcept;
    int  getNumModules() const noexcept;

    double getLatencySamples() const noexcept { return 0.0; }   // zero latency by design

private:
    ChainMode mode = ChainMode::Vocal;
    const ChainSpec* spec = nullptr;
    std::vector<Module*> modules;
    double sr = 48000.0;

    void clear() noexcept;
};

} // namespace JdxxChains
