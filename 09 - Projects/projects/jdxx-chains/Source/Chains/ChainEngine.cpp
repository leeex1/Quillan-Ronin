/*
    JDXX Chains — chain engine implementation.
    AGPL-3.0. See LICENSE.
*/
#include "ChainEngine.h"
#include "Modules.h"      // createModule / destroyModule / Module
#include <algorithm>

namespace JdxxChains
{

void ChainInstance::build (ChainMode newMode, double sampleRate)
{
    clear();
    mode = newMode;
    spec = &getChainSpec (mode);
    sr = sampleRate;
    modules.reserve ((size_t) spec->numModules);

    for (int i = 0; i < spec->numModules; ++i)
    {
        const ModuleSpec& ms = spec->modules[i];
        Module* m = createModule (ms, sr);
        if (m != nullptr) modules.push_back (m);
    }
}

void ChainInstance::clear() noexcept
{
    for (Module* m : modules) destroyModule (m);
    modules.clear();
}

ChainInstance::~ChainInstance() { clear(); }

void ChainInstance::setSampleRate (double sampleRate)
{
    if (sampleRate == sr) return;
    sr = sampleRate;
    build (mode, sr);
}

void ChainInstance::reset() noexcept
{
    for (Module* m : modules) if (m) m->reset();
}

void ChainInstance::processBlock (float* L, float* R, int nFrames) noexcept
{
    for (Module* m : modules)
        if (m && m->enabled)
            m->processBlock (L, R, nFrames);
}

void ChainInstance::setModuleEnabled (int index, bool enabled)
{
    if (index >= 0 && index < (int) modules.size() && modules[(size_t) index])
        modules[(size_t) index]->enabled = enabled;
}

bool ChainInstance::isModuleEnabled (int index) const noexcept
{
    if (index < 0 || index >= (int) modules.size()) return false;
    const Module* m = modules[(size_t) index];
    return m && m->enabled;
}

int ChainInstance::getNumModules() const noexcept { return (int) modules.size(); }

} // namespace JdxxChains
