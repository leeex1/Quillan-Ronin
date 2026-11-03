/*
    JDXX Chains — chain topology data (the five §4 presets, JDXX spec 2026-11-02)

    Every value here comes from the live BandLab preset tables extracted
    2026-11-02 (see docs/JDXX_CHAIN_SPEC.md, "Source register"). Where BandLab
    shows an ambiguous scale, the assumption is flagged inline with
    // ASSUMPTION: ...
    and mirrored in docs/JDXX_CHAIN_SPEC.md §"Assumptions ledger".

    Chain order is load-bearing. Do not reorder.
*/
#pragma once

#include "Modules.h"
#include <array>

namespace JdxxChains
{

enum class ChainMode : int
{
    Vocal = 0,   // recording JDXX (main + reused for main 2)
    Adlib,       // adlib
    Intro,       // intro (hypeVox basis)
    Pan,         // Pan (slowPan basis)
    Mastering,   // Mastering JDXX
    NumModes
};

struct ChainSpec
{
    const char* displayName;   // host-facing name
    const char* jdxxRole;      // e.g. "Vocals main 1", from the doc role map
    const ModuleSpec* modules;
    int numModules;
};

const ChainSpec& getChainSpec (ChainMode mode) noexcept;

} // namespace JdxxChains
