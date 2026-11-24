# Requires a C++17 compiler. Uses clang++ from LLVM if present, else PATH.
# Native tools write progress to stderr; keep PowerShell from treating that as a terminating error.
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$tmp = Join-Path $env:TEMP 'nextverse-verify'
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

$clang = 'C:\Program Files\LLVM\bin\clang++.exe'
if (-not (Test-Path $clang)) { $clang = 'clang++' }

function Invoke-Step([string]$name, [scriptblock]$body) {
    Write-Output "`n=== $name ==="
    & $body
    if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) {
        throw "$name failed with exit code $LASTEXITCODE"
    }
}

Invoke-Step 'Quantum Core (C++17 native)' {
    $core = Join-Path $root 'nextverse_alpha_core_design'
    $exe = Join-Path $tmp 'quantum_core_native.exe'
    & $clang -std=c++17 -I"$core" `
        (Join-Path $core 'qcc_alpha.cpp') `
        (Join-Path $core 'qsvm_alpha.cpp') `
        (Join-Path $core 'tests\native_verify.cpp') -o $exe
    & $exe 2>&1 | Select-String '\[PASS\]|\[FAIL\]'
}

Invoke-Step 'DAW conceptual (C++20 native)' {
    $daw = Join-Path $root 'nextverse-platform\daw\conceptual_logic'
    $exe = Join-Path $tmp 'daw_native.exe'
    $sources = @('AudioEffect','AudioTrack','MidiTrack','EditSession','Transport',
                 'ReverbEffect','DelayEffect','main') | ForEach-Object { Join-Path $daw "src\$_.cpp" }
    & $clang -std=c++20 -I"$daw\include" $sources -o $exe
    & $exe > (Join-Path $tmp 'daw_run.log') 2>&1
    Write-Output "DAW exit=$LASTEXITCODE log_lines=$((Get-Content (Join-Path $tmp 'daw_run.log')).Count)"

    $verify = Join-Path $tmp 'daw_verify.exe'
    $libSources = @('AudioEffect','AudioTrack','MidiTrack','EditSession','Transport',
                    'ReverbEffect','DelayEffect') | ForEach-Object { Join-Path $daw "src\$_.cpp" }
    & $clang -std=c++20 -I"$daw\include" `
        (Join-Path $daw 'tests\native_verify.cpp') $libSources -o $verify
    & $verify 2>&1 | Select-String '\[PASS\]|\[FAIL\]'
}

Invoke-Step 'Game engine portable units (C++17 native)' {
    $ge = Join-Path $root 'game_engine'
    $exe = Join-Path $tmp 'game_engine_native.exe'
    & $clang -std=c++17 -I"$ge\include" `
        (Join-Path $ge 'tests\native_verify.cpp') `
        (Join-Path $ge 'src\PhysicsEngine.cpp') `
        (Join-Path $ge 'src\DVVE.cpp') `
        (Join-Path $ge 'src\AICommandServer.cpp') -lws2_32 -o $exe
    & $exe 2>&1 | Select-String '\[PASS\]|\[FAIL\]'
}

Invoke-Step 'Quantum Core Python suite' {
    Push-Location (Join-Path $root 'nextverse-platform\quantum-core')
    python -m pytest test_quantum_core.py -q --disable-warnings
    Pop-Location
}

Invoke-Step 'Quantum VM unit + live smoke' {
    $vm = Join-Path $root 'nextverse-platform\quantum-vm'
    Push-Location $vm
    python -m unittest discover -s tests -p 'test_*.py'
    python (Join-Path $vm 'tests\smoke_live_vm.py')
    Pop-Location
}

$godotExe = 'C:\Users\Admin\AppData\Local\Temp\opencode\sdk\godot\Godot_v4.4.1-stable_win64_console.exe'
$godotProject = Join-Path $root 'nextverse-platform\godot'
if (Test-Path $godotExe) {
    Invoke-Step 'Godot 4.x runtime skeleton (headless)' {
        & $godotExe --headless --path $godotProject --script res://tests/verify_runtime.gd 2>&1 |
            Select-String '\[PASS\]|\[FAIL\]|ERROR'
    }
} else {
    Write-Output "`nGodot binary not present at $godotExe - Godot check skipped"
}

$juceSource = 'C:\Users\Admin\AppData\Local\Temp\opencode\sdk\juce\JUCE'
$dawJuce = Join-Path $root 'nextverse-platform\daw\source\NextVerseDAW'
if (Test-Path $juceSource) {
    Invoke-Step 'JUCE audio graph (offline render)' {
        $juceBuild = Join-Path $tmp 'daw_juce_build'
        cmake -S $dawJuce -B $juceBuild "-DJUCE_SOURCE_DIR=$juceSource" -DJUCE_BUILD_EXAMPLES=OFF -DJUCE_BUILD_EXTRAS=OFF | Out-Null
        cmake --build $juceBuild --target NextVerseDAW_AudioGraphTest --config Release | Out-Null
        ctest --test-dir $juceBuild -C Release --output-on-failure
    }
} else {
    Write-Output "`nJUCE source not present at $juceSource - JUCE audio-graph check skipped"
}

Invoke-Step 'REAL acceleration layer (measured)' {
    python (Join-Path $root 'nextverse-platform\quantum-vm\tests\verify_real_vm.py')
}

Write-Output "`nALL NATIVE VERIFICATIONS COMPLETED"
