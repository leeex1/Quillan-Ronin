# ==============================================================================
# Production Launcher for Quillan-Ronin v5.4.0-ONI Sovereign SFT Training
# ==============================================================================
$ErrorActionPreference = "Stop"

$PYTHON_EXE = "C:\Users\Admin\AppData\Local\Programs\Python\Python314\python.exe"
$SCRIPTS_DIR = "C:\02_QUILLAN\03 - Training & Model\scripts"
$RUNNER_SCRIPT = "$SCRIPTS_DIR\launch_sovereign_sft.py"

Write-Host "=======================================================================" -ForegroundColor Cyan
Write-Host "   QUILLAN-RONIN v5.4.0-ONI — SOVEREIGN SFT PRODUCTION RUNNER (REV 2)" -ForegroundColor Cyan
Write-Host "=======================================================================" -ForegroundColor Cyan

# Verify runtime dependencies
if (-not (Test-Path $PYTHON_EXE)) {
    Write-Error "Python executable not found at: $PYTHON_EXE"
}
if (-not (Test-Path $RUNNER_SCRIPT)) {
    Write-Error "Launcher script not found at: $RUNNER_SCRIPT"
}

# Training Hyperparameters
$STEPS = 1000
$BATCH_SIZE = 1
$GRAD_ACCUM = 4
$SEQ_LEN = 512
$PEAK_LR = "3e-4"
$MIN_LR = "3e-6"
$WARMUP = 50
$EVAL_EVERY = 50
$SAVE_EVERY = 50
$LOG_EVERY = 5
$THREADS = 4

Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Steps:              $STEPS"
Write-Host "  Effective Batch:    $($BATCH_SIZE * $GRAD_ACCUM) (Tokens/Step: $($BATCH_SIZE * $GRAD_ACCUM * $SEQ_LEN))"
Write-Host "  Learning Rate:      $PEAK_LR -> $MIN_LR (Warmup: $WARMUP)"
Write-Host "  CPU Threads:        $THREADS (with denormal flushing and BELOW_NORMAL priority)"
Write-Host '  Telemetry:          C:\02_QUILLAN\checkpoints\checkpoints_oni\sft_runs\sft_telemetry.jsonl'
Write-Host '=======================================================================' -ForegroundColor Cyan

# Launch command
& $PYTHON_EXE -u $RUNNER_SCRIPT `
    --device cpu `
    --threads $THREADS `
    --steps $STEPS `
    --batch-size $BATCH_SIZE `
    --grad-accum $GRAD_ACCUM `
    --seq-len $SEQ_LEN `
    --lr $PEAK_LR `
    --min-lr $MIN_LR `
    --warmup-steps $WARMUP `
    --eval-every $EVAL_EVERY `
    --save-every $SAVE_EVERY `
    --log-every $LOG_EVERY
