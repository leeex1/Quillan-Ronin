# =============================================================================
# C14-KAIDO / SPEED PATH - prefix caching + residency config
# Brief #896 (request_id 8d2bb927-7f04-47a2-b685-f6e4a750ace0), 2026-11-20
# NEW FILE. Nothing existing is modified.
# -----------------------------------------------------------------------------
# INVARIANTS (non-negotiable, board #855/#861/#863/#874):
#   * num_ctx stays 2097152 in the Modelfile / API declaration (metadata).
#   * The full 171,434-token system prompt stays intact, byte-for-byte.
#     C:\02_QUILLAN\06 - Deployment & Platforms\system prompts\Quillan-Samurai.md
#     measured 630,767 chars -> 171,434 tokens (Llama-3.2 tokenizer), board #874.
#   * -c 262144 is the ACTIVE attention window (256k), NOT a num_ctx change.
#     q4_0 KV = 9,216 B/token (board #863), so 171,434 tok = 1.47 GiB and
#     262,144 tok = 2.25 GiB. A literal -c 2097152 allocation = 18.0 GiB and
#     cannot fit 7.78 GiB free host RAM / 4 GiB GPU (measured this session).
#   * Declared capacity != realized attention span. Do not claim otherwise.
#
# EVIDENCE FOR EVERY FLAG (llama-server.exe --help, build b9631-6e14286ed):
#   --cache-prompt  "whether to enable prompt caching (default: enabled)"
#   --cache-reuse N "min chunk size to attempt reusing from the cache via KV
#                    shifting, requires prompt caching to be enabled (default: 0)"
#   -sps F          "how much the prompt of a request must match the prompt of a
#                    slot in order to use that slot (default: 0.10)"
#   -np N           "number of server slots (default: -1, -1 = auto)"
#   -c / -ctk / -ctv / -fa / -ngl / -kvo / -t / --metrics / --slots
#   GPU facts (board #863, llama-bench -v): offloaded 17/17 layers, CUDA
#   sm_61 supported by direct llama.cpp, pp128 = 1100.69 t/s, tg8 = 45.86 t/s.
#   Ollama's sm_61 exclusion is packaging-only (board #874, ollama_headless_serve.log).
# =============================================================================

$ErrorActionPreference = 'Stop'

# --- Model: standard-arch Llama-3.2 1B WSP GGUF. Its tokenizer is the one used
#     to measure the 171,434-token prompt, so prompt length is exact, not inferred.
$GGUF = 'C:\02_QUILLAN\exports\quillan-1b-sovereign-wsp-v23-q8_0.gguf'
$LLAMA_SERVER = 'C:\01_PROGRAMS\Open_Source\AI_ML\llama.cpp\build\llama-server.exe'
$PROMPT_FILE = 'C:\02_QUILLAN\06 - Deployment & Platforms\system prompts\Quillan-Samurai.md'
$PORT = 8099   # non-default port: never collide with an existing :8080 serving slot

if (-not (Test-Path $LLAMA_SERVER)) { throw "llama-server not found: $LLAMA_SERVER" }
if (-not (Test-Path $GGUF)) { throw "GGUF not found: $GGUF" }
if (-not (Test-Path $PROMPT_FILE)) { throw "system prompt not found: $PROMPT_FILE" }

# --- Ollama-compatible environment (keeps declared num_ctx = 2097152 in the
#     Modelfile; these only cap the RUNTIME allocation and pin residency).
$env:OLLAMA_HOST             = '127.0.0.1:11434'
$env:OLLAMA_KV_CACHE_TYPE    = 'q4_0'     # 18.0 GiB @ 2M -> 2.25 GiB @ 256k
$env:OLLAMA_FLASH_ATTENTION  = '1'        # board #863: FA auto-enables on sm_61
$env:OLLAMA_CONTEXT_LENGTH   = '262144'   # runtime active window cap
$env:OLLAMA_KEEP_ALIVE       = '24h'      # pin model in VRAM: no reload between requests
$env:OLLAMA_MAX_LOADED_MODELS = '1'       # never evict the pinned model/KV
$env:OLLAMA_NUM_PARALLEL     = '1'        # one slot = one cacheable prefix

# --- llama-server equivalent of keep_alive: process residency is permanent;
#     the slot retains its KV prefix for the life of the server. There is no
#     --keep-alive flag on llama-server; do not invent one.
$serverArgs = @(
  '-m',              $GGUF,
  '--host',          '127.0.0.1',
  '--port',          "$PORT",
  '-c',              '262144',     # active window; declared num_ctx untouched
  '-ctk',            'q4_0',
  '-ctv',            'q4_0',
  '-fa',             'on',         # fallback '-fa off' if sm_61 rejects q4_0 FA
  '-ngl',            '99',         # measured 17/17 layers on GTX 1050
  '-kvo',                          # weights on CUDA0, KV in host RAM
  '--cache-prompt',                # THE lever: keep longest common prefix in KV
  '--cache-reuse',   '256',        # default 0 disables KV-shift chunk reuse
  '-sps',            '0.10',       # default; single slot makes this a no-op
  '-np',             '1',          # one slot; N slots fragment the cached prefix
  '-t',              '4',
  '-tb',             '4',
  '--no-mmap',                     # avoid pageouts under 7.78 GiB free RAM
  '--metrics',                     # /metrics for cache-hit observability
  '--slots'                        # /slots to inspect retained prefix length
)

Write-Host "KAIDO_SERVE_CMD: $LLAMA_SERVER $($serverArgs -join ' ')"
& $LLAMA_SERVER @serverArgs
