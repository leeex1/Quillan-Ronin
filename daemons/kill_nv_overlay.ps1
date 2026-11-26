# Quillan-Ronin user-level NVIDIA Overlay suppressor.
# Terminates the NVIDIA App CEF overlay processes that consume VRAM/CPU
# on this 4GB GTX 1050 host. Reversible: Disable-ScheduledTask -TaskName Quillan-DisableNVIDIAOverlay
Get-Process -Name 'NVIDIA Overlay' -ErrorAction SilentlyContinue |
    Stop-Process -Force -ErrorAction SilentlyContinue
