@echo off
set LOG=C:\Users\Admin\QuillanWorker\data\torch-install.log
set PY=C:\Users\Admin\AppData\Local\Comfy-Desktop\ComfyUI-Installs\CrashoverrideX\ComfyUI\.venv\Scripts\python.exe
echo === torch swap start %date% %time% === >> "%LOG%"
uv pip install --python "%PY%" torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126 >> "%LOG%" 2>&1
echo === done exit=%errorlevel% %time% === >> "%LOG%"
