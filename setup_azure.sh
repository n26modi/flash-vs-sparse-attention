#!/bin/bash
# Run on Azure GPU VM after SSH in
pip install -r requirements-gpu.txt
python -c "import flash_attn; print('FA2 ready:', flash_attn.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0))"
