# Local system and model configuration

- OS: Ubuntu 24.04.4 LTS in WSL2, Linux kernel `6.18.40.1-microsoft-standard-WSL2`, x86_64.
- Main Python: 3.12.3; pip 26.2.1.
- CUDA toolkit: 12.6.85 (`nvcc`).
- GPU: NVIDIA GeForce RTX 4070 Ti; WSL driver 610.88; reported memory 12,282 MiB. `nvidia-smi` is available at `/usr/lib/wsl/lib/nvidia-smi` in this image.
- llama.cpp: `ggml-org/llama.cpp`, tag `b11379`, commit `1537a0a8b2f8711d840878b0a0677ab2213c882c`; source checkout is excluded from the parent GitHub repo and should be recreated from the tag.

Qwen backend (server was stopped at freeze):

- Model: `Qwen/Qwen3.5-27B`, GGUF repository `bartowski/Qwen_Qwen3.5-27B-GGUF`.
- File: `Qwen_Qwen3.5-27B-Q4_K_M.gguf`, SHA256 `81657841d62f1821c748d0fea6c260b7d3508844fe4e9250253ef81c4e4d9edf`.
- Local file path: `/home/cody/models/qwen3.5-27b-q4/Qwen_Qwen3.5-27B-Q4_K_M.gguf` (not committed).
- Quantization: Q4_K_M; llama.cpp `-ngl 36`, remainder CPU offload.
- Context 4096, batch 512, ubatch 128, parallel 1, Flash Attention on, K/V cache Q8_0.
- Former endpoint `http://127.0.0.1:8080/v1`; alias `qwen3.5-27b-q4`.
- Defaults: temperature 0, top-p 1, seed 42; completion budget task-specific.
- Embeddings remain baseline-specific and are not replaced by Qwen.
- Model startup timing was not recorded. No server is running.

IRIDIS must download the exact GGUF and verify the hash; do not transfer the
local 18 GB file through GitHub.
