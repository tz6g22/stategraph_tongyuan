# Shared local LLM backend

Status: running and smoke-verified.

- Endpoint: `http://127.0.0.1:8080/v1`
- Model alias: `qwen3.5-27b-q4`
- GGUF: `/home/cody/models/qwen3.5-27b-q4/Qwen_Qwen3.5-27B-Q4_K_M.gguf`
- SHA256: `81657841d62f1821c748d0fea6c260b7d3508844fe4e9250253ef81c4e4d9edf`
- llama.cpp: `b11379`, commit `1537a0a8b2f8711d840878b0a0677ab2213c882c`
- Quantization: Q4_K_M; 36 GPU layers with remaining layers on CPU; RTX 4070 Ti 12 GB, WSL2
- Context 4096; batch 512; ubatch 128; parallel 1; Flash Attention on; K/V cache Q8_0
- Generation: temperature 0, top-p 1, seed 42; each task keeps its own token budget.
- Embeddings remain each baseline's existing model and run separately from Qwen generation.

`GET /v1/models` returned the pinned alias. A local chat completion returned `READY` when using llama.cpp's `chat_template_kwargs.enable_thinking=false`; without that template option, a short request used its full output budget in `reasoning_content` and returned no final `content`. The adapter therefore sends this model-template option through OpenAI SDK `extra_body`. No external provider is used or permitted by the local client.
