# vLLM Metal (Apple Silicon) smoke test for DeepSeek-OCR

This folder is a **best-effort** attempt to run **`deepseek-ai/DeepSeek-OCR`** on **Apple Silicon (macOS arm64)** using **`vllm-metal`** (MLX backend).

Important:
- `vllm-metal` is **alpha** and only supports a subset of models (via `mlx-lm` / `mlx-vlm`).
- DeepSeek-OCR is a **multimodal** model with custom code; it may fail to load/run on `vllm-metal` today.
- If it fails, the scripts print diagnostics so we can decide whether it’s a model support issue vs. a config issue.

## 0) Requirements
- macOS on Apple Silicon (arm64)
- Python **3.12+** (recommended to match `vllm-metal`’s constraints)

## 1) Install vllm-metal (recommended: official installer)
From anywhere:

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
source ~/.venv-vllm-metal/bin/activate
```

Then install this repo’s runtime deps (PDF/image utilities):

```bash
pip install -r ../../requirements.txt
```

## 2) Quick smoke test (text-only)
This verifies that `vllm-metal` is activated and vLLM runs on your Mac.

```bash
python smoke_test_metal.py --model Qwen/Qwen2.5-0.5B-Instruct --text "Hello!"
```

You should see `Platform: MetalPlatform` in the output.

## 3) DeepSeek-OCR image smoke test (multimodal)

```bash
python smoke_test_metal.py \
  --model deepseek-ai/DeepSeek-OCR \
  --image /path/to/your/image.png \
  --prompt "<image>\n<|grounding|>Convert the document to markdown."
```

## 4) DeepSeek-OCR PDF → Markdown

```bash
python run_dpsk_ocr_pdf_metal.py \
  --model deepseek-ai/DeepSeek-OCR \
  --pdf /path/to/your.pdf \
  --out /path/to/out.mmd
```

If it works, `out.mmd` will contain per-page markdown separated by:

```
<--- Page Split --->
```

