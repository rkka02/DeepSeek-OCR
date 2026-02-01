import argparse
import io
import os
import platform
import re
import sys
from pathlib import Path
from typing import Iterable


REF_BLOCK_PATTERN = re.compile(
    r"(<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>)", re.DOTALL
)


def clean_markdown(text: str, *, strip_refs: bool = True) -> str:
    # DeepSeek uses this token sometimes when skip_special_tokens=False.
    text = text.replace("<｜end▁of▁sentence｜>", "")

    if strip_refs:
        text = REF_BLOCK_PATTERN.sub("", text)

    # Small latex operator cleanup (keeps original behavior from vLLM script).
    text = text.replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:")

    # Collapse excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def pdf_to_images(pdf_path: Path, *, dpi: int) -> list["Image.Image"]:
    import fitz  # PyMuPDF
    from PIL import Image

    images: list[Image.Image] = []
    doc = fitz.open(str(pdf_path))
    try:
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        for page_num in range(doc.page_count):
            page = doc[page_num]
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            images.append(img)
    finally:
        doc.close()
    return images


def batched(items: list, batch_size: int) -> Iterable[list]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _ensure_allowed_layer_types() -> None:
    """Work around transformers/vLLM mismatches for `ALLOWED_LAYER_TYPES`."""
    try:
        import transformers
        from transformers import configuration_utils as cu

        if hasattr(cu, "ALLOWED_LAYER_TYPES"):
            return

        cu.ALLOWED_LAYER_TYPES = (
            "full_attention",
            "sliding_attention",
            "chunked_attention",
            "linear_attention",
        )
        print(
            "WARNING: Injected transformers.configuration_utils.ALLOWED_LAYER_TYPES "
            f"(transformers={transformers.__version__})."
        )
    except Exception:
        return


def main() -> int:
    parser = argparse.ArgumentParser(
        description="DeepSeek-OCR PDF→Markdown using vLLM Metal (Apple Silicon / MLX)."
    )
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-OCR")
    parser.add_argument("--pdf", required=True, help="Input PDF path")
    parser.add_argument("--out", required=True, help="Output markdown path (.mmd/.md)")
    parser.add_argument(
        "--prompt",
        default="<image>\n<|grounding|>Convert the document to markdown.",
        help="Must include <image>.",
    )
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument(
        "--keep-refs",
        action="store_true",
        help="Keep <|ref|>/<|det|> blocks instead of stripping them.",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if "<image>" not in args.prompt:
        print("ERROR: --prompt must include '<image>' for PDF->image inference.")
        return 2

    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Model: {args.model}")
    print(f"PDF: {pdf_path}")
    print(f"Out: {out_path}")

    # macOS fork-safety (also handled by vllm-metal plugin, but safe to set here too).
    if sys.platform == "darwin":
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    _ensure_allowed_layer_types()

    try:
        from vllm import LLM, SamplingParams
        from vllm.platforms import current_platform
    except Exception as e:
        print("ERROR: vLLM is not importable in this environment.")
        print(f"Details: {type(e).__name__}: {e}")
        msg = str(e)
        if "ALLOWED_LAYER_TYPES" in msg and "transformers.configuration_utils" in msg:
            print("Likely cause: `transformers` is too old for vLLM 0.13.x.")
            print("Fix (inside this venv):")
            print('  pip install -U "transformers>=4.56.0,<5"')
            print("Note: don't install this repo's root requirements.txt in this venv (it pins transformers==4.46.3).")
        elif "operator torchvision::nms does not exist" in msg:
            print("Likely cause: broken/mismatched torchvision build (C++ ops not loaded).")
            print("Check versions:")
            print('  python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"')
            print("Fix (reinstall matching torch/torchvision in this venv):")
            print("  pip uninstall -y torchvision torchaudio || true")
            print("  pip install -U --force-reinstall --no-cache-dir torch torchvision torchaudio")
            print("Tip: avoid mixing conda envs when running this venv (deactivate conda if possible).")
        else:
            print("Hint (Apple Silicon): install vllm-metal via:")
            print("  curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash")
            print("  source ~/.venv-vllm-metal/bin/activate")
        return 2

    print(f"vLLM platform class: {current_platform.__class__.__name__}")
    try:
        print(f"vLLM device name: {current_platform.get_device_name(0)}")
    except Exception:
        pass

    print(f"Rendering PDF pages to images (dpi={args.dpi})...")
    images = pdf_to_images(pdf_path, dpi=args.dpi)
    print(f"Pages: {len(images)}")

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        skip_special_tokens=False,
    )

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
    )

    page_split = "\n<--- Page Split --->\n"
    out_chunks: list[str] = []

    for batch in batched(images, max(1, args.batch_size)):
        prompts = [
            {
                "prompt": args.prompt,
                "multi_modal_data": {"image": image},
            }
            for image in batch
        ]
        outputs = llm.generate(prompts, sampling_params)
        for out in outputs:
            text = out.outputs[0].text
            out_chunks.append(clean_markdown(text, strip_refs=not args.keep_refs))

    out_path.write_text(page_split.join(out_chunks) + page_split, encoding="utf-8")
    print(f"Saved markdown: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
