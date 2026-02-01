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


def _ensure_torchvision_stub_if_broken() -> None:
    """See smoke_test_metal.py for rationale."""
    try:
        import torchvision  # noqa: F401

        return
    except Exception:
        pass

    import enum
    import types

    import numpy as np
    import torch

    class InterpolationMode(enum.Enum):
        NEAREST = "nearest"
        BILINEAR = "bilinear"
        BICUBIC = "bicubic"
        LANCZOS = "lanczos"

    class Compose:
        def __init__(self, transforms):
            self.transforms = list(transforms or [])

        def __call__(self, x):
            for t in self.transforms:
                x = t(x)
            return x

    class ToTensor:
        def __call__(self, pic):
            if isinstance(pic, torch.Tensor):
                return pic.to(dtype=torch.float32)

            if hasattr(pic, "mode") and hasattr(pic, "size"):
                arr = np.array(pic, copy=False)
            else:
                arr = np.array(pic, copy=False)

            if arr.ndim == 2:
                arr = arr[:, :, None]
            if arr.ndim != 3:
                raise ValueError(f"Unsupported input shape for ToTensor: {arr.shape}")

            if arr.dtype != np.float32:
                arr = arr.astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0

            arr = np.transpose(arr, (2, 0, 1))
            return torch.from_numpy(arr)

    class Normalize:
        def __init__(self, mean, std):
            self.mean = torch.tensor(mean, dtype=torch.float32)[:, None, None]
            self.std = torch.tensor(std, dtype=torch.float32)[:, None, None]

        def __call__(self, tensor):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("Normalize expects a torch.Tensor")
            return (tensor - self.mean) / self.std

    class Resize:
        def __init__(self, size, interpolation=InterpolationMode.BILINEAR):
            if isinstance(size, int):
                self.size = (size, size)
            else:
                self.size = tuple(size)
            self.interpolation = interpolation

        def __call__(self, img):
            try:
                from PIL import Image
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "Pillow is required for torchvision Resize stub"
                ) from e

            if not isinstance(img, Image.Image):
                raise TypeError("Resize expects a PIL.Image.Image")

            pil_interp = {
                InterpolationMode.NEAREST: Image.Resampling.NEAREST,
                InterpolationMode.BILINEAR: Image.Resampling.BILINEAR,
                InterpolationMode.BICUBIC: Image.Resampling.BICUBIC,
                InterpolationMode.LANCZOS: Image.Resampling.LANCZOS,
            }.get(self.interpolation, Image.Resampling.BILINEAR)

            height, width = self.size
            return img.resize((width, height), resample=pil_interp)

    tv = types.ModuleType("torchvision")
    transforms = types.ModuleType("torchvision.transforms")
    functional = types.ModuleType("torchvision.transforms.functional")

    transforms.Compose = Compose
    transforms.ToTensor = ToTensor
    transforms.Normalize = Normalize
    transforms.Resize = Resize
    transforms.InterpolationMode = InterpolationMode

    functional.InterpolationMode = InterpolationMode

    tv.transforms = transforms
    tv.__dict__["__version__"] = "0.0.0-stub"

    sys.modules["torchvision"] = tv
    sys.modules["torchvision.transforms"] = transforms
    sys.modules["torchvision.transforms.functional"] = functional

    print("WARNING: torchvision is broken; using a minimal torchvision.transforms stub.")


def _ensure_auto_image_processor_stub_if_missing() -> None:
    """Ensure `from transformers import AutoImageProcessor` won't crash vLLM import."""
    try:
        from transformers import AutoImageProcessor  # noqa: F401

        return
    except Exception as e:
        try:
            import transformers
        except Exception:
            return

        class _AutoImageProcessorStub:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                raise RuntimeError(
                    "AutoImageProcessor is not available in this environment. "
                    "Install vision deps (at least `Pillow`) and ensure your "
                    "Transformers install is healthy. "
                    f"Original error: {type(e).__name__}: {e}"
                )

        transformers.AutoImageProcessor = _AutoImageProcessorStub  # type: ignore[attr-defined]
        print(
            "WARNING: AutoImageProcessor could not be imported; installed a stub. "
            "Text-only models should work; some vision models may not."
        )


def _ensure_transformers_auto_stubs_if_missing() -> None:
    """Ensure transformers auto-* imports won't crash vLLM import."""
    try:
        import transformers
    except Exception:
        return

    def _install_stub(attr_name: str, original_exc: Exception) -> None:
        class _Stub:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                raise RuntimeError(
                    f"{attr_name} is not available in this environment. "
                    "Install missing vision/audio deps and ensure your Transformers "
                    "install is healthy. "
                    f"Original error: {type(original_exc).__name__}: {original_exc}"
                )

        setattr(transformers, attr_name, _Stub)  # type: ignore[attr-defined]
        print(
            f"WARNING: {attr_name} could not be imported; installed a stub. "
            "Text-only models should work; some multimodal models may not."
        )

    for name in (
        "AutoProcessor",
        "AutoImageProcessor",
        "AutoFeatureExtractor",
        "AutoVideoProcessor",
    ):
        try:
            getattr(transformers, name)
        except Exception as e:
            _install_stub(name, e)


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
    _ensure_torchvision_stub_if_broken()
    _ensure_auto_image_processor_stub_if_missing()
    _ensure_transformers_auto_stubs_if_missing()

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
