import argparse
import platform
import sys
from pathlib import Path


def _print_env_diagnostics() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.system()} {platform.machine()}")


def _ensure_torchvision_stub_if_broken() -> None:
    """Avoid hard failure when torchvision is installed but unusable.

    On some macOS setups, importing torchvision can fail with:
      RuntimeError: operator torchvision::nms does not exist

    vLLM imports some multi-modal processors that depend on
    `torchvision.transforms` (ToTensor/Normalize/Resize). If torchvision is
    broken, we inject a tiny pure-Python stub module that provides the subset
    vLLM needs.
    """
    try:
        import torchvision  # noqa: F401

        return
    except Exception as e:
        msg = str(e)
        if "operator torchvision::nms does not exist" not in msg:
            # If torchvision import fails for other reasons (or isn't installed),
            # we still provide a minimal stub so vLLM can import.
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

            # PIL.Image.Image
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

            # HWC -> CHW
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

            # PIL expects (width, height)
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
    """Ensure `from transformers import AutoImageProcessor` won't crash vLLM import.

    On some environments, Transformers' vision stack is partially missing and
    `AutoImageProcessor` fails to import with a LazyModule error. vLLM imports
    `AutoImageProcessor` unconditionally, even for text-only models.

    We install a tiny stub so vLLM can import; models that truly require
    AutoImageProcessor will still fail at runtime with a clear error.
    """
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


def _try_print_vllm_platform() -> None:
    try:
        from vllm.platforms import current_platform

        print(f"vLLM platform class: {current_platform.__class__.__name__}")
        try:
            print(f"vLLM device name: {current_platform.get_device_name(0)}")
        except Exception:
            pass
    except Exception as e:
        print(f"WARNING: could not import vLLM platform info ({type(e).__name__}: {e})")


def _ensure_allowed_layer_types() -> None:
    """Work around transformers/vLLM mismatches for `ALLOWED_LAYER_TYPES`.

    Some vLLM builds import `ALLOWED_LAYER_TYPES` from
    `transformers.configuration_utils`. If it's missing, vLLM fails to import.
    We inject a conservative default so the smoke test can proceed.
    """
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
    parser = argparse.ArgumentParser(description="vLLM Metal smoke test (text or image).")
    parser.add_argument("--model", required=True, help="HF model name or local path")
    parser.add_argument(
        "--prompt",
        default="<image>\n<|grounding|>Convert the document to markdown.",
        help="Prompt for multimodal run (must include <image> if --image is used).",
    )
    parser.add_argument("--text", default=None, help="Text-only prompt (no image).")
    parser.add_argument("--image", default=None, help="Path to an image file for VLM test.")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--dtype", default="auto", help="vLLM dtype (e.g. auto/float16/bfloat16)")
    args = parser.parse_args()

    _print_env_diagnostics()

    # macOS fork-safety (also handled by vllm-metal plugin, but safe to set here too).
    if sys.platform == "darwin":
        import os

        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    _ensure_allowed_layer_types()
    _ensure_torchvision_stub_if_broken()
    _ensure_auto_image_processor_stub_if_missing()

    try:
        from vllm import LLM, SamplingParams
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
            print("Hint: install vllm-metal using:")
            print("  curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash")
            print("  source ~/.venv-vllm-metal/bin/activate")
        return 2

    _try_print_vllm_platform()

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

    if args.text is not None:
        prompts = [args.text]
    else:
        if args.image is None:
            print("ERROR: Provide either --text or --image.")
            return 2
        if "<image>" not in args.prompt:
            print("ERROR: --prompt must include '<image>' when using --image.")
            return 2
        try:
            from PIL import Image
        except Exception as e:
            print(f"ERROR: Pillow not installed ({type(e).__name__}: {e})")
            print("Hint: pip install Pillow")
            return 2

        image_path = Path(args.image)
        if not image_path.exists():
            print(f"ERROR: image not found: {image_path}")
            return 2

        image = Image.open(image_path).convert("RGB")
        prompts = [
            {
                "prompt": args.prompt,
                "multi_modal_data": {"image": image},
            }
        ]

    outputs = llm.generate(prompts, sampling_params)
    if not outputs:
        print("No outputs returned.")
        return 1

    text = outputs[0].outputs[0].text
    print("\n===== MODEL OUTPUT (first request) =====\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
