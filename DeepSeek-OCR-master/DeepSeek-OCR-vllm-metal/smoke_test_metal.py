import argparse
import platform
import sys
from pathlib import Path


def _print_env_diagnostics() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.system()} {platform.machine()}")


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
