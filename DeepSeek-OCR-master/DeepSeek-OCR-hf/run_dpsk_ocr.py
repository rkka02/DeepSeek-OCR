from transformers import AutoModel, AutoTokenizer
import torch
import os
from pathlib import Path


def pick_device_dtype_and_attn():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16, "flash_attention_2"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16, "sdpa"
    return torch.device("cpu"), torch.float32, "eager"


model_name = 'deepseek-ai/DeepSeek-OCR'


tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
device, dtype, attn_implementation = pick_device_dtype_and_attn()

model_kwargs = dict(trust_remote_code=True, use_safetensors=True, _attn_implementation=attn_implementation)
try:
    model = AutoModel.from_pretrained(model_name, **model_kwargs)
except TypeError:
    # Older transformers may not support `_attn_implementation`.
    model_kwargs.pop("_attn_implementation", None)
    model = AutoModel.from_pretrained(model_name, **model_kwargs)
except Exception:
    # flash_attention_2 may be unavailable even on CUDA; fallback to SDPA/eager.
    model_kwargs["_attn_implementation"] = "sdpa" if device.type != "cpu" else "eager"
    try:
        model = AutoModel.from_pretrained(model_name, **model_kwargs)
    except TypeError:
        model_kwargs.pop("_attn_implementation", None)
        model = AutoModel.from_pretrained(model_name, **model_kwargs)

model = model.eval().to(device=device, dtype=dtype)
print(f"Using device={device.type}, dtype={dtype}, attn={model_kwargs.get('_attn_implementation', 'default')}")



# prompt = "<image>\nFree OCR. "
prompt = "<image>\n<|grounding|>Convert the document to markdown. "
image_file = 'your_image.jpg'
output_path = 'your/output/dir'



# infer(self, tokenizer, prompt='', image_file='', output_path = ' ', base_size = 1024, image_size = 640, crop_mode = True, test_compress = False, save_results = False):

# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 640, image_size = 640, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False

# Gundam: base_size = 1024, image_size = 640, crop_mode = True

res = model.infer(
    tokenizer,
    prompt=prompt,
    image_file=image_file,
    output_path=output_path,
    base_size=1024,
    image_size=640,
    crop_mode=True,
    save_results=False,
    test_compress=True,
)

Path(output_path).mkdir(parents=True, exist_ok=True)
out_file = Path(output_path) / f"{Path(image_file).stem}.mmd"
out_file.write_text(res if isinstance(res, str) else str(res), encoding="utf-8")
print(f"Saved markdown: {out_file}")
