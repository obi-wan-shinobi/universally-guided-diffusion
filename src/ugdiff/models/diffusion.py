from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

torch_device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)  # Or "cuda" / "cpu"


def load_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32
    ).to(torch_device)

    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    return pipe


def generate_image(prompt: str, output_path: str = "output.png"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pipe = load_pipeline()
    image = pipe(
        prompt, height=512, width=512, num_inference_steps=50, guidance_scale=10
    ).images[0]
    image.save(output_path)
