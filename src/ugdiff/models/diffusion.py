from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline


def load_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32
    ).to("mps")

    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    return pipe


def generate_image(prompt: str, output_path: str = "output.png"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pipe = load_pipeline()
    image = pipe(prompt, height=512, width=768, num_inference_steps=75).images[0]
    image.save(output_path)
