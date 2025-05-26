import numpy as np
import torch
from PIL import Image


def seed_generator(seed=0, device="cpu"):
    return torch.Generator(device=device).manual_seed(seed)


def latents_to_pil(image_tensor):
    image = (image_tensor / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    return [Image.fromarray(img) for img in images]
