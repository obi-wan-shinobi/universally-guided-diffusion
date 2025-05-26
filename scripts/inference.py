import torch

from ugdiff.guidance.segmentation_guidance import apply_segmentation_guidance
from ugdiff.inference.pipeline import generate_image_from_prompt
from ugdiff.models.stable_diffusion import (
    create_scheduler,
    load_text_encoder,
    load_unet,
    load_vae,
)
from ugdiff.utils import latents_to_pil

# Set device (MPS or "cuda"/"cpu")
torch_device = "mps"

# Load components
vae = load_vae(torch_device)
tokenizer, text_encoder = load_text_encoder(torch_device)
unet = load_unet(torch_device)
scheduler = create_scheduler()

# Prompt to generate image from
prompt = [
    "an anime painting of starry night",
    "a photo of an astronaut riding a horse on mars",
]

# Run the generation pipeline
image_tensor = generate_image_from_prompt(
    prompt=prompt,
    vae=vae,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    unet=unet,
    scheduler=scheduler,
    guidance_fn=apply_segmentation_guidance,  # Optional hook
    device=torch_device,
    height=512,
    width=512,
    num_inference_steps=100,
    guidance_scale=7.5,
    seed=0,
)

# Convert to PIL and save
pil_images = latents_to_pil(image_tensor)
for i, image in enumerate(pil_images):
    image.save(f"tmp/sample_{i}.png")
