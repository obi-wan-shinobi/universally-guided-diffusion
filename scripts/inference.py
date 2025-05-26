import torch
from tqdm.auto import tqdm

# from ugdiff.guidance.segmentation_guidance import apply_segmentation_guidance
from ugdiff.models.stable_diffusion import (
    create_scheduler,
    load_text_encoder,
    load_unet,
    load_vae,
)
from ugdiff.utils import latents_to_pil, seed_generator

torch_device = "mps"

# Load models
vae = load_vae(torch_device)
tokenizer, text_encoder = load_text_encoder(torch_device)
unet = load_unet(torch_device)
scheduler = create_scheduler()

prompt = ["a photograph of an astronaut riding a horse"]
batch_size = len(prompt)
height = width = 512
num_inference_steps = 100
guidance_scale = 7.5
generator = seed_generator(0, device=torch_device)

# Tokenize text
text_input = tokenizer(
    prompt,
    padding="max_length",
    max_length=tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt",
)
text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

# Empty prompt for classifier-free guidance
uncond_input = tokenizer(
    [""] * batch_size,
    padding="max_length",
    max_length=text_input.input_ids.shape[-1],
    return_tensors="pt",
)
uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

# Latents
latents = torch.randn(
    (batch_size, unet.in_channels, height // 8, width // 8),
    generator=generator,
    device=torch_device,
)
scheduler.set_timesteps(num_inference_steps)
latents *= scheduler.init_noise_sigma

# Denoising loop
for t in tqdm(scheduler.timesteps):
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

    with torch.no_grad():
        noise_pred = unet(
            latent_model_input,
            torch.tensor([t], dtype=torch.float32, device=torch_device),
            encoder_hidden_states=text_embeddings,
        ).sample

    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (
        noise_pred_text - noise_pred_uncond
    )

    # Optional segmentation guidance
    # noise_pred = apply_segmentation_guidance(noise_pred, t, text_embeddings)

    latents = scheduler.step(noise_pred, t, latents).prev_sample

# Decode
latents = 1 / 0.18215 * latents
with torch.no_grad():
    image = vae.decode(latents).sample

pil_images = latents_to_pil(image)
pil_images[0].save("tmp/another_sample.png")
