import torch
from tqdm.auto import tqdm

from ugdiff.utils import seed_generator


def generate_image_from_prompt(
    prompt,
    vae,
    tokenizer,
    text_encoder,
    unet,
    scheduler,
    guidance_fn=None,
    device="cpu",
    height=512,
    width=512,
    num_inference_steps=50,
    guidance_scale=7.5,
    seed=0,
):
    batch_size = len(prompt)
    generator = seed_generator(seed=seed, device=device)

    # Tokenize input
    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

    # Empty prompt for classifier-free guidance
    uncond_input = tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=text_input.input_ids.shape[-1],
        return_tensors="pt",
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # Prepare latents
    latents = torch.randn(
        (batch_size, unet.in_channels, height // 8, width // 8),
        generator=generator,
        device=device,
    )
    scheduler.set_timesteps(num_inference_steps)
    latents *= scheduler.init_noise_sigma

    for t in tqdm(scheduler.timesteps):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

        with torch.no_grad():
            noise_pred = unet(
                latent_model_input,
                torch.tensor([t], dtype=torch.float32, device=device),
                encoder_hidden_states=text_embeddings,
            ).sample

        # Classifier-free guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        # Apply optional external guidance
        if guidance_fn is not None:
            noise_pred = guidance_fn(
                noise_pred=noise_pred,
                timestep=t,
                latents=latents,
                embeddings=text_embeddings,
            )

        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # Decode latents
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(latents).sample

    return image
