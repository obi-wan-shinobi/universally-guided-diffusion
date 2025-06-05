import torch
from tqdm.auto import tqdm

from ugdiff.utils import seed_generator


class SegmentationGuidedDiffusionPipeline:
    def __init__(
        self,
        vae,
        tokenizer,
        text_encoder,
        unet,
        scheduler,
        device="cpu",
        guidance_module=None,
    ):
        self.vae = vae
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.unet = unet
        self.scheduler = scheduler
        self.device = device
        self.guidance_module = guidance_module

        for param in self.unet.parameters():
            param.requires_grad = False

        for param in self.vae.parameters():
            param.requires_grad = False

        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def encode_prompt(self, prompt):
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        return text_embeddings

    def generate(
        self,
        prompt,
        height=512,
        width=512,
        num_inference_steps=50,
        seed=32,
        segmentation_maps=None,
        class_id=12,
        guidance_scale=2,
    ):
        if segmentation_maps is not None:
            batch_size = len(segmentation_maps)
        else:
            batch_size = len(prompt)
        generator = seed_generator(seed=seed, device=self.device)

        # --- Encode both conditional and unconditional prompts
        text_embeddings = self.encode_prompt(prompt)  # Conditional
        uncond_prompt = [""] * batch_size
        uncond_embeddings = self.encode_prompt(uncond_prompt)  # Unconditional

        latents = torch.randn(
            (batch_size, self.unet.in_channels, height // 8, width // 8),
            generator=generator,
            device=self.device,
        )
        self.scheduler.set_timesteps(num_inference_steps)
        latents = latents * self.scheduler.init_noise_sigma

        recurrent_steps = 3

        for t in tqdm(self.scheduler.timesteps):
            for k in range(recurrent_steps):
                if self.guidance_module is not None and segmentation_maps is not None:
                    latents = latents.detach().requires_grad_()
                else:
                    latents = latents.detach()

                latent_input = self.scheduler.scale_model_input(latents, t)

                # torch.no_grad() won't work if we want to use universally guided diffusion
                # with torch.no_grad():
                noise_pred_text = self.unet(
                    latent_input,
                    torch.tensor([t], dtype=torch.float32, device=self.device),
                    encoder_hidden_states=text_embeddings,
                ).sample

                with torch.no_grad():
                    latent_input_uncond = self.scheduler.scale_model_input(
                        latents.detach(), t
                    )
                    noise_pred_uncond = self.unet(
                        latent_input_uncond,
                        torch.tensor([t], dtype=torch.float32, device=self.device),
                        encoder_hidden_states=uncond_embeddings,
                    ).sample

                # CFG interpolation
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

                if self.guidance_module is not None and segmentation_maps is not None:
                    latents.requires_grad_(True)
                    alpha_t = self.scheduler.alphas_cumprod[t.long()].to(self.device)
                    pred_z0 = (
                        latents - (1 - alpha_t).sqrt() * noise_pred
                    ) / alpha_t.sqrt()
                    loss = self.guidance_module.compute_loss(
                        latents=pred_z0,
                        target_masks=segmentation_maps.to(self.device),
                        class_id=class_id,
                    )
                    loss.backward()

                    if latents.grad is None:
                        raise RuntimeError(
                            "Gradient is None. Ensure latents require grad."
                        )

                    gradients = latents.grad.detach()
                    noise_pred = noise_pred + self.guidance_strength(t) * gradients

                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                if k < recurrent_steps - 1:
                    noise = torch.randn_like(latents)
                    alpha_prev = self.scheduler.alphas_cumprod[t.long() - 1].to(
                        self.device
                    )
                    alpha_ratio = alpha_t / alpha_prev

                    # Clamp to ensure the square root gets only non-negative input
                    alpha_ratio = alpha_ratio.clamp(max=1.0)

                    latents = (alpha_ratio).sqrt() * latents + (1 - alpha_ratio).clamp(
                        min=0
                    ).sqrt() * noise

        # Decode
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample

        return image

    def generate_with_cfg(
        self,
        prompt,
        height=512,
        width=512,
        num_inference_steps=50,
        seed=32,
        guidance_scale=7.5,
    ):
        batch_size = len(prompt)
        generator = seed_generator(seed=seed, device=self.device)

        # Encode conditional and unconditional prompts
        text_embeddings = self.encode_prompt(prompt)  # (B, T, D)
        uncond_embeddings = self.encode_prompt([""] * batch_size)

        # Concatenate for classifier-free guidance
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings], dim=0)

        # Prepare initial latents
        latents = torch.randn(
            (batch_size, self.unet.in_channels, height // 8, width // 8),
            generator=generator,
            device=self.device,
        )
        self.scheduler.set_timesteps(num_inference_steps)
        latents = latents * self.scheduler.init_noise_sigma

        for t in tqdm(self.scheduler.timesteps):
            latent_model_input = self.scheduler.scale_model_input(latents, t)

            # Duplicate for unconditional + conditional
            latent_model_input = torch.cat([latent_model_input] * 2, dim=0)

            # UNet forward pass
            noise_pred = self.unet(
                latent_model_input,
                torch.tensor([t], dtype=torch.float32, device=self.device),
                encoder_hidden_states=text_embeddings,
            ).sample

            # Split into unconditional and conditional predictions
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

            # Apply classifier-free guidance
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # Scheduler step
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # Decode
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample

        return image

    def guidance_strength(self, t):
        alpha_t = self.scheduler.alphas_cumprod[t.long()].to(self.device)

        return 400 * (1 - alpha_t).sqrt()
