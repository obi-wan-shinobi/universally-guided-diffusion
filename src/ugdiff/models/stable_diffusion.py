from diffusers import (
    AutoencoderKL,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer


def load_vae(device):
    vae = AutoencoderKL.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="vae"
    )

    return vae.to(device)


def load_text_encoder(device):
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    return tokenizer, text_encoder.to(device)


def load_unet(device):
    unet = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="unet"
    )
    return unet.to(device)


def create_LMS_scheduler(num_train_timesteps=1000):
    return LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=num_train_timesteps,
    )


def create_PNDM_scheduler(num_train_timesteps=1000):
    return PNDMScheduler.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="scheduler"
    )
    # return PNDMScheduler(
    #     beta_start=0.00085,
    #     beta_end=0.012,
    #     beta_schedule="scaled_linear",
    #     num_train_timesteps=num_train_timesteps,
    # )
