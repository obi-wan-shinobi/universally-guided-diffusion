import torch
import torch.nn as nn

# I think we should define a basic Diffusion model here
# A class that takes in a UNet and scheduler. We can use the Unet from Compvis
# Based on the paper, we need to have a forward, denoise, add_noise steps
