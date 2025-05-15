import torch
from torch.nn import functional as F
from tqdm import tqdm

# I think the training loop should take in the model, dataloader, optimizer along with the text encoder and decoder
# and the guidance functions along with the models, I'm not sure exactly how we should go about this
