import argparse
import os
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from torchvision import transforms, utils
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import random
import clip
import torch.nn as nn
import torchvision.transforms.functional as TF

from ldm.util import instantiate_from_config
from .helper import OptimizerDetails

import json
import hashlib
from torchvision import transforms
from data_loader.ms_coco import MSCOCO1kDataset
from torchvision.transforms import ToPILImage
import traceback
import datetime

def normalize_tensor(t):
    return (t * 2) - 1


class ImageFolderDataset(Dataset):
    def __init__(self, folder, image_size, exts=('jpg', 'jpeg', 'png')):
        self.paths = [p for ext in exts for p in Path(folder).rglob(f'*.{ext}')]
        random.shuffle(self.paths)
        self.transform = transforms.Compose([
            transforms.Resize(image_size, Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Lambda(normalize_tensor)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img)


class ClipGuidance(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def forward(self, x, text):
        img = (x + 1) * 0.5
        img = TF.resize(img, (224, 224), interpolation=TF.InterpolationMode.BICUBIC)
        img = self.normalize(img)
        logits_per_image, _ = self.model(img, text)
        return -logits_per_image


def load_model_from_config(config, ckpt_path, verbose=False):
    print(f"Loading model from {ckpt_path}")
    state = torch.load(ckpt_path, map_location='cpu')
    if 'global_step' in state:
        print(f"Global Step: {state['global_step']}")
    sd = state['state_dict']
    model = instantiate_from_config(config.model)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if verbose:
        if missing:
            print('Missing keys:', missing)
        if unexpected:
            print('Unexpected keys:', unexpected)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    return model


def save_with_border(img_tensor, path, border=10, color=(255,255,255)):
    img = (img_tensor + 1) * 0.5
    utils.save_image(img, path, nrow=1)
    print(f"Saved: {path}")
    bgr = cv2.imread(path)
    return cv2.copyMakeBorder(bgr, border, border, border, border,
                              cv2.BORDER_CONSTANT, value=color)


def create_folder(path):
    os.makedirs(path, exist_ok=True)


def get_optimation_details(args, device):
    clip_model, _ = clip.load('RN50')
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)
    guidance = ClipGuidance(clip_model)
    guidance = nn.DataParallel(guidance).to(device)
    operation = OptimizerDetails()
    operation.num_steps = args.optim_num_steps
    operation.guidance_func = None
    operation.optimizer = 'Adam'
    operation.lr = args.optim_lr
    operation.loss_func = guidance
    operation.max_iters = args.optim_max_iters
    operation.loss_cutoff = args.optim_loss_cutoff
    operation.tv_loss = args.optim_tv_loss
    operation.guidance_3 = args.optim_forward_guidance
    operation.guidance_2 = args.optim_backward_guidance
    operation.original_guidance = args.optim_original_conditioning
    print(f"guidance_3: {operation.guidance_3}, guidance_2: {operation.guidance_2}, original_guidance: {operation.original_guidance}")
    
    operation.mask_type = args.optim_mask_type
    operation.optim_guidance_3_wt = args.optim_forward_guidance_wt
    operation.do_guidance_3_norm = args.optim_do_forward_guidance_norm
    operation.warm_start = args.optim_warm_start
    operation.print = args.optim_print
    operation.print_every = 500
    operation.folder = args.optim_folder
    return operation, guidance

def create_dataloader(batch_size, image_transforms):
    dataset = MSCOCO1kDataset(
        image_transform=image_transforms,
        text_transform=None
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False)
    return loader

def main():
    parser = argparse.ArgumentParser()

    # sampling / model flags
    parser.add_argument("--laion400m", action="store_true", default=False,
                        help="use the LAION400M model")
    parser.add_argument("--fixed_code", action="store_true", default=False,
                        help="use the same starting code across samples")
    parser.add_argument("--ddim_eta", type=float, default=0.0,
                        help="DDIM eta (0.0 = deterministic sampling)")
    parser.add_argument("--n_iter", type=int, default=2,
                        help="how many sampling iterations to run")
    parser.add_argument("--H", type=int, default=256,
                        help="image height in pixels")
    parser.add_argument("--W", type=int, default=256,
                        help="image width in pixels")
    parser.add_argument("--C", type=int, default=4,
                        help="number of latent channels")
    parser.add_argument("--f", type=int, default=8,
                        help="downsampling factor")
    parser.add_argument("--n_samples", type=int, default=3,
                        help="batch size (samples per prompt)")

    # config & checkpoint
    parser.add_argument("--config", type=str,
                        default="configs/stable-diffusion/v1-inference.yaml",
                        help="path to model config")
    parser.add_argument("--ckpt", type=str,
                        default="models/checkpoints/sd-v1-4/sd-v1-4.ckpt",
                        help="path to model checkpoint")

    # training / optimization
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for reproducibility")
    parser.add_argument("--steps", type=int, default=100,
                        help="number of diffusion steps")
    parser.add_argument("--mode", type=int, default=0,
                        help="mode selector")
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of training epochs")
    parser.add_argument("--save_image_folder", type=str, default="./sanity_check/",
                        help="where to save generated images")

    # optimization / guidance settings
    parser.add_argument("--optim_lr", type=float, default=1e-2,
                        help="learning rate for optimizer")
    parser.add_argument("--optim_max_iters", type=int, default=1,
                        help="max iterations per optimization step")
    parser.add_argument("--optim_mask_type", type=int, default=1,
                        help="mask type for guidance")
    parser.add_argument("--optim_loss_cutoff", type=float, default=1e-5,
                        help="loss cutoff threshold")
    parser.add_argument("--optim_forward_guidance", action="store_true", default=False,
                        help="enable forward guidance")
    parser.add_argument("--optim_backward_guidance", action="store_true", default=False,
                        help="enable backward guidance")
    parser.add_argument("--optim_original_conditioning", action="store_true", default=False,
                        help="use original conditioning")
    parser.add_argument("--optim_forward_guidance_wt", type=float, default=5.0,
                        help="weight for forward guidance")
    parser.add_argument("--optim_do_forward_guidance_norm", action="store_true", default=False,
                        help="normalize forward guidance")
    parser.add_argument("--optim_tv_loss", type=float, default=None,
                        help="total variation loss weight")
    parser.add_argument("--optim_warm_start", action="store_true", default=False,
                        help="warm start optimization")
    parser.add_argument("--optim_print", action="store_true", default=False,
                        help="print optimization progress")
    parser.add_argument("--optim_folder", type=str, default="./results",
                        help="folder for optimization outputs")
    parser.add_argument("--optim_num_steps", nargs="+", type=int, default=[1],
                        help="number of steps for each optimization phase")

    # text / batches
    parser.add_argument("--text", type=str, default=None,
                        help="text prompt (overrides text_type presets)")
    
    parser.add_argument("--text_type", type=int, default=1,
                        help="preset text type index")
    parser.add_argument("--batches", type=int, default=0,
                        help="how many batches to process before exiting")
    
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt)
    model = nn.DataParallel(model).to(device)

    operation, guidance = get_optimation_details(args, device)

    # ! new
    date_str = datetime.datetime.now().strftime("%m-%d")
    gt_img_dir = f'test_clip/{date_str}/GT/'
    output_img_dir = f'test_clip/{date_str}/output/'
    prompt_dir = f'test_clip/{date_str}/input/prompts/'
    create_folder(gt_img_dir)
    create_folder(output_img_dir)
    create_folder(prompt_dir)

    batch_size = 1
    num_workers = 0
    dtype = torch.float16
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image_transforms = transforms.Compose([
        transforms.Resize((args.W, args.H)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    dataset = MSCOCO1kDataset(
        image_transform=image_transforms,
        text_transform=None
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False, num_workers=num_workers)

    to_pil = ToPILImage()
    prompts = []

    cnt = 0
    while True:
        try:
            for batch_idx, (images, captions, image_ids) in enumerate(loader):
                images = images.to(device)
                for i in range(images.size(0)):
                    img_tensor = images[i].cpu()
                    img = img_tensor * torch.tensor(std).view(3,1,1) + torch.tensor(mean).view(3,1,1)
                    img = torch.clamp(img, 0, 1)
                    pil_img = to_pil(img)
                    img_name = f"{image_ids[i]}.png" 
                    pil_img.save(os.path.join(gt_img_dir, img_name))
                    prompts.append({"image": img_name, "text": captions[i]})
                
                xc = images.size(0) * [""]
                cond = model.module.get_learned_conditioning(xc).to(device)
                # output = model.module.operation_diffusion(
                #     og_img=images,
                #     operated_image=clip.tokenize(list(captions)).to(device),
                #     cond=cond,
                #     operation=operation
                # )
                # conditional generation
                output = model.module.operation_diffusion(
                    og_img=images,
                    operated_image=clip.tokenize(list(captions)).to(device),
                    cond=model.module.get_learned_conditioning([captions[i] for i in range(images.size(0))]).to(device),
                    operation=operation
                )
                for i, out in enumerate(output):
                    out_img = (out + 1) * 0.5
                    out_img = torch.clamp(out_img, 0, 1)
                    out_pil = to_pil(out_img.cpu())
                    img_name = f"{image_ids[i]}.png"
                    out_pil.save(os.path.join(output_img_dir, img_name))
                cnt += images.size(0)

                if cnt % 10 == 0:
                    break

                if args.batches and batch_idx >= args.batches:
                    break
            break
        except Exception as e:
            traceback.print_exc()
            print(f"DataLoader error at batch {batch_idx}, restarting DataLoader. Error: {e}")
            loader = create_dataloader(batch_size, image_transforms)
            continue

    prompt_path = os.path.join(prompt_dir, "prompts.json")
    with open(prompt_path, "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)
    print(f"Prompts saved to {prompt_path}")

    # root = 'test_clip/inputs/'
    # ds = ImageFolderDataset(root, args.W)
    # loader = DataLoader(ds, batch_size=1, shuffle=False, pin_memory=True, drop_last=True)

    # cnt = 0
    # for batch_idx, img in enumerate(loader):
    #     cond = model.module.get_learned_conditioning([''] * img.size(0))
    #     output = model.module.operation_diffusion(og_img=img, operated_image=clip.tokenize([args.text or '' ]).to(device),
    #                                             cond=cond, operation=operation)
    #     for out in output:
    #         save_with_border(out, os.path.join(args.optim_folder, f'out_{cnt}.png'))
    #         cnt += 1
    #     if batch_idx == args.batches:
    #         break


if __name__ == '__main__':
    main()
