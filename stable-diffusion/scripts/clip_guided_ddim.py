import argparse
import os
import torch
import json
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torchvision import transforms
from torch.utils.data import DataLoader
import clip
import torch.nn as nn
import torchvision.transforms.functional as TF
from ldm.util import instantiate_from_config
from .helper import OptimizerDetails
from data_loader.ms_coco import MSCOCO1kDataset
from torchvision.transforms import ToPILImage
from ldm.models.diffusion.ddim_with_grad import DDIMSamplerWithGrad

import datetime
from PIL import Image

def create_folder(path):
    os.makedirs(path, exist_ok=True)

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

def get_optimation_details(args, device):
    clip_model, _ = clip.load('RN50')
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)
    guidance = ClipGuidance(clip_model)
    guidance = nn.DataParallel(guidance).to(device)
    operation = OptimizerDetails()
    operation.num_steps = args.optim_num_steps

    operation.optimizer = 'Adam'
    operation.lr = args.optim_lr
    operation.guidance_fun = None
    operation.loss_func = guidance
    operation.max_iters = args.optim_max_iters
    operation.loss_cutoff = args.optim_loss_cutoff

    operation.tv_loss = args.optim_tv_loss
    operation.guidance_3 = args.optim_forward_guidance
    operation.guidance_2 = args.optim_backward_guidance
    operation.original_guidance = args.optim_original_conditioning
    print(f"forward guidance: {operation.guidance_3}, backward guidance: {operation.guidance_2}, original conditioning: {args.optim_original_conditioning}")
    operation.mask_type = args.optim_mask_type
    operation.optim_guidance_3_wt = args.optim_forward_guidance_wt
    operation.warm_start = args.optim_warm_start
    operation.do_guidance_3_norm = args.optim_do_forward_guidance_norm

    operation.print = args.optim_print
    operation.print_every = 500
    operation.folder = args.optim_folder
    
    return operation, guidance

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

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ddim_steps", type=int, default=50, help="number of ddim sampling steps")
    parser.add_argument("--ddim_eta", type=float, default=0.0, help="ddim eta (eta=0.0 is deterministic)")
    parser.add_argument("--H", type=int, default=256, help="image height in pixels")
    parser.add_argument("--W", type=int, default=256, help="image width in pixels")
    parser.add_argument("--C", type=int, default=4, help="number of latent channels")
    parser.add_argument("--f", type=int, default=8, help="downsampling factor")
    parser.add_argument("--scale", type=float, default=7.5, help="unconditional guidance scale")
    parser.add_argument("--config", type=str, default="configs/stable-diffusion/v1-inference.yaml", help="path to model config")
    parser.add_argument("--ckpt", type=str, default="models/checkpoints/sd-v1-4/sd-v1-4.ckpt", help="path to model checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    parser.add_argument("--optim_lr", default=1e-2, type=float)
    parser.add_argument('--optim_max_iters', type=int, default=1)
    parser.add_argument('--optim_mask_type', type=int, default=1)
    parser.add_argument("--optim_loss_cutoff", default=0.00001, type=float)
    parser.add_argument('--optim_forward_guidance', action='store_true', default=False)
    parser.add_argument('--optim_backward_guidance', action='store_true', default=False)
    parser.add_argument('--optim_original_conditioning', action='store_true', default=False)
    parser.add_argument("--optim_forward_guidance_wt", default=5.0, type=float)
    parser.add_argument("--optim_tv_loss", default=None, type=float)
    parser.add_argument('--optim_warm_start', action='store_true', default=False)
    parser.add_argument("--optim_do_forward_guidance_norm", action="store_true", default=False,
                        help="normalize forward guidance")
    parser.add_argument('--optim_print', action='store_true', default=False)
    parser.add_argument('--optim_folder', default='./results/')
    parser.add_argument("--optim_num_steps", nargs="+", default=[1], type=int)
    parser.add_argument('--text_type', type=int, default=1)
    parser.add_argument("--text", default=None)
    parser.add_argument('--batches', type=int, default=1, help="how many batches to process before exiting")

    args = parser.parse_args()

    seed_everything(args.seed)

    config = OmegaConf.load(f"{args.config}")
    model = load_model_from_config(config, f"{args.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.eval()

    sampler = DDIMSamplerWithGrad(model)
    operation, guidance = get_optimation_details(args, device)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image_transforms = transforms.Compose([
        transforms.Resize((args.W, args.H)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    batch_size = args.batches

    dataset = MSCOCO1kDataset(
        image_transform=image_transforms,
        text_transform=None
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False, num_workers=2)

    to_pil = ToPILImage()
    prompts = []
    date_str = datetime.datetime.now().strftime("%m-%d")
    
    # input_img_path = f'test_clip_ddim/input/image/cat.jpg'
    gt_img_dir = f'test_clip_ddim/{date_str}/GT/'
    output_img_dir = f'test_clip_ddim/{date_str}/output/'
    prompt_dir = f'test_clip_ddim/{date_str}/input/prompts/'
    create_folder(gt_img_dir)
    create_folder(output_img_dir)
    create_folder(prompt_dir)
    # input_img_pil = Image.open(input_img_path).convert("RGB")
    # input_img_tensor = image_transforms(input_img_pil).unsqueeze(0).to(device)  # [1, C, H, W]

    cnt = 0

    for batch_idx, (images, captions, image_ids) in enumerate(loader):
        images = images.to(device)
        for i in range(images.size(0)):
            img_tensor = images[i].cpu()
            img = img_tensor * torch.tensor(std).view(3,1,1) + torch.tensor(mean).view(3,1,1)
            img = torch.clamp(img, 0, 1)
            pil_img = to_pil(img)
            img_name = f"{image_ids[i]}.png"
            pil_img.save(os.path.join(gt_img_dir, img_name))
            print(f"Saved GT image {img_name}")
            prompts.append({"image": img_name, "text": captions[i]})

        # DDIM sampling
        prompt_texts = list(captions)
        print(prompt_texts)
        uc = None
        if args.scale != 1.0:
            uc = model.module.get_learned_conditioning(batch_size * [""])
        c = model.module.get_learned_conditioning(prompt_texts)
        shape = [args.C, args.H // args.f, args.W // args.f]
        samples_ddim, start_zt = sampler.sample(
            S=args.ddim_steps,
            conditioning=c,
            batch_size=batch_size,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=args.scale,
            unconditional_conditioning=uc,
            eta=args.ddim_eta,
            operated_image=clip.tokenize(list(captions)).to(device),
            operation=operation
        )
        x_samples_ddim = model.module.decode_first_stage(samples_ddim)
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        for i, out in enumerate(x_samples_ddim):
            out_pil = to_pil(out.cpu())
            out_pil.save(os.path.join(output_img_dir, f"{image_ids[i]}.png"))
            print(f"Saved output image {image_ids[i]}.png to {output_img_dir}")
        cnt += images.size(0)

        break

        if args.batches and batch_idx >= args.batches:
            break

    prompt_path = os.path.join(prompt_dir, "prompts.json")
    with open(prompt_path, "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)
    print(f"Prompts saved to {prompt_path}")

if __name__ == '__main__':
    main()