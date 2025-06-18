import sys
import torch
import clip
from PIL import Image
import os
import matplotlib.pyplot as plt

def main():
    # Hardcoded directories and prompt file
    generated_dir = "evaluate/eval_data/generated"
    gt_dir = "evaluate/eval_data/gt-ddim-50"
    prompt_file = "evaluate/prompt.txt"
    res_dir = "evaluate/results"

    # Read prompts
    with open(prompt_file, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]

    # Get subdirectories (assume sorted order matches prompt order)
    subdirs = sorted([d for d in os.listdir(generated_dir) if os.path.isdir(os.path.join(generated_dir, d))])

    assert len(subdirs) == len(prompts), "Number of subdirs and prompts must match"

    device = "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    for idx, subdir in enumerate(subdirs):
        prompt = prompts[idx]
        gen_subdir = os.path.join(generated_dir, subdir)
        gt_subdir = os.path.join(gt_dir, subdir)

        # Get first image from generated
        gen_imgs = sorted([f for f in os.listdir(gen_subdir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not gen_imgs:
            print(f"No images found in {gen_subdir}")
            continue
        gen_img_path = os.path.join(gen_subdir, gen_imgs[0])

        # Get all images from gt
        gt_imgs = sorted([f for f in os.listdir(gt_subdir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        gt_img_paths = [os.path.join(gt_subdir, f) for f in gt_imgs]

        # Load and preprocess images
        images = []
        img_names = []
        img_labels = []
        # First: generated image
        img = Image.open(gen_img_path).convert("RGB")
        images.append(preprocess(img).unsqueeze(0))
        img_names.append(f"generated/{subdir}/{gen_imgs[0]}")
        img_labels.append("universal-guided-diffusion")
        # Then: all gt images
        for gt_path in gt_img_paths:
            img = Image.open(gt_path).convert("RGB")
            images.append(preprocess(img).unsqueeze(0))
            img_names.append(f"gt-ddim-50/{subdir}/{os.path.basename(gt_path)}")
            img_labels.append("conditional stable diffusion")
        images = torch.cat(images, dim=0).to(device)

        # Tokenize text
        text_tokens = clip.tokenize([prompt]).to(device)

        # Compute features and cosine similarity
        with torch.no_grad():
            img_feats = model.encode_image(images)
            txt_feats = model.encode_text(text_tokens)
            img_feats = img_feats / img_feats.norm(dim=1, keepdim=True)
            txt_feats = txt_feats / txt_feats.norm(dim=1, keepdim=True)
            sims = (img_feats @ txt_feats.T).squeeze(1).cpu().numpy()

        # Prepare bar plot data
        # x = list(range(1, len(sims) + 1))
        # legends = {"universal-guided-diffusion": "tab:blue", "conditional stable diffusion": "tab:orange"}

        # plt.figure(figsize=(6, 4))
        # # Plot bars for each group
        # for i, (score, label) in enumerate(zip(sims, img_labels)):
        #     plt.bar(x[i], score, color=legends[label], label=label if i == 0 or (i == 1 and len(sims) > 1) else "")
        #     plt.text(x[i], score + 0.02, f"{score:.3f}", ha='center', va='bottom', fontsize=10)
        # plt.ylim(0, 1)
        # plt.xlabel("Image Index")
        # plt.ylabel("Cosine Similarity")
        # plt.title(f"Prompt {idx+1}")
        # handles = [
        #     plt.Rectangle((0,0),1,1,color=legends["universal-guided-diffusion"]),
        #     plt.Rectangle((0,0),1,1,color=legends["conditional stable diffusion"])
        # ]
        # plt.legend(handles, ["universal-guided-diffusion", "conditional stable diffusion"])
        # plt.tight_layout()
        # plt.savefig(os.path.join(res_dir, f"prompt_{idx+1}.png"))
        # plt.close()

        # Print results
        print(f"Prompt {idx+1}: {prompt}")
        for name, score in zip(img_names, sims):
            print(f"  {name}: {score.item():.4f}")
        print("-" * 40)

if __name__ == "__main__":
    main()