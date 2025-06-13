import csv
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from ugdiff.guidance.segmentation_guidance import SegmentationGuidance

# Setup device and model
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
guidance = SegmentationGuidance(vae=None, device=device)

# Define paths
experiments_folder = Path("experiments")
base_folder = experiments_folder / "analysis-data"
output_folder = experiments_folder / "results"
output_folder.mkdir(parents=True, exist_ok=True)
csv_path = output_folder / "segmentation_accuracy_results.csv"

prompt_folders = [base_folder / f"prompt-{i}" for i in range(1, 5)]
label_files = [f"og_img_{i}.png" for i in range(3)]
num_images = 3

dog_class_id = 12


def segment_and_get_mask(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = guidance.transform(img).unsqueeze(0).to(device)
    probs = guidance.segment_tensor(img_tensor)
    dog_mask = guidance.extract_class_mask(probs, dog_class_id)
    return (dog_mask > 0.5).float()


# Prepare CSV file
with open(csv_path, mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Image", "Prompt Folder", "Pixel Accuracy", "IoU"])

    all_accuracies = []
    all_ious = []

    for i in range(num_images):
        og_path = base_folder / f"og_img_{i}.png"
        gt_mask = segment_and_get_mask(og_path)

        for folder in prompt_folders:
            gen_path = folder / f"new_img_{i}_0.png"
            pred_mask = segment_and_get_mask(gen_path)

            # Resize to match shapes
            pred_mask = torch.nn.functional.interpolate(
                pred_mask, size=gt_mask.shape[-2:], mode="nearest"
            )

            # Pixel accuracy
            correct = (pred_mask == gt_mask).float().sum()
            total = torch.numel(gt_mask)
            acc = (correct / total).item()
            all_accuracies.append(acc)

            # IoU computation
            intersection = (pred_mask * gt_mask).sum()
            union = ((pred_mask + gt_mask) > 0).float().sum()
            iou = (intersection / union).item() if union > 0 else 1.0
            all_ious.append(iou)

            # Write to CSV
            writer.writerow([f"Image {i}", folder.name, f"{acc:.4f}", f"{iou:.4f}"])
            print(
                f"[Image {i} | {folder.name}] Pixel Accuracy: {acc:.4f}, IoU: {iou:.4f}"
            )

    # Averages
    avg_acc = sum(all_accuracies) / len(all_accuracies)
    avg_iou = sum(all_ious) / len(all_ious)
    writer.writerow(["Average", "All", f"{avg_acc:.4f}", f"{avg_iou:.4f}"])

    print(f"\n[*] Average Pixel Accuracy: {avg_acc:.4f}")
    print(f"[*] Average IoU: {avg_iou:.4f}")

print(f"\n[-] Results saved to: {csv_path}")
