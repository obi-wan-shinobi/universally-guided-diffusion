import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


def load_metrics_with_pandas(csv_path):
    df = pd.read_csv(csv_path)
    metrics = (
        df[~df["Image"].str.contains("Average")]
        .set_index(["Image", "Prompt Folder"])
        .to_dict("index")
    )
    return metrics


def get_prompt_text(prompt_folder):
    with open(prompt_folder / "prompt.txt", "r", encoding="utf-8") as file:
        prompt = file.readline().strip()
    return prompt


def add_metrics_annotations(axes, metrics):
    """Add metrics annotations to each generated image cell"""
    for i in range(num_images):
        # Add metrics to prompt-generated images (columns 2+)
        for j, prompt_folder in enumerate(prompt_folders, start=2):
            key = (f"Image {i}", prompt_folder)
            if key in metrics:
                # Create annotation text with both metrics
                acc = metrics[key]["Pixel Accuracy"]
                iou = metrics[key]["IoU"]
                text = f"Acc: {acc:.3f}\nIoU: {iou:.3f}"

                # Add annotation at bottom center of image
                axes[i, j].text(
                    0.5,
                    0.02,  # x, y position (relative to axes)
                    text,
                    transform=axes[i, j].transAxes,  # Use axes coordinates
                    ha="center",
                    va="bottom",  # Alignment
                    color="white",
                    bbox=dict(
                        facecolor="black",
                        alpha=0.7,  # Semi-transparent
                        pad=1,  # Padding
                        boxstyle="round",  # Rounded corners
                    ),
                    fontsize=8,
                )


# Folder structure
experiments_folder = Path("experiments")
base_folder = experiments_folder / "analysis-data"
output_folder = experiments_folder / "results" / "visualization"
output_folder.mkdir(parents=True, exist_ok=True)

prompt_folders = ["prompt-1", "prompt-2", "prompt-3", "prompt-4"]
num_images = 3

csv_path = experiments_folder / "results" / "segmentation_accuracy_results.csv"
# Get and wrap prompt texts
prompt_texts = [
    get_prompt_text(experiments_folder / prompt_folder)
    for prompt_folder in prompt_folders
]
wrapped_prompts = [
    "\n".join(textwrap.wrap(text, width=20)) for text in prompt_texts
]  # Adjust width as needed

# Column titles
column_titles = ["Original", "Segmentation Map"] + wrapped_prompts

# Create figure with adjusted spacing
fig, axes = plt.subplots(
    num_images,
    2 + len(prompt_folders),
    figsize=(20, 8),  # Reduced height
    gridspec_kw={
        "wspace": 0.00,
        "hspace": 0.00,
        "width_ratios": [1] * (2 + len(prompt_folders)),
    },  # Minimize gaps
)

# Set column titles with smaller font and better positioning
for ax, col_title in zip(axes[0], column_titles):
    ax.set_title(col_title, y=1.1, fontsize=10, pad=2)  # Smaller font, less padding

for i in range(num_images):
    # Load original image
    og_path = base_folder / f"og_img_{i}.png"
    axes[i, 0].imshow(Image.open(og_path))
    axes[i, 0].axis("off")

    # Load label image
    label_path = base_folder / f"label_{i}.png"
    axes[i, 1].imshow(Image.open(label_path))
    axes[i, 1].axis("off")

    # Load prompt images
    for j, prompt_folder in enumerate(prompt_folders, start=2):
        prompt_path = base_folder / prompt_folder / f"new_img_{i}_0.png"
        try:
            axes[i, j].imshow(Image.open(prompt_path))
        except FileNotFoundError:
            print(f"Warning: Could not find {prompt_path}")
        axes[i, j].axis("off")

metrics = load_metrics_with_pandas(csv_path)

add_metrics = False
if add_metrics:
    add_metrics_annotations(axes, metrics)
    output_path = output_folder / "visualization_with_metrics.png"
else:
    output_path = output_folder / "visualization.png"

plt.subplots_adjust(top=0.85, bottom=0.01, left=0.01, right=0.99)  # Tight layout
# plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
plt.show()
