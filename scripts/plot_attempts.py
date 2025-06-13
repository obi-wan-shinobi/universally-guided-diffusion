from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# --- PATHS ---
experiments_dir = Path("experiments")  # (typo stays, deal with it)
attempts_dir = experiments_dir / "attempts"
results_dir = experiments_dir / "results" / "visualization"
results_dir.mkdir(parents=True, exist_ok=True)

# --- GET ALL IMAGES ---
image_files = list(attempts_dir.glob("*.png"))
if not image_files:
    raise FileNotFoundError("I'M DONE WITH YOUR MISSING FILES.")


# --- EXTRACT PARAMS (NOW WITH *3* FORMATS) ---
def extract_params(filename):
    stem = filename.stem
    if "attempt" in stem:  # attempt-k3-steps10
        parts = stem.split("-")
        k = int(parts[1][1:])
        T = int(parts[2][5:])
    elif "k1" in stem:  # k1-10steps
        parts = stem.split("-")
        k = int(parts[0][1:])
        T = int(parts[1].replace("steps", ""))
    else:  # 50-steps
        T = int(stem.split("-")[0])
        k = None  # No k value
    return k, T


params = [extract_params(f) for f in image_files]

# --- PLOT 1: MAIN GRID (k3, k10) ---
main_k_values = sorted({k for k, T in params if k is not None and k != 1})
all_T_values = sorted({T for k, T in params})

if main_k_values:
    fig1, axes1 = plt.subplots(
        len(main_k_values),
        len(all_T_values),
        figsize=(12, 8),
        gridspec_kw={"wspace": 0.1, "hspace": 0.3},
    )

    for i, k in enumerate(main_k_values):
        axes1[i, 0].text(
            -0.1,
            0.5,
            f"k = {k}",
            fontsize=12,
            ha="right",
            va="center",
            transform=axes1[i, 0].transAxes,
        )

    for j, T in enumerate(all_T_values):
        axes1[0, j].set_title(f"T = {T}", fontsize=12)

    for (k, T), img_file in zip(params, image_files):
        if k in main_k_values:
            i = main_k_values.index(k)
            j = all_T_values.index(T)
            img = np.array(Image.open(img_file))
            axes1[i, j].imshow(img)
            axes1[i, j].axis("off")

    plt.tight_layout()
    plt.savefig(results_dir / "main_grid.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved main_grid.png")

# --- PLOT 2: k=1 ROW ---
k1_files = [f for f in image_files if extract_params(f)[0] == 1]
if k1_files:
    k1_T_values = sorted({extract_params(f)[1] for f in k1_files})
    fig2 = plt.figure(figsize=(12, 4))
    gs = fig2.add_gridspec(1, len(k1_T_values), left=0.15)

    plt.figtext(0.05, 0.5, "k = 1", fontsize=12, ha="center", va="center")

    for j, T in enumerate(k1_T_values):
        ax = fig2.add_subplot(gs[0, j])
        img = np.array(Image.open(k1_files[j]))
        ax.imshow(img)
        ax.set_title(f"T = {T}", fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(results_dir / "k1_row.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved k1_row.png")

# --- PLOT 3: STEPS-ONLY ROW ---
steps_files = [f for f in image_files if extract_params(f)[0] is None]
if steps_files:
    steps_T = sorted({extract_params(f)[1] for f in steps_files})
    fig3, axes3 = plt.subplots(1, len(steps_T), figsize=(12, 4))

    for j, T in enumerate(steps_T):
        img = np.array(Image.open(steps_files[j]))
        axes3[j].imshow(img)
        axes3[j].set_title(f"T = {T}", fontsize=12)
        axes3[j].axis("off")

    plt.tight_layout()
    plt.savefig(results_dir / "steps_row.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved steps_row.png")
