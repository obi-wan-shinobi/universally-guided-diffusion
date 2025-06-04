import torch
import torch.functional as F
from torch.utils.data import DataLoader

from ugdiff.data.datasets import COCODataset
from ugdiff.guidance.segmentation_guidance import SegmentationGuidance
from ugdiff.inference.pipeline import SegmentationGuidedDiffusionPipeline
from ugdiff.models.stable_diffusion import (
    create_LMS_scheduler,
    create_PNDM_scheduler,
    load_text_encoder,
    load_unet,
    load_vae,
)
from ugdiff.utils import latents_to_pil

torch_device = "mps"  # Or "cuda" / "cpu"
target_class_id = 12  # Dog in COCO/VOC

vae = load_vae(torch_device)
tokenizer, text_encoder = load_text_encoder(torch_device)
unet = load_unet(torch_device)
scheduler = create_LMS_scheduler()
# scheduler = create_PNDM_scheduler()

# dataset = COCODataset()
# loader = DataLoader(dataset, batch_size=1, shuffle=False)
dataset = COCODataset(image_dir="segmentation-data/Walker", use_annotations=False)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

segmentation_module = SegmentationGuidance(vae, device=torch_device)

multi_class = True
target_masks_list = []
images = []

for image_tensor, _ in loader:
    image_tensor = image_tensor.to(torch_device)

    with torch.no_grad():
        probs = segmentation_module.segment_tensor(image_tensor)  # (B, C, H, W)

    if multi_class:
        # Full multi-class prediction
        predicted_class = probs.argmax(dim=1)  # (B, H, W)
        target_mask = predicted_class.unsqueeze(1).float()  # (B, 1, H, W)
        image = segmentation_module.segmentation_map_to_image(
            predicted_class.squeeze(0)
        )
    else:
        # Binary mask for a specific target class
        predicted_class = probs.argmax(dim=1).squeeze(0)  # (H, W)
        if (predicted_class == target_class_id).any():
            target_mask = (
                (predicted_class == target_class_id).float().unsqueeze(0).unsqueeze(0)
            )  # (1, 1, H, W)
            image = segmentation_module.segmentation_map_to_image(
                target_mask.squeeze(0).squeeze(0).long()
            )

            # target_mask = F.one_hot(target_mask.squeeze(1), num_classes=21)
            # target_mask = target_mask.permute(0, 3, 1, 2).float()
        else:
            continue  # Or raise error if needed

    images.append(image)
    target_masks_list.append(target_mask)

if not target_masks_list:
    raise RuntimeError("No image with target class found in dataset.")

target_masks = torch.cat(target_masks_list, dim=0)[1].unsqueeze(0)

print(f"{target_masks.shape=}")

for i, img in enumerate(images):
    img.save(f"tmp/target_map_{i}.png")

pipeline = SegmentationGuidedDiffusionPipeline(
    vae=vae,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    unet=unet,
    scheduler=scheduler,
    device=torch_device,
    guidance_module=segmentation_module,
)

prompt = ["Walker hound, Walker foxhound under water."]
generated_tensor = pipeline.generate(
    prompt=prompt,
    height=512,
    width=512,
    num_inference_steps=100,
    seed=32,
    # segmentation_maps=target_masks,
    # class_id=target_class_id,
)

pil_images = latents_to_pil(generated_tensor)
for i, img in enumerate(pil_images):
    img.save(f"tmp/segmented_sample_{i}.png")
