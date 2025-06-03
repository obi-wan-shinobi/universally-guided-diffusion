import torch
from torch.utils.data import DataLoader

from ugdiff.data.datasets import COCODataset
from ugdiff.guidance.segmentation_guidance import SegmentationGuidance
from ugdiff.inference.pipeline import SegmentationGuidedDiffusionPipeline
from ugdiff.models.stable_diffusion import (
    create_scheduler,
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
scheduler = create_scheduler()

dataset = COCODataset()
loader = DataLoader(dataset, batch_size=1, shuffle=False)

segmentation_module = SegmentationGuidance(vae, device=torch_device)

target_masks_list = []
images = []
for image_tensor, _ in loader:
    image_tensor = image_tensor.to(torch_device)

    with torch.no_grad():
        probs = segmentation_module.segment_tensor(image_tensor)

    predicted_class = probs.argmax(dim=1).squeeze(0)

    if (predicted_class == target_class_id).any():
        target_mask = (predicted_class == target_class_id).float()
        target_mask = target_mask.unsqueeze(0)
        image = segmentation_module.segmentation_map_to_image(
            target_mask.squeeze(0).long()
        )
        images.append(image)
        target_masks_list.append(target_mask)
        break
else:
    raise RuntimeError("No image with target class found in dataset.")

target_masks = torch.stack(target_masks_list, dim=0)

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

prompt = ["a dog in space with an astronaut"]
generated_tensor = pipeline.generate(
    prompt=prompt,
    height=512,
    width=512,
    num_inference_steps=50,
    seed=42,
    segmentation_maps=target_masks,
    class_id=target_class_id,
)

pil_images = latents_to_pil(generated_tensor)
for i, img in enumerate(pil_images):
    img.save(f"tmp/segmented_sample_{i}.png")
