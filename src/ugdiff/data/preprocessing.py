from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.models.segmentation as seg_models
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import COCODataset

# Refer: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.segmentation.lraspp_mobilenet_v3_large.html

cmap = plt.get_cmap("tab20", 21)
VOC_COLORMAP = (cmap(range(21))[:, :3] * 255).astype(np.uint8)


class SegmentationPreprocessor:
    def __init__(
        self,
        image_dir="coco/val2017",
        output_dir="segmented/val2017",
        batch_size=1,
        device=None,
        skip_existing=True,
    ):
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.skip_existing = skip_existing

        self.transform = self.get_transform()
        self.model = self.load_model()

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_transform(self):
        return T.Compose(
            [
                T.Resize((520, 520)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def load_model(self):
        model = seg_models.lraspp_mobilenet_v3_large(
            weights="LRASPP_MobileNet_V3_Large_Weights.DEFAULT"
        )
        model.eval().to(self.device)
        return model

    def get_dataset(self):
        return COCODataset(image_dir=self.image_dir, transform=self.transform)

    def segment(self, images, filter_classes=None):
        """
        Runs segmentation on input images.

        Args:
            images (Tensor): Input images (B, 3, H, W).
            filter_classes (List[int], optional): If provided, zeroes out all class probabilities
                except those listed.

        Returns:
            Tensor: Segmentation probability maps (B, C, H, W), filtered if requested.
        """
        self.model.eval()
        images = images.to(self.device)
        with torch.no_grad():
            output = self.model(images)["out"]
            probs = torch.softmax(output, dim=1)

        if filter_classes is not None:
            # Zero out non-selected classes
            mask = torch.zeros_like(probs)
            for cls_id in filter_classes:
                mask[:, cls_id, :, :] = probs[:, cls_id, :, :]
            probs = mask

        return probs

    def segmentation_map_to_image(self, class_mask, filter_classes=None):
        """
        Converts a class mask to a color image.

        Args:
            class_mask (Tensor): (H, W) mask of class IDs.
            filter_classes (List[int], optional): If provided, only visualize those classes.

        Returns:
            PIL.Image: Colored segmentation map.
        """
        class_mask = class_mask.cpu()

        if filter_classes is not None:
            # Set everything not in filter_classes to background (0)
            mask = torch.zeros_like(class_mask)
            for cls_id in filter_classes:
                mask[class_mask == cls_id] = cls_id
            class_mask = mask

        colored = VOC_COLORMAP[class_mask.numpy()]
        return Image.fromarray(colored.astype(np.uint8))

    def preprocess(self):
        dataset = self.get_dataset()
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        for images, fnames in tqdm(dataloader, desc="Preprocessing"):
            images = images.to(self.device)

            with torch.no_grad():
                outputs = self.model(images)["out"]
                probs = torch.softmax(outputs, dim=1)

            for i in range(images.size(0)):
                fname = fnames[i]
                base = fname.replace(".jpg", "")

                out_soft = self.output_dir / f"{base}_soft.pt"
                out_mask = self.output_dir / f"{base}_mask.pt"
                out_colored = self.output_dir / f"{base}_seg.png"

                if (
                    self.skip_existing
                    and out_soft.exists()
                    and out_mask.exists()
                    and out_colored.exists()
                ):
                    continue

                prob_map = probs[i].cpu()
                torch.save(prob_map, out_soft)

                class_mask = prob_map.argmax(dim=0).byte()
                torch.save(class_mask, out_mask)

                colored = VOC_COLORMAP[class_mask.numpy()]
                img = Image.fromarray(colored.astype(np.uint8))
                img.save(out_colored)

        print(f"Done. Saved to {self.output_dir}")


if __name__ == "__main__":

    preprocessor = SegmentationPreprocessor(
        image_dir="coco/val2017",
        output_dir="segmented/val2017",
        batch_size=32,
        skip_existing=True,
        device="mps",
    )
    preprocessor.preprocess()
