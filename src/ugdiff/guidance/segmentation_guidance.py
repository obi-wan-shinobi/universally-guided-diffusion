import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models.segmentation as seg_models
import torchvision.transforms as T
from matplotlib import pyplot as plt
from PIL import Image

# Refer: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.segmentation.lraspp_mobilenet_v3_large.html

cmap = plt.get_cmap("tab20", 21)
VOC_COLORMAP = (cmap(range(21))[:, :3] * 255).astype(np.uint8)


class SegmentationGuidance:
    def __init__(self, vae, device="cpu"):
        self.device = device
        self.vae = vae
        self.model = self.load_model()
        self.transform = self.get_transform()

    def load_model(self):
        model = seg_models.lraspp_mobilenet_v3_large(
            weights="LRASPP_MobileNet_V3_Large_Weights.DEFAULT"
        )
        model.eval().to(self.device)
        return model

    def get_transform(self):
        return T.Compose(
            [
                T.Resize((520, 520)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def segment_tensor(self, images):
        """
        Run segmentation on input tensor of images.

        Args:
            images (Tensor): (B, 3, H, W)

        Returns:
            probs (Tensor): (B, C, H, W) class probabilities
        """
        images = images.to(self.device)
        output = self.model(images)["out"]
        probs = torch.softmax(output, dim=1)
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
            mask = torch.zeros_like(class_mask)
            for cls_id in filter_classes:
                mask[class_mask == cls_id] = cls_id
            class_mask = mask

        colored = VOC_COLORMAP[class_mask.numpy()]
        return Image.fromarray(colored.astype(np.uint8))

    def extract_class_mask(self, probs, class_id):
        """
        Extract mask for the given class index.

        Args:
            probs (Tensor): (B, C, H, W)
            class_id (int): class index

        Returns:
            mask (Tensor): (B, 1, H, W)
        """
        return probs[:, class_id : class_id + 1, :, :]

    def compute_loss(self, latents, target_masks, class_id, eps=1e-6):
        # Decode latents into images
        images = self.vae.decode(latents).sample  # (B, 3, H, W), range [-1, 1]

        # Normalize images from [-1, 1] to [0, 1]
        images = (images + 1) / 2
        images = images.clamp(0, 1)

        # Resize images to 520x520 using tensor interpolation (compatible with tensor inputs)
        images = F.interpolate(
            images, size=(520, 520), mode="bilinear", align_corners=False
        )

        # Normalize images
        normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        images = normalize(images)

        # Now segment the images and compute loss
        probs = self.segment_tensor(images)
        pred_masks = self.extract_class_mask(probs, class_id)

        target_masks = F.interpolate(target_masks, size=(520, 520), mode="nearest")

        loss = F.binary_cross_entropy(
            pred_masks.clamp(eps, 1 - eps),
            target_masks,
            reduction="none",
        )
        loss = loss.mean(dim=[1, 2, 3])

        return loss
