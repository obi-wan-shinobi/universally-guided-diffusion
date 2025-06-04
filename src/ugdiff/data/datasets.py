import os
import urllib.request
import zipfile
from pathlib import Path

import torchvision.transforms as T
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class COCODataset(Dataset):
    def __init__(
        self,
        image_dir=Path("coco/val2017"),
        annotation_file=Path("coco/annotations/instances_val2017.json"),
        category_names=["dog"],
        transform=None,
        use_annotations=True,
    ):
        self.image_dir = Path(image_dir)
        self.annotation_file = Path(annotation_file) if annotation_file else None
        self.transform = transform or T.Compose([T.Resize((512, 512)), T.ToTensor()])
        self.use_annotations = use_annotations
        self.category_names = category_names or []

        if self.use_annotations:
            self._ensure_dataset()
            self._load_annotations()
        else:
            self.filtered_filenames = [
                f
                for f in os.listdir(self.image_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

    def _ensure_dataset(self):
        if not self.image_dir.exists():
            print(f"[*] COCO images not found at '{self.image_dir}'. Downloading...")
            os.makedirs(self.image_dir.parent, exist_ok=True)

            coco_zip_url = "http://images.cocodataset.org/zips/val2017.zip"
            zip_path = self.image_dir.parent / "val2017.zip"

            if not zip_path.exists():
                self._download_with_progress(coco_zip_url, zip_path)
                print(f"Downloaded {zip_path}")

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(self.image_dir.parent)
                print(f"Extracted to {self.image_dir}")

        if not self.annotation_file.exists():
            print(f"[*] COCO annotations not found. Downloading...")
            os.makedirs(self.annotation_file.parent, exist_ok=True)

            anno_url = (
                "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
            )
            zip_path = self.annotation_file.parent / "annotations_trainval2017.zip"

            if not zip_path.exists():
                self._download_with_progress(anno_url, zip_path)
                print(f"Downloaded {zip_path}")

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(self.annotation_file.parent.parent)
                print(f"Extracted annotations to {self.annotation_file.parent.parent}")

    def _download_with_progress(self, url, output_path):
        from tqdm import tqdm

        class DownloadProgressBar(tqdm):
            def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)

        with DownloadProgressBar(
            unit="B", unit_scale=True, miniters=1, desc=os.path.basename(output_path)
        ) as t:
            urllib.request.urlretrieve(
                url, filename=output_path, reporthook=t.update_to
            )

    def _load_annotations(self):
        self.coco = COCO(self.annotation_file)

        # Get category IDs for desired category names
        cat_ids = self.coco.getCatIds(catNms=self.category_names)
        img_ids = self.coco.getImgIds(catIds=cat_ids)
        imgs = self.coco.loadImgs(img_ids)

        self.filtered_filenames = [img["file_name"] for img in imgs]

    def __len__(self):
        return len(self.filtered_filenames)

    def __getitem__(self, idx):
        fname = self.filtered_filenames[idx]
        path = self.image_dir / fname
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, fname


if __name__ == "__main__":
    import torchvision.transforms.functional as F
    from torch.utils.data import DataLoader

    dataset = COCODataset(category_names=["dog"])
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for image, fname in loader:
        img = F.to_pil_image(image.squeeze(0))
        print(f"Filename: {fname[0]}, Shape: {image.shape}")
        img.show()
        break
