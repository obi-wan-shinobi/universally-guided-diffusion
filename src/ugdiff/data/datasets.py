import os
import urllib.request
import zipfile
from pathlib import Path

import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


class COCODataset(Dataset):
    def __init__(self, image_dir=Path("coco/val2017"), transform=None):
        self.image_dir = image_dir
        self.transform = transform or T.Compose([T.Resize((512, 512)), T.ToTensor()])

        self._ensure_dataset()

        self.filenames = sorted(
            [fname for fname in os.listdir(image_dir) if fname.endswith(".jpg")]
        )

    def __len__(self):
        return len(self.filenames)

    def _ensure_dataset(self):
        if os.path.exists(self.image_dir):
            return

        print(f"[*] COCO images not found at '{self.image_dir}'. Downloading...")
        os.makedirs(os.path.dirname(self.image_dir), exist_ok=True)

        coco_zip_url = "http://images.cocodataset.org/zips/val2017.zip"
        zip_path = os.path.join(os.path.dirname(self.image_dir), "val2017.zip")

        if not os.path.exists(zip_path):
            self._download_with_progress(coco_zip_url, zip_path)
            print(f"Downloaded {zip_path}")

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(os.path.dirname(self.image_dir))
            print(f"Extracted to {self.image_dir}")

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

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        path = os.path.join(self.image_dir, fname)
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, fname


if __name__ == "__main__":
    import torchvision.transforms.functional as F
    from torch.utils.data import DataLoader

    dataset = COCODataset()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for image, fname in loader:
        img = F.to_pil_image(image.squeeze(0))
        print(f"Filename: {fname[0]}, Shape: {image.shape}")
        img.show()
        break
