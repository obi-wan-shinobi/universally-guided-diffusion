import os
import hashlib
from datasets import load_dataset
from torch.utils.data import Dataset

class MSCOCO1kDataset(Dataset):
    """
    A PyTorch Dataset for the MSCOCO-1k dataset from Hugging Face.
    Each sample is a tuple (image, caption).

    Args:
        split (str): Which split to load. Default is 'test'.
        image_transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g., torchvision.transforms.
        text_transform (callable, optional): A function/transform that takes in a caption string
            and returns a processed version (e.g., tokenized tensor).
    """
    def __init__(self, split='train', image_transform=None, text_transform=None):
        super().__init__()
        # Load the specified split of MSCOCO-1k from Hugging Face
        # Available splits: 'train', 'validation', 'test' (if provided by the dataset)
        self.dataset = load_dataset("JotDe/mscoco_1k", split=split)

        # Optional transforms for image and text
        self.image_transform = image_transform
        self.text_transform = text_transform

        self.unique_images = []
        self.unique_captions = []
        self.hash_to_index = {}
        self.image_ids = []

        self._deduplicate_and_select_captions()
    
    def _deduplicate_and_select_captions(self):
        for item in self.dataset:
            pil_image = item["image"]
            caption = item["text"]  

            img_rgb = pil_image.convert("RGB")
            img_bytes = img_rgb.tobytes()
            img_hash = hashlib.md5(img_bytes).hexdigest()
            
            if img_hash not in self.hash_to_index:
                index = len(self.unique_images)
                self.hash_to_index[img_hash] = index
                self.unique_images.append(img_rgb)
                self.unique_captions.append(caption)
                self.image_ids.append(f"{index:04d}") 
            else:
                # Already seen this image: check if this caption is longer
                index = self.hash_to_index[img_hash]
                existing_caption = self.unique_captions[index]
                if len(caption) > len(existing_caption):
                    self.unique_captions[index] = caption

    def __len__(self):
        return len(self.unique_images)

    def __getitem__(self, idx):
        image = self.unique_images[idx]
        caption = self.unique_captions[idx]
        image_id = self.image_ids[idx]
        # select
        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.text_transform is not None:
            caption = self.text_transform(caption)

        return image, caption, image_id

if __name__ == "__main__":
    import torch
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from torchvision.transforms import ToPILImage

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    image_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                            std=std),
    ])

    def denormalize(tensor, mean, std):
        mean = torch.tensor(mean).view(-1, 1, 1)
        std = torch.tensor(std).view(-1, 1, 1)
        return tensor * std + mean

    dataset = MSCOCO1kDataset(
        split='train',                
        image_transform=image_transforms,
        text_transform=None 
    )   

    print(f"Number of unique images in dataset: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4, 
    )
    to_pil = ToPILImage() 
    for batch_idx, (images, captions) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print("  Images tensor size:", images.size())
        print("  First caption in batch:", captions[0])
        img = images[0].cpu()
        img = denormalize(img, mean, std)
        img = torch.clamp(img, 0, 1)
        pil_img = to_pil(img) 
        pil_img.save(f"sample_image_{batch_idx}.png")
        break