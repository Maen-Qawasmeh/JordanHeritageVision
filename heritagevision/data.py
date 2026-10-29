"""Dataset reading and consistent image preprocessing.

The component responsible for converting images stored on disk
 into data that a deep learning model can understand.


Original Image --> Resize / Crop / Augmentation --> ToTensor + Normalize --> DataLoader → GPU → Model


"""
import csv
from pathlib import Path

from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(training: bool, image_size: int = 224):
    """
    Use augmentation only for training; deterministic preprocessing otherwise.

    It selects a random crop of the image, with an area between
    75% and 100% of the original image's area, and then resizes it to 224×224.

    """
    if training:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(round(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class LandmarkDataset(Dataset):
    """Images remain on disk; load each image only when needed."""
    def __init__(self, manifest: str | Path, data_dir: str | Path,
                 class_names: list[str], training: bool = False,
                 image_size: int = 224):
        self.data_dir = Path(data_dir)
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.transform = build_transform(training, image_size)
        with open(manifest, newline='', encoding='utf-8') as fh:
            self.samples = list(csv.DictReader(fh))
        if not self.samples:
            raise ValueError(f'No images in {manifest}. Run prepare_data.py first.')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        record = self.samples[index]
        image_path = self.data_dir / Path(record['rel_path'])
        with Image.open(image_path) as img:
            # EXIF orientation fixes rotated phone photos.
            image = ImageOps.exif_transpose(img).convert('RGB')
            tensor = self.transform(image)
        return tensor, self.class_to_idx[record['label']]
