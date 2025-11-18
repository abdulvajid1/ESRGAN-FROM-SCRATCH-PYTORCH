from torch.utils.data import Dataset, DataLoader
import numpy as np
import albumentations as A
from pathlib import Path
from config import IMG_SIZE
import config
from PIL import Image
import cv2
from albumentations.pytorch import ToTensorV2

both_transform = A.Compose([
    A.RandomCrop(height=IMG_SIZE, width=IMG_SIZE),
    A.HorizontalFlip(p=0.5),
])

highres_transform = A.Compose([
    A.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ToTensorV2()
])

lowres_transform = A.Compose([
    A.Resize(height=IMG_SIZE//4, width=IMG_SIZE//4, interpolation=cv2.INTER_CUBIC),
    A.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ToTensorV2()
])


class CustomDataset(Dataset):
    def __init__(self, data_dir="data"):
        self.images = list(Path(data_dir).glob("*"))

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        image = np.array(image)
        image = both_transform(image=image)["image"]
        highres_image = highres_transform(image=image)["image"]
        lowres_image = lowres_transform(image=image)["image"]
        return highres_image, lowres_image
    
def get_dataloader(batch_size=config.BATCH_SIZE, num_workers=config.DATALOADER_WORKERS, pin_memory=config.PIN_MEMORY):
    return DataLoader(CustomDataset(), batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers)
    

if __name__ == "__main__":
    arr = np.random.randint(low=0, high=255, size=(512, 512, 3)).astype(np.uint8)
    print(lowres_transform(image=arr)['image'].shape)
    print(highres_transform(image=arr)['image'].shape)
    print(both_transform(image=arr)['image'].shape)
    print(lowres_transform(image=arr)['image'].min())
    print(lowres_transform(image=arr)['image'].max())

    dataloader = get_dataloader()
    for high, low in dataloader:
        break

    print(high.shape)
    print(low.shape)