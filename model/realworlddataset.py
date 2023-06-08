import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

class RealWorldData(Dataset):
    def __init__(self, data_dir, transform=None):
        self.datadir = data_dir
        self.transform = transform

        directories = [os.path.join(self.datadir, file) for file in os.listdir(self.datadir)
                    if os.path.isdir(os.path.join(self.datadir, file))]
        if not os.name == "nt":
            directories.sort(key = lambda x: int(x.split("/")[-1]))
            self.image_paths = [os.path.join(x, f"{x.split('/')[-1]}.jpg") for x in directories]
            self.textfiles = [os.path.join(x, f"{x.split('/')[-1]}.txt") for x in directories]
        else:
            directories.sort(key = lambda x: int(os.path.basename(x)))
            self.image_paths = [os.path.join(x, f'{os.path.basename(x)}.jpg') for x in directories]
            self.textfiles = [os.path.join(x, f'{os.path.basename(x)}.txt') for x in directories]

    def __getitem__(self, index):
        image =  Image.open(self.image_paths[index]).convert("RGB")
        with open(self.textfiles[index]) as f:
            label = f.readline()
            f.close

        if self.transform:
            image = self.transform(image)

        return (image, torch.tensor(int(label)))

    def __len__(self):
        return len(self.image_paths)
