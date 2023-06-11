import os
import torch
from torch.utils.data import Dataset
from pysolotools.consumers import Solo
from PIL import Image
import itertools

class CustomData(Dataset):
    def __init__(self, data_dir, transform=None):
        self.datadir = data_dir
        self.dataset = Solo(data_path=data_dir)
        self.dataset_iterator = [frame for frame in self.dataset.frames()]
        # self.dataset_iterator = list( self.dataset.frames())
        self.img_filename = "step1.camera.png"
        self.transform = transform
        self.captures = self.dataset.metadata.totalSequences

        print(f"finished setting up dataset\nSamples in dataset: {self.captures}")

    def __getitem__(self, index):
        filename = os.path.join(self.datadir + f"/sequence.{index}", self.img_filename)
        # capture = next(itertools.islice(self.dataset_iterator, (index * 2) + 1, None))
        capture = self.dataset_iterator[ (index * 2) + 1]
        position = capture.captures[0].position

        image = Image.open(filename).convert("RGB")

        if self.transform:
            image = self.transform(image)

        def get_class(position):
            if position[0] > 90:
                position[0] = 90

            if position[2] > 60:
                position[2] = 60

            if position[0] < 0:
                position[0] = 0

            if position[2] < 0:
                position[2] = 0

            return round(position[0]) + (91 * round(position[2]))

        return (image, torch.tensor(int(get_class(position))))

    def __len__(self):
        return self.captures
