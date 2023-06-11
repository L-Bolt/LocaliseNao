import os
import torch
from torch.utils.data import Dataset
from pysolotools.consumers import Solo
from PIL import Image
import itertools
import json

class CustomData(Dataset):
    def __init__(self, data_dir, transform=None, save=False):
        self.datadir = data_dir
        self.dataset = Solo(data_path=data_dir)
        self.img_filename = "step1.camera.png"
        self.transform = transform
        self.captures = self.dataset.metadata.totalSequences
        self.loaded = False

        if os.path.isfile(f"Dataset_metadata{self.captures}.json"):
            with open(f"Dataset_metadata{self.captures}.json") as json_file:
                data = json.load(json_file)

                self.positions = data["positions"]
                print("Loaded filenames and positions from JSON file.")
                self.loaded = True
        else:
            print("Did not find JSON file, generating metadata...")
            self.positions = [capture.captures[0].position for capture in itertools.islice(self.dataset.frames(), 1, None, 2)]

        # Save filenames and positions to a JSON file if the save parameter is
        # set to true.
        if save and not self.loaded:
            data = {"positions": self.positions}
            with open(f"Dataset_metadata{self.captures}.json", "w") as f:
                json.dump(data, f)
                print("Saved filenames and positions to JSON file.")

        print(f"finished setting up dataset\nSamples in dataset: {self.captures}")

    def __getitem__(self, index):
        position = self.positions[index]
        filename = os.path.join(self.datadir + f"/sequence.{index}", self.img_filename)
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
