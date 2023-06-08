import os
import torch
from torch.utils.data import Dataset
from pysolotools.consumers import Solo
import torchvision.transforms as transforms
from PIL import Image
import json

class CustomData(Dataset):
    def __init__(self, data_dir, transform=None, save=False):
        loaded = False
        self.dataset = Solo(data_path=data_dir)
        self.img_filename = "step1.camera.png"
        self.transform = transform

        self.captures = self.dataset.metadata.totalSequences

        if os.path.isfile(f"Dataset_metadata{self.captures}.json"):
            with open(f"Dataset_metadata{self.captures}.json") as json_file:
                data = json.load(json_file)

                self.filenames = data["filenames"]
                self.positions = data["positions"]
                print("Loaded filenames and positions from JSON file.")
                loaded = True
        else:
            print("Did not find JSON file, generating metadata...")
            self.filenames = [os.path.join((data_dir + f"/sequence.{i}"), self.img_filename) for i in range(self.captures)]
            self.positions = [capture[0].position for capture in [frame.captures for frame in self.dataset.frames()] if len(capture) > 0]

        # Save filenames and positions to a JSON file if the save parameter is
        # set to true.
        if save and not loaded:
            data = {"filenames": self.filenames, "positions": self.positions}
            with open(f"Dataset_metadata{self.captures}.json", "w") as f:
                json.dump(data, f)
                print("Saved filenames and positions to JSON file.")

        print(f"finished setting up dataset\nItems in dataset: {self.captures}")

    def __getitem__(self, index):
        image = Image.open(self.filenames[index]).convert("RGB")

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

        return (image, torch.tensor(int(get_class(self.positions[index]))))

    def __len__(self):
        return self.captures


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    test = CustomData("S:\datasets\solo10kv2", transform=transform, save=True)
