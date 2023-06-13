import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import itertools
import json
from typing import Dict
from pysolotools.core import BoundingBox2DAnnotationDefinition, DatasetMetadata
from pysolotools.core.models import (
    BoundingBox3DAnnotationDefinition,
    DatasetAnnotations,
    Frame,
    InstanceSegmentationAnnotationDefinition,
)
import glob
import logging
import time

logger = logging.getLogger(__name__)

class CustomFramesIterator:
    SENSORS = [
        {
            "sensor": "type.unity.com/unity.solo.RGBCamera"
            # "annotations": [
            #     BoundingBox2DAnnotation,
            #     BoundingBox3DAnnotation,
            #     InstanceSegmentationAnnotation,
            #     SemanticSegmentationAnnotation,
            # ],
        }
    ]

    def __init__(
        self,
        data_path: str,
        metadata: DatasetMetadata,
        start: int = 0,
        end: int = None,
    ):
        """
        Constructor for an Iterator that loads a Solo Frame from a Solo Dataset.

        Args:
            data_path (str): Path to dataset. This should have all sequences.
            metadata (DatasetMetadata): DatasetMetadata object
            start (int): Start sequence
            end (int): End sequence


        """
        super().__init__()
        self.frame_pool = list()
        self.data_path = os.path.normpath(data_path)
        self.frame_idx = start
        pre = time.time()
        self.metadata = metadata
        logger.info("DONE (t={:0.5f}s)".format(time.time() - pre))

        self.total_frames = self.metadata.totalFrames
        self.total_sequences = self.metadata.totalSequences
        self.steps_per_sequence = int(self.total_frames / self.total_sequences)

        self.end = end or self.__len__()

    def parse_frame(self, f_path: str) -> Frame:
        """
        Parses a json file to a pysolo Frame model.

        Args:
            f_path (str): Path to a step in a sequence for a frame.

        Returns:
            Frame:

        """
        with open(f_path, "r") as f:
            frame = Frame.from_json(f.read())
            return frame

    def __iter__(self):
        self.frame_idx = 0
        return self

    def __next__(self):
        if self.frame_idx >= self.end:
            raise StopIteration
        return self.__load_frame__(self.frame_idx)

    def __len__(self):
        return self.total_frames

    def __load_frame__(self, frame_id: int) -> Frame:
        sequence = int(frame_id / self.steps_per_sequence)
        step = frame_id % self.steps_per_sequence
        self.sequence_path = os.path.join(f"{self.data_path}", f"sequence.{sequence}")
        filename_pattern = os.path.join(f"{self.sequence_path}", f"step{step}.frame_data.json")
        # files = glob.glob(filename_pattern)
        # There should be exactly 1 frame_data for a particular sequence.
        # if len(files) != 1:
        #     raise Exception(f"Metadata file not found for sequence {sequence}")
        self.frame_idx += 1
        return self.parse_frame(filename_pattern)

class Solo:
    def __init__(
        self,
        data_path: str,
        metadata_file: str = None,
        annotation_definitions_file: str = None,
        start: int = 0,
        end: int = None,
        **kwargs
    ):
        """
        Constructor for Unity SOLO helper class.
        Args:
            data_path (str): Location for the root folder of Solo dataset
            metadata_file (str): Location for the metadata.json file for the Solo dataset
            start (Optional[int]): Start index for frames in the dataset
            end (Optional[int]): End index for frames in the dataset
        """

        self.data_path = data_path
        self.start = start
        self.end = end

        self.metadata = self.__open_metadata__(metadata_file)
        self.annotation_definitions = self.__open_annotation_definitions__(
            annotation_definitions_file
        )

    def frames(self) -> CustomFramesIterator:
        """
        Return a Frames Iterator

        Returns:
            FramesIterator
        """
        return CustomFramesIterator(
            self.data_path,
            self.metadata,
            self.start,
            self.end,
        )

    def categories(self) -> Dict[int, str]:
        categories = {}
        for d in self.annotation_definitions.annotationDefinitions:
            if isinstance(d, BoundingBox2DAnnotationDefinition):
                for s in d.spec:
                    categories[s.label_id] = s.label_name
                return categories
            elif isinstance(d, BoundingBox3DAnnotationDefinition):
                for s in d.spec:
                    categories[s.label_id] = s.label_name
                return categories
            elif isinstance(d, InstanceSegmentationAnnotationDefinition):
                for s in d.spec:
                    categories[s.label_id] = s.label_name
                return categories
        return None

    def frame_ids(self):
        return list(map(lambda f: f.frame, self.frames()))

    def get_metadata(self) -> DatasetMetadata:
        """
        Get metadata for the Solo dataset

        Returns:
            DatasetMetadata: Returns metadata of SOLO Dataset
        """
        return self.metadata

    def get_annotation_definitions(self) -> DatasetAnnotations:
        """
        Get annotation definitions for the Solo dataset

        Returns:
            DatasetAnnotations

        """
        return self.annotation_definitions

    def __open_metadata__(self, metadata_file: str = None) -> DatasetMetadata:
        """
        Default metadata location is expected at root/metadata.json but
        if an metadata_file path is provided that is used as the metadata file path.

        Metadata can be in one of two locations, depending if it was a part of a singular build,
        or if it was a part of a distributed build.

        Args:
            metadata_file (str): Path to solo annotation file

        Returns:
            DatasetMetadata

        """
        if metadata_file:
            discovered_path = [metadata_file]
        else:
            discovered_path = glob.glob(
                self.data_path + "/metadata.json", recursive=True
            )
            if len(discovered_path) != 1:
                raise Exception("Found none or multiple metadata files.")

        with open(discovered_path[0]) as metadata_f:
            return DatasetMetadata.from_json(metadata_f.read())

    def __open_annotation_definitions__(
        self, annotation_definitions_file: str = None
    ) -> DatasetAnnotations:
        """
        Default annotation_definitions.json is expected in the root folder of the Solo dataset. If a custom
        `annotation_definitions_file` is provided then that is used instead.

        Args:
            annotation_definitions_file (str): Custom path for annotation_definitions.json file

        Returns:
            DatasetAnnotations

        """
        if annotation_definitions_file:
            discovered_path = [annotation_definitions_file]
        else:
            discovered_path = glob.glob(
                self.data_path + "/annotation_definitions.json", recursive=True
            )
            if len(discovered_path) != 1:
                raise Exception("Found none or multiple annotation definition files.")
        with open(discovered_path[0]) as metadata_f:
            return DatasetAnnotations.from_json(metadata_f.read())


class CustomData(Dataset):
    def __init__(self, data_dir, transform=None, save=False):
        self.datadir = data_dir
        self.dataset = Solo(data_path=data_dir)
        self.img_filename = "step1.camera.png"
        self.transform = transform
        self.captures = self.dataset.metadata.totalSequences
        self.loaded = False

        if os.path.isfile(os.path.join("dataset_dumps", f"Dataset_metadata{self.captures}.json")):
            with open(os.path.join("dataset_dumps", f"Dataset_metadata{self.captures}.json")) as json_file:
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
            with open(os.path.join("dataset_dumps", f"Dataset_metadata{self.captures}.json"), "w") as f:
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


if __name__ == "__main__":
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    test = CustomData("S:\datasets\solo250kv2", transform=transform, save=True)
    print(test.__getitem__(0))