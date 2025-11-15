import numpy as np
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
import albumentations as A


class CovidDataset(Dataset):
    def __init__(self, data_dict, image_size, training=True) -> None:
        super().__init__()
        self.root_data_path = Path(data_dict["data dir"])
        self._label_names = data_dict["label names"]
        self.image_size = tuple(data_dict["image size"][1:])
        self.training = training
        self.imgPath_lbl = []

    def __len__(self):
        return len(self.imgPath_lbl)

    def __getitem__(self, idx):
        image_path = self.imgPath_lbl[idx][0]
        label = self.imgPath_lbl[idx][1]
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        if self.training:
            transform = A.Compose(
                [
                    A.RandomCropFromBorders(),
                    A.HorizontalFlip(p=0.5),
                    A.Resize(self.image_size[0], self.image_size[1]),
                    A.Normalize(),
                    A.ToTensorV2(),
                ]
            )

        else:
            transform = A.Compose(
                [
                    A.Resize(self.image_size[0], self.image_size[1]),
                    A.Normalize(),
                    A.ToTensorV2(),
                ]
            )
        transformed_image = transform(image=image)["image"]

        return transformed_image, torch.tensor(label, dtype=torch.uint8)


class Covid_Xray(CovidDataset):
    def __init__(self, data_dict, image_size, training) -> None:
        super().__init__(data_dict, image_size, training)
        self.imgPath_lbl = [
            (image_path, label)
            for label, name in enumerate(self._label_names)
            for image_path in self.root_data_path.joinpath(name).rglob("*.png")
        ]


class Covid_QU_Ex(CovidDataset):
    def __init__(self, data_dict, training, mode="train") -> None:
        super().__init__(data_dict, training)

        if mode == "train":
            self.root_data_path = self.root_data_path.joinpath("Train")
        elif mode == "val":
            self.root_data_path = self.root_data_path.joinpath("Val")
        elif mode == "test":
            self.root_data_path = self.root_data_path.joinpath("Test")
        else:
            raise ValueError("mode must be train or val or test")

        self.imgPath_lbl = [
            (image_path, label)
            for label, name in enumerate(self._label_names)
            for image_path in self.root_data_path.joinpath(name, "images").rglob(
                "*.png"
            )
        ]
