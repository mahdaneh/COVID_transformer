
import numpy as np
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
import albumentations as A

LBL_NAMES = {'Covid':0,'Normal':1, 'Lung_Opacity':2}
class CovidDataset(Dataset):
    def __init__(self, root_data_path:str, image_size , training)->None:
        super().__init__()
        self.image_size = tuple(image_size)
        self.root_data_path = Path(root_data_path)
        self.training = training

        self.imgPath_lbl = [(image_path,label) for name,label in LBL_NAMES.items()
                            for image_path in self.root_data_path.joinpath(name).rglob('*.png')]
        self.labels = np.array([d[1] for d in self.imgPath_lbl])


    def __len__(self):

        return len(self.imgPath_lbl)

    def __getitem__(self, idx):
        image_path = self.imgPath_lbl[idx][0]
        label = self.imgPath_lbl[idx][1]
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)


        if self.training:

            transform = A.Compose([
                A.RandomCropFromBorders(),
                A.HorizontalFlip(p=0.5),
                A.Resize(self.image_size[0],self.image_size[1]),

                A.ToTensorV2(),
            ])


        else:
            transform = v2.Compose([
                A.Resize(self.image_size[0],self.image_size[1]),
                A.ToTensorV2(),

            ])
        transformed_image = transform(image=image)['image']/255.

        return transformed_image, torch.tensor(label, dtype=torch.uint8)







