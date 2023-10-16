import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset

from utlis import rle_decode


# 数据读取
class build_dataset(Dataset):
    def __init__(self, df, train=True, transforms=None) -> None:
        self.train = train
        self.df = df
        self.transforms = transforms
        self.image_path = df["image_path"].values.tolist()
        self.mask = df["mask"].values.tolist()

        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.image_path)

    def __getitem__(self, idx):
        img_path = self.image_path[idx]

        image = cv2.imread(img_path)

        if self.train:
            mask = self.mask[idx]
            mask = rle_decode(self.mask[idx])
            mask = np.array(mask)

            if self.transforms:
                image = self.transforms(image)
                mask = self.transforms(mask)

            return image, mask

        else:
            if self.transforms:
                image = self.transforms(image)

            return image


def build_dataloader(df, fold, data_transforms):
    train_df = df[df["fold"] != fold]
    valid_df = df[df["fold"] == fold]
    train_dataset = build_dataset(
        train_df, train=True, transforms=data_transforms["train"]
    )
    valid_dataset = build_dataset(
        valid_df, train=True, transforms=data_transforms["valid_test"]
    )
    data_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    valid_data_loader = DataLoader(
        valid_dataset, batch_size=8, shuffle=False, num_workers=4
    )
    return data_loader, valid_data_loader
