from typing import Callable, List, NoReturn, Optional, Tuple
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


class TripletDataset(Dataset):
    def __init__(
        self,
        list_path: List[str],
        transform_anchor: Optional[Callable] = None,
        transform_positive: Optional[Callable] = None,
        transform_negative: Optional[Callable] = None,
    ) -> NoReturn:
        """Triplet dataset.

        Args:
            list_path (List[str]): List of image path
            transform_anchor (Optional[Callable], optional): Image transform for anchor. Defaults to None.
            transform_positive (Optional[Callable], optional): Image transform for positive example.  Defaults to None.
            transform_negative (Optional[Callable], optional): Image transform for negative example. Defaults to None.

        Example:
        
            >>> from glob import glob
            >>> train_path = sorted(glob("/kaggle/input/hackathon-isae-2021-patch-retrieval/Train/Train/*.png"))
            >>> dataset = TripletDataset(list_path=train_path)
            
        References:
       
            1. DEEP METRIC LEARNING USING TRIPLET NETWORK: https://arxiv.org/pdf/1412.6622.pdf
        """        
        self.list_path = list_path
        
        if transform_anchor is None:
            self.transform_anchor = transforms.ToTensor()
        else: 
            self.transform_anchor = transform_anchor
        
        if transform_positive is None:
            self.transform_positive = transforms.ToTensor()
        else:
            self.transform_positive = transform_positive
        
        if transform_negative is None:
            self.transform_negative = transforms.ToTensor()
        else:
            self.transform_negative = transform_negative

    def __len__(self) -> int:
        return len(self.list_path)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # Anchor part
        anchor_path: str = self.list_path[index]
        # blabla/train/time_series_24_0.png -> [blabla/train/time, series, 24, 0.png] -> "24" -> 24
        anchor_time_series_number: int = int(anchor_path.split("_")[-2])
        # blabla/train/time_series_24_0.png -> [blabla/train/time, series, 24, 0.png] -> "0.png" -> [0, .png] -> "0" -> 0
        anchor_image_number: int = int(anchor_path.split("_")[-1].split(".")[0])

        # Positive part
        # Select a patch different from the anchor in the time series
        positive_image_number: int = random.randint(
            0, 8
        )  # We have 8 image in each time series (train)
        while positive_image_number == anchor_image_number:
            positive_time_series_number: int = random.randint(0, 7)

        positive_path = self.__edit_path(
            path=anchor_path,
            time_series_number=anchor_time_series_number,
            image_number=positive_image_number,
        )

        # Negative part
        # Select a patch from a different time series
        negative_time_series_number: int = random.randint(
            0, 4999
        )  # We have 5000 time series
        while negative_time_series_number == anchor_time_series_number:
            negative_time_series_number: int = random.randint(0, 4999)

        negative_image_number: int = random.randint(0, 7)

        negative_path = self.__edit_path(
            path=anchor_path,
            time_series_number=negative_time_series_number,
            image_number=negative_image_number,
        )

        anchor_image: Image.Image = Image.open(anchor_path)
        positive_image: Image.Image = Image.open(positive_path)
        negative_image: Image.Image = Image.open(negative_path)
        
        # Apply transformations
        anchor_tensor: torch.Tensor = self.transform_anchor(anchor_image)
        positive_tensor: torch.Tensor = self.transform_positive(positive_image)
        negative_tensor: torch.Tensor = self.transform_negative(negative_image)
        
        return anchor_tensor, positive_tensor, negative_tensor

    @classmethod
    def __edit_path(cls, path: str, time_series_number: int, image_number: int) -> str:
        prefix = "/".join(path.split("/")[:-1])
        return f"{prefix}/time_series_{time_series_number}_{image_number}.png"

