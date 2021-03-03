from typing import Callable, List, NoReturn, Optional, Tuple
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from glob import glob

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from torch import optim
import numpy as np
from torch.optim import lr_scheduler
import torch.optim as optim
import DeepHash.datasets
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from DeepHash.trainer import fit
import torch.nn as nn

import matplotlib
import matplotlib.pyplot as plt
from DeepHash.utils import freeze_model, list_trainable, del_last_layers, save, load, create_embeddings

class TripletDataset(Dataset):
    def __init__(
            self,
            list_path: List[str],
            proba_augment: float = 0,
            isTest: bool = False,
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
            >>> train_path = sorted(glob("/kaggle/input/hackathon-isae-2021-patch-retrieval/Train/*.png"))
            >>> dataset = TripletDataset(list_path=train_path)

        References:

            1. DEEP METRIC LEARNING USING TRIPLET NETWORK: https://arxiv.org/pdf/1412.6622.pdf
        """
        self.list_path = list_path
        self.isTest = isTest
        self.proba_augment = proba_augment

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
        if self.isTest:
            positive_image_number: int = random.randint(6, 7)
        else:
            positive_image_number: int = random.randint(0, 5)

        while positive_image_number == anchor_image_number:
            if self.isTest:
                positive_image_number: int = random.randint(6, 7)
            else:
                positive_image_number: int = random.randint(0, 5)

        positive_path = self.__edit_path(
            path=anchor_path,
            time_series_number=anchor_time_series_number,
            image_number=positive_image_number,
        )

        # Negative part
        # Select a patch from a different time series
        negative_time_series_number1: int = random.randint(
            0, 4999
        )  # We have 5000 time series
        negative_time_series_number2: int = random.randint(0,4999)
        negative_time_series_number3: int = random.randint(0,4999)
        while negative_time_series_number1 == anchor_time_series_number:
            negative_time_series_number1: int = random.randint(0, 4999)

        negative_image_number1: int = random.randint(0, 7)
        negative_image_number2: int = random.randint(0, 7)
        negative_image_number3: int=  random.randint(0,7)

        negative_path1 = self.__edit_path(
            path=anchor_path,
            time_series_number=negative_time_series_number1,
            image_number=negative_image_number1,
        )
        negative_path2 = self.__edit_path(
            path=anchor_path,
            time_series_number=negative_time_series_number2,
            image_number=negative_image_number2,
        )
        negative_path3 = self.__edit_path(
            path=anchor_path,
            time_series_number=negative_time_series_number3,
            image_number=negative_image_number3,
        )

        anchor_image: Image.Image = Image.open(anchor_path)
        positive_image: Image.Image = Image.open(positive_path)
        negative_image1: Image.Image = Image.open(negative_path1)
        negative_image2: Image.Image = Image.open(negative_path2)
        negative_image3: Image.Image = Image.open(negative_path3)


        # Apply transformations
        if random.random()<self.proba_augment:  
            anchor_tensor: torch.Tensor = self.transform_anchor(anchor_image)
        else:anchor_tensor = transforms.ToTensor()(anchor_image)
        if random.random()<self.proba_augment:  
            positive_tensor: torch.Tensor = self.transform_positive(positive_image)
        else:positive_tensor = transforms.ToTensor()(positive_image)
        if random.random()<self.proba_augment:  
            negative_tensor1: torch.Tensor = self.transform_negative(negative_image1)
        else:negative_tensor1 = transforms.ToTensor()(negative_image1)
        if random.random()<self.proba_augment:
            negative_tensor2: torch.Tensor = self.transform_negative(negative_image2)
        else:negative_tensor2 = transforms.ToTensor()(negative_image2)
        if random.random()<self.proba_augment:
            negative_tensor3: torch.Tensor = self.transform_negative(negative_image3)
        else:negative_tensor3 = transforms.ToTensor()(negative_image3)

        return (anchor_tensor, positive_tensor, negative_tensor1, negative_tensor2, negative_tensor3), []

    @classmethod
    def __edit_path(cls, path: str, time_series_number: int, image_number: int) -> str:
        prefix = "/".join(path.split("/")[:-1])
        return f"{prefix}/time_series_{time_series_number}_{image_number}.png"

    class TestDataset(Dataset):
        def __init__(self,
                     list_path: List[str]):
            self.list_path = list_path

        def __len__(self) -> int:
            return len(self.list_path)

        def __getitem__(self, index: int) -> torch.Tensor:
            test_path: str = self.list_path[index]
            test_time_series_number: int = int(test_path.split("_")[-2])
            test_image_number: int = int(test_path.split("_")[-1].split(".")[0])

            test_image: Image.Image = Image.open(test_path)

            # Apply transformations
            test_tensor: torch.Tensor = transforms.ToTensor(test_image)

            return test_tensor


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3, x4, x5):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        output4 = self.embedding_net(x4)
        output5 = self.embedding_net(x5)
        return output1, output2, output3, output4, output5

    def get_embedding(self, x):
        return self.embedding_net(x)
    def get_embedding_net(self):
        return self.embedding_net

class MetricResnet(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(512 * 7 * 7, 128)
        self.activation = nn.ReLU()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.flatten(x)
        x = self.activation(x)
        x = self.linear_1(x)
        x = F.normalize(x, p=2, dim=1)
        return x

# Define model
resnet18 = models.resnet18(pretrained=True)


# Freeze all the parameters in the model
def freeze_model(model):
    for params in model.parameters():
        params.requires_grad = False


### LOAD TRAIN DATA

train_path = sorted(glob("Train/*.png"))
print("""HEEEEEEY""")
print(len(train_path))
# The two last images of each timeseries is set as part of the test dataset.
train = [train_path[i] for i in range(40000) if i % 8 < 6]
test = [train_path[i] for i in range(40000) if i % 8 >= 6]
print(test[0:30])

contrast_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.ColorJitter(brightness=0.5, contrast=1, hue=0.2, saturation=0.8),
])

triplet_train_dataset = TripletDataset(list_path=train, proba_augment=0., isTest=False, transform_anchor=contrast_transform, transform_positive=contrast_transform, transform_negative=contrast_transform)
triplet_test_dataset = TripletDataset(list_path=test, isTest=True)
print (len(test), len(train))


### SELECT CUDA
cuda = torch.cuda.is_available()

model_input = torch.nn.Sequential(torch.nn.Sequential(*list(resnet18.children())[:-2]))

cuda = torch.cuda.is_available()
if cuda: print("cuda")
else: print("no cuda")
"""triplet_train_dataset = TripletDataset(image_dataset['train']) # Returns triplets of images
triplet_test_dataset = TripletCifar1(image_dataset['test'])"""
batch_size = 128
kwargs = {'num_workers':8, 'pin_memory': True} if cuda else {}
triplet_train_loader = torch.utils.data.DataLoader(triplet_train_dataset, batch_size=batch_size, shuffle=False, **kwargs)
triplet_test_loader = torch.utils.data.DataLoader(triplet_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# Set up the network and training parameters
#from DeepHash.networks import EmbeddingNet, TripletNet
from DeepHash.losses import TripletLoss

margin = 2.
embedding_net = MetricResnet(model_input)
#embedding_net = resnet18
model = TripletNet(embedding_net)
if cuda:
    model.cuda()
loss_fn = TripletLoss(margin)
lr = 5e-5 
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 5
log_interval = 50

from DeepHash.trainer import fit
fit(triplet_train_loader, triplet_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)

torch.save(model.embedding_net,'multi3_noaug_5ep_m2_lr5-5_batch128_w8.pt')

