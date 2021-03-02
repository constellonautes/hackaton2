import os

from typing import List, NoReturn, Union

import numpy as np
import pandas as pd


class Sentinel2TimeSeriesDataframe:
    """This class creates a dataframe containing Sentinel-2 time series"""

    def __init__(self, directory: str) -> NoReturn:
        """Inits the class providing a directory containing the data.
        Args:
            directory (str): Path to the image time series.
        Returns:
            NoReturn
        """
        self.extension = ".png"
        self.data_separator = "_"
        self.columns = ["Path", "Time series ID", "Image ID"]
        self.directory = directory
        self.dataframe = self.get_dataframe()
        self.id_list = self.get_time_series_id_list()

    def is_png(self, filename: str) -> bool:
        return filename.lower().endswith(self.extension)

    def get_path_list(self) -> List[str]:
        path_list = []
        for filename in os.listdir(self.directory):
            if self.is_png(filename):
                path_list.append(os.path.join(self.directory, filename))
            else:
                raise Exception("No png file found: {}".format(filename))

        return path_list

    def create_dataframe(
        self, data: List[List[Union[str, int]]]
    ) -> pd.DataFrame:  # noqa: E501
        dataframe = pd.DataFrame(data=data, columns=self.columns)

        return dataframe

    def split_information(self, path: str) -> List[str]:
        filename = os.path.basename(path)
        filename = filename.split(self.extension)[0]
        information = filename.split(self.data_separator)
        information = [int(information[-2]), int(information[-1])]
        information = [path] + information

        return information

    def get_data(self, path_list: List[str]) -> List[List[Union[str, int]]]:
        data = [self.split_information(path) for path in path_list]

        return data

    def sort_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe = dataframe.sort_values([self.columns[1], self.columns[2]])

        return dataframe

    def get_dataframe(self) -> pd.DataFrame:
        path_list = self.get_path_list()
        data = self.get_data(path_list)
        dataframe = self.create_dataframe(data)
        dataframe = self.sort_dataframe(dataframe)

        return dataframe

    def get_time_series_id_list(self) -> List[int]:
        id_list = self.dataframe[self.columns[1]].unique()

        return id_list

    def get_paths_as_list(self, dataframe) -> List[str]:
        path_list = dataframe[self.columns[0]].tolist()

        return path_list

    def get_time_series_dataframe(self, time_series_id: int) -> pd.DataFrame:
        time_series_dataframe = self.dataframe.loc[
            self.dataframe[self.columns[1]] == time_series_id
        ]

        return time_series_dataframe

    def get_time_series_paths(self, time_series_id: int) -> List[str]:
        time_series_dataframe = self.get_time_series_dataframe(time_series_id)
        path_list = self.get_paths_as_list(time_series_dataframe)

        return path_list


class Sentinel2ImagesDataframe:
    """This class creates a dataframe containing Sentinel-2 images to retrieve"""

    def __init__(self, directory: str) -> NoReturn:
        """Inits the class providing a directory containing the data.
        Args:
            directory (str): Path to the images to be retrieved.
        Returns:
            NoReturn
        """
        self.extension = ".png"
        self.columns = ["Path"]
        self.directory = directory
        self.dataframe = self.get_dataframe()
        self.id_list = self.create_id_list()

    def is_png(self, filename: str) -> bool:
        return filename.lower().endswith(self.extension)

    def get_path_list(self) -> List[str]:
        path_list = []
        for filename in os.listdir(self.directory):
            if self.is_png(filename):
                path_list.append(os.path.join(self.directory, filename))
            else:
                raise Exception("No png file found: {}".format(filename))

        return path_list

    def create_dataframe(
        self, data: List[List[Union[str, int]]]
    ) -> pd.DataFrame:  # noqa: E501
        dataframe = pd.DataFrame(data=data, columns=self.columns)

        return dataframe

    def get_dataframe(self) -> pd.DataFrame:
        data = self.get_path_list()
        dataframe = self.create_dataframe(data)

        return dataframe

    def create_id_list(self) -> List[int]:
        id_list = [i for i in range(len(self.dataframe))]

        return id_list

    def get_paths_as_list(self, dataframe) -> List[str]:
        path_list = dataframe[self.columns[0]].tolist()

        return path_list

    def get_image_dataframe(self, image_id: int) -> pd.DataFrame:
        image_dataframe = self.dataframe.iloc[[image_id]]

        return image_dataframe

    def get_image_path(self, image_id: int) -> List[str]:
        image_dataframe = self.get_image_dataframe(image_id)
        path_list = self.get_paths_as_list(image_dataframe)
        path_list = path_list[0]

        return path_list
      
   
import torch
from skimage import io
from typing import List, NoReturn, Union
from torch.utils.data import Dataset


class TimeSeries(Dataset):
    """Creates a dataset containing the Sentinel-2 time series.
    Args:
        Dataset ([PyTorch Dataset])
    """

    def __init__(
        self,
        dataframe: Sentinel2TimeSeriesDataframe,
        transform=None,
    ) -> NoReturn:
        """Creates a time series dataset.
        Args:
            dataframe ([Sentinel2TimeSeriesDataframe]): Dataframe containing the time series information.
            transform ([type], optional): [description]. Optional transform to be applied.
        """
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe.id_list)

    def get_time_series(self, path_list: str) -> np.ndarray:
        time_series = []
        for path in path_list:
            image = io.imread(path)
            time_series.append(image)
        time_series = np.stack(time_series, axis=3)

        return time_series

    def __getitem__(self, index: int) -> torch.Tensor:
        index = self.dataframe.id_list[index]
        time_series_paths = self.dataframe.get_time_series_paths(index)
        time_series = self.get_time_series(time_series_paths)

        if self.transform:
            time_series = self.transform(time_series)
        time_series = torch.tensor(time_series)
        return time_series


class Images(Dataset):
    """Creates a dataset containing the Sentinel-2 images to retrieve.
    Args:
        Dataset ([PyTorch Dataset])
    """

    def __init__(
        self,
        dataframe: Sentinel2ImagesDataframe,
        transform=None,
    ) -> NoReturn:
        """Create an image dataset
        Args:
            dataframe ([Sentinel2ImagesDataframe]): Dataframe containing the image information.
            transform ([type], optional): [description]. Optional transform to be applied.
        """
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe.id_list)

    def get_image(self, image_path: str) -> np.ndarray:
        image = io.imread(image_path)

        return image

    def __getitem__(self, index: int) -> np.ndarray:
        index = self.dataframe.id_list[index]
        image_path = self.dataframe.get_image_path(index)
        image = self.get_image(image_path)

        if self.transform:
            image = self.transform(image)
        image = torch.tensor(image)
        return image