import os
import logging
import random
from collections import defaultdict
from PIL import Image
import pydicom

import numpy as np
import pandas as pd
import torch
from torchvision import transforms

import pickle
from sklearn.model_selection import StratifiedShuffleSplit

from .base import GEORGEDataset

import pdb


class PmxDataset(GEORGEDataset):
    """ Pneumothorax dataset """

    _normalization_stats = {"mean": 0.48865, "std": 0.24621}

    def __init__(
        self, root, split, transform=None, download=False, augment=False, seed=0
    ):
        self.seed = seed
        transform = transform = get_transform()
        super().__init__("Pmx", root, split, transform=transform)

    def _check_exists(self):
        """Checks whether or not the CXR filemarkers exist."""
        return os.path.isfile(
            "/home/ksaab/Documents/observational/file_markers/cxr/trainval_list_gold.pkl"
        )

    def _download(self):
        """Raises an error if the raw dataset has not yet been downloaded."""
        raise ValueError("Follow the README instructions to download the dataset.")

    def _load_samples(self):
        """ Loads Pneumothorax dataset """
        file_dir = "/home/ksaab/Documents/observational/file_markers/cxr"
        # load tube annotations
        with open(
            "/home/ksaab/Documents/observational/cxr_tube_dict.pkl", "rb"
        ) as pkl_f:
            cxr_tube_dict = pickle.load(pkl_f)

        if self.split in ["train", "val"]:
            file_markers_dir = os.path.join(file_dir, "trainval_list_gold.pkl")
            with open(file_markers_dir, "rb") as fp:
                file_markers = pickle.load(fp)

            labels = [fm[1] for fm in file_markers]
            sss = StratifiedShuffleSplit(
                n_splits=1, test_size=0.2, random_state=self.seed
            )

            for train_ndx, val_ndx in sss.split(np.zeros(len(labels)), labels):
                file_markers_train = [file_markers[ndx] for ndx in train_ndx]
                file_markers_val = [file_markers[ndx] for ndx in val_ndx]

            if self.split == "train":
                file_markers = file_markers_train
            else:
                file_markers = file_markers_val

            true_subclass_labels = np.array(
                [
                    int(
                        "chest tube"
                        in cxr_tube_dict[fm[0].split("/")[-1].split(".dcm")[0]]
                    )
                    for fm in file_markers
                ]
            )

        elif self.split == "test":
            file_markers_dir = os.path.join(file_dir, "test_list.pkl")
            with open(file_markers_dir, "rb") as fp:
                file_markers = pickle.load(fp)

            true_subclass_labels = np.random.randint(0, 2, size=len(file_markers))

        print(f"{len(file_markers)} files in {self.split} split...")

        X = np.array([fm[0] for fm in file_markers])
        superclass_labels = np.array([fm[1] for fm in file_markers])

        Y_dict = {
            "superclass": torch.from_numpy(superclass_labels),
            "true_subclass": torch.from_numpy(true_subclass_labels),
        }

        return X, Y_dict

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
        Returns:
            tuple: (x: torch.Tensor, y: dict) where X is a tensor representing an image
                and y is a dictionary of possible labels.
        """
        image_path = os.path.join(self.data_dir, self.X[idx])
        ds = pydicom.dcmread(image_path)
        image = ds.pixel_array
        image = Image.fromarray(np.uint8(image))

        if self.transform is not None:
            image = self.transform(image)
        image = torch.cat([image, image, image])
        x = image

        y_dict = {name: label[idx] for name, label in self.Y_dict.items()}
        return x, y_dict


def get_transform():
    target_resolution = (224, 224)

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize(**PmxDataset._normalization_stats),
        ]
    )
    return transform
