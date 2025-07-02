import os
from re import S
import pandas as pd
import numpy as np

from tqdm import tqdm

import torch
import torch.utils.data as data
import h5py
from sklearn.utils.class_weight import compute_class_weight


class Dataset_Subtyping(data.Dataset):
    def __init__(self, root, csv_file, feature, encoder_pipeline):
        self.feature = feature
        self.encoder_pipeline = encoder_pipeline
        # there are multiple roots for some specific datasets
        if "," in root:
            self.root = root.split(",")
        else:
            self.root = [root]
        self.csv_file = csv_file
        self.data = pd.read_csv(csv_file)
        # if there is only one fold, then it is a fixed split
        if "fold" in self.data.columns:
            self.split = "fixed"
            self.num_folds = 1
        else:
            self.split = "cv"
            num_folds = self.data.columns.str.startswith("fold").sum()
            self.num_folds = num_folds
        # convert "label" column to discrete values
        self.data["label"] = pd.Categorical(self.data["label"])
        self.data["label"] = self.data["label"].cat.codes
        # get number of classes
        self.num_classes = len(self.data["label"].unique())
        # get the dimension of WSI features from "slide" column
        for root in self.root:
            if encoder_pipeline == "HKUST": # features in pt
                if os.path.exists(os.path.join(root, self.feature, str(self.data["slide"].values[0]) + "_patches.pt")):
                    self.n_features = torch.load(os.path.join(root, self.feature, str(self.data["slide"].values[0]) + "_patches.pt")).shape[-1]
                    print("[dataset] feature dimension: %d" % self.n_features)
                    break
            elif encoder_pipeline == "Trident": # features in h5
                print("****** using Trident encoder pipeline ******8")
                if os.path.exists(os.path.join(root, self.feature, str(self.data["slide"].values[0]) + ".h5")):
                    with h5py.File(os.path.join(root, self.feature, str(self.data["slide"].values[0]) + ".h5"), "r") as f:
                        h5_features = torch.tensor(f["features"][:])
                    self.n_features = h5_features.shape[-1]
                    print("[dataset] feature dimension: %d" % self.n_features)
                    break
            else:
                raise ValueError("Unknown encoder pipeline: {}".format(encoder_pipeline))
        self.cases = [] # list of cases, each case is a list of [ID, Slide, Label]
        for idx in range(len(self.data)):
            case = self.data.iloc[idx, :].values.tolist()[:3]
            self.cases.append(case)
        print("[dataset] dataset from %s" % (self.csv_file))
        print("[dataset] number of cases=%d" % (len(self.cases)))
        print("[dataset] number of classes=%d" % (self.num_classes))
        print("[dataset] number of features=%d" % self.n_features)
        if self.split == "cv":
            self.train = []
            self.test = []
            for fold in range(self.num_folds):
                split = self.data["fold{}".format(fold + 1)].values.tolist()
                train_split = [i for i, x in enumerate(split) if x == "train"]
                test_split = [i for i, x in enumerate(split) if x == "test"]
                self.train.append(train_split)
                self.test.append(test_split)
                print("[dataset] fold %d, training split: %d, test split: %d" % (fold, len(train_split), len(test_split)))
        else:
            split = self.data["fold"].values.tolist()
            self.train = [i for i, x in enumerate(split) if x == "train"]
            self.val = [i for i, x in enumerate(split) if x == "val"]
            self.test = [i for i, x in enumerate(split) if x == "test"]
            print("[dataset] training split: {}, validation split: {}, test split: {}".format(len(self.train), len(self.val), len(self.test)))

    def get_fold(self, fold=0):
        if self.split == "fixed":
            assert fold == 0, "fold should be 0"
            print("[fetch *] training split: {}, validation split: {}, test split: {}".format(len(self.train), len(self.val), len(self.test)))
            return self.train, self.val, self.test
        elif self.split == "cv":
            assert fold >= 0, "fold should be greater than 0"
            print("[fetch *] fold %d, training split: %d, test split: %d" % (fold, len(self.train[fold]), len(self.test[fold])))
            return self.train[fold], self.test[fold]
        
    def get_class_weights(self, train_indices):
        train_labels = self.data.loc[train_indices, "label"].values

        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(self.num_classes),
            y=train_labels
        )

        return torch.tensor(class_weights, dtype=torch.float32)

    def __getitem__(self, index):
        case = self.cases[index]
        ID, Slide, Label = case
        slide = []
        for root in self.root:
            for s in str(Slide).split(";"):
                if self.encoder_pipeline == "HKUST":
                    if os.path.exists(os.path.join(root, self.feature, s + "_patches.pt")):
                        slide.append(torch.load(os.path.join(root, self.feature, s + "_patches.pt")))
                elif self.encoder_pipeline == "Trident":
                    if os.path.exists(os.path.join(root, self.feature, s + ".h5")):
                        with h5py.File(os.path.join(root, self.feature, s + ".h5"), "r") as f:
                            h5_features = torch.tensor(f["features"][:])
                        slide.append(h5_features)
                else:
                    raise ValueError("Unknown encoder pipeline: {}".format(self.encoder_pipeline))
        Slide = torch.cat(slide, dim=0)
        if type(Slide) is not torch.Tensor:
            raise ValueError("Slide is not a tensor")
        Label = torch.tensor(Label, dtype=torch.int64)
        return ID, Slide, Label

    def __len__(self):
        return len(self.cases)
