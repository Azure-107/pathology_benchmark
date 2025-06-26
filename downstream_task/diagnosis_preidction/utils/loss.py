import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight

def define_loss(args):
    if args.loss == "bce":
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == "ce":
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'weighted_ce':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = nn.CrossEntropyLoss(weight=args.class_weights.to(device))
    else:
        raise NotImplementedError
    return criterion
