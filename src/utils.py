import os
import torch

import pickle
from pathlib import Path
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from src.eval.evaluate import AverageMeter, accuracy
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def eval_loss(model, loader, criterion, device, SSL=False):
    losses = AverageMeter()
    score = AverageMeter()
    model.eval()

    t = tqdm(loader)
    with torch.no_grad():  # no gradient needed
        for images, labels in t:

            if SSL:
                block_size = int(images.size()[-1]/2)
                block_locs = np.random.random_integers(low=0, high=images.size()[-1] - block_size, size=2*images.size()[0])
                transform_tensors = torch.cat([F.pad(torch.zeros((1, 3, block_size, block_size)),
                                                     (block_locs[j], images.size()[-1] - block_size - block_locs[j],
                                                      block_locs[j+images.size()[0]],
                                                      images.size()[-1] - block_size - block_locs[j+images.size()[0]]),
                                                     "constant", 1)
                                               for j in range(images.size()[0])])
                labels = images
                images = images*transform_tensors
                y = torch.ones(images.size()[0])

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            if not SSL:
                acc = accuracy(outputs, labels)
                score.update(acc.item(), images.size(0))
            losses.update(loss.item(), images.size(0))

            t.set_description('(=> Test) Score: {:.4f} (=> Test) Loss: {:.4f}'.format(score.avg, losses.avg))

    return score.avg, losses.avg

def save_result(filename: str, obj: object) -> None:
    """Save object to disk as pickle file.

    Args:
        filename: Name of file in ./results directory to write object to.
        obj: The object to write to file.

    """
    # make sure save directory exists
    save_path = Path("results")
    os.makedirs(save_path, exist_ok=True)

    # save the python objects as bytes
    with (save_path / f"{filename}.pkl").open('wb') as fh:
        pickle.dump(obj, fh)

import torch
import torch.nn


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: https://github.com/delijati/pytorch-siamese/blob/master/contrastive.py
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def check_type_forward(self, in_types):
        assert len(in_types) == 3

        x0_type, x1_type, y_type = in_types
        assert x0_type.size() == x1_type.shape
        assert x1_type.size()[0] == y_type.shape[0]
        assert x1_type.size()[0] > 0
        # print(x0_type.size())
        # print(x0_type.dim())
        assert x0_type.dim() == 2
        assert x1_type.dim() == 2
        assert y_type.dim() == 1

    def forward(self, x0, x1):
        # self.check_type_forward((x0, x1, y))

        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        # dist = torch.sqrt(dist_sq)

        # mdist = self.margin - dist
        # dist = torch.clamp(mdist, min=0.0)
        # loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(dist_sq) / 2.0 / x0.size()[0] / (x0.size()[-1]**2)
        return loss


class SSLTransform:
    def __init__(self):
        self.angles = [0, 90, 180, 270]

    def __call__(self, image):
        images = [TF.to_tensor(TF.rotate(image, angle)) for angle in self.angles]
        labels = list(range(len(self.angles)))
        return images, labels