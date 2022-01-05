import os
import torch

from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from src.eval.evaluate import AverageMeter, accuracy

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def eval_loss(model, loader, criterion, device):
    losses = AverageMeter()
    score = AverageMeter()
    model.eval()

    t = tqdm(loader)
    with torch.no_grad():  # no gradient needed
        for images, labels in t:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            acc = accuracy(outputs, labels)
            score.update(acc.item(), images.size(0))
            losses.update(loss.item(), images.size(0))

            t.set_description('(=> Test) Score: {:.4f} (=> Test) Loss: {:.4f}'.format(score.avg, losses.avg))

    return score.avg, losses.avg
