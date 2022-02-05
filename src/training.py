from tqdm import tqdm
import time
import torch
import torch.nn.functional as F
import numpy as np

from src.eval.evaluate import AverageMeter, accuracy


def train_fn(model, optimizer, criterion, loader, device, scheduler = None, epoch = 0, learning_rates = [], SSL=False):
    """
  Training method
  :param model: model to train
  :param optimizer: optimization algorithm
  :param criterion: loss function
  :param loader: data loader for either training or testing set
  :param device: torch device
  :return: (accuracy, loss) on the data
  """
    time_begin = time.time()
    score = AverageMeter()
    losses = AverageMeter()
    model.train()
    time_train = 0

    iter = len(loader)
    t = tqdm(loader)
    # learning_rates = []
    for i, (images, labels) in enumerate(t):
        # print(images.size())

        # for j in range(images.size()[0]):

        if SSL:
            block_size = int(images.size()[-1]/2)
            block_locs = np.random.random_integers(low=0, high=images.size()[-1] - block_size, size=2*images.size()[0])
            # set a square of size (block_size, block_size) to zeros
            transform_tensors = torch.cat([F.pad(torch.zeros((1, 3, block_size, block_size)),
                                           (block_locs[j], images.size()[-1] - block_size - block_locs[j],
                                           block_locs[j+images.size()[0]],
                                           images.size()[-1] - block_size - block_locs[j+images.size()[0]]),
                                                 "constant", 1)
                                           for j in range(images.size()[0])])
            labels = images
            images = images*transform_tensors
        # else:
        #     labels = F.one_hot(labels, num_classes=model.fc2.out_features).float()
        images = images.to(device)
        # print()
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        # print(logits.size())
        # print(labels.size())
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        if type(scheduler).__name__ == 'OneCycleLR' or type(scheduler).__name__ == 'CyclicLR':
            scheduler.step()
            learning_rates.append(scheduler.get_last_lr())

        if type(scheduler).__name__ == 'CosineAnnealingWarmRestarts':
            scheduler.step(epoch + i / iter)
            # print('SGDR')

        n = images.size(0)
        if not SSL:
            acc = accuracy(logits, labels)
            score.update(acc.item(), n)
        losses.update(loss.item(), n)

        t.set_description('(=> Training) Loss: {:.4f}'.format(losses.avg))

    time_train += time.time() - time_begin
    print('training time: ' + str(time_train))
    return score.avg, losses.avg
