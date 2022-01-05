from tqdm import tqdm
import time

from src.eval.evaluate import AverageMeter, accuracy


def train_fn(model, optimizer, criterion, loader, device, scheduler = None, epoch = 0, learning_rates = []):
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
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        if type(scheduler).__name__ == 'OneCycleLR' or type(scheduler).__name__ == 'CyclicLR':
            scheduler.step()
            learning_rates.append(scheduler.get_last_lr())

        if type(scheduler).__name__ == 'CosineAnnealingWarmRestarts':
            scheduler.step(epoch + i / iter)
            # print('OneCycle')

        acc = accuracy(logits, labels)
        n = images.size(0)
        losses.update(loss.item(), n)
        score.update(acc.item(), n)

        t.set_description('(=> Training) Loss: {:.4f}'.format(losses.avg))

    time_train += time.time() - time_begin
    print('training time: ' + str(time_train))
    return score.avg, losses.avg
