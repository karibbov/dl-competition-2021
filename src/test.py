import torch
import os
import pathlib
from src.cnn import SkipModel
from src.data_augmentations import size, resize
from tqdm import tqdm
from src.eval.evaluate import AverageMeter
from src.plotting import plot_images

import torchvision.transforms.functional as TF
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def get_mistakes(model, loader, device):
    """
    Evaluation method
    :param model: model to evaluate
    :param loader: data loader for either training or testing set
    :param device: torch device
    :return: accuracy on the data
    """
    score = AverageMeter()
    model.eval()

    t = tqdm(loader)
    mistake_img = []
    mistake_class = []
    wrong_class = []
    nn_outs = []
    with torch.no_grad():  # no gradient needed
        for images, labels in t:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            # acc = accuracy(outputs, labels)
            preds = torch.argmax(outputs, dim=1)
            mistake_img = mistake_img + [TF.to_pil_image(image) for image in images[preds != labels]]
            nn_outs = nn_outs + [output.detach().numpy() for output in outputs[preds != labels]]
            mistake_class = mistake_class + [int(label.detach().numpy()) for label in labels[preds != labels]]
            wrong_class = wrong_class + [int(label.detach().numpy()) for label in preds[preds != labels]]
            # score.update(acc.item(), images.size(0))

            t.set_description('(=> Test) Score: {:.4f}'.format(score.avg))

    return mistake_img, mistake_class, wrong_class, nn_outs


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_save_dir = os.path.join(os.getcwd(), "models")

skip_model_loc_64 = "default_model_1643298044"
skip_model_2loc_64 = "default_model_1643301573"
skip_model_colab = "skip_model_1643303123"
skip_model_2colab = "skip_model_1643299483"
# skip_model_2colab = "default_model_1643371384"

save_model_str = os.path.join(model_save_dir, skip_model_2colab)

model = SkipModel(input_shape=[3] + list(size))

model.load_state_dict(torch.load(save_model_str, map_location=torch.device(device)))

data = ImageFolder(os.path.join(os.getcwd(), 'dataset', 'val'), transform=resize)

val_loader = DataLoader(dataset=data,
                         batch_size=128,
                         shuffle=False)

mis_img, mis_class, w_c, out = get_mistakes(model, val_loader, device)

print(mis_class)

print(w_c)
print(out)
plot_images(mis_img)


