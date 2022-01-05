import matplotlib.pyplot as plt
import numpy as np
import os
import time
import math


def plot_accuracy(train_acc, val_acc, save=False, **kwargs):
    train_acc = np.array(train_acc)
    val_acc = np.array(val_acc)
    fig = plt.figure()
    plt.plot(np.arange(len(train_acc)), train_acc, label="train accuracy")
    plt.plot(np.arange(len(val_acc)), val_acc, label="validation accuracy")
    plt.xlabel("epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title(f"{kwargs.get('model', '')} {kwargs.get('opt', '')} "
              f"{kwargs.get('scheduler', 'None')} lr={kwargs.get('lr', '')}")

    if save:
        save_plot(fig, **kwargs)
    fig.show()


def plot_loss(train_loss, val_loss, save=False, **kwargs):
    train_acc = np.array(train_loss)
    val_acc = np.array(val_loss)
    fig = plt.figure()
    plt.plot(np.arange(len(train_acc)), train_acc, label="train loss")
    plt.plot(np.arange(len(val_acc)), val_acc, label="validation loss")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"{kwargs.get('model', '')} {kwargs.get('opt', '')} "
              f"{kwargs.get('scheduler', 'None')} lr={kwargs.get('lr', '')}")

    if save:
        save_plot(fig, info='Loss', **kwargs)
    fig.show()


def plot_lr(lrs, save=False, **kwargs):
    train_acc = np.array(lrs)
    # val_acc = np.array(val_loss)
    fig = plt.figure()
    plt.plot(np.arange(len(lrs)), lrs, label=f"{kwargs.get('scheduler', 'Learning rate')}")
    # plt.plot(np.arange(len(val_acc)), val_acc, label="validation loss")
    plt.xlabel("epochs")
    plt.ylabel("Learning Rate")
    plt.legend()
    plt.title(f"{kwargs.get('model', '')} {kwargs.get('opt', '')} "
              f"{kwargs.get('scheduler', 'None')} lr={kwargs.get('lr', '')}")

    if save:
        save_plot(fig, info='Learning Rate', **kwargs)
    fig.show()

def plot_augs(image, transforms):
    images = [transform(image) for transform in transforms]
    plot_images(images)


def plot_images(images, **imshow_kwargs):
    orig_img = None
    cols = int(math.sqrt(len(images)))
    rows = math.ceil(len(images)/cols)
    fig = plt.figure(figsize=(cols, rows))
    for i in range(1, len(images)):
        fig.add_subplot(rows, cols, i)
        # print(images[i])
        plt.imshow(images[i])
    plt.tight_layout()
    plt.show()

def save_plot(fig, **kwargs):
    plot_save_dir = os.path.join(os.getcwd(), 'plots')

    if not os.path.exists(plot_save_dir):
        os.mkdir(plot_save_dir)
    name = f"{kwargs.get('model', 'default')}_{kwargs.get('opt', '')}_" \
           f"{kwargs.get('scheduler', 'None')}_{kwargs.get('lr', '')}_" \
           f"{kwargs.get('model_id', str(int(time.time())))}_{kwargs.get('info', '')}"
    save_plot_str = os.path.join(plot_save_dir,  name + '.png')
    print(f"Saving to: {save_plot_str}")
    fig.savefig(save_plot_str)



if __name__ == '__main__':
    plot_accuracy(np.arange(100), np.negative(np.arange(100)), save=True)