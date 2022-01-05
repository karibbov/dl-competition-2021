from PIL import Image
from pathlib import Path
from torchvision import transforms
from src.plotting import plot_images

size = (128, 128)

resize_to_64x64 = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

resize_and_colour_jitter = transforms.Compose([
    transforms.Resize(size),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor()
])

rotate = transforms.Compose([
    transforms.Resize(size),
    transforms.RandomRotation(30),
    transforms.ToTensor()
])

rotate_128 = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomRotation(30),
    transforms.ToTensor()
])

affine = transforms.Compose([
    transforms.Resize(size),
    transforms.RandomApply(
        [transforms.RandomAffine(10, translate=(0.2, 0.2), scale=(0.85, 1.5), shear=(-10, 10, -10, 10))], p=0.7),
    transforms.ToTensor()
])

logical_affine = transforms.RandomApply([transforms.RandomChoice([transforms.RandomRotation(15),
                                                                  transforms.RandomAffine(0, (0.2, 0.2)),
                                                                  transforms.RandomAffine(0, scale=(0.75, 1))])], p=0.75)

compose = transforms.Compose([
    resize_and_colour_jitter,
    # transforms.ToPILImage(),
    transforms.RandomAdjustSharpness(0.2, p=0.7),
    transforms.RandomAutocontrast(p=0.7),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToPILImage(),
    affine
    # transforms.ToTensor()

])

if __name__ == '__main__':
    img = Image.open(Path('/media/samir/F/Uni-Freiburg/DL/Exercises'
                          '/dl2021-competition-dl2021-redducks/dataset'
                          '/train/1/image_0002.jpg'))
    images = [transforms.ToPILImage()(compose(img)) for i in range(100)]
    images.insert(0, img)
    plot_images(images)
