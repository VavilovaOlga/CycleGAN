import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np
import torchvision.transforms as tt
from skimage.io import imread
from skimage.color import gray2rgb
import torch
import random


def denorm(img_tensors, stats=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))):
    return img_tensors * stats[1][0] + stats[0][0]


def show_images(images, nmax=4):
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(np.rollaxis(make_grid(denorm(images[:nmax]),
                                    nrow=4).numpy(), 0, 3))
    return fig


def singe_image_transform(image_path, stats=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                          image_size=256, resize_size=256, padding=0):
    image = imread(image_path)
    if image.ndim == 2:
        image = gray2rgb(image)
    transforms = tt.Compose([
        tt.ToPILImage(),
        tt.Resize(resize_size),
        tt.Pad(padding),
        tt.CenterCrop(image_size),
        tt.ToTensor(),
        tt.Normalize(*stats)])
    image_transformed = transforms(image)
    return image_transformed.unsqueeze(0)


class Buffer:
    def __init__(self, max_size=500):
        self.max_size = max_size
        self.content = []

    def take_from_buffer(self, images):
        return_images = []
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if len(self.content) < self.max_size:
                self.content.append(image)
                return_images.append(image)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    return_images.append(self.content[i].clone())
                    self.content[i] = image
                else:
                    return_images.append(image)
        return torch.cat(return_images)
