import cv2
import numpy as np
from torchvision import transforms


class RandomGaussianBlur(object):
    """
    https://github.com/facebookresearch/swav/blob/master/src/multicropdataset.py
    """

    def __call__(self, img):
        if np.random.rand() > 0.5:
            return img
        sigma = np.random.uniform(0.1, 2.0)
        return cv2.GaussianBlur(np.asarray(img), (23, 23), sigma)  # 23 is for imagenet that has size of 224 x 224.


def create_simclr_data_augmentation(strength: float, size: int) -> transforms.Compose:
    """
    Create SimCLR's data augmentation.

    :param strength: strength parameter for colorjiter.
    :param size: `RandomResizedCrop`'s size parameter.
    :return: Compose of transforms.

    """
    color_jitter = transforms.ColorJitter(
        brightness=0.8 * strength,
        contrast=0.8 * strength,
        saturation=0.8 * strength,
        hue=0.2 * strength
    )

    rnd_color_jitter = transforms.RandomApply(transforms=[color_jitter], p=0.8)

    common_transforms = [
        transforms.RandomResizedCrop(size=size),
        transforms.RandomHorizontalFlip(p=0.5),
        # the following two are `color_distort`
        rnd_color_jitter,
        transforms.RandomGrayscale(0.2),
        # end of color_distort
    ]
    if size == 224:  # imagenet or pet dataset
        common_transforms.append(RandomGaussianBlur())
    elif size == 32:
        pass
    else:
        raise ValueError("`size` must be either `32` or `224`.")
    common_transforms.append(transforms.ToTensor())

    return transforms.Compose(
        common_transforms
    )


class SimCLRTransforms(object):
    def __init__(self, strength: float = 0.5, size: int = 32, num_views: int = 2):
        # Definition is from Appendix A. of SimCLRv1 paper:
        # https://arxiv.org/pdf/2002.05709.pdf

        self.transform = create_simclr_data_augmentation(strength, size)

        if num_views <= 1:
            raise ValueError("`num_views` must be greater than 1.")

        self.num_views = num_views

    def __call__(self, x) -> list:
        return [self.transform(x) for _ in range(self.num_views)]
