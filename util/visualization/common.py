import numpy as np
from PIL import Image

import torch
import torchvision.transforms.functional as F

def tensor2pilimage(image, width=None, height=None, minus1to1_normalized=False):
    r"""Convert a 3 dimensional torch tensor to a PIL image with the desired
    width and height.

    Args:
        image (3 x W1 x H1 tensor): Image tensor
        width (int): Desired width for the result PIL image.
        height (int): Desired height for the result PIL image.
        minus1to1_normalized (bool): True if the tensor values are in [-1,
        1]. Otherwise, we assume the values are in [0, 1].

    Returns:
        (PIL image): The resulting PIL image.
    """
    if len(image.size()) != 3:
        raise ValueError('Image tensor dimension does not equal = 3.')
    if image.size(0) != 3:
        raise ValueError('Image has more than 3 channels.')
    if minus1to1_normalized:
        # Normalize back to [0, 1]
        image = (image + 1) * 0.5
    image = image.detach().cpu().squeeze().numpy()
    image = np.transpose(image, (1, 2, 0)) * 255
    output_img = Image.fromarray(np.uint8(image))
    if width is not None and height is not None:
        output_img = output_img.resize((width, height), F.InterpolationMode.BICUBIC)
    return output_img


def tensor2im(image_tensor, imtype=np.uint8, normalize=True,
              three_channel_output=True):
    r"""Convert tensor to image.

    Args:
        image_tensor (torch.tensor or list of torch.tensor): If tensor then
            (NxCxHxW) or (NxTxCxHxW) or (CxHxW).
        imtype (np.dtype): Type of output image.
        normalize (bool): Is the input image normalized or not?
            three_channel_output (bool): Should single channel images be made 3
            channel in output?

    Returns:
        (numpy.ndarray, list if case 1, 2 above).
    """
    if image_tensor is None:
        return None
    if isinstance(image_tensor, list):
        return [tensor2im(x, imtype, normalize) for x in image_tensor]
    if image_tensor.dim() == 5 or image_tensor.dim() == 4:
        return [tensor2im(image_tensor[idx], imtype, normalize)
                for idx in range(image_tensor.size(0))]

    if image_tensor.dim() == 3:
        image_numpy = image_tensor.cpu().float().numpy()
        if normalize:
            image_numpy = (np.transpose(
                image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        else:
            image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
        image_numpy = np.clip(image_numpy, 0, 255)
        if image_numpy.shape[2] == 1 and three_channel_output:
            image_numpy = np.repeat(image_numpy, 3, axis=2)
        elif image_numpy.shape[2] > 3:
            image_numpy = image_numpy[:, :, :3]
        return image_numpy.astype(imtype)



def tensor2label(segmap, n_label=None, imtype=np.uint8,
                 colorize=True, output_normalized_tensor=False):
    r"""Convert segmentation tensor to color image.

    Args:
        segmap (tensor) of
        If tensor then (NxCxHxW) or (NxTxCxHxW) or (CxHxW).
        n_label (int): If None, then segmap.size(0).
        imtype (np.dtype): Type of output image.
        colorize (bool): Put colors in.

    Returns:
        (numpy.ndarray or normalized torch image).
    """
    if segmap is None:
        return None
    if isinstance(segmap, list):
        return [tensor2label(x, n_label,
                             imtype, colorize,
                             output_normalized_tensor) for x in segmap]
    if segmap.dim() == 5 or segmap.dim() == 4:
        return [tensor2label(segmap[idx], n_label,
                             imtype, colorize,
                             output_normalized_tensor)
                for idx in range(segmap.size(0))]

    segmap = segmap.float()
    if not output_normalized_tensor:
        segmap = segmap.cpu()
    if n_label is None:
        n_label = segmap.size(0)
    if n_label > 1:
        segmap = segmap.max(0, keepdim=True)[1]

    if output_normalized_tensor:
        segmap = Colorize(n_label)(segmap).to('cuda')
        return 2 * (segmap.float() / 255) - 1
    else:
        if colorize:
            segmap = Colorize(n_label)(segmap)
            segmap = np.transpose(segmap.numpy(), (1, 2, 0))
        else:
            segmap = segmap.cpu().numpy()
        return segmap.astype(imtype)


class Colorize(object):
    """Class to colorize segmentation maps."""

    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, seg_map):
        r"""

        Args:
            seg_map (tensor): Input Segmentation maps to be colorized.
        """
        size = seg_map.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)
        for label in range(0, len(self.cmap)):
            mask = (label == seg_map[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]
        return color_image



def labelcolormap(N):
    r"""Create colors for segmentation label ids.

    Args:
        N (int): Number of labels.
    """
    if N == 35:  # GTA/cityscape train
        cmap = np.array([(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0),
                         (111, 74, 0), (81, 0, 81), (128, 64, 128),
                         (244, 35, 232), (250, 170, 160), (230, 150, 140),
                         (70, 70, 70), (102, 102, 156), (190, 153, 153),
                         (180, 165, 180), (150, 100, 100), (150, 120, 90),
                         (153, 153, 153), (153, 153, 153), (250, 170, 30),
                         (220, 220, 0), (107, 142, 35), (152, 251, 152),
                         (70, 130, 180), (220, 20, 60), (255, 0, 0),
                         (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 0, 90),
                         (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32),
                         (0, 0, 142)],
                        dtype=np.uint8)
    elif N == 20:  # GTA/cityscape eval
        cmap = np.array([(128, 64, 128), (244, 35, 232), (70, 70, 70),
                         (102, 102, 156), (190, 153, 153), (153, 153, 153),
                         (250, 170, 30), (220, 220, 0), (107, 142, 35),
                         (152, 251, 152), (220, 20, 60), (255, 0, 0),
                         (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100),
                         (0, 0, 230), (119, 11, 32), (70, 130, 180), (0, 0, 0)],
                        dtype=np.uint8)
    else:
        cmap = np.zeros([N, 3]).astype(np.uint8)
        for i in range(N):
            r, g, b = np.zeros(3)
            for j in range(8):
                r = r + (1 << (7 - j)) * ((i & (1 << (3 * j))) >> (3 * j))
                g = g + (1 << (7 - j)) * \
                    ((i & (1 << (3 * j + 1))) >> (3 * j + 1))
                b = b + (1 << (7 - j)) * \
                    ((i & (1 << (3 * j + 2))) >> (3 * j + 2))
            cmap[i, :] = np.array([r, g, b])
    return cmap
