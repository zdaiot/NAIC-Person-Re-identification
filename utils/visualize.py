# 可视化操作
import torch
import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw


def image_with_mask_torch(image, label, mean=None, std=None):
    """返回numpy形式的样本和掩膜
    :param image: 样本，tensor
    :param label: 类标，tensor
    :param mean: 样本均值
    :param std: 样本标准差
    """
    if mean and std:
        for i in range(3):
            image[i] = image[i] * std[i]
            image[i] = image[i] + mean[i]
    image = image * 255.0
    image = image.permute(1, 2, 0).numpy()
    image = Image.fromarray(np.uint8(image))
    draw = ImageDraw.Draw(image)
    label = label.item()
    draw.text((10, 10), str(label), fill=(255, 0, 0))

    return image
