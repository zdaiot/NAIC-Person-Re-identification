import math
import random
import numpy as np
import torchvision.transforms as T
from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, CLAHE, RandomRotate90, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma, OneOf,
    ToFloat, ShiftScaleRotate, GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightnessContrast, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise,CenterCrop,
    IAAAdditiveGaussianNoise,GaussNoise,Cutout,Rotate, Normalize, Crop, RandomCrop, Resize
)
from PIL import Image


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.485, 0.456, 0.406)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.shape[0] * img.shape[1]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.shape[1] and h < img.shape[0]:
                x1 = random.randint(0, img.shape[0] - h)
                y1 = random.randint(0, img.shape[1] - w)
                if img.shape[2] == 3:
                    img[x1:x1 + h, y1:y1 + w, 0] = self.mean[0] * 255
                    img[x1:x1 + h, y1:y1 + w, 1] = self.mean[1] * 255
                    img[x1:x1 + h, y1:y1 + w, 2] = self.mean[2] * 255
                else:
                    img[x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


class DataAugmentation(object):
    def __init__(self, erase_flag=False, full_aug=True):
        """
        Args:
            full_aug: 是否对整幅图片进行随机增强
        """
        self.full_aug = full_aug
        self.erase_flag = erase_flag
        if erase_flag:
            self.random_erase = RandomErasing()

    def __call__(self, image):
        """

        :param image: 传入的图片
        :return: 经过数据增强后的图片
        """
        if self.erase_flag:
            image = self.random_erase(image)
        if self.full_aug:
            image = self.data_augmentation(image)
        
        return image

    def data_augmentation(self, original_image):
        """ 进行样本和掩膜的随机增强
        Args:
            original_image: 原始图片
            original_mask: 原始掩膜
        Return:
            image_aug: 增强后的图片
            mask_aug: 增强后的掩膜
        """
        augmentations = Compose([
            Resize(224, 224),
            HorizontalFlip(p=0.4),
            ShiftScaleRotate(shift_limit=0.07, rotate_limit=0, p=0.4),
            # 直方图均衡化
            CLAHE(p=0.3),

            # 亮度、对比度
            RandomGamma(gamma_limit=(80, 120), p=0.1),
            RandomBrightnessContrast(p=0.1),
            
            # 模糊
            OneOf([
                    MotionBlur(p=0.1),
                    MedianBlur(blur_limit=3, p=0.1),
                    Blur(blur_limit=3, p=0.1),
                ], p=0.3),
            
            OneOf([
                    IAAAdditiveGaussianNoise(),
                    GaussNoise(),
                ], p=0.2)
        ])
        
        augmented = augmentations(image=original_image)
        image_aug = augmented['image']

        return image_aug

