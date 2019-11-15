import os
import random
import numpy as np
import tqdm
import math
import shutil
from PIL import Image
from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, CLAHE, RandomRotate90, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma, OneOf,
    ToFloat, ShiftScaleRotate, GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightnessContrast, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise,CenterCrop,
    IAAAdditiveGaussianNoise,GaussNoise,Cutout,Rotate, Normalize, Crop, RandomCrop
)


class DatasetAmplify:
    def __init__(self, data_root, id_txt_path, target_root, sample_id_path):
        """

        Args:
            data_root: 原始数据集的根目录
            id_txt_path: 记录ID信息的txt文件
        """
        self.data_root = data_root
        self.id_txt_path = id_txt_path
        self.target_root = target_root
        self.sample_id_path = sample_id_path
        # 各个ID包含的样本个数
        self.ids_numbers = self.get_ids_numbers()
        self.samples_ids = self.get_samples_ids()
        self.prepare()

    def do_amplify(self, augmentations=[], augment_times=4):
        """
        Args:
            augmentations: list，候选的数据扩增方式, 确保len(augmentations) >= augment_times
            augment_times: int, 样本扩增次数
        """
        assert(len(augmentations) > augment_times)
        tbar = tqdm.tqdm(self.samples_ids.items())
        for [sample_name, id] in tbar:
            # 当前id的样本数目
            id_sample_number = self.ids_numbers[id]
            if id_sample_number >= augment_times:
                # 如果样本数目大于等于扩增的次数，只复制
                shutil.copy(os.path.join(self.data_root, sample_name), os.path.join(self.target_root, sample_name))
                with open(self.sample_id_path, 'a+') as f:
                    line = '/' + sample_name + ' ' + str(id) + '\n'
                    f.write(line) 
                continue
            augmentations_selected = self.select_augmentation(augmentations, id_sample_number, augment_times)
            image = Image.open(os.path.join(self.data_root, sample_name))
            # 保存原始图片
            image.save(os.path.join(self.target_root, sample_name))
            with open(self.sample_id_path, 'a+') as f:
                line = '/' + sample_name + ' ' + str(id) + '\n'
                f.write(line)            
                for index, augmentation in enumerate(augmentations_selected):
                    self.augment_and_save(augmentation, image, sample_name, index, id, f)
            tbar.set_description(desc=sample_name)

    def select_augmentation(self, augmentations, id_samples_num, augment_times):
        """依据ID样本数和目标扩增次数计算当前样本的扩增次数

        Args:
            augmentations: 待选扩增方法
            id_samplses_num: 当前ID的样本数目
            augment_times: 目标扩增次数
        """
        augment_times_per_sample = augment_times / id_samples_num
        augment_times_per_sample = math.floor(augment_times_per_sample)
        augmentations = random.sample(augmentations, augment_times_per_sample)

        return augmentations

    def augment_and_save(self, augmentation, image, image_name, augment_index, id, f):
        """对输入图片进行扩增并保存图片、将ID写入txt文件

        Args:
            augmentation: 数据扩增的方式
            image: 输入图片, PIL
            image_name: 图片对应的名称
            augment_index: 当前图片第几次增强
            id: 图片对应的类标
            f: 打开的文件指针
        """
        image = np.asarray(image)
        agumented = augmentation(image=image)
        image_augmented = agumented['image']
        image_augmented = Image.fromarray(image_augmented)
        image_name = image_name.split('.')[0] + '_' + str(augment_index) + '.' + image_name.split('.')[1]
        image_aug_name = os.path.join(self.target_root, image_name)
        image_augmented.save(image_aug_name)
        line = '/' + image_name + ' ' + str(id) + '\n'
        f.write(line)       

    def get_samples_ids(self):
        """得到所有样本及其对应的类标

        Return:
            samples_ids: dir, {sample_name: id}
        """
        samples_ids = {}
        with open(self.id_txt_path, 'r') as f:
            for sample_id in f:
                sample_name = sample_id.split(' ')[0].split('/')[1]
                sample_id = int(sample_id.split(' ')[1].strip())
                samples_ids[sample_name] = int(sample_id)
        return samples_ids

    def get_ids_numbers(self):
        """ 统计每一个id含有多少张图片

        Args:
            id_numbers: 每一个id含有多少张图片，类型为dict
        """
        ids_numbers = dict()
        with open(self.id_txt_path) as fread:
            for line in fread.readlines():
                pic_name, label = line.strip().split(' ')
                label = eval(label)
                if not ids_numbers.__contains__(label):
                    ids_numbers[label] = 0
                ids_numbers[label] += 1
        return ids_numbers

    def prepare(self):
        if os.path.exists(self.target_root):
            print('Removing %s' % self.target_root)
            shutil.rmtree(self.target_root)
            print('Building %s' % self.target_root)
            os.makedirs(self.target_root)
        else:
            print('Building %s' % self.target_root)
            os.makedirs(self.target_root)

        if os.path.exists(self.sample_id_path):
            print('Removing %s' % self.sample_id_path)
            os.remove(self.sample_id_path)


if __name__ == "__main__":
    data_root = 'dataset/NAIC_data/初赛训练集/train_set'
    id_txt_path = 'dataset/NAIC_data/初赛训练集/train_list.txt'
    target_root = 'dataset/NAIC_data/train_amplify/train_set'
    sample_id_path = 'dataset/NAIC_data/train_amplify/train_list.txt'
    augmentations = [
        HorizontalFlip(p=1.0), 
        RandomBrightness(limit=0.15, p=1.0, always_apply=True),
        ShiftScaleRotate(shift_limit=0.07, scale_limit=0.1, rotate_limit=15, always_apply=True), 
        Blur(p=1.0, always_apply=True), 
        RandomCrop(height=256, width=128, always_apply=True)
        ]
    augment_times = 4

    dataset_amplify = DatasetAmplify(data_root, id_txt_path, target_root, sample_id_path)
    dataset_amplify.do_amplify(augmentations, augment_times)
