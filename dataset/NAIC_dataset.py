import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
from utils.transforms import DataAugmentation
from utils.visualize import image_with_mask_torch
from dataset.triplet_sampler import RandomIdentitySampler
from utils.data_analysis import get_folds_id, get_all_id


class TrainDataset(Dataset):
    def __init__(self, root, train_list_txt_path, train_id, augmentation_flag, erase_prob, gray_prob, mean, std):
        """ 训练数据集的Dataset类

        :param root: 训练数据集的根目录；类型为str
        :param train_list_txt_path: 存储全部数据集对应的id的txt文件；类型为str
        :param train_id: 筛选用于训练集的id，类型为list
        :param augmentation_flag: 是否对样本使用额外的数据增强；类型为bool
        :param erase_prob: 数据增强中随机擦除的概率；类型为float
        :param gray_prob: 数据增强中转为灰度图的概率；类型为float
        :param mean: 每个通道的均值；类型为tuple
        :param std: 每个通道的方差；类型为tuple
        """
        super(TrainDataset, self).__init__()
        self.root = root
        self.train_list_txt_path = train_list_txt_path
        self.train_id = train_id
        # 因为样本的id可能是不连续的，所以要将样本id映射为类标
        self.id_to_label = {id: label for label, id in enumerate(sorted(train_id))}
        self.samples_list = self.parse_id_list()
        if augmentation_flag:
            self.augmentation = DataAugmentation(erase_prob=erase_prob, gray_prob=gray_prob)
        else:
            self.augmentation = None

        self.mean = mean
        self.std = std

    def __getitem__(self, idx):
        """

        :param idx: 索引下标
        :return image: 该索引对应的图片
        :return sample_label: 该索引对应的类标
        """
        sample_name = os.path.join(self.root, self.samples_list[idx][0])
        sample_label = self.samples_list[idx][1]
        try:
            image = Image.open(sample_name).convert('RGB')
        except IOError:
            raise IOError('Reading image %s failed.' % sample_name)
        # 数据增强
        if self.augmentation:
            image = np.array(image)
            image = self.augmentation(image)
            image = Image.fromarray(image)

        transform_train_list = [
            T.Resize((256, 128), interpolation=3),
            T.Pad(10),
            T.RandomCrop((256, 128)),
            T.ToTensor(),
            T.Normalize(self.mean, self.std)
        ]
        transform_compose = T.Compose(transform_train_list)
        image = transform_compose(image)

        sample_label = torch.tensor(sample_label).long()

        return image, sample_label

    def __len__(self):
        """

        :return: 返回训练集一共有多少个样本
        """
        return len(self.samples_list)

    def parse_id_list(self):
        """ 筛选train_id中每个id对应的所有样本
        :return samples_list: 筛选train_id中每个id对应的所有样本后结果，类型为list；长度不定；
                             list中的每一个元素也为list，表示每一个样本，其第一个值为图片名，第二个值为label
        """
        samples_list = []
        if not os.path.exists(self.train_list_txt_path):
            raise FileExistsError('Please ensure %s exists.' % self.train_list_txt_path)
        with open(self.train_list_txt_path, 'r') as id_file:
            for sample in id_file.readlines():
                sample_list = []
                # 依据id进行样本筛选
                sample_id = int(sample.split(' ')[1].strip('\n'))
                if sample_id in self.train_id:
                    sample_name = sample.split(' ')[0].split('/')[1]
                    sample_list.append(sample_name)
                    sample_list.append(self.id_to_label[sample_id])
                    samples_list.append(sample_list)

        return samples_list


class ValidateDataset(Dataset):
    def __init__(self, root, samples_list, mean, std):
        """ 验证数据集的Dataset类

        :param root: 验证数据集的根目录；类型为str
        :param samples_list: [[sample_name, label], [sample_name, label]]
        :param mean: 每个通道的均值；类型为tuple
        :param std: 每个通道的方差；类型为tuple
        """
        super(ValidateDataset, self).__init__()
        self.root = root
        self.samples_list = samples_list

        self.mean = mean
        self.std = std

    def __getitem__(self, idx):
        """

        :param idx: 索引下标
        :return image: 该索引对应的图片
        :return sample_label: 该索引对应的类标
        :return sample_path: 该索引对应的图片路径
        """
        sample_path = os.path.join(self.root, self.samples_list[idx][0])
        sample_label = self.samples_list[idx][1]
        try:
            image = Image.open(sample_path).convert('RGB')
        except IOError:
            raise IOError('Reading image %s failed.' % sample_path)
        to_tensor = T.ToTensor()
        normalize = T.Normalize(self.mean, self.std)
        transform_compose = T.Compose([to_tensor, normalize])
        image = transform_compose(image)

        sample_label = torch.tensor(sample_label).long()

        return image, sample_label, sample_path

    def __len__(self):
        """

        :return: 返回一共有多少个样本
        """
        return len(self.samples_list)


class queryGallerySeparate:
    def __init__(self, root, train_list_txt_path, class_id):
        """ 划分查询集与数据库

        :param root: 训练数据集的根目录；类型为str
        :param train_list_txt_path: 存储全部数据集对应的id的txt文件；类型为str
        :param class_id: 筛选用于验证集的id，类型为list
        """
        self.root = root
        self.train_list_txt_path = train_list_txt_path
        self.class_id = class_id
        # 因为样本的id可能是不连续的，所以要将样本id映射为类标
        self.id_to_label = {id: label for label, id in enumerate(sorted(class_id))}
        self.samples_list, self.labels_list = self.parse_id_list()

    def query_gallery_separate(self):
        """拆分查询集和数据库

        :return query_list: 查询集 [[sample_name, label], [sample_name, label]]
        :return gallery_list: 数据库 [[sample_name, label], [sample_name, label]]
        :return len(query_list): 查询集的样本总数
        """
        # 依据类标从小到大进行排序，并返回对应的原始下标 [原始下标，从小到大排序结果]
        sorted_labels = sorted(enumerate(self.labels_list), key=lambda x: x[1])
        sorted_index = []
        sorted_labels_list = []
        for item in sorted_labels:
            sorted_index.append(item[0])
            sorted_labels_list.append(item[1])
        sorted_samples_list = [self.samples_list[index] for index in sorted_index]
        seed_label = -1
        query_list = []
        gallery_list = []
        for (sample, label) in zip(sorted_samples_list, sorted_labels_list):
            # 当前类标第一次出现，加入query
            if label != seed_label:
                query_list.append([sample, label])
                seed_label = label
            else:
                gallery_list.append([sample, label])

        return query_list, gallery_list, len(query_list)

    def parse_id_list(self):
        """ 筛选class_id中id对应的所有样本以及类标

        :return samples_list: class_id中id对应的所有样本
        :return labels_list: class_id中id对应的所有样本的类标
        """
        samples_list = []
        labels_list = []
        if not os.path.exists(self.train_list_txt_path):
            raise FileExistsError('Please ensure %s exists.' % self.train_list_txt_path)
        with open(self.train_list_txt_path, 'r') as id_file:
            for sample in id_file.readlines():
                # 依据id进行样本筛选
                sample_id = int(sample.split(' ')[1].strip('\n'))
                if sample_id in self.class_id:
                    sample_name = sample.split(' ')[0].split('/')[1]
                    samples_list.append(sample_name)
                    labels_list.append(self.id_to_label[sample_id])

        return samples_list, labels_list


class TestDataset(Dataset):
    def __init__(self, pic_list, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        """ 测试数据集的Dataset类

        :param pic_list: 数据集路径组成的list；类型为list
        :param mean: 每个通道的均值；类型为tuple
        :param std: 每个通道的方差；类型为tuple
        """
        super(TestDataset, self).__init__()
        self.pic_list = pic_list
        self.mean = mean
        self.std = std

    def __getitem__(self, idx):
        """

        :param idx: 索引下标
        :return image: 该索引对应的图片
        :return sample_name: 该索引对应的图片名称
        """
        sample_name = self.pic_list[idx].split('/')[-1]
        try:
            image = Image.open(self.pic_list[idx]).convert('RGB')
        except IOError:
            raise IOError('Reading image %s failed.' % self.pic_list[idx])
        to_tensor = T.ToTensor()
        normalize = T.Normalize(self.mean, self.std)
        transform_compose = T.Compose([to_tensor, normalize])
        image = transform_compose(image)

        return image, sample_name

    def __len__(self):
        """

        :return: 返回总共有多少张图片
        """
        return len(self.pic_list)


def get_loaders(root, n_splits, batch_size, num_instances, num_works, augmentation_flag, erase_prob, gray_prob,
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """ 获得各个折的训练集、验证集的Dataloader，以及各个折的查询集个数

    :param root: 训练数据集的根目录；类型为str
    :param n_splits: 要划分多少折；类型为int
    :param batch_size: batch的大小；类型为int
    :param num_instances: 每一类选取多少个数据；类型为int
    :param num_works: 读取数据时的线程数；类型为int
    :param augmentation_flag: 是否对训练集进行额外的数据增强；类型为bool
    :param erase_prob: 数据增强中随机擦除的概率；类型为float
    :param gray_prob: 数据增强中转为灰度图的概率；类型为float
    :param mean: 每个通道的均值；类型为tuple
    :param std: 每个通道的方差；类型为tuple
    :return train_dataloader_folds: 所有折训练集的Dataloader；类型为list
    :return valid_dataloader_folds: 所有折验证集的Dataloader；类型为list
    :return num_query_folds: 与valid_dataloader_folds搭配使用，表示每一个valid_dataloader中前多少个样本是查询集；类型为list
    :return num_classes_folds: 所有折的训练集类别数,验证集类别数；类型为list；第i个值为第i折的[训练集类别数, 验证集类别数]
    """
    train_list_txt_path = os.path.join(root, 'train_list.txt')
    root_pic = os.path.join(root, 'train_set')
    train_dataloader_folds, valid_dataloader_folds = list(), list()
    num_query_folds, num_classes_folds = list(), list()

    # 分层交叉验证
    train_id_folds, valid_id_folds = get_folds_id(train_list_txt_path, n_splits)
    for train_id_fold, valid_id_fold in zip(train_id_folds, valid_id_folds):
        train_dataset = TrainDataset(
            root=root_pic,
            train_list_txt_path=train_list_txt_path,
            train_id=train_id_fold,
            augmentation_flag=augmentation_flag,
            erase_prob=erase_prob,
            gray_prob=gray_prob,
            mean=mean, std=std
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_works,
            pin_memory=True,
            sampler=RandomIdentitySampler(train_dataset.samples_list, batch_size, num_instances),
        )

        query_gallery_separate = queryGallerySeparate(
            root=root_pic,
            train_list_txt_path=train_list_txt_path,
            class_id=valid_id_fold
        )
        query_list, gallery_list, num_query = query_gallery_separate.query_gallery_separate()
        valid_dataset = ValidateDataset(
            root=root_pic,
            samples_list=query_list + gallery_list,
            mean=mean, std=std
        )
        # 注意可以根据num_query来划分出查询集和数据库
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            num_workers=num_works,
            pin_memory=True
        )

        train_dataloader_folds.append(train_dataloader)
        valid_dataloader_folds.append(valid_dataloader)
        num_query_folds.append(num_query)
        num_classes_folds.append([len(train_id_fold), len(valid_id_fold)])

    return train_dataloader_folds, valid_dataloader_folds, num_query_folds, num_classes_folds


def get_baseline_loader(root, batch_size, num_instances, num_works, augmentation_flag, erase_prob, gray_prob,
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """ 获得训练集的Dataloader，以及总的类别数

    :param root: 训练数据集的根目录；类型为str
    :param batch_size: batch的大小；类型为int
    :param num_instances: 每一类选取多少个数据；类型为int
    :param num_works: 读取数据时的线程数；类型为int
    :param augmentation_flag: 是否对训练集进行额外的数据增强；类型为bool
    :param erase_prob: 数据增强中随机擦除的概率；类型为float
    :param gray_prob: 数据增强中转为灰度图的概率；类型为float
    :param mean: 每个通道的均值；类型为tuple
    :param std: 每个通道的方差；类型为tuple
    :return train_dataloader: 训练集的Dataloader；
    :return num_classes: 训练集的类别数；类型为int
    """
    train_list_txt_path = os.path.join(root, 'train_list.txt')
    root_pic = os.path.join(root, 'train_set')
    train_id = get_all_id(train_list_txt_path)
    train_dataset = TrainDataset(
        root=root_pic,
        train_list_txt_path=train_list_txt_path,
        train_id=train_id,
        augmentation_flag=augmentation_flag,
        erase_prob=erase_prob,
        gray_prob=gray_prob,
        mean=mean, std=std
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_works,
        pin_memory=True,
        sampler=RandomIdentitySampler(train_dataset.samples_list, batch_size, num_instances),
    )
    num_classes = len(train_id)
    return train_dataloader, num_classes


def get_test_loader(test_dataset_root, batch_size, num_workers):
    """

    :param test_dataset_root: 测试数据集的根目录；类型为str
    :param batch_size: batch的大小；类型为int
    :param num_workers: 读取数据时的线程数；类型为int
    :return: test_dataloader: 测试数据集的Dataloader
    :return num_query: 查询集数量；类型为int
    """
    if '初赛B榜测试集' in test_dataset_root:
        pic_path_query = os.path.join(test_dataset_root, 'query_b')
        pic_path_gallery = os.path.join(test_dataset_root, 'gallery_b')
    else:
        pic_path_query = os.path.join(test_dataset_root, 'query_a')
        pic_path_gallery = os.path.join(test_dataset_root, 'gallery_a')

    pic_list_query = glob.glob(pic_path_query + '/*.png')
    pic_list_gallery = glob.glob(pic_path_gallery + '/*.png')

    pic_list = pic_list_query + pic_list_gallery

    test_dataset = TestDataset(pic_list)
    num_query = len(pic_list_query)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=False)
    return test_dataloader, num_query


if __name__ == "__main__":
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    root = 'dataset/NAIC_data/初赛训练集'
    n_splits = 4

    train_dataloader_folds, valid_dataloader_folds, num_query_folds, num_classes_folds = \
        get_loaders(
            root,
            n_splits,
            batch_size=8,
            num_instances=4,
            num_works=8,
            augmentation_flag=True,
            erase_prob=0.3,
            gray_prob=0.3
        )
    for train_dataloader, valid_dataloader, num_query, num_classes in zip(train_dataloader_folds,
                                                                          valid_dataloader_folds,
                                                                          num_query_folds,
                                                                          num_classes_folds):
        for images, labels in train_dataloader:
            for index in range(images.size(0)):
                image, label = images[index], labels[index]
                image = image_with_mask_torch(image, label, mean, std)
                plt.imshow(image)
                plt.show()
