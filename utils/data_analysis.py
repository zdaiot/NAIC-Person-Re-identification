import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import KFold, StratifiedKFold
import random


def dataset_statics(train_list_path):
    """ 统计每一个id含有多少张图片
    :param train_list_path: 存储全部数据集对应的id的txt文件；类型为str
    :return id_numbers: 每一个id含有多少张图片，类型为dict
    """
    id_numbers = dict()
    with open(train_list_path) as fread:
        for line in fread.readlines():
            pic_name, label = line.strip().split(' ')
            label = eval(label)
            if not id_numbers.__contains__(label):
                id_numbers[label] = 0
            id_numbers[label] += 1
    return id_numbers


def id_numbers_statics(id_numbers):
    """ 对id对应的图片张数进行数据统计

    :param id_numbers: dict, {id: samples_num}
    :return: numbers_statics_sort: dict, {images_num: ids_num}, 例： {256： 3}表示有256张图片的ID数目为3个
    """
    numbers_statics = dict()
    for value in id_numbers.values():
        if not numbers_statics.__contains__(value):
            numbers_statics[value] = 0
        numbers_statics[value] += 1

    # 对统计结果画图
    numbers_statics_sort = sorted(numbers_statics.items(), key=lambda x: x[0])
    x, y = list(), list()
    for number_statics in numbers_statics_sort:
        x.append(number_statics[0])
        y.append(number_statics[1])

    ax1 = plt.subplot(111)
    x_axis = range(len(x))
    rects = ax1.bar(x=x_axis, height=y, width=0.8, label='ID Number')
    plt.ylabel('ID Number')
    plt.xticks([index + 0.2 for index in x_axis], x)
    plt.xlabel('Sample Number')
    plt.title('ID Number of Each Sample Number')
    plt.legend()

    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")

    plt.show()

    return numbers_statics_sort


def get_folds_id(train_list_path, n_splits):
    """
    :param train_list_path: 存储全部数据集对应的id的txt文件；类型为str
    :param n_splits: 要划分多少折；类型为int
    :return train_id_folds: 各个折的训练集id；类型为list；第i个值表示第i折的训练集id
    :return valid_id_folds: 各个折的验证集id；类型为list；第i个值表示第i折的验证集id
    """
    id_numbers = dataset_statics(train_list_path)
    train_id_pin, valid_id_pin = list(), list()
    train_valid_id = list()
    train_valid_id_number = list()
    for key, value in id_numbers.items():
        if value == 1:
            train_id_pin.append(key)
        # elif value >= 100:
            # pass
        else:
            train_valid_id.append(key)
            train_valid_id_number.append(value)

    train_id_folds, valid_id_folds = list(), list()
    # 注意这里的随机种子要固定
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2019)
    # skf = KFold(n_splits, shuffle=True, random_state=2019)
    for train_id_fold, valid_id_fold in skf.split(train_valid_id, train_valid_id_number):
        train_id_fold, valid_id_fold = train_id_fold.tolist(), valid_id_fold.tolist()
        train_id_fold = train_id_fold + train_id_pin
        valid_id_fold = valid_id_fold + valid_id_pin

        train_id_folds.append(train_id_fold)
        valid_id_folds.append(valid_id_fold)
    return train_id_folds, valid_id_folds


def get_all_id(train_list_path):
    """ 返回训练数据集的所有类别
    :param train_list_path: 存储全部数据集对应的id的txt文件；类型为str
    :return: 所有数据集的类别
    """
    train_id = list()
    id_numbers = dataset_statics(train_list_path)
    for key, value in id_numbers.items():
        if value == 1:
            pass
        elif value >= 100:
            pass
        else:
            train_id.append(key)
    return train_id


def demo_id(id, number, number_of_ids, data_dir, sample_id_txt, ids_samples):
    """展示指定ID的样本, 当指定的ID所包含的样本数目少于number时，全部显示

    Args:
        id: 指定的ID编号
        number: 显示的样本数目
        number_of_ids: dir, 各个ID对应的样本数目
        data_dir: 样本根目录
        sample_id_txt: 存放各个样本的ID的txt文件的路径
        ids_samples: 各个ID对应的样本的名称
    """
    # 当前ID的样本数目
    number_of_id = number_of_ids[id]
    if number_of_id < number:
        show_number = number_of_id
    else:
        show_number = number

    samples = ids_samples[id]
    samples_selected = random.sample(samples, show_number)
    if show_number < 10:
        number_per_line = show_number
    else:
        number_per_line = 10
    lines = (show_number + 10) / 10
    for index, sample_name in enumerate(samples_selected):
        sample_path = os.path.join(data_dir, sample_name)
        image = Image.open(sample_path)
        plt.subplot(lines, number_per_line, index + 1)
        plt.imshow(image)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()


def get_ids_samples(sample_id_txt):
    """得到各个ID对应的所有样本的名称

    Args:
        sample_id_txt: 存放各个样本的ID的txt文件的路径
    """
    ids_samples = {}
    watched_id = set()
    with open(sample_id_txt, 'r') as f:
        for sample_id in f:
            sample_name = sample_id.split(' ')[0].split('/')[1]
            sample_id = int(sample_id.split(' ')[1].strip())
            if sample_id in watched_id:
                # 如果遇到过当前id，则直接进行列表扩增
                ids_samples[sample_id].append(sample_name)
            else:
                ids_samples[sample_id] = []
                ids_samples[sample_id].append(sample_name)
            watched_id.add(sample_id)

    return ids_samples


def image_channels_statistics(images_path):
    images_list = os.listdir(images_path)
    for image_name in images_list:
        image_path = os.path.join(images_path, image_name)
        image = Image.open(image_path).convert('RGB')
        image = np.asarray(image)
        print(image)


if __name__ == '__main__':
    train_list_path = 'dataset/NAIC_data/初赛训练集/train_list.txt'
    train_data_dir = 'dataset/NAIC_data/初赛训练集/train_set'
    id_numbers = dataset_statics(train_list_path)
    show_id = []
    show_number = 864
    for (id, number) in id_numbers.items():
        if number == show_number:
            show_id.append(id)
    # image_channels_statistics(train_data_dir)
    numbers_statics = id_numbers_statics(id_numbers)
    print(numbers_statics)

    # train_id_folds, valid_id_folds = get_folds_id(train_list_path, 3)

    ids_samples = get_ids_samples(train_list_path)
    ids = show_id
    for id in ids:
        demo_id(id, 8, id_numbers, train_data_dir, train_list_path, ids_samples)
