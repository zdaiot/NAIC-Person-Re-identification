import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold


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

    :param id_numbers: 每一个id含有多少张图片，类型为dict
    :return: numbers_statics: 对id对应的图片张数进行数据统计，类型为dict
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
    plt.plot(x, y)
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
    train_id_pin = list()
    train_valid_id = list()
    train_valid_id_number = list()
    for key, value in id_numbers.items():
        if value == 1:
            pass
        # elif value == 2:
        #     train_id_pin.append(key)
        else:
            train_valid_id.append(key)
            train_valid_id_number.append(value)
    
    train_id_folds, valid_id_folds = list(), list()
    # 注意这里的随机种子要固定
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
    for train_id_fold, valid_id_fold in skf.split(train_valid_id, train_valid_id_number):
        train_id_fold, valid_id_fold = train_id_fold.tolist(), valid_id_fold.tolist()
        train_id_fold = train_id_fold + train_id_pin

        train_id_folds.append(train_id_fold)
        valid_id_folds.append(valid_id_fold)
    return train_id_folds, valid_id_folds
        

if __name__ == '__main__':
    train_list_path = 'dataset/NAIC_data/初赛训练集/train_list.txt'
    id_numbers = dataset_statics(train_list_path)
    # print(id_numbers)

    numbers_statics = id_numbers_statics(id_numbers)
    print(numbers_statics)

    train_id_folds, valid_id_folds = get_folds_id(train_list_path, 3)


