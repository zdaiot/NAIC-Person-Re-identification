import torch
import os
import glob
import json
import codecs
import numpy as np
import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
from dataset.NAIC_dataset import TestDataset, get_loaders, get_baseline_loader
from config import get_config
from models.model import build_model, get_model
from models.sync_bn.batchnorm import convert_model
from evaluate import euclidean_dist, re_rank, cos_dist
from solver import Solver


class CreateSubmission(object):
    def __init__(self, config, num_classes, pth_path):
        """

        :param config: 配置参数
        :param num_classes: 类别数；类型为int
        :param pth_path: 权重文件路径；类型为str
        """
        self.num_classes = num_classes

        self.model_name = config.model_name
        self.last_stride = config.last_stride
        self.test_dataset_root = os.path.join(config.dataset_root, '初赛A榜测试集')
        self.dist = config.dist
        self.num_gpus = torch.cuda.device_count()
        print('Using {} GPUS'.format(self.num_gpus))

        # 加载模型，只要有GPU，则使用DataParallel函数，当GPU有多个GPU时，调用sync_bn函数
        self.model = get_model(self.model_name, self.num_classes, self.last_stride)
        if torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model)
            if self.num_gpus > 1:
                self.model = convert_model(self.model)
            self.model = self.model.cuda()

        # 实例化实现各种子函数的 solver 类
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.solver = Solver(self.model, self.device)

        # 加载权重矩阵
        self.model = self.solver.load_checkpoint(pth_path)

        # 每一个查询样本从数据库中取出最近的10个样本
        self.num_choose = 200

        # 加载test Dataloader
        self.pic_path_query = os.path.join(self.test_dataset_root, 'query_a')
        self.pic_path_gallery = os.path.join(self.test_dataset_root, 'gallery_a')

        pic_list_query = glob.glob(self.pic_path_query + '/*.png')
        pic_list_gallery = glob.glob(self.pic_path_gallery + '/*.png')

        pic_list = pic_list_query + pic_list_gallery

        test_dataset = TestDataset(pic_list)
        self.num_query = len(pic_list_query)

        self.test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=config.num_workers,
                                          pin_memory=True, shuffle=False)

        self.demo_names = os.listdir('dataset/demo_data')
        self.demo_results_path = './results'
        if not os.path.exists(self.demo_results_path):
            os.makedirs(self.demo_results_path)

    def get_result(self, show):
        """

        :param show: 是否显示查询出的结果
        :return None
        """
        self.model.eval()
        tbar = tqdm.tqdm(self.test_dataloader)
        features_all, names_all = [], []
        with torch.no_grad():
            for i, (images, names) in enumerate(tbar):
                # 完成网络的前向传播
                labels_predict, global_features, features = self.solver.forward(images)

                features_all.append(features.detach().cpu())
                names_all.extend(names)

        features_all = torch.cat(features_all, dim=0)
        query_features = features_all[:self.num_query]
        gallery_features = features_all[self.num_query:]

        query_names = np.array(names_all[:self.num_query])
        gallery_names = np.array(names_all[self.num_query:])

        if self.dist == 're_rank':
            distmat = re_rank(query_features, gallery_features)
        elif self.dist == 'cos_dist':
            distmat = cos_dist(query_features, gallery_features)
        elif self.dist == 'euclidean_dist':
            distmat = euclidean_dist(query_features, gallery_features)
        else:
            assert "Not implemented :{}".format(self.dist)

        result = {}
        for query_index, query_dist in enumerate(distmat):
            # 注意若使用的是cos_dist需要从大到小将序排列（加负号），其余为从小到大升序排列（不加负号）
            if self.dist == 'cos_dist':
                choose_index = np.argsort(-query_dist)[:self.num_choose]
            else:
                choose_index = np.argsort(query_dist)[:self.num_choose]
            query_name = query_names[query_index]
            gallery_name = gallery_names[choose_index]
            result[query_name] = gallery_name.tolist()
            if query_name in self.demo_names:
                self.show_result(query_name, gallery_name, 5, show)

        with codecs.open('./result.json', 'w', "utf-8") as json_file:
            json.dump(result, json_file, ensure_ascii=False)

    def show_result(self, query_name, gallery_names, top_rank, show):
        """

        :param query_name: 待查询样本的名称；类型为str
        :param gallery_names: 检索到的样本名称；类型为list
        :param top_rank: 显示检索到的前多少张图片；类型为int
        :param show: 是否显示结果；类型为bool
        :return None
        """
        # 将索引转换为样本名称
        query_image = Image.open(os.path.join(self.pic_path_query, query_name))
        plt.figure(figsize=(14, 10))
        plt.subplot(1, top_rank + 1, 1)
        plt.imshow(query_image)
        plt.text(30, -10.0, query_name)
        for i, gallery_name in enumerate(gallery_names):
            if i == top_rank:
                break
            gallery_image = Image.open(os.path.join(self.pic_path_gallery, gallery_name))
            plt.subplot(1, top_rank + 1, i + 1 + 1)
            plt.imshow(gallery_image)
            plt.text(30, -10.0, gallery_name)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.savefig(os.path.join(self.demo_results_path, query_name))
        if show:
            plt.show()


if __name__ == "__main__":
    test_baseline = True
    config = get_config()
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_dataset_root = os.path.join(config.dataset_root, '初赛训练集')

    if test_baseline:
        # 测试baseline
        train_loader, num_classes = get_baseline_loader(train_dataset_root, config.batch_size, config.num_workers,
                                                        config.shuffle_train, mean, std)
        pth_path = os.path.join(config.save_path, config.model_name, '{}.pth'.format(config.model_name))
        create_submission = CreateSubmission(config, num_classes, pth_path)
        create_submission.get_result(show=False)
    else:
        # 测试加了各种trick的模型
        train_dataloader_folds, valid_dataloader_folds, num_query_folds, num_classes_folds, train_valid_ratio_folds = get_loaders(
            train_dataset_root, config.n_splits, config.batch_size, config.num_workers, config.shuffle_train, config.use_erase, mean, std)

        for fold_index, [train_loader, valid_loader, num_query, num_classes] in enumerate(zip(train_dataloader_folds,
                                                      valid_dataloader_folds, num_query_folds, num_classes_folds)):
            if fold_index not in config.selected_fold:
                continue
            pth_path = os.path.join(config.save_path, config.model_name, '{}_fold{}_best.pth'.format(config.model_name, fold_index))
            # 注意fold之间的因为类别数不同所以模型也不同，所以均要实例化TrainVal
            create_submission = CreateSubmission(config, num_classes, pth_path)
            create_submission.get_result(show=True)

