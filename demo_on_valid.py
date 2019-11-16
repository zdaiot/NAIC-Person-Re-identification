import torch
import os
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from dataset.NAIC_dataset import get_loaders, get_baseline_loader
from config import get_config
from models.model import build_model, get_model
from models.sync_bn.batchnorm import convert_model
from evaluate import euclidean_dist, re_rank, cos_dist
from solver import Solver


class Demo(object):
    def __init__(self, config, num_classes, pth_path, valid_dataloader, num_query):
        """

        :param config: 配置参数
        :param num_classes: 类别数；类型为int
        :param pth_path: 权重文件路径；类型为str
        :param valid_dataloader: 验证数据集的Dataloader
        :param num_query: 查询集数量；类型为int
        """
        self.num_classes = num_classes

        self.model_name = config.model_name
        self.last_stride = config.last_stride
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
        self.model.eval()

        # 每一个查询样本从数据库中取出最近的200个样本
        self.num_choose = 10
        self.num_query = num_query
        self.valid_dataloader = valid_dataloader

        self.demo_results_path = './results/valid'
        if not os.path.exists(self.demo_results_path):
            os.makedirs(self.demo_results_path)

    def get_result(self, show):
        """

        :param show: 是否显示查询出的结果
        :return None
        """
        tbar = tqdm.tqdm(self.valid_dataloader)
        features_all, labels_all, paths_all = [], [], []
        with torch.no_grad():
            for i, (images, labels, paths) in enumerate(tbar):
                # 完成网络的前向传播
                # labels_predict, global_features, features = self.solver.forward(images)
                features = self.solver.tta(images)

                features_all.append(features.detach().cpu())
                labels_all.extend(labels)
                paths_all.extend(paths)

        features_all = torch.cat(features_all, dim=0)
        query_features = features_all[:self.num_query]
        gallery_features = features_all[self.num_query:]

        query_lables = np.array(labels_all[:self.num_query])
        gallery_labels = np.array(labels_all[self.num_query:])

        query_paths = np.array(paths_all[:self.num_query])
        gallery_paths = np.array(paths_all[self.num_query:])

        if self.dist == 're_rank':
            distmat = re_rank(query_features, gallery_features)
        elif self.dist == 'cos_dist':
            distmat = cos_dist(query_features, gallery_features)
        elif self.dist == 'euclidean_dist':
            distmat = euclidean_dist(query_features, gallery_features)
        else:
            assert "Not implemented :{}".format(self.dist)

        for query_index, query_dist in enumerate(distmat):
            # 注意若使用的是cos_dist需要从大到小将序排列（加负号），其余为从小到大升序排列（不加负号）
            if self.dist == 'cos_dist':
                choose_index = np.argsort(-query_dist)[:self.num_choose]
            else:
                choose_index = np.argsort(query_dist)[:self.num_choose]
            query_path = query_paths[query_index]
            gallery_path = gallery_paths[choose_index]
            query_label = query_lables[query_index]
            gallery_label = gallery_labels[choose_index]
            self.show_result(query_path, gallery_path, query_label, gallery_label, 5, show)

    def show_result(self, query_path, gallery_paths, query_label, gallery_labels, top_rank, show):
        """

        :param query_path: 待查询样本的路径；类型为str
        :param gallery_paths: 检索到的样本路径；类型为list
        :param query_label: 待检索样本的类标；类型为int
        :param gallery_labels: 检索到的样本类标；类型为list
        :param top_rank: 显示检索到的前多少张图片；类型为int
        :param show: 是否显示结果；类型为bool
        :return None
        """
        # 将索引转换为样本名称
        query_image = Image.open(query_path)
        plt.figure(figsize=(14, 10))
        plt.subplot(1, top_rank + 1, 1)
        plt.imshow(query_image)
        plt.text(30, -10.0, query_path.split('/')[-1])
        plt.text(30, -20.0, query_label)
        for i, (gallery_path, gallery_label) in enumerate(zip(gallery_paths, gallery_labels)):
            if i == top_rank:
                break
            gallery_image = Image.open(gallery_path)
            plt.subplot(1, top_rank + 1, i + 1 + 1)
            plt.imshow(gallery_image)
            plt.text(30, -20.0, gallery_label)
            plt.text(30, -10.0, gallery_path.split('/')[-1])
        plt.savefig(os.path.join(self.demo_results_path, query_path.split('/')[-1]))
        if show:
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            plt.show()
        plt.close()


if __name__ == "__main__":
    demo_on_baseline =True
    config = get_config()
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_dataset_root = os.path.join(config.dataset_root, '初赛训练集')

    _, valid_dataloader_folds, num_query_folds, num_classes_folds = get_loaders(
        train_dataset_root,
        config.n_splits,
        config.batch_size,
        config.num_instances,
        config.num_workers,
        config.augmentation_flag,
        config.erase_prob,
        config.gray_prob,
        mean, std
    )

    for fold_index, [_, valid_loader, num_query, num_classes] in enumerate(zip(valid_dataloader_folds, num_query_folds, num_classes_folds)):
        if fold_index not in config.selected_fold:
            continue
        num_train_classes = num_classes[0]
        pth_path = os.path.join(config.save_path, config.model_name, '{}_fold{}_best.pth'.format(config.model_name, fold_index))
        # 注意fold之间的因为类别数不同所以模型也不同，所以均要实例化TrainVal
        if demo_on_baseline:
            _, num_train_classes = get_baseline_loader(train_dataset_root, config.batch_size, config.num_workers,
                                                            True, mean, std)
            pth_path = os.path.join(config.save_path, config.model_name,
                                    '{}.pth'.format(config.model_name))
        create_submission = Demo(config, num_train_classes, pth_path, valid_loader, num_query)
        create_submission.get_result(show=True)

