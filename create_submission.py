import torch
import torch.nn as nn
import os
import glob
import json
import codecs
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset.data import TestDataset
from main import parse_args
from config import cfg
from dataset.data import get_loaders
from evaluate import eval_func, euclidean_dist, re_rank
from utils import setup_logger
from model import build_model, convert_model

# 每一个查询样本从数据库中取出最近的10个样本
num_choose = 200
root = 'dataset/NAIC_data/初赛A榜测试集'
pic_path_query = os.path.join(root, 'query_a')
pic_path_gallery = os.path.join(root, 'gallery_a')

pic_list_query = glob.glob(pic_path_query + '/*.png')
pic_list_gallery = glob.glob(pic_path_gallery + '/*.png')

pic_list = pic_list_query + pic_list_gallery

test_dataset = TestDataset(pic_list)
num_query = len(pic_list_query)

val_dl = DataLoader(test_dataset, batch_size=8, num_workers=8, pin_memory=True, shuffle=False)
args = parse_args()
if args.config_file != "":
    cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)
cfg.freeze()

logger = setup_logger('reid_baseline.eval', cfg.OUTPUT_DIR, 0, train=False)

logger.info('Running with config:\n{}'.format(cfg))

train_dataloader_folds, valid_dataloader_folds, num_query_folds, num_classes_folds = get_loaders(cfg.DATASETS.DATA_PATH,
                                                                                                 n_splits=3,
                                                                                                 batch_size=32,
                                                                                                 num_works=8,
                                                                                                 shuffle_train=True)
num_classes = num_classes_folds[0]

model = build_model(cfg, num_classes)
if cfg.TEST.MULTI_GPU:
    model = nn.DataParallel(model)
    model = convert_model(model)
    logger.info('Use multi gpu to inference')
para_dict = torch.load(cfg.TEST.WEIGHT)
model.load_state_dict(para_dict)
model.cuda()
model.eval()

feats, paths = [], []
with torch.no_grad():
    for batch in tqdm(val_dl, total=len(val_dl), leave=False):
        data, path = batch
        paths.extend(path)
        data = data.cuda()
        feat = model(data).detach().cpu()
        feats.append(feat)
feats = torch.cat(feats, dim=0)

query_feat = feats[:num_query]
query_path = np.array(paths[:num_query])

gallery_feat = feats[num_query:]
gallery_path = np.array(paths[num_query:])

# distmat = euclidean_dist(query_feat, gallery_feat)
distmat = re_rank(query_feat, gallery_feat)

result = {}
for query_index, query_dist in enumerate(distmat):
    choose_index = np.argsort(query_dist)[:num_choose]
    query_name = query_path[query_index]
    gallery_names = gallery_path[choose_index]
    result[query_name] = gallery_names.tolist()

print(result)
with codecs.open('./result.json', 'w', "utf-8") as json_file:
    json.dump(result, json_file, ensure_ascii=False)
