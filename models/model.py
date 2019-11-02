from models.baseline import Baseline


def build_model(model_name, num_classes, last_stride, model_pretrain_path='dataset/NAIC_data/resnet50-19c8e357.pth'):
    if model_name == 'resnet50':
        model = Baseline(num_classes, last_stride, model_pretrain_path)
    return model

