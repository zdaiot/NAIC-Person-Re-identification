## Requirements
- Pytorch 1.1.0
- Torchvision 0.3.0
- Python3.7
- pretrainedmodels

## How to run
### Clone Our Project
```bash
git clone https://github.com/zdaiot/NAIC-Person-Re-identification.git
cd NAIC-Person-Re-identification
```

### Prepare Dataset
Download dataset, unzip and put them into `../Input` directory.

Structure of the ../Input folder can be like:
```bash
初赛训练集
初赛A榜测试集
submission_example_A.json
```
Create soft links of datasets in the following directories:

```bash
cd dataset
mkdir NAIC_data
cd NAIC_data
ln -s ../../../Input/初赛训练集/ ./初赛训练集
ln -s ../../../Input/初赛A榜测试集/ ./初赛A榜测试集
ln -s ../../../Input/submission_example_A.json ./submission_example_A.json
# TODO
ln -s ~/.cache/torch/checkpoints/resnet50-19c8e357.pth ./ 
``` 

### Make Eval Code

```bash
pip install Cython
cd evaluate/eval_cylib
make
# if you want to test the code
cd ../..
python evaluate/eval_cylib/test_cython.py
```

### Train baseline
```bash
python train_baseline.py
```

### Train Cross validation
```bash
python train.py
```