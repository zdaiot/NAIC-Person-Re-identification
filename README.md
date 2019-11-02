## Prepare Dataset
Download dataset, unzip and put them into `../Input` directory.

Structure of the ../Input folder can be like:
```bash

```
Create soft links of datasets in the following directories:

```bash
cd dataset/NAIC_data
ln -s ../../../Input/初赛训练集/ ./初赛训练集
ln -s ../../../Input/初赛A榜测试集/ ./初赛A榜测试集
ln -s ../../../Input/submission_example_A.json ./submission_example_A.json
# TODO
ln -s ~/.cache/torch/checkpoints/resnet50-19c8e357.pth ./ 
``` 

```bash
pip install pyyaml
pip install visdom
```

## Usage

1. Clone the repo using `git clone `
2. Compile the code for Cython accelerated evaluation code `cd evaluate/eval_cylib && make`
3. the [SyncBN](https://github.com/zdaiot/Synchronized-BatchNorm-PyTorch) module is pure pytorch implementation, so no need to compile once you have pytorch.
4. Modify the training config in configs folder.
5. Start training:

```bash

```


```bash
python evaluate/eval_cylib/test_cython.py
```