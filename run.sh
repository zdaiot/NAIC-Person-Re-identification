model=resnet50

python train.py --model=${model}  --optimizer_name=SGD --scheduler_name=Cosine --config_lr_optim_path='./config1.json'
time=$(date "+%m-%d %H:%M:%S")
mv checkpoints/${model} checkpoints/${model}_"${time}"

python train.py --model=${model}  --optimizer_name=SGD --scheduler_name=StepLR --config_lr_optim_path='./config2.json'
time=$(date "+%m-%d %H:%M:%S")
mv checkpoints/${model} checkpoints/${model}_"${time}"