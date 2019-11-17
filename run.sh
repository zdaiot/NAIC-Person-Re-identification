model=resnet50

python train.py --optimizer_name=Adam --base_lr=1e-4 --scheduler_name=COS --cos_max=20
time=$(date "+%m-%d %H:%M:%S")
mv checkpoints/${model} checkpoints/${model}_"${time}"

python train.py --optimizer_name=Adam --base_lr=1e-4 --scheduler_name=StepLR --step=20
time=$(date "+%m-%d %H:%M:%S")
mv checkpoints/${model} checkpoints/${model}_"${time}"

python train.py --optimizer_name=SGD --base_lr=5e-2 --scheduler_name=COS --cos_max=20
time=$(date "+%m-%d %H:%M:%S")
mv checkpoints/${model} checkpoints/${model}_"${time}"

python train.py --optimizer_name=SGD --base_lr=5e-2 --scheduler_name=StepLR --step=20
time=$(date "+%m-%d %H:%M:%S")
mv checkpoints/${model} checkpoints/${model}_"${time}"
