# Official Implementation of CVPR'25 paper "Rethinking Epistemic and Aleatoric Uncertainty for Active Open-Set Annotation: An Energy-Based Approach"

by **Chen-Chen Zong, Sheng-Jun Huang**

## Run 
### CIFAR-10

```
python main.py --query-strategy eaoa_sampling --init-percent 1 --known-class 2 --query-batch 1500 --seed 1 --model resnet18 --dataset cifar10 --max-query 11 --max-epoch 200 --gpu 0

python main.py --query-strategy eaoa_sampling --init-percent 1 --known-class 3 --query-batch 1500 --seed 1 --model resnet18 --dataset cifar10 --max-query 11 --max-epoch 200 --gpu 0

python main.py --query-strategy eaoa_sampling --init-percent 1 --known-class 4 --query-batch 1500 --seed 1 --model resnet18 --dataset cifar10 --max-query 11 --max-epoch 200 --gpu 0
```


### CIFAR-100

```
python main.py --query-strategy eaoa_sampling --init-percent 8 --known-class 20 --query-batch 1500 --seed 1 --model resnet18 --dataset cifar100 --max-query 11 --max-epoch 200 --gpu 0

python main.py --query-strategy eaoa_sampling --init-percent 8 --known-class 30 --query-batch 1500 --seed 1 --model resnet18 --dataset cifar100 --max-query 11 --max-epoch 200 --gpu 0

python main.py --query-strategy eaoa_sampling --init-percent 8 --known-class 40 --query-batch 1500 --seed 1 --model resnet18 --dataset cifar100 --max-query 11 --max-epoch 200 --gpu 0
```


### Tiny-ImageNet

```
python main.py --query-strategy eaoa_sampling --init-percent 8 --known-class 20 --query-batch 1500 --seed 1 --model resnet18 --dataset tinyimagenet --max-query 11 --max-epoch 200 --gpu 0

python main.py --query-strategy eaoa_sampling --init-percent 8 --known-class 30 --query-batch 1500 --seed 1 --model resnet18 --dataset tinyimagenet --max-query 11 --max-epoch 200 --gpu 0

python main.py --query-strategy eaoa_sampling --init-percent 8 --known-class 40 --query-batch 1500 --seed 1 --model resnet18 --dataset tinyimagenet --max-query 11 --max-epoch 200 --gpu 0
```
