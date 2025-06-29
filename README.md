# Official Implementation of CVPR'25 paper "Rethinking Epistemic and Aleatoric Uncertainty for Active Open-Set Annotation: An Energy-Based Approach"

by **Chen-Chen Zong, Sheng-Jun Huang**

[[Paper link]](https://openaccess.thecvf.com/content/CVPR2025/html/Zong_Rethinking_Epistemic_and_Aleatoric_Uncertainty_for_Active_Open-Set_Annotation_An_CVPR_2025_paper.html)

## Abstract

Active learning (AL), which iteratively queries the most informative examples from a large pool of unlabeled candidates for model training, faces significant challenges in the presence of open-set classes. Existing methods either prioritize query examples likely to belong to known classes, indicating low epistemic uncertainty (EU), or focus on querying those with highly uncertain predictions, reflecting high aleatoric uncertainty (AU). However, they both yield suboptimal performance, as low EU corresponds to limited useful information, and closed-set AU metrics for unknown class examples are less meaningful. In this paper, we propose an Energy-based Active Open-set Annotation (EAOA) framework, which effectively integrates EU and AU to achieve superior performance. EAOA features a (C+1)-class detector and a target classifier, incorporating an energy-based EU measure and a margin-based energy loss designed for the detector, alongside an energy-based AU measure for the target classifier. Another crucial component is the target-driven adaptive sampling strategy. It first forms a smaller candidate set with low EU scores to ensure closed-set properties, making AU metrics meaningful. Subsequently, examples with high AU scores are queried to form the final query set, with the candidate set size adjusted adaptively. Extensive experiments show that EAOA achieves state-of-the-art performance while maintaining high query precision and low training overhead.


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

## Citation

If you find this repo useful for your research, please consider citing the paper.

```bibtex
@misc{zong2025rethinkingepistemicaleatoricuncertainty,
      title={Rethinking Epistemic and Aleatoric Uncertainty for Active Open-Set Annotation: An Energy-Based Approach}, 
      author={Chen-Chen Zong and Sheng-Jun Huang},
      year={2025},
      eprint={2502.19691},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.19691}, 
}
```

## Acknowledgement

Thanks to Safaei et al. for publishing their code for [EOAL](https://github.com/bardisafa/EOAL). Our implementation is heavily based on their work.
