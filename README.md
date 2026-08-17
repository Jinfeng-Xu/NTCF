# Rethinking Neural Graph Collaborative Filtering as Neural Tree Collaborative Filtering with Bakry-Emery Curvature-Aware Propagation Depth

## Requirements
```
Python>=3.9.18
Pytorch>=1.13.1
```

## Dataset

| Datasets  | #Users | #Items | #Interactions | Sparsity |
| --------- | ------ | ------ | ------------- | -------- |
| Kindle    | 60,468 | 57,212 | 880,859       | 99.975%  |
| Pinterest | 55,188 | 9,912  | 1,445,622     | 99.736%  |
| Yelp      | 45,477 | 30,708 | 1,777,765     | 99.873%  |


## Training
```
python main.py
```

## Citing NTCF

If you find NTCF useful in your research, please consider citing our [NTCF paper](https://arxiv.org/pdf/2608.10297).

```
@article{xu2026neural,
  title={Neural Tree Collaborative Filtering: Rethinking Graph Collaborative Filtering as Tree Collaborative Filtering with Curvature-Aware Propagation Depth},
  author={Xu, Jinfeng and Chen, Zheyu and Peng, Ziyue and Yang, Shuo and Li, Jinze and Yuan, Wenhao and Chen, Jian and Ngai, Edith CH},
  journal={arXiv preprint arXiv:2608.10297},
  year={2026}
}
```
