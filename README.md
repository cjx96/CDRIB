CDRIB
===

The source code is for the paper: “Cross-Domain Recommendation to Cold-Start Users
via Variational Information Bottleneck” accepted in ICDE 2022 by Jiangxia Cao, Jiawei Sheng, Xin Cong, Tingwen Liu and Bin Wang.

```
@inproceedings{cao2022cdrib,
  title={Cross-Domain Recommendation to Cold-Start Users via Variational Information Bottleneck},
  author={Cao, Jiangxia and Sheng, Jiawei and Cong, Xin and Liu, Tingwen and Wang, Bin},
  booktitle={IEEE International Conference on Data Engineering (ICDE)},
  year={2022}
}
```

Requirements
---

Python=3.7.9

PyTorch=1.6.0

Scipy = 1.5.2

Numpy = 1.19.1

Usage
---

To run this project, please make sure that you have the following packages being downloaded. Our experiments are conducted on a PC with an Intel Xeon E5 2.1GHz CPU, 256 RAM and a Tesla V100 32GB GPU. 

Running example:

```shell
CUDA_VISIBLE_DEVICES=1 python -u train_rec.py --id gv --dataset game_video
```


