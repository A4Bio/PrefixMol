# PrefixMol: Target- and Chemistry-aware Molecule Design via Prefix Embedding

## 📢 News
  - Code: https://github.com/ALEEEHU/MolDesign
  - Paper: https://arxiv.org/abs/2302.07120

<img src="./assets/overall_framework.jpg" alt="overall_framework"  width="70%"/>
We propose PrefixMol, inserting learnable conditional feature vectors into the attention module to unify multi-conditional molecule generative models to support the modeling of customized requirements.

## Installation

### Dependency
The codes have been tested in the following environment:
Package  | Version
--- | ---
Python | 3.7.12
PyTorch | 1.10.0
CUDA | 11.3.1
PyTorch Geometric | 2.0.3
RDKit | 2021.09.4 

### Install via conda yaml file (cuda 11.3)
```bash
conda env create -f env_cuda113.yml
conda activate PrefixMol
```

### Install manually

``` bash
conda create -n PrefixMol python=3.7.12
conda activate PrefixMol

# Install PyTorch (for cuda 11.3)
conda install pytorch==1.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
# Install PyTorch Geometric (>=2.0.0)
conda install pyg -c pyg

# Install other tools
conda install -c conda-forge rdkit
conda install pyyaml easydict python-lmdb -c conda-forge
```
## Datasets

Please refer to [`README.md`](./data/README.md) in the `data` folder.

## Training

```
python train.py --config ./configs/train.yml --logdir ./logs
```

## Testing

```
python train.py --config ./configs/train.yml --logdir ./logs
```


## Citation
```
@article{gao2023prefixmol,
  title={PrefixMol: Target-and Chemistry-aware Molecule Design via Prefix Embedding},
  author={Gao, Zhangyang and Hu, Yuqi and Tan, Cheng and Li, Stan Z},
  journal={arXiv preprint arXiv:2302.07120},
  year={2023}
}
```

## Contact 
Zhangyang Gao (gaozhangyang@westlake.edu.cn)
Yuqi Hu (hyqale1024@gmail.com)

