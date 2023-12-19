# Inferring Hybrid Neural Fluid Fields from Videos
This is the official code for Inferring Hybrid Neural Fluid Fields from Videos (NeurIPS 2023).

![teaser](assets/demo.gif)

**[[Paper](https://arxiv.org/pdf/2312.06561.pdf)] [[Project Page](https://kovenyu.com/hyfluid/)]**

## Installation
Install with conda:
```bash
conda env create -f environment.yml
conda activate hyfluid
```
or with pip:
```bash
pip install -r requirements.txt
```

## Data
The demo data is available at [data/ScalarReal](data/ScalarReal). 
The full ScalarFlow dataset can be downloaded [here](https://ge.in.tum.de/publications/2019-scalarflow-eckert/).

## Quick Start
To learn the hybrid neural fluid fields from the demo data, firstly reconstruct the density field by running (~40min):
```bash
bash scripts/train.sh
```
Then, reconstruct the velocity field by jointly training with the density field (~15 hours on a single A6000 GPU.):
```bash
bash scripts/train_j.sh
```
Finally, add vortex particles and optimize their physical parameters (~40min):
```bash
bash scripts/train_vort.sh
```
The results will be saved in `./logs/exp_real`. With the learned hybrid neural fluid fields, you can re-simulate the fluid by using the velocity fields to advect density:
```bash
bash scripts/test_resim.sh
```
Or, you can predict the future states by extrapolating the velocity fields:
```bash
bash scripts/test_future_pred.sh
```

## Citation
If you find this code useful for your research, please cite our paper:
```
@article{yu2023inferring,
  title={Inferring Hybrid Neural Fluid Fields from Videos},
  author={Yu, Hong-Xing and Zheng, Yang and Gao, Yuan and Deng, Yitong and Zhu, Bo and Wu, Jiajun},
  journal={NeurIPS},
  year={2023}
}
```