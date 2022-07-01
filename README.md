<h2 align="center">
<p> Generalizing to Evolving Domains with Latent Structure-Aware Sequential Autoencoder</p>
</h2>

<div align="center">

[![](https://img.shields.io/github/stars/WonderSeven/LSSAE)](https://github.com/WonderSeven/LSSAE)
[![](https://img.shields.io/github/forks/WonderSeven/LSSAE)](https://github.com/WonderSeven/LSSAE)
[![](https://img.shields.io/github/license/WonderSeven/LSSAE)](https://github.com/WonderSeven/LSSAE/blob/main/LICENSE)

</div>

---

<p align="center">
    <img width=400 src="./figs/DAG.png">
    <!-- <br>Fig 1. The overview of network architecture for LSSAE.</br> -->
</p>

This repo contains official PyTorch implementation of:

- [Generalizing to Evolving Domains with Latent Structure-Aware Sequential Autoencoder](https://arxiv.org/abs/2205.07649) (ICML 2022) 
  <br>Tiexin Qin, Shiqi Wang and Haoliang Li.</br>


<!-- <center>
<img src="./figs/framework.png" width="90%" height="50%" />
</center> -->




### LSSAE

<p align="center">
    <img width=500 src="./figs/framework_LSSAE.png">
    <!-- <br>Fig 1. .</br> -->
</p>

> The overview of network architecture for LSSAE.

<br>

### News:
1. [2022-6-30] WEIFZH reports an [issue](https://github.com/WonderSeven/LSSAE/issues/2) for reproduction, so I add a table for recording test results under different envs

<br>

### Datasets
- [Circle](https://drive.google.com/file/d/1kWyunwxMXGJI5lARqTuJUFP8_gZ3nFA-/view?usp=sharing)/[Circle-C](https://drive.google.com/file/d/1LM2aWS-d4d47syWROkM57oI2AGZ-hnD2/view?usp=sharing)  [![DOI](https://zenodo.org/badge/DOI/10.1007/978-3-319-46227-1_7.svg)](https://doi.org/10.1007/978-3-319-46227-1_7)
- [Sine](https://drive.google.com/file/d/1E0Z4wxPjQKvWESlZdmt70A6B9SBOXSsw/view?usp=sharing)/[Sine-C](https://drive.google.com/file/d/1l15E_RX9zlvicSYur_Bwdqm7t-LbcKri/view?usp=sharing) [![DOI](https://zenodo.org/badge/DOI/10.1007/978-3-319-46227-1_7.svg)](https://doi.org/10.1007/978-3-319-46227-1_7)
- [RMNIST](http://yann.lecun.com/exdb/mnist/) [![DOI](https://zenodo.org/badge/DOI/10.1109/ICCV.2015.293.svg)](https://doi.org/10.1109/ICCV.2015.293)
- [Portraits](https://drive.google.com/file/d/1nvKn2pwaU6vr7Zmo6DTSts2i5Ik_--DW/view?usp=sharing) [![DOI](https://zenodo.org/badge/DOI/10.1109/TCI.2017.2699865.svg)](https://doi.org/10.1109/TCI.2017.2699865)
- [Caltran](https://drive.google.com/file/d/1x-23eDB1ksE2qKDbpA8vwmBRsWD6jiJw/view?usp=sharing) [![DOI](https://zenodo.org/badge/DOI/10.1109/CVPR.2014.116.svg)](https://doi.org/10.1109/CVPR.2014.116)
- [PowerSupply](https://drive.google.com/file/d/11AXm-kcSWk2LBhaNEMm56UVm7Evhj793/view?usp=sharing) [![DOI](https://zenodo.org/badge/DOI/10.1109/JAS.2019.1911747.svg)](https://doi.org/10.1109/JAS.2019.1911747)

We provide the Google Drive links here, so you can download these datasets directly and move them to your own file path for storage.


### Requirements

- python 3.8
- Pytorch 1.11 or above
- Pyyaml
- tqdm

### Quick Start

#### 1. Toy Circle/-C
```
cd ./LSSAE
chmod +x ./scripts/*
./scripts/train_circle.sh or ./scripts/train_circle_c.sh
```


#### 2. Other Datasets

We can copy the script of train_circle.sh directly, then change the file path for '--data_path', such as 

```
# RMNIST
cp ./scripts/train_circle.sh  ./scripts/train_rmnist.sh
--data_path "/data/DataSets" # the root of rmnist
```

For different datasets, the feature_extractor (model_func in our implementation), classifier (cla_func in our implementation) and hyper-parameters need to be specified. We provide the detailed description of network architectures and most of the hyper-parameters in our [Appendix](https://arxiv.org/abs/2205.07649). As this is a reproduced version by myself, I cannot keep the code totally unchanged, so the results could be a little different.


### Test Environment

| Platform | CUDA Driver| CUDA Version | Python Version | Pytorch Version | Saved Ckpt | Status |
| :------: | :--------: | :----------: | :------------: | :-------------: | :--------: | :----: |
|  Ubantu  | 470.57.02  |     11.3     |      3.8.3     |      1.11       |[./logs/ToyCircle/0](./logs/ToyCircle/0)| :heavy_check_mark:|
| Window10 |   512.96   |     11.6     |      3.8.3     |      1.12       | TODO | - |




### Citation    
If you find this repo useful for your research, please cite the following paper:

    @inproceedings{Qin2022LSSAE,
    title={Generalizing to Evolving Domains with Latent Structure-Aware Sequential Autoencoder},
    author={Tiexin Qin and Shiqi Wang and Haoliang Li},
    booktitle={ICML},
    year={2022}

---

### Acknowledgments

Our codes are influenced by the following repos: [DomainBed](https://github.com/facebookresearch/DomainBed) and [Disentangled Sequential Autoencoder](https://github.com/yatindandi/Disentangled-Sequential-Autoencoder).




