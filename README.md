Generalizing to Evolving Domains with Latent Structure-Aware Sequential Autoencoder
---

### Introduction
This reposity contains the official implementation for the paper:

[Generalizing to Evolving Domains with Latent Structure-Aware Sequential Autoencoder](*)

Tiexin Qin, Haoliang Li and Shiqi Wang.


<!-- <center>
<img src="./figs/framework.png" width="90%" height="50%" />
</center> -->


### Abstract
Domain generalization aims to improve the generalization capability of machine learning systems to out-of-distribution (OOD) data. Existing domain generalization techniques embark upon stationary and discrete environments to tackle the generalization issue caused by OOD data. However, many real-world tasks in non-stationary environments (e.g. self-driven car system, sensor measures) involve more complex and continuously evolving domain drift, which raises new challenges for the problem of domain generalization. In this paper, we formulate the aforementioned setting as the problem of evolving domain generalization. Specifically, we propose to introduce a probabilistic framework called Latent Structure-aware Sequential Autoencoder (LSSAE) to tackle the problem of evolving domain generalization via exploring the underlying continuous structure in the latent space of deep neural networks, where we aim to identify two major factors namely covariate shift and concept shift accounting for distribution shift in non-stationary environments. Experimental results on both synthetic and real-world datasets show that LSSAE can lead to superior performances based on the evolving domain generalization setting.

### Datasets
- [Circle](https://drive.google.com/file/d/1kWyunwxMXGJI5lARqTuJUFP8_gZ3nFA-/view?usp=sharing)/[Circle-C](https://drive.google.com/file/d/1LM2aWS-d4d47syWROkM57oI2AGZ-hnD2/view?usp=sharing)
- [Sine](https://drive.google.com/file/d/1E0Z4wxPjQKvWESlZdmt70A6B9SBOXSsw/view?usp=sharing)/[Sine-C](https://drive.google.com/file/d/1l15E_RX9zlvicSYur_Bwdqm7t-LbcKri/view?usp=sharing)
- [RMNIST](http://yann.lecun.com/exdb/mnist/)
- [Portraits](https://drive.google.com/file/d/1nvKn2pwaU6vr7Zmo6DTSts2i5Ik_--DW/view?usp=sharing)
- [Caltran](https://drive.google.com/file/d/1x-23eDB1ksE2qKDbpA8vwmBRsWD6jiJw/view?usp=sharing)
- [PowerSupply](https://drive.google.com/file/d/11AXm-kcSWk2LBhaNEMm56UVm7Evhj793/view?usp=sharing)


---

#### Note
Currently, I am still organizing my code, I will update this reponsitory once I finish my organization.
