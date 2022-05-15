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
Domain generalization aims to improve the generalization capability of machine learning systems to out-of-distribution (OOD) data. Existing domain generalization techniques embark upon stationary and discrete environments to tackle the generalization issue caused by OOD data. However, many real-world tasks in non-stationary environments ~(\eg~self-driven car system, sensor measures) involve more complex and continuously evolving domain drift, which raises new challenges for the problem of domain generalization. In this paper, we formulate the aforementioned setting as the problem of \emph{evolving domain generalization}. Specifically, we propose to introduce a probabilistic framework called Latent Structure-aware Sequential Autoencoder~(LSSAE) to tackle the problem of evolving domain generalization via exploring the underlying continuous structure in the latent space of deep neural networks, where we aim to identify two major factors namely \emph{covariate shift} and \emph{concept shift} accounting for distribution shift in non-stationary environments. Experimental results on both synthetic and real-world datasets show that LSSAE can lead to superior performances based on the evolving domain generalization setting.
