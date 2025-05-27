# Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning

In [our work](https://github.com/DLFundamentals/understanding-ssl), we make progress towards addressing the following question:

> **How does self-supervsied CL learn representations similar to supervised learning, despite lacking explicit supervision?**

We acknowledge the following works for their open-source contributions:

* [SimCLR](https://github.com/google-research/simclr)
* [SimCLR PyTorch](https://github.com/Spijkervet/SimCLR/tree/master)
* [MoCo](https://github.com/facebookresearch/moco)
* [LightlySSL](https://github.com/lightly-ai/lightly)

## Abstract

Despite its empirical success, the theoretical foundations of self-supervised contrastive learning (CL) are not yet fully established. In this work, we address this gap by showing that standard CL objectives implicitly approximate a supervised variant we call the negatives-only supervised contrastive loss (NSCL), which excludes same-class contrasts. We prove that the gap between the CL and NSCL losses vanishes as the number of semantic classes increases, under a bound that is both label-agnostic and architecture-independent.

We characterize the geometric structure of the global minimizers of the NSCL loss: the learned representations exhibit augmentation collapse, within-class collapse, and class centers that form a simplex equiangular tight frame. We further introduce a new bound on the few-shot error of linear-probing. This bound depends on two measures of feature variability—within-class dispersion and variation along the line between class centers. We show that directional variation dominates the bound and that the within-class dispersion's effect diminishes as the number of labeled samples increases. These properties enable CL and NSCL-trained representations to support accurate few-shot label recovery using simple linear probes.

Finally, we empirically validate our theoretical findings: the gap between CL and NSCL losses decays at a rate of $\mathcal{O}(\frac{1}{\text{classes}})$; the two losses are highly correlated; minimizing the CL loss implicitly brings the NSCL loss close to the value achieved by direct minimization; and the proposed few-shot error bound provides a tight estimate of probing performance in practice.

## Instructions

To get started, follow these steps:

```bash
git clone https://github.com/DLFundamentals/understanding-ssl.git
cd understanding-ssl
```

The packages that we use are straightforward to install. Please run the following command:

```bash
conda env create -f requirements.yml
conda activate contrastive
```

We provide documentation for [pretraining](https://github.com/DLFundamentals/understanding-ssl/blob/main/docs/pretraining.md) SSL models, [linear probing](https://github.com/DLFundamentals/understanding-ssl/blob/main/docs/linear_probing.md) pretrained encoders, and [evaluation](https://github.com/DLFundamentals/understanding-ssl/blob/main/docs/evaluation.md) scripts used for reproducing experiments shown in our paper.

## License

This project is licensed under the [Apache-2.0](https://github.com/DLFundamentals/understanding-ssl?tab=Apache-2.0-1-ov-file) license.

## 📚 Citation

If you find our work useful in your research or applications, please cite us using the following BibTeX:

```bibtex
@article{yourkey2025, 
title = {Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning}, 
author = {Luthra, Achleshwar and Yang, Tianbao and Galanti, Tomer},
journal = {arxiv TODO}, 
year = {2025},
}
```