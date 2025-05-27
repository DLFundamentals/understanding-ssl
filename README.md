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

Finally, we empirically validate our theoretical findings: the gap between CL and NSCL losses decays at a rate of $\mathcal{O}(\frac{1}{\#\text{classes}})$; the two losses are highly correlated; minimizing the CL loss implicitly brings the NSCL loss close to the value achieved by direct minimization; and the proposed few-shot error bound provides a tight estimate of probing performance in practice.

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

We provide documentation for [pretraining]() SSL models, [linear probing]() and [evaluation]() scripts used for reproducing experiments shown in our paper. 

In the following section, we have defined the code directory structure for easily understanding our work and reproducing our experiments:

```text
understanding-ssl/
| —— configs/                            # Config files for different algorithms and datasets
|    | —— moco.yaml                          
|    | —— datasets.yaml
|    | —— simclr.yaml
| —— data/
|    | —— cifar10/
| —— docs/
| —— | —— / evaluation.md
| —— | —— / linear_probing.md
| —— | —— / pretraining.md
| —— experiments/                        # Experiments directory 
|    | —— moco/
|    |    | —— logs/
|    |    | —— checkpoints/
|    |    | —— visualizations/
|    | —— simclr/
|    |    | —— logs/
|    |    | —— checkpoints/
|    |    | —— visualizations/
| —— models/                           
|    | —— base_encoder.py               # Backbones (ResNet50, ViT, etc.)
|    | —— moco.py                       # moco-specifc model
|    | —— projector.py                  # Projection head
|    | —— simclr.py                     # SimCLR-specific model
| —— notebooks/                         # Jupyter notebooks for random experiments
|    | —— simclr.ipynb
|    | —— moco.ipynb
| —— scripts/                        
|    | —— evaluate.py
|    | —— train.py
|    | —— visualize.py
| —— utils/                             # Utility functions
|    | —— augmentations.py
|    | —— dataset_loader.py
|    | —— losses.py
|    | —— metrics.py
| —— README.md
| —— requirements.txt
```
