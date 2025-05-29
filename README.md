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

## Installation

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

## Pretraining SSL models

### Training on Single GPU

Run the following command to train SimCLR on single GPU.

```bash
python scripts/train.py --config <path-to-yaml-config>
```

### Distributed Training on Multiple GPUs

Run the following command to train SimCLR on multiple GPUs.
> **NOTE:** In our experiments, we used 2 GPUs for training. You can adjust the number of GPUs based on your hadrware setup.

```bash
torchrun --nproc_per_node=N_GPUs --standalone scripts/multigpu_train_simclr.py --config <path-to-yaml-config>
```

Replace `N_GPUs` with the number of GPUs you want to use and `<path-to-yaml-config>` with the path to your configuration file.

Please refer to [docs/pretraining](https://github.com/DLFundamentals/understanding-ssl/blob/main/docs/pretraining.md) for more details.

## Linear Probing

To evaluate pretrained encoders via linear probing, you can run:

```bash
python scripts/linear_probe.py --config <path-to-config-file> --ckpt_path <path-to-ckpt-dir> --output_path <path-to-save-logs> --N <n_samples>
```

For example,

```bash
python scripts/linear_probe.py --config configs/simclr_DCL_cifar10_b1024.yaml --ckpt_path experiments/simclr/cifar10_dcl/checkpoints/ --output_path logs/cifar10/ --N 500
```

## Evaluation

To validate our Theorem [1], you can run:

```bash
python scripts/losses_eval.py --config <path-to-config-file> --ckpt_path <path-to-ckpt-dir> --output_path <path-to-save-logs>
```

For example,

```bash
python scripts/losses_eval.py --config configs/simclr_DCL_cifar10_b1024.yaml --ckpt_path experiments/simclr/cifar10_dcl/checkpoints/ --output_path logs/cifar10/simclr/exp1/
```

This will log `losses.csv` file to your `output_path` directory. You can analyse losses as a function of epochs and verify our proposed bound.

Please refer to [docs/evaluation](https://github.com/DLFundamentals/understanding-ssl/blob/main/docs/evaluation.md) scripts for reproducing additional experiments shown in our paper.

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
